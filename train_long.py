"""Time-budgeted CPU training with isolated trials and durable checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from flood import probability
from train import select, fetch_pair, read_pair, evaluate, scores


def stamp():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(tmp, path):
    """Keep atomic writes while tolerating brief Windows reader/sync locks."""
    for attempt in range(9):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    replace_with_retry(tmp, path)


def save_model(path, value):
    tmp = path.with_suffix('.tmp')
    joblib.dump(value, tmp, compress=3)
    replace_with_retry(tmp, path)


def sample(rows, limit, seed, feature_version='v1'):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for row in rows:
        x, y, valid = read_pair(row, feature_version)
        idx = np.flatnonzero(valid)
        if not len(idx):
            continue
        idx = rng.choice(idx, min(limit, len(idx)), replace=False)
        xs.append(x[idx])
        ys.append(y.ravel()[idx])
    return np.concatenate(xs), np.concatenate(ys)


def evaluate_bundle(bundle, rows):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    per_event = {}
    for i, row in enumerate(rows):
        x, y, valid = read_pair(row, bundle.get('feature_version', 'v1'))
        pred = probability(bundle['model'], x[valid.ravel()]) >= bundle['threshold']
        result = scores(y[valid], pred)
        event = per_event.setdefault(row[0].split('_')[0], dict(tp=0, fp=0, fn=0, tn=0))
        for key in counts:
            counts[key] += result[key]
            event[key] += result[key]
        if i % 10 == 0:
            print(f'Evaluated {i + 1}/{len(rows)} chips', flush=True)
    def finish(c):
        tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
        return dict(c, water_iou=tp / max(tp + fp + fn, 1),
                    water_f1=2 * tp / max(2 * tp + fp + fn, 1),
                    precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                    accuracy=(tp + tn) / max(tp + fp + fn + tn, 1))
    return dict(finish(counts), per_event={k: finish(c) for k, c in per_event.items()})


def worker(run, mode):
    settings_path = run / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    feature_version = settings.get('feature_version', 'v1')
    baseline_path = settings.get('baseline', 'models/water_model.joblib')
    if mode == 'evaluate':
        bundle = joblib.load(run / 'best_model.joblib')
        splits = json.loads((run / 'splits.json').read_text(encoding='utf-8'))
        # Only the final chosen model sees test labels.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch_pair, splits['test']))
        report = {'selected_configuration': bundle.get('configuration'), 'threshold': bundle['threshold'],
                  'feature_version': bundle.get('feature_version', 'v1'), 'evaluation_complete': False}
        for name, evaluated_bundle in [('selected', bundle), ('baseline', joblib.load(baseline_path))]:
            report[name] = {}
            for split in ('valid', 'test'):
                print(f'Evaluating {name} on full {split} split', flush=True)
                report[name][split] = evaluate_bundle(evaluated_bundle, splits[split])
                save_json(run / 'final_metrics.json', report)
        report['evaluation_complete'] = True
        report['completed_at'] = stamp()
        save_json(run / 'final_metrics.json', report)
        return
    x = np.load(run / 'train_x.npy', mmap_mode='r')
    y = np.load(run / 'train_y.npy', mmap_mode='r')
    vx = np.load(run / 'valid_x.npy', mmap_mode='r')
    vy = np.load(run / 'valid_y.npy', mmap_mode='r')
    if mode == 'baseline':
        bundle = joblib.load(baseline_path)
        p = probability(bundle['model'], vx[:, :bundle['model'].n_features_in_])
        metrics = scores(vy, p >= bundle['threshold'])
        bundle['sampled_validation'] = metrics
        bundle['configuration'] = dict(bundle.get('configuration', {}), baseline=True)
        save_model(run / 'candidate.joblib', bundle)
        save_json(run / 'candidate.json', {'configuration': bundle['configuration'], 'metrics': metrics})
        return
    config = json.loads((run / 'configuration.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(config['random_state'])
    indices = rng.choice(len(y), min(config['sample_pixels'], len(y)), replace=False)
    params = {k: v for k, v in config.items() if k != 'sample_pixels'}
    model = HistGradientBoostingClassifier(**params, early_stopping=False)
    started = time.time()
    model.fit(x[indices], y[indices])
    p = probability(model, vx)
    threshold = float(max(np.arange(.05, .81, .025), key=lambda t: scores(vy, p >= t)['water_iou']))
    metrics = scores(vy, p >= threshold)
    bundle = {'model': model, 'threshold': threshold, 'configuration': config, 'feature_version': feature_version,
              'sampled_validation': metrics, 'sklearn_version': sklearn.__version__}
    save_model(run / 'candidate.joblib', bundle)
    save_json(run / 'candidate.json', {'configuration': config, 'metrics': metrics,
                                      'threshold': threshold, 'seconds': time.time() - started})


def run_child(run, mode, timeout):
    with (run / 'worker.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', __file__, '--worker', mode, '--run', str(run)],
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            return child.wait(timeout=max(.1, timeout)) == 0
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
            else:
                child.kill()
            child.wait()
            return False


def supervise(run, hours, baseline='models/water_model.joblib', feature_version='v1'):
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + hours * 3600
    state = {'pid': os.getpid(), 'started_at': stamp(), 'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
             'hours': hours, 'status': 'preparing_data', 'completed_trials': 0, 'threads': 4}
    save_json(run / 'status.json', state)
    save_json(run / 'settings.json', {'baseline': str(Path(baseline).resolve()),
                                     'feature_version': feature_version, 'sampling_seed': 42})
    # Capture exactly the training code used in this run for later provenance.
    import shutil
    for source in ('train_long.py', 'train.py', 'flood.py'):
        shutil.copyfile(source, run / ('source_' + source))
    # This temporary request lasts only while this process is alive; no power settings change.
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        splits = {s: select(s, 10000) for s in ('train', 'valid', 'test')}
        sets = [set(r[0] for r in rows) for rows in splits.values()]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
        save_json(run / 'splits.json', splits)
        # Keep the selected checkpoint durable even if preparation is interrupted.
        save_model(run / 'best_model.joblib', joblib.load(baseline))
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, _ in enumerate(pool.map(fetch_pair, splits['train'] + splits['valid'])):
                if i % 20 == 0:
                    print(f'Data pairs ready: {i + 1}', flush=True)
        for split, per_chip in [('train', 12000), ('valid', 4000)]:
            x, y = sample(splits[split], per_chip, 42, feature_version)
            np.save(run / f'{split}_x.npy', x)
            np.save(run / f'{split}_y.npy', y)
            state[f'{split}_pixels'] = len(y)
            del x, y
        state['chip_counts'] = {s: len(rows) for s, rows in splits.items()}
        state['feature_version'] = feature_version
        state['status'] = 'training'
        save_json(run / 'status.json', state)
        print(json.dumps(state), flush=True)
        if not run_child(run, 'baseline', min(300, deadline - time.time())):
            raise RuntimeError('Baseline evaluation failed; inspect worker.log')
        baseline = json.loads((run / 'candidate.json').read_text(encoding='utf-8'))
        best = baseline['metrics']['water_iou']
        save_json(run / 'baseline_validation.json', baseline)
        save_model(run / 'best_model.joblib', joblib.load(run / 'candidate.joblib'))
        # Full evaluation previously took 12 minutes per model; budget both models.
        train_deadline = deadline - min(2400, hours * 3600 * .2)
        configs = list(itertools.product([31, 63], [300, 600, 1000], [.035, .07],
                                         [1., 5.], [750000, 2000000], [None, 'balanced'], [20, 100]))
        np.random.default_rng(20261028).shuffle(configs)
        for trial, (leaves, iterations, rate, l2, pixels, weight, min_leaf) in enumerate(itertools.cycle(configs), 1):
            if time.time() >= train_deadline:
                break
            config = {'max_leaf_nodes': leaves, 'max_iter': iterations, 'learning_rate': rate,
                      'l2_regularization': l2, 'random_state': 42 + trial,
                      'sample_pixels': pixels, 'class_weight': weight, 'min_samples_leaf': min_leaf}
            save_json(run / 'configuration.json', config)
            state.update(current_trial=trial, configuration=config, best_validation_iou=best, updated_at=stamp())
            save_json(run / 'status.json', state)
            ok = run_child(run, 'trial', min(900, train_deadline - time.time()))
            record = {'trial': trial, 'completed_at': stamp(), 'success': ok, 'configuration': config}
            if ok:
                record.update(json.loads((run / 'candidate.json').read_text(encoding='utf-8')))
                state['completed_trials'] += 1
                if record['metrics']['water_iou'] > best:
                    best = record['metrics']['water_iou']
                    save_model(run / 'best_model.joblib', joblib.load(run / 'candidate.joblib'))
                    save_json(run / 'best_validation.json', record)
                print(f'Trial {trial}: validation IoU={record["metrics"]["water_iou"]:.4f}; best={best:.4f}', flush=True)
            with (run / 'trials.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(record) + '\n')
            state.update(best_validation_iou=best, updated_at=stamp())
            save_json(run / 'status.json', state)
            if not ok and time.time() < train_deadline - 30:
                # Abort persistent failures instead of spinning for five hours.
                failures = state.get('consecutive_failures', 0) + 1
                state['consecutive_failures'] = failures
                if failures >= 3:
                    raise RuntimeError('Three consecutive trial failures; inspect worker.log')
            elif ok:
                state['consecutive_failures'] = 0
        state['status'] = 'final_evaluation'
        save_json(run / 'status.json', state)
        ok = run_child(run, 'evaluate', deadline - time.time())
        state['status'] = 'completed' if ok else 'completed_checkpoint_saved_evaluation_incomplete'
        state['finished_at'] = stamp()
        save_json(run / 'status.json', state)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished_at=stamp())
        save_json(run / 'status.json', state)
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=5)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', default='models/water_model.joblib')
    parser.add_argument('--feature-version', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--worker', choices=['baseline', 'trial', 'evaluate'])
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('--hours must be positive')
    if args.worker:
        worker(args.run, args.worker)
    else:
        supervise(args.run, args.hours, args.baseline, args.feature_version)
