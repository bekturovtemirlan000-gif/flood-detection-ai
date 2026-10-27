"""Download a deterministic subset of official Sen1Floods11 splits and train."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

os.environ.setdefault('OMP_NUM_THREADS', '4')
import joblib
import numpy as np
from PIL import Image, ImageDraw
import rasterio
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from flood import features, probability

BASE = 'https://storage.googleapis.com/sen1floods11/v1.1/'
DATA = Path('data')
SEED = 42


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
                temp = path.with_suffix(path.suffix + '.tmp')
                temp.write_bytes(content)
                temp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return path


def select(split, count):
    path = download(BASE + f'splits/flood_handlabeled/flood_{split}_data.csv', DATA / f'{split}.csv')
    rows = [r for r in csv.reader(io.StringIO(path.read_text())) if r]
    # Round-robin events after deterministic shuffling avoids taking just the first country.
    rng = np.random.default_rng(SEED)
    groups = {}
    for row in rows:
        groups.setdefault(row[0].split('_')[0], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while any(groups.values()) and len(chosen) < count:
        for event in sorted(groups):
            if groups[event] and len(chosen) < count:
                chosen.append(groups[event].pop())
    return chosen


def fetch_pair(row):
    for name, layer in zip(row, ('S1Hand', 'LabelHand')):
        download(BASE + f'data/flood_events/HandLabeled/{layer}/{name}', DATA / name)


def read_pair(row):
    x, valid, profile = features(DATA / row[0])
    with rasterio.open(DATA / row[1]) as src:
        y = src.read(1)
        assert (src.transform.almost_equals(profile['transform'], precision=1e-10)
                and src.crs == profile['crs'] and src.shape == valid.shape)
    valid &= np.isin(y, [0, 1])
    return x, y, valid


def scores(y, pred):
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return {'water_iou': tp / max(tp + fp + fn, 1),
            'water_f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'accuracy': (tp + tn) / max(tp + fp + fn + tn, 1),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate(model, rows):
    labels, probs = [], []
    for row in rows:
        x, y, valid = read_pair(row)
        labels.append(y[valid])
        probs.append(probability(model, x[valid.ravel()]))
    return np.concatenate(labels), np.concatenate(probs)


def preview(model, threshold, row):
    x, y, valid = read_pair(row)
    pred = probability(model, x).reshape(y.shape) >= threshold
    with rasterio.open(DATA / row[0]) as src:
        vv = src.read(1)
    gray = (np.clip((np.nan_to_num(vv) + 25) / 25, 0, 1) * 255).astype('uint8')
    radar = np.repeat(gray[..., None], 3, axis=2)
    panels = [radar]
    for m in (y == 1, pred):
        rgb = radar.copy()
        rgb[m] = (20, 180, 240)
        rgb[~valid] = (65, 65, 65)
        panels.append(rgb)
    w, h = y.shape[1], y.shape[0]
    canvas = Image.new('RGB', (w * 3, h + 60), '#101d2d')
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, ['Sentinel-1 VV', 'Hand-labeled water', 'Predicted water'])):
        canvas.paste(Image.fromarray(panel), (i * w, 60))
        draw.text((i * w + 15, 15), title, fill='white')
    canvas.save('reports/test_preview.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-chips', type=int, default=96)
    parser.add_argument('--valid-chips', type=int, default=24)
    parser.add_argument('--test-chips', type=int, default=48)
    parser.add_argument('--pixels-per-chip', type=int, default=2000)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error('All counts must be positive')
    start = time.time()
    splits = {s: select(s, n) for s, n in [('train', args.train_chips), ('valid', args.valid_chips), ('test', args.test_chips)]}
    sets = [set(r[0] for r in rows) for rows in splits.values()]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    rows = sum(splits.values(), [])
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(fetch_pair, rows)):
            if i % 12 == 0:
                print(f'Downloaded {i + 1}/{len(rows)} image/label pairs', flush=True)
    print('Extracting training features...', flush=True)
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for row in splits['train']:
        x, y, valid = read_pair(row)
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        indices = rng.choice(indices, min(len(indices), args.pixels_per_chip), replace=False)
        xs.append(x[indices])
        ys.append(y.ravel()[indices])
    x, y = np.concatenate(xs), np.concatenate(ys)
    print(f'Training on {len(y):,} pixels...', flush=True)
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, learning_rate=0.08,
                                           l2_regularization=1, early_stopping=False, random_state=SEED)
    model.fit(x, y)
    print('Selecting threshold on validation images...', flush=True)
    vy, vp = evaluate(model, splits['valid'])
    thresholds = np.arange(0.2, 0.81, 0.05)
    threshold = float(max(thresholds, key=lambda t: scores(vy, vp >= t)['water_iou']))
    print('Evaluating held-out test images...', flush=True)
    ty, tp = evaluate(model, splits['test'])
    Path('models').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    joblib.dump({'model': model, 'threshold': threshold, 'features': 'VV,VH,VV-VH,mean/std3,mean/std7',
                 'sklearn_version': sklearn.__version__}, 'models/water_model.joblib', compress=3)
    manifest = {s: [{'image': row[0], 'label': row[1],
                    'image_sha256': hashlib.sha256((DATA / row[0]).read_bytes()).hexdigest(),
                    'label_sha256': hashlib.sha256((DATA / row[1]).read_bytes()).hexdigest()}
                   for row in rows] for s, rows in splits.items()}
    Path('reports/data_manifest.json').write_text(json.dumps(manifest, indent=2))
    # Choose an interpretable example by labels only, never prediction performance.
    preview_row = next((row for row in splits['test']
                        if preview_eligible(row)), splits['test'][0])
    results = {'dataset': BASE, 'seed': SEED, 'chip_counts': {s: len(r) for s, r in splits.items()},
               'training_pixels': len(y), 'threshold': threshold,
               'validation': scores(vy, vp >= threshold), 'test': scores(ty, tp >= threshold),
               'all_land_test_baseline': scores(ty, np.zeros(len(ty), dtype=bool)),
               'test_preview_chip': preview_row[0],
               'python': platform.python_version(), 'sklearn': sklearn.__version__,
               'elapsed_seconds': round(time.time() - start, 1)}
    Path('reports/metrics.json').write_text(json.dumps(results, indent=2))
    preview(model, threshold, preview_row)
    print(json.dumps(results, indent=2), flush=True)


def preview_eligible(row):
    with rasterio.open(DATA / row[1]) as src:
        y = src.read(1)
    return (y == 1).mean() >= .01 and (y == 0).mean() >= .01


if __name__ == '__main__':
    main()
