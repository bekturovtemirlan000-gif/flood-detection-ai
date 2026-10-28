# Flood Detection AI

Train a satellite surface-water segmentation model and export georeferenced masks.
This project uses **real Sentinel-1 radar imagery and human water labels** from
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).

The trained model is a CPU-friendly histogram gradient boosting classifier, not a
neural network. It classifies each pixel using VV/VH radar backscatter and local
texture. A single scene detects **surface water**, including permanent rivers and
lakes. With an aligned pre-flood scene, the tool reports **candidate newly flooded
pixels** by subtracting previously detected water.

## Setup

Python 3.11 or newer is recommended. Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied model's exact training environment is recorded in
`requirements-lock.txt`. Use that file to reproduce the released model environment;
scikit-learn model serialization is version-sensitive. Load only trusted `.joblib`
files because their format can execute Python code.

## Predict

Input must be a two-band Sentinel-1 GeoTIFF: **band 1 VV, band 2 VH in decibels**,
processed consistently with the Sen1Floods11 dataset. JPEGs, optical RGB images,
linear-power radar values and arbitrary band orders are not supported. The program
checks the band count; the caller must verify units and preprocessing.

```bash
python flood.py path/to/sentinel1.tif --output outputs/water.tif
python flood.py path/to/after.tif --before path/to/before.tif --output outputs/candidate_flood.tif
```

Outputs preserve CRS, pixel grid and extent. Values: **0 = other, 1 = water**
(or candidate new flooding with `--before`), **255 = invalid/no-data**.
Before/after rasters must have identical CRS, extent and pixel alignment, and should
use the same orbit and preprocessing. The model processes a scene in memory; tile
large scenes with at least a 3-pixel context halo for v1 models, or a 15-pixel
halo for v2 models, before inference. The model artifact selects its feature
version automatically; existing artifacts default to v1.

## Reproduce training

```bash
python train.py
python -m unittest -v
```

The default run downloads 96 training, 24 validation and 48 test image/label pairs
from the public dataset bucket into `data/` (several hundred MB). It samples up to
2,000 valid training pixels per image, with seed 42. Images are selected by cycling
through events within each official split after deterministic shuffling. No image
is shared across the three splits. The 11 features are VV, VH, VV minus VH, and
3x3/7x7 local means and standard deviations for each band. Invalid pixels do not
contribute to local statistics or training labels.

The probability threshold is selected using validation water IoU. Test metrics
use all valid labeled pixels in the held-out test subset, with no test tuning.
They are pooled pixel metrics, not event-averaged scores. Events can occur in
multiple official splits, so these results do not establish performance on unseen
flood events. This subset is **not the full published benchmark**.

To use all available chips in each official split:

```bash
python train.py --train-chips 10000 --valid-chips 10000 --test-chips 10000
```

The counts are capped at the number of available images. Training regenerates the
model and reports. `reports/data_manifest.json` records exact selected filenames
and SHA-256 checksums. `reports/metrics.json` records scores and configuration.

## Five-hour training run

```bash
python train_long.py --hours 5 --run runs/my-five-hour-run
```

Use a new output directory for each run. This CPU search uses all official training
and validation chips, up to 12,000 sampled pixels per training chip, and a fixed
4,000-pixel-per-chip validation sample for model selection. It explores tree size,
boosting iterations, learning rate, regularization and training sample size. It
compares against the published baseline on the same validation sample, saves
improving checkpoints atomically, and leaves the published model untouched.

The wall-clock budget includes data preparation and reserves up to 40 minutes
(20% of the budget for short runs) for full evaluation of the selected checkpoint
and baseline on identical validation/test chips. Each
trial has a 15-minute cap and is terminated at the training deadline if needed.
If final evaluation exceeds the remaining budget, the saved checkpoint remains
usable and status reports incomplete evaluation. Preparation failures abort the
run. Hyperparameter search can overfit validation data; more runtime does not
guarantee better held-out results.

Inspect `status.json`, `trials.jsonl`, `worker.log`, `best_model.joblib`, and
`final_metrics.json` inside the run folder. Windows runs temporarily request that
the machine stay awake while the supervisor is alive; closing the lid or shutting
down can still interrupt training. This does not modify persistent power settings.

For the wider-context improvement search:

```bash
python train_long.py --hours 5 --run path/outside/OneDrive/new-run --baseline models/water_model_20261027_partial.joblib --feature-version v2
python monitor.py --run path/outside/OneDrive/new-run --log outputs/improved-training.log --port 8766
```

The log argument points to the file where you redirect training output, if desired.
Using a local folder outside cloud synchronization reduces checkpoint contention.
V2 adds VV/VH means and standard deviations over 15x15 and 31x31 neighborhoods,
giving 19 features while retaining the original 11 as its leading columns. The
search also tries class balancing and minimum leaf sizes in shuffled configuration
order. Baseline selection uses the exact same sampled validation pixels and
retains the baseline if no candidate improves that score. The final report
includes per-event results and matched baseline comparisons. Repeated use of
the published test split does not constitute a fresh independent benchmark.

## Results

See [measured metrics](reports/metrics.json) and [model card](MODEL_CARD.md).

### V2 improvement run (October 28–29, 2026)

The new run completed successfully **within its five-hour budget**, taking
4h 39m 10s including final evaluation. It completed 265 training trials; the last
in-progress trial was stopped at the planned cutoff to leave time for evaluation.

On the **same full 90-chip test set**, compared with the October 27 extended model:

| Metric | Previous extended model | V2 model |
|---|---:|---:|
| Water IoU | 63.65% | **65.41%** |
| Water F1 | 77.79% | **79.09%** |
| Recall | 74.05% | **76.99%** |
| Precision | **81.91%** | 81.31% |
| Pixel accuracy | 94.80% | **94.99%** |

V2 improves IoU by 1.76 percentage points and F1 by 1.30 points, with a small
precision tradeoff. Event results are not uniformly better: Spain's test water
IoU decreases by 0.45 points. See the [complete V2 report](reports/run-20261028-v2/README.md)
for validation results, event breakdowns, limitations and reproducibility.
Existing models and the default prediction model are preserved. Select V2 explicitly:

```bash
python flood.py path/to/sentinel1.tif --model models/water_model_20261028_v2.joblib --output outputs/water_v2.tif
```

### Interrupted extended run (October 27–28, 2026)

The requested five-hour search **stopped early after 2h 03m 12s and 182 completed
trials**, when Windows denied an atomic status-file replacement. Training was not
restarted. The best saved checkpoint is published separately as
[`models/water_model_20261027_partial.joblib`](models/water_model_20261027_partial.joblib).
The original baseline remains the default. See the
[extended-run report](reports/run-20261027-partial/README.md) for evaluation and provenance.

```bash
python flood.py path/to/sentinel1.tif --model models/water_model_20261027_partial.joblib --output outputs/water.tif
```

The best checkpoint improved water IoU from 59.41% to 61.58% on the **same sampled
validation set**; this is a model-selection result, not evidence of test-set
improvement. The selected checkpoint was evaluated after the interruption without
further fitting. A bounded retry for transient Windows file-access errors has
since been added to the training code, with regression tests.

To display the original local run's progress in a browser, run `python monitor.py`
and open `http://127.0.0.1:8765`. The monitor is read-only, Windows-specific, and
expects the original `runs/five-hour-20261027` directory; run data are not included
in a fresh clone. Published results are available in the report above.

![Held-out test image: radar, ground truth and model output](reports/test_preview.png)

## Limitations

This is a research baseline, not a validated emergency warning system. Radar
shadow, smooth dry surfaces, urban areas, vegetation, acquisition differences and
speckle can cause errors. Before/after differencing is a heuristic and is not
separately validated as flood-change detection. Model probabilities are not
calibrated confidence estimates. Confirm results with local information and
independent imagery before making operational decisions.

## Data and attribution

Bonafilia, D., Tellman, B., Anderson, T., and Issenberg, E. (2020).
*Sen1Floods11: A georeferenced dataset to train and test deep learning flood
algorithms for Sentinel-1*. CVPR Workshops, pp. 210–211.
[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html).

Source: `https://storage.googleapis.com/sen1floods11/v1.1/`.
Raw datasets are excluded from this repository. Consult the upstream project for
data licensing and reuse terms; this repository does not relicense its data.
