# Model card: Sentinel-1 water baseline

## Intended use and outputs

Research and education: delineate surface water in two-band Sentinel-1 VV/VH dB
GeoTIFFs. Output is a georeferenced binary water mask with an invalid-pixel class.
This model is **not validated for emergency response**, optical imagery, or
floodwater hidden under vegetation or buildings. Single-scene water is not the
same as newly flooded land. Optional pre/post subtraction produces candidate
change only; no change-detection benchmark was run.

## Training

- Data: Sen1Floods11 v1.1, official hand-labeled splits.
- Selected chips: 96 train / 24 validation / 48 test, sampled across events.
- Training pixels: 192,000, uniformly sampled within each training chip.
- Model: scikit-learn HistGradientBoostingClassifier, 120 boosting iterations,
  maximum 31 leaves, learning rate 0.08, L2 regularization 1, no early stopping.
- Inputs: 11 backscatter and local texture features; see `train.py` / `flood.py`.
- Seed: 42. Threshold: 0.20, selected from 0.20–0.80 in steps of 0.05 using
  validation water IoU only. The selected threshold is at the grid's lower bound;
  this is a limited search, not a claim of optimal calibration.
- Runtime: CPU. Python 3.14.0, scikit-learn 1.9.1; full dependency lock included.

## Measured performance

| Metric | Validation | Held-out test |
|---|---:|---:|
| Water IoU | 58.23% | 62.05% |
| Water F1 | 73.60% | 76.58% |
| Water precision | 77.78% | 73.21% |
| Water recall | 69.84% | 80.27% |
| Pixel accuracy | 93.70% | 92.94% |

Test evaluation includes 10,325,218 valid labeled pixels. An all-land classifier
achieves 85.63% accuracy but zero water IoU/F1, demonstrating why overall accuracy
alone is misleading. Metrics are pooled over valid pixels; unlabeled and no-data
pixels are excluded. Chips with no valid labels contribute no evaluation pixels.

These are results on a selected subset, not the entire official benchmark. Splits
are disjoint by chip but not necessarily by event; neighboring scenes and shared
events limit claims of geographical generalization. No confidence intervals or
independent regional validation are provided. The displayed example is the first
selected test chip containing at least 1% labeled water and 1% labeled land,
chosen using labels only, not model performance. Its filename is in metrics.json.

## Reproducibility and provenance

`reports/data_manifest.json` records image and label filenames plus SHA-256 hashes.
`reports/metrics.json` contains the full confusion counts and training configuration.
`requirements-lock.txt` records the exact installed packages. The recorded runtime
of 54 seconds is for a run with the dataset already cached and excludes the
initial download and dependency installation.

The trained artifact is `models/water_model.joblib`. Only load model files you
trust. The raw upstream data are not redistributed; see the source attribution
and data-terms note in README.md. The preview is a derived visualization of one
upstream test chip.

## Known failure modes

Radar shadows and smooth dry surfaces can resemble water; vegetation, wind and
urban structures can obscure it. Acquisition and preprocessing shifts may
degrade performance. Probabilities are uncalibrated classifier scores. Flood
change estimates compound errors from both scenes. No protected-group or
population-impact evaluation was conducted; this model should not be used to
allocate aid or determine individual eligibility.
