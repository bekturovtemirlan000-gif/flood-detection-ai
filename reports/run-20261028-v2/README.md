# V2 water model: completed improvement run

The five-hour-budget run completed successfully in **4h 39m 10s**, from
October 28, 2026 at 21:59:26 to October 29 at 02:38:37 (Asia/Bishkek).
Model search ran until the planned 02:19:26 cutoff, then final evaluation finished
before the 02:59:26 deadline. There were **265 completed trials**. Trial 266 was
terminated at the time limit and did not replace the selected checkpoint.

## Model

Artifact: [`water_model_20261028_v2.joblib`](../../models/water_model_20261028_v2.joblib).
The previous models are unchanged and the original baseline remains the CLI default.

This is CPU histogram gradient boosting for surface-water segmentation, not a
neural network or a validated emergency flood-warning system. V2 uses 19 inputs:
VV, VH, VV minus VH, and VV/VH local means and standard deviations at 3x3, 7x7,
15x15 and 31x31 scales. The first 11 columns exactly preserve the v1 features.
Input remains a two-band Sentinel-1 VV/VH dB GeoTIFF. Inference reads the feature
version from the model artifact. Large scenes need tiles with a 15-pixel context
halo. Do not apply the model to arbitrary RGB images or linear-power SAR data.

Selected trial 70 used 600 boosting iterations, up to 63 leaves, learning rate
0.035, L2 regularization 5, balanced class weights, minimum leaf size 20, random
seed 112, and 750,000 training pixels sampled from the training pool. Threshold
0.80 was chosen on the validation sample. This is the upper edge of the searched
threshold grid (0.05 to 0.80), so it is not a claim of globally optimal calibration.
The probability scores are not calibrated confidence estimates.

## Data and selection

- Sen1Floods11 v1.1 official splits: 252 train, 89 validation, 90 test chips.
- Training pool: 3,001,892 valid pixels, up to 12,000 per chip, sampling seed 42.
- Model-selection validation sample: 344,000 pixels, up to 4,000 per chip, seed 42.
- Configuration search varied tree size, iterations, learning rate, regularization,
  training sample size, class weighting, and minimum leaf size in a seeded shuffle.
- The reference was `models/water_model_20261027_partial.joblib`, evaluated on
  exactly the same sampled validation pixels. Baseline sample IoU was 61.58%;
  selected V2 sample IoU was 64.07%. Selection used no test labels.
- Both checkpoints were then evaluated on all valid labeled pixels in identical
  full validation and test splits. Evaluation completed successfully.

## Matched full evaluation

| Metric | Previous validation | V2 validation | Previous test | V2 test |
|---|---:|---:|---:|---:|
| Water IoU | 60.54% | 63.36% | 63.65% | 65.41% |
| Water F1 | 75.42% | 77.57% | 77.79% | 79.09% |
| Precision | 80.86% | 80.75% | 81.91% | 81.31% |
| Recall | 70.66% | 74.64% | 74.05% | 76.99% |
| Pixel accuracy | 94.93% | 95.25% | 94.80% | 94.99% |

The test set has 20,443,959 valid labeled pixels. Test water IoU increases by
**1.76 percentage points**, F1 by **1.30 points**, and recall by **2.94 points**.
Precision decreases by **0.61 points**. These are pooled pixel metrics, not
event-averaged scores. Test event water IoU improves for 9 of 10 named events;
Spain decreases by 0.45 points. All per-event counts and scores, including
regressions, are included in `final_metrics.json`.

These results are not a fresh independent benchmark: this public test split has
been evaluated previously. Chips are disjoint, but events can span splits.
Repeated validation selection can overfit, and there are no uncertainty intervals
or external-region validation. Wider context and class weighting were changed
together, so this run does not isolate their individual causal contributions.
Surface water includes permanent rivers/lakes; before/after differencing remains
an unvalidated candidate-flood heuristic. All original model-card limitations apply.

## Verification and provenance

- All 12 behavior/regression tests passed, including v1 feature compatibility,
  invalid pixels, georeferencing and file-lock retries.
- The packaged v2 artifact successfully predicted a real Sentinel-1 test chip.
- Split isolation, trial counts, best-checkpoint selection and per-event confusion
  count sums were verified during packaging.
- `final_metrics.json` contains both models' full results and `evaluation_complete: true`.
- `comparison_percentage_points.json` contains the paired metric differences.
- `trials.jsonl`, `best_validation.json`, `baseline_validation.json`,
  `run_status.json`, and sanitized `settings.json` document the run.
- `splits.json` and `data_manifest.json` contain exact filenames and SHA-256 hashes.
- `source_*.py` are the exact source snapshots captured at launch;
  `checksums.json` includes their hashes and the model hash.
- Dependencies are pinned in the root `requirements-lock.txt`.

Live checkpoints were stored outside OneDrive, and atomic writes use bounded
retries for transient access errors. No raw satellite imagery, feature caches,
credentials, or machine-specific paths are included. See the root README for
Sen1Floods11 attribution and data terms. Only load trusted joblib files.
