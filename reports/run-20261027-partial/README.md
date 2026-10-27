# Extended training run — interrupted, checkpoint recovered

This is **not a completed five-hour training run**. The process started October 27,
2026 at 23:14:36 Bishkek time and failed October 28 at 01:17:48, after 2h 03m 12s.
Windows returned `PermissionError [WinError 5]` while replacing the status JSON
file. The exact source of that access denial is unknown. There were 182 completed
trials; trial 183 was being prepared when the supervisor stopped. Training was not
restarted or extended. Evaluation of the saved checkpoint was performed afterward.

## Model and selection

- Artifact: `models/water_model_20261027_partial.joblib` (repository root).
- Selected trial: 100; 1,000 boosting iterations, up to 63 leaves, learning rate
  0.035, L2 regularization 5, random seed 142, 2,000,000 sampled training pixels.
- Decision threshold: 0.325, chosen on sampled validation pixels.
- Training pool: 3,001,892 pixels from 252 official training chips.
- Validation selection sample: 344,000 pixels from 89 official validation chips.
- Final evaluation: all valid labeled pixels in 89 validation and 90 test chips.
- Original `models/water_model.joblib` is preserved and remains the default.

| Same sampled validation set | Water IoU | Water F1 |
|---|---:|---:|
| Original baseline | 59.41% | 74.54% |
| Selected extended checkpoint | 61.58% | 76.22% |

These selection scores are directly comparable because they use the same pixels.
Repeated model selection can overfit validation data. Do not compare the final
test score directly with the original 48-chip test-subset score: this evaluation
uses 90 test chips. This remains surface-water segmentation, not a separately
validated flood-change detector.

## Post-interruption evaluation

The saved checkpoint was evaluated without further fitting, completing at 01:33:27
Bishkek time on October 28, 2026.

| Full labeled-pixel evaluation | Validation (89 chips) | Test (90 chips) |
|---|---:|---:|
| Water IoU | 60.54% | 63.65% |
| Water F1 | 75.42% | 77.79% |
| Precision | 80.86% | 81.91% |
| Recall | 70.66% | 74.05% |
| Pixel accuracy | 94.93% | 94.80% |

Test evaluation includes 20,443,959 valid labeled pixels. These are pooled pixel
metrics; the official chip splits are not an event-disjoint generalization test.
Only the saved selected checkpoint was evaluated on this full test set.

## Artifacts

- `final_metrics.json`: post-interruption full validation/test evaluation, with
  confusion counts, selected configuration, threshold and completion time.
- `best_validation.json` and `baseline_validation.json`: comparable sampled
  validation results.
- `trials.jsonl`: completed trial history and model configurations.
- `run_status.json`: sanitized termination record and actual elapsed time.
- `splits.json` and `data_manifest.json`: exact chip lists and SHA-256 input hashes.
- `checksums.json`: published checkpoint and original training-source SHA-256.
- `training_source.py`: exact supervisor source used for this interrupted run.

The root `train_long.py` now includes bounded retries for transient file-access
errors. That fix was made after termination and was not used to train this model.
The test suite covers successful retry, persistent failure, and other I/O errors.
The model successfully generated a georeferenced mask from a real test image.

Dependency versions are in the root `requirements-lock.txt`. Raw satellite data,
feature caches and machine-specific error traces are not published. Sen1Floods11
attribution and data-terms notes are in the root README. Only load trusted joblib
files. All baseline model-card limitations also apply to this checkpoint.
