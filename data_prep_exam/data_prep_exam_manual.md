# data_prep_exam Manual (Preprocessed Dataset EDA)

This module inspects the output of offline preprocessing (the `data_prep` results). It is header-only: it reads NIfTI headers, does not load voxel arrays, and produces a JSON report plus a Markdown report with PNG figures.

## 1. Scope

Covered:
- Image/label pairing consistency
- Header readability
- Spacing/shape/orientation statistics
- Image/label geometry consistency (spacing/shape/orientation/affine)
- Distribution figures (spacing/shape/FOV/orientation/dtype)

Not covered:
- Voxel-level intensity distribution
- Label semantic correctness
- Empty mask ratio / foreground occupancy

## 2. Inputs and Outputs

### 2.1 Input directory

Configured in `config_prep_exam.py`:
- `PREPROCESSED_DIR` (default: `data/preprocessed/processed_MSD_Task7`)

### 2.2 Output files

Outputs go to `data_prep_exam_report`:
- `stats_prep_<dataset>.json`
- `report_prep_<dataset>.md`
- `figures/<dataset>/*.png` (referenced in the Markdown report only)

## 3. What is checked

- Pairing: `_image_prep.nii.gz` and `_label_prep.nii.gz` must exist for each sample
- Header readability: each NIfTI header must be readable
- Spacing/shape/orientation: computed for all images
- Geometry consistency: image vs label spacing/shape/orientation/affine
- Dtype distribution: image/label dtype counts

## 4. How to run

From the repository root:

```bash
python data_prep_exam/run_prep_exam.py
```

## 5. How to interpret results

- `geometry_consistency.mismatch_sample_count` should be 0
- Spacing should be highly consistent after preprocessing
- Orientation should be a single code (typically RAS)
- Label dtype should be discrete (often uint8)

If mismatches exist, fix preprocessing outputs before training to avoid misaligned supervision.
