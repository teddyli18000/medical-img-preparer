# Task7 Preprocessing Validation

This folder contains an independent acceptance checker for the preprocessed MSD
Task07 Pancreas dataset.

It does not depend on `nibabel`, `monai`, or `SimpleITK`. The validator reads
NIfTI `.nii.gz` headers and voxel arrays directly with Python standard library,
`numpy`, and `Pillow` for overlay PNGs.

## Run

From the repository root:

```powershell
python task7_preprocess_validation\validate_task7_preprocess.py
```

Outputs are written to:

```text
task7_preprocess_validation\output\
```

Main artifacts:

- `report.md`: meeting-ready Markdown report.
- `summary.json`: machine-readable full validation summary.
- `samples.csv`: one row per image/mask pair.
- `overlays\*.png`: sampled image/mask overlay checks.

## Validation Layers

1. Pairing: every `*_image_prep.nii.gz` has exactly one matching
   `*_label_prep.nii.gz`.
2. Spatial geometry: image and mask have matching shape, spacing, orientation,
   and affine matrices.
3. Target preprocessing: spacing is near `(0.8027, 0.8027, 2.5)`, orientation is
   `RAS`, images are finite and normalized to `[0, 1]`, and labels are discrete.
4. Voxel-level mask sanity: masks are non-empty, contain expected label values,
   and overlap the non-background image region.
