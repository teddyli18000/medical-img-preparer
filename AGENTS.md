# Project Context

This repository prepares and validates MSD Task07 Pancreas medical image data.
Treat the checked-in source code and documentation as the durable project state.
Treat `data/`, `logs/`, report output folders, IDE settings, and Python caches as
local/generated state unless an explicit archival task says otherwise.

## Modules

- `data_exam/`: header-level EDA for the raw Task07 dataset. Configure
  `ConfigExam.RAW_DATA_DIR`, then run `python data_exam\run_exam.py`.
- `data_prep/`: deterministic offline preprocessing from raw NIfTI image/label
  pairs into `data/preprocessed/processed_MSD_Task7`. Configure
  `ConfigPrep.RAW_DATA_DIR`, then run `python data_prep\run_preprocess.py`.
- `data_prep_exam/`: header-level EDA for the preprocessed output. Run
  `python data_prep_exam\run_prep_exam.py` after preprocessing.
- `task7_preprocess_validation/`: independent acceptance checker that reads
  NIfTI files directly without MONAI/nibabel. Run
  `python task7_preprocess_validation\validate_task7_preprocess.py`.
- `train_prep/`: training data loader and placeholder training entry point
  consuming the preprocessed output.
- `log_prep/`: shared print/log mirroring helper used by preprocessing and
  training scripts.

## Data Contracts

- Raw data is expected under a directory containing `imagesTr/*.nii.gz` and
  `labelsTr/*.nii.gz`.
- `data_prep` assumes CT HU values, target spacing `(0.8027, 0.8027, 2.5)`,
  RAS orientation, image normalization to `[0, 1]`, and label values fitting in
  `{0, 1, 2}`.
- Preprocessed files are discovered by suffix:
  `*_image_prep.nii.gz` and `*_label_prep.nii.gz`.
- Image/label pairing must be verified by basename. List-order pairing can hide
  mismatches when file counts differ.

## Local State and Backup Notes

- The project has historically used `data/preprocessed/processed_MSD_Task7` for
  generated MSD Task07 outputs. That folder is large and intentionally ignored
  by Git.
- GitHub regular repository storage is for source and lightweight documents.
  Large medical image artifacts should be restored from an explicit archival
  release asset or regenerated from raw data.
- Do not commit `.env`, credentials, raw logs, IDE metadata, `__pycache__`, or
  generated validation output unless the user explicitly asks for an archival
  snapshot.

## Validation

Fast source-level checks:

```powershell
python -m compileall data_exam data_prep data_prep_exam log_prep task7_preprocess_validation train_prep
```

Full data validation after restoring or regenerating preprocessed data:

```powershell
python task7_preprocess_validation\validate_task7_preprocess.py
```
