# medical-img-preparer

Utilities for examining, preprocessing, validating, and eventually training on
MSD Task07 Pancreas NIfTI medical image data.

## Repository Layout

- `data_exam/`: raw dataset header EDA.
- `data_prep/`: offline preprocessing pipeline.
- `data_prep_exam/`: preprocessed dataset header EDA.
- `task7_preprocess_validation/`: independent acceptance validation and
  presentation artifacts.
- `train_prep/`: training loader and training entry point scaffold.
- `log_prep/`: shared runtime logging helper.

Generated data, logs, IDE metadata, and report output folders are intentionally
ignored by Git.

## Environment

Python 3.10+ is recommended. Install the checked-in dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using GPU training, install the PyTorch build that matches the local CUDA
runtime before installing the remaining dependencies.

## Main Workflows

### Raw Dataset EDA

Edit `data_exam/config_exam.py` so `ConfigExam.RAW_DATA_DIR` points to a raw
Task07 directory containing `imagesTr/` and `labelsTr/`, then run:

```powershell
python data_exam\run_exam.py
```

### Offline Preprocessing

Edit `data_prep/config_prep.py` so `ConfigPrep.RAW_DATA_DIR` points to the raw
Task07 dataset, then run:

```powershell
python data_prep\run_preprocess.py
```

The default output directory is:

```text
data/preprocessed/processed_MSD_Task7
```

### Preprocessed Dataset EDA

After preprocessing:

```powershell
python data_prep_exam\run_prep_exam.py
```

### Acceptance Validation

After preprocessing or restoring archived outputs:

```powershell
python task7_preprocess_validation\validate_task7_preprocess.py
```

The validator checks pairing, shape, spacing, affine/orientation, image range,
label values, non-empty masks, and mask/image overlap.

## Restore From GitHub

1. Clone the repository and check out `develop`:

   ```powershell
   git clone https://github.com/teddyli18000/medical-img-preparer.git
   cd medical-img-preparer
   git switch develop
   ```

2. Create the Python environment and install `requirements.txt`.
3. Restore `data/` from the archival release asset if one exists, or regenerate
   it by running `data_prep/run_preprocess.py` against the raw MSD Task07
   dataset.
4. Run `python task7_preprocess_validation\validate_task7_preprocess.py` before
   trusting restored or regenerated data.

## Important Local Paths

The current checked-in configs contain the original local raw dataset path used
on the development machine:

```text
E:\Python_Projects\Swin-UNETR-for-MSD-task7\data\Task07_Pancreas
```

Change that path after cloning onto a new machine.
