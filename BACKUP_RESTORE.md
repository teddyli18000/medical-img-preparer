# Backup and Restore Notes

This project separates source backup from generated data backup.

## Git Repository

The GitHub repository is the source-of-truth for code, manuals, validation
scripts, and lightweight report material:

```text
https://github.com/teddyli18000/medical-img-preparer
```

Use the `develop` branch for the latest project state.

## Large Local Data

The local `data/` directory is generated medical image data and is ignored by
Git. At the time this note was written, it contained the preprocessed MSD
Task07 output at approximately 9.8 GiB, with individual `.nii.gz` files larger
than 100 MiB.

Do not add that folder to normal Git history. Use one of these routes:

1. Restore from a private GitHub Release archive asset if one was uploaded for
   this archival handoff.
2. Regenerate the folder from the raw MSD Task07 dataset by configuring
   `data_prep/config_prep.py` and running `python data_prep\run_preprocess.py`.

After either route, run:

```powershell
python task7_preprocess_validation\validate_task7_preprocess.py
```

## Expected Generated Layout

```text
data/
  preprocessed/
    processed_MSD_Task7/
      **/*_image_prep.nii.gz
      **/*_label_prep.nii.gz
```
