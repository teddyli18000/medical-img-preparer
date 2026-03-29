import os
import glob
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, SaveImaged, CastToTyped
)
from monai.data import Dataset, DataLoader
from config_prep import ConfigPrep


# Reference: MONAI Developers. (2024). Transforms - Dictionary Transforms.
# Documentation explicitly recommends mapping intensities before spatial resampling
# to avoid zero-padding artifacts in CT scans.
# [Ref. 1, Section: Dictionary Transforms]

def run_offline_preprocessing():
    img_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "imagesTr", "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(img_files, lbl_files)]

    preprocess_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # 核心修正 1：先归一化，让所有背景空气变为 0.0
        ScaleIntensityRanged(
            keys=["image"], a_min=ConfigPrep.HU_MIN, a_max=ConfigPrep.HU_MAX,
            b_min=0.0, b_max=1.0, clip=True
        ),

        # 核心修正 2：此时再做重采样，产生的 Padding 默认是 0.0，与背景完美融合
        Spacingd(
            keys=["image", "label"],
            pixdim=ConfigPrep.SPACING,
            mode=("bilinear", "nearest")
        ),

        # 核心修正 3：由于背景全是 0.0，安全地裁剪掉所有大于 0 的区域外的黑边
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > 0,
            margin=5
        ),

        # 核心修正 4：极限压缩标签体积，提升后续 IO 速度
        CastToTyped(keys=["label"], dtype=np.uint8),

        # 独立保存，杜绝覆盖
        SaveImaged(
            keys="image",
            output_dir=ConfigPrep.OUTPUT_DIR,
            output_postfix="image_prep",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=True
        ),
        SaveImaged(
            keys="label",
            output_dir=ConfigPrep.OUTPUT_DIR,
            output_postfix="label_prep",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=True
        )
    ])

    ds = Dataset(data=data_dicts, transform=preprocess_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    print(f"开始预处理，目标目录: {ConfigPrep.OUTPUT_DIR}")
    for i, _ in enumerate(loader):
        print(f"已完成: {i + 1}/{len(data_dicts)}")
    print("离线预处理全部完成。")


if __name__ == "__main__":
    run_offline_preprocessing()