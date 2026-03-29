import os
import glob
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, SaveImaged, CastToTyped
)
from monai.data import Dataset, DataLoader
from config_prep import ConfigPrep


def run_offline_preprocessing():
    # 强制创建输出目录，防止 Windows 路径异常中断
    os.makedirs(ConfigPrep.OUTPUT_DIR, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "imagesTr", "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(img_files, lbl_files)]

    preprocess_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # 核心修正 1：先归一化，让所有背景空气变为 0.0
        # 【优化 1】：尽早裁剪！在庞大的原始矩阵上直接切除空气（>-500 HU 为人体组织）。
        # 矩阵体积瞬间暴减，极大地减轻了后续环节的 CPU 和内存压力。
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > -500,
            margin=5
        ),

        # 【优化 2】：在“瘦身”后的矩阵上进行归一化，残留的空气背景被映射为 0.0
        ScaleIntensityRanged(
            keys=["image"], a_min=ConfigPrep.HU_MIN, a_max=ConfigPrep.HU_MAX,
            b_min=0.0, b_max=1.0, clip=True
        ),

        # 【优化 3】：在最小、最干净的矩阵上做极其耗时的三维重采样。
        # 此时产生的补边（Padding）默认也是 0.0，与背景完美融合，无缝衔接。
        Spacingd(
            keys=["image", "label"],
            pixdim=ConfigPrep.SPACING,
            mode=("bilinear", "nearest")
        ),

        # 极限压缩标签体积，提升训练时的 IO 吞吐量
        CastToTyped(keys=["label"], dtype=np.uint8),

        # 独立且安全地保存 Image 和 Label，防覆盖
        SaveImaged(
            keys="image", output_dir=ConfigPrep.OUTPUT_DIR, output_postfix="image_prep",
            output_ext=".nii.gz", resample=False, separate_folder=True
        ),
        SaveImaged(
            keys="label", output_dir=ConfigPrep.OUTPUT_DIR, output_postfix="label_prep",
            output_ext=".nii.gz", resample=False, separate_folder=True
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