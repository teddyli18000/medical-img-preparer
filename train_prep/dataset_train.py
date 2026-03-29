import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from config_train import ConfigTrain


# Reference: Stevens et al. (2020). Deep Learning with PyTorch.
# Chapter 4 emphasizes robust file pairing mechanisms to prevent silent data misalignment.
# [Ref. 2, Chapter 4: Data Loaders]

def get_train_loader():
    # 1. 自动寻找子文件夹中的预处理后文件
    # 核心修正 5：废弃双 glob 排序，改用绝对的一一对应机制
    all_images = sorted(
        glob.glob(os.path.join(ConfigTrain.PREPROCESSED_DIR, "**", "*_image_prep.nii.gz"), recursive=True))
    data_dicts = []
    for img_path in all_images:
        lbl_path = img_path.replace("_image_prep.nii.gz", "_label_prep.nii.gz")
        if os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})
        else:
            print(f"warning：未找到 {img_path} 对应的标签文件，已跳过。")

    split = int(len(data_dicts) * 0.8)
    train_files = data_dicts[:split]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        SpatialPadd(keys=["image", "label"], spatial_size=ConfigTrain.ROI_SIZE),

        # 此时的 image_threshold=0 将完美避开 0.0 的空气背景，精准在人体内采样
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=ConfigTrain.ROI_SIZE, pos=1, neg=1, num_samples=2,
            image_key="image", image_threshold=0
        ),

        # 恢复独立的轴翻转概率
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),

        # 核心修正 6：确保最终喂给显卡的是原生 PyTorch Tensor 格式
        EnsureTyped(keys=["image", "label"], data_type="tensor")
    ])

    ds = Dataset(data=train_files, transform=train_transforms)
    return DataLoader(ds, batch_size=ConfigTrain.BATCH_SIZE, shuffle=True, num_workers=ConfigTrain.NUM_WORKERS)