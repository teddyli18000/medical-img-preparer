import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd
)
from monai.data import Dataset, DataLoader
from config_train import ConfigTrain


def get_train_loader():
    # 1. 自动寻找子文件夹中的预处理后文件
    # [Ref. 3, Section: Data Pipelines] - 递归搜索程序一生成的 NIfTI 文件
    all_images = sorted(
        glob.glob(os.path.join(ConfigTrain.PREPROCESSED_DIR, "**", "*_image_prep.nii.gz"), recursive=True))
    all_labels = sorted(
        glob.glob(os.path.join(ConfigTrain.PREPROCESSED_DIR, "**", "*_label_prep.nii.gz"), recursive=True))

    data_dicts = [{"image": i, "label": l} for i, l in zip(all_images, all_labels)]

    # 方案 A：在训练程序中进行 80/20 动态划分
    split = int(len(data_dicts) * 0.8)
    train_files = data_dicts[:split]

    # 2. 定义在线随机采样与增强流水线 (Transform 7-13)
    # 注意：不再需要 Orientation 和 Spacing，因为程序一已经做过了
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # [Ref. 2, Section 3.3] - 针对 3D Patch 训练的补边与采样策略
        SpatialPadd(keys=["image", "label"], spatial_size=ConfigTrain.ROI_SIZE),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=ConfigTrain.ROI_SIZE, pos=1, neg=1, num_samples=2,
            image_key="image", image_threshold=0
        ),

        # 随机增强
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])

    ds = Dataset(data=train_files, transform=train_transforms)
    return DataLoader(ds, batch_size=ConfigTrain.BATCH_SIZE, shuffle=True, num_workers=ConfigTrain.NUM_WORKERS)