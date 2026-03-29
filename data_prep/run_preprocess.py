import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, SaveImaged
)
from monai.data import Dataset, DataLoader
from config_prep import ConfigPrep


def run_offline_preprocessing():
    # 1. 搜集原始文件
    img_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "imagesTr", "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(ConfigPrep.RAW_DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(img_files, lbl_files)]

    # 2. 定义确定性变换流水线 (Transform 1-6)
    # [Ref. 3, Section: Dictionary Transforms] - 字典操作保证 Image/Label 空间同步
    preprocess_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=ConfigPrep.SPACING,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=ConfigPrep.HU_MIN, a_max=ConfigPrep.HU_MAX,
            b_min=0.0, b_max=1.0, clip=True
        ),
        # 裁剪前景以缩小文件体积，减轻程序二的 IO 压力

        # 1. 先用真实 CT 值裁剪外围空气 (>-500 HU 代表人体组织)
        # margin=5 给边界留一点缓冲余地，防止把贴近边缘的器官切没
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > -500,
            margin=5
        ),

        # 2. 裁剪完之后，再安心地将特定软组织窗口映射到 0.0 ~ 1.0
        ScaleIntensityRanged(
            keys=["image"], a_min=ConfigPrep.HU_MIN, a_max=ConfigPrep.HU_MAX,
            b_min=0.0, b_max=1.0, clip=True
        ),

        # 3. 保存结果：每个病人一个文件夹
        # [Ref. 3, Section: IO Transforms] - SaveImaged 将中间态持久化至磁盘
        # [Ref. 1, Section: IO Transforms] - 分离保存逻辑以防止同名元数据覆写
        # save image, 后缀指定为 image_prep
        SaveImaged(
            keys="image",
            output_dir=ConfigPrep.OUTPUT_DIR,
            output_postfix="image_prep",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=True
        ),
        # save label, 后缀指定为 label_prep
        SaveImaged(
            keys="label",
            output_dir=ConfigPrep.OUTPUT_DIR,
            output_postfix="label_prep",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=True
        )
    ])

    # 3. 执行处理 (方案 A：处理全部数据)
    ds = Dataset(data=data_dicts, transform=preprocess_transforms)
    # 使用 DataLoader 配合 num_workers 开启多线程预处理
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    print(f"开始预处理，目标目录: {ConfigPrep.OUTPUT_DIR}")
    for i, _ in enumerate(loader):
        print(f"已完成: {i + 1}/{len(data_dicts)}")
    print("离线预处理全部完成。")


if __name__ == "__main__":
    run_offline_preprocessing()