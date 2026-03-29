import os


class ConfigPrep:
    # --- 路径配置 (请在此处填写您的本地路径) ---
    # [Ref. 3, Section: Data Pipelines] - 离线预处理建议与代码逻辑解耦
    RAW_DATA_DIR = r"E:\Python_Projects\Swin-UNETR-for-MSD-task7\data\Task07_Pancreas"  # TODO: 填入原数据集路径

    # 程序一生成数据的存放位置
    OUTPUT_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "preprocessed", "processed_MSD_Task7"
    ))

    # --- 空间对齐参数 (由中位数脚本算出) ---
    # [Ref. 1, Section 2.1] - 使用中位数物理间距以保证特征稳定性
    SPACING = (0.8027, 0.8027, 2.5000)

    # 强度归一化参数 (针对胰腺 CT)
    HU_MIN = -87
    HU_MAX = 199