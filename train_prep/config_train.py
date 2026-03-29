import os
import torch


class ConfigTrain:
    # 这里的输入是data_prep的输出
    PREPROCESSED_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "preprocessed", "processed_MSD_Task7"
    ))

    ROI_SIZE = (96, 96, 96)  # 针对 8GB 显存 [Ref. 2, Section 3.3]
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")