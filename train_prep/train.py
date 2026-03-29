import os
import sys


# 预警,防止无限递归报错
from dataset_train import get_train_loader


#log function
from config_train import ConfigTrain


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from log_prep.runtime_logger import PrintMirrorLogger, infer_dataset_name
#log function


#dataset_train写了num_workers，必须套在main里
#经过测试，Windows下必须使用单进程，否则不稳定
if __name__ == "__main__":



    #log function
    dataset_name = infer_dataset_name(ConfigTrain.PREPROCESSED_DIR)
    with PrintMirrorLogger(module_name="train_prep", dataset_name=dataset_name, project_root=PROJECT_ROOT):
    #log function



        loader = get_train_loader()
        # 开始写模型的 for 循环...
