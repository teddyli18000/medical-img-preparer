# 预警,防止无限递归报错
from dataset_train import get_train_loader

#dataset_train写了num_workers，必须套在main里
if __name__ == "__main__":
    loader = get_train_loader()
    # 开始写模型的 for 循环...