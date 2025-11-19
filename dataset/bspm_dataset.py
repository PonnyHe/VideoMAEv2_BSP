import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BSPMVideoDataset(Dataset):
    """
    读取 BSPM 单通道视频数据
    输入格式：npy 文件，每个文件形状为 (1, T, H, W)
    """
    def __init__(self, root, list_file):
        self.root = root
        with open(list_file, "r") as f:
            self.samples = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        path = os.path.join(self.root, filename)

        arr = np.load(path)  # shape: (1, T, H, W)
        video = torch.tensor(arr, dtype=torch.float32)

        return video
