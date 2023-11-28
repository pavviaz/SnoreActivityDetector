import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class LoadDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self.__load_features()

    def __load_features(self):
        data = [
            os.path.join(self.dataset_path, el)
            for el in tqdm(os.listdir(self.dataset_path))
        ]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = np.load(self.data[idx], allow_pickle=True)
        np.random.shuffle(arr)

        features = []
        labels = []
        [[features.append(el["features"]), labels.append(el["label"])] for el in arr]

        return np.array(features), np.array(labels, dtype=np.int64)
