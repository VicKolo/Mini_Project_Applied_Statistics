import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


class ProjectDataset(Dataset):
    def __init__(self, path, one_hot_encode=False, debug=False):
        if debug:
            a = np.zeros((1000))
            b = np.ones((1000))
            self.y = np.concatenate((a, b))
            print(self.y.shape)
            a = np.zeros((1000, 100))
            b = np.ones((1000, 100))
            self.X = np.concatenate((a, b))
            print(self.X.shape)
        else:
            loaded = pd.read_csv(path)
            self.y = loaded.iloc[:, 0].to_numpy()
            self.X = loaded.drop(columns=loaded.columns[0], axis=1).to_numpy()
            assert len(self.y) == len(self.X)

        if one_hot_encode:
            self.y = np.eye(10)[self.y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        y = self.y[i]
        x = self.X[i]
        # img = x.reshape((28, 28))
        # plt.imshow(img)
        # plt.title(str(y))
        # plt.show()
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
