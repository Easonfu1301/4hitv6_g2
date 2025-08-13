import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import *


class FilteringData(Dataset):
    def __init__(self, data):
        # data_mean = np.mean(data[:, 0:-1], axis=0)
        # data_std = np.std(data[:, 0:-1], axis=0) + 1e-6
        #
        # data[:, 0:-1] = (data[:, 0:-1] - data_mean) / data_std

        # data[:, 0:6] = np.log(data[:, 0:6] * 10000 +1) # *2000
        data[:, 0:6] = data[:, 0:6] * 5000  # *2000
        # data[:, 6:9] = np.log(data[:, 6:9] /10+1)
        data[:, 6:9] = data[:, 6:9] / 10
        data[:, 9:11] = data[:, 9:11] / 750/2+0.5

        # x = data[:, 0]
        # x = x[x<1]

        # plt.hist(x, bins=100, color='b', alpha=0.5)
        # plt.hist(data[:, 9], bins=100, color='r', alpha=0.7)
        # plt.show()

        # data[:, 0] = data[:, 0] / 5
        # data[:, 1] = data[:, 1] / 5

        # plt.hist(data[:, 1], bins=100, color='b', alpha=0.7)
        # plt.show()

        # data[:, 0:8] = np.

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(data[:, 5], bins=100, color='b', alpha=0.7)
        #
        # plt.show()

        data = torch.asarray(data, dtype=torch.float32)
        # print(data.shape)
        data[:, -1] = torch.tensor(data[:, -1], dtype=torch.long)
        # print(data.shape)

        # print(data.shape)
        # print(data)
        # print(torch.max(data))
        self.filter = data
        self.filter = self.filter.to(device)

    def __len__(self):
        return self.filter.shape[0]

    def __getitem__(self, idx):
        feature = self.filter[idx, 0:11]
        label = self.filter[idx, 11:12]
        # print(feature)
        return feature, label


if __name__ == "__main__":
    train_data = np.load(r"D:\files\pyproj\exatrkx_copyer\Net\Embed_split_data\train.npy")
    val_data = np.load(r"D:\files\pyproj\exatrkx_copyer\Net\Embed_split_data\val.npy")

    # train_dataset = EmbedData(train_data)
    # val_dataset = EmbedData(val_data)
    # print(len(train_dataset))
    # # print(dataset[0])
    #
    # train_iter = DataLoader(val_dataset, batch_size=2)
    # for x,z in train_iter:
    #     print(x, z)
    #     break
