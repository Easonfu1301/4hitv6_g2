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

        datax = np.zeros((data.shape[0], 14))

        datax[:, 0:6] = data[:, 0:6] * 2000 # *2000
        datax[:, 6:9] = data[:, 6:9] # /10
        datax[:, 9] = data[:, 0] - data[:, 1]
        datax[:, 10] = data[:, 0] - data[:, 2]
        datax[:, 11] = data[:, 0] - data[:, 3]
        datax[:, 12] = data[:, 0] - data[:, 4]
        datax[:, 13] = data[:, 0] - data[:, 5]

        datax[:, 9:13] = (datax[:, 9:13] - min(datax[:, 9:13].min(), 0))*1000
        datax[:, 13] = data[:, -1]
        # plt.hist(datax[:, 0], bins=100, color='b', alpha=0.7)
        # plt.hist(datax[:, 8], bins=100, color='r', alpha=0.7)
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

        data = torch.asarray(datax, dtype=torch.float32)
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
        feature = self.filter[idx, 0:13]
        label = self.filter[idx, 13:14]
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
