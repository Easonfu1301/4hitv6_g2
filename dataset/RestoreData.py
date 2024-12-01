import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RestoreData(Dataset):
    def __init__(self, data, hit_data):

        # data_mean = np.mean(data[:, 0:-1], axis=0)
        # data_std = np.std(data[:, 0:-1], axis=0) + 1e-6
        #
        # data[:, 0:-1] = (data[:, 0:-1] - data_mean) / data_std

        data[:, 0:6] = data[:, 0:6] * 2000
        data[:, 6:9] = data[:, 6:9] / 10

        # plt.hist(data[:, 1], bins=100, color='b', alpha=0.7)
        # plt.show()

        # data[:, 0:8] = np.


        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(data[:, 5], bins=100, color='b', alpha=0.7)
        #
        # plt.show()

        data = torch.asarray(data, dtype=torch.float32)
        data[:, -1] = torch.tensor(data[:, -1], dtype=torch.long)
        # print(data.shape)
        # print(data)
        # print(torch.max(data))
        self.filter = data
        self.hit_data = hit_data
        self.filter = self.filter.to(device)



    def __len__(self):
        return self.filter.shape[0]

    def __getitem__(self, idx):
        feature = self.filter[idx, 0:9]
        label = self.filter[idx, 9:10]
        # print(self.hit_data)
        hit = self.hit_data[idx, 0:4]
        return feature, label, hit



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
