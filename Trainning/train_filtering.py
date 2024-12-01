import os.path

import torch

import numpy as np
from dataset import FilteringData
from Net import Filter_MLP
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import  auc
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from utils import *

from sklearn.metrics import  roc_curve


def train_one_epoch(model, train_loader, optimizer):
    loss_epoch = 0

    for embed_link, label in tqdm(train_loader):
        # print(batch_hit1.shape, batch_hit2.shape, true_dist.shape)

        model.train()
        optimizer.zero_grad()
        link_pred = model(embed_link)

        # print(link_pred, torch.tensor(label[:, 0], dtype=torch.int))
        # print(label)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(link_pred, label[:, 0].long())


        # print(loss)

        # print(loss.item())

        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch / len(train_loader)


def eval_one_epoch(model, valid_loader):
    model.eval()
    loss_epoch = 0
    # print(len(valid_loader))
    # preds = []
    # labels = []
    for embed_link, label in valid_loader:
        # print(batch_hit1.shape, batch_hit2.shape, true_dist.shape)
        link_pred = model(embed_link)
        # preds.append(link_pred)
        # labels.append(label)
        # print(embed_link[:, 7])
        # print(torch.max(embed_link), torch.max(link_pred), torch.min(link_pred), torch.max(label), torch.min(label))

    # preds = torch.cat(preds, dim=0)
    # labels = torch.cat(labels, dim=0)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(link_pred, label[:, 0].long())

        fpr = dict()
        tpr = dict()
        # roc_auc = dict()

        y = label_binarize(label[:, 0].long().cpu(), classes=[0, 1, 2, 3, 4, 5, 6, 7])  # 三个类别
        # print(y)

        link_pred = nn.functional.softmax(link_pred, dim=1)
        link_pred = link_pred.cpu().detach().numpy()

        for i in range(8):

            fpr[i], tpr[i], _ = roc_curve(y[:, i], link_pred[:, i])

        fpr[8], tpr[8], _ = roc_curve(y.ravel(), link_pred.ravel())

                # roc_auc[i] = auc(fpr[i], tpr[i])

            # fpr, tpr, thresholds = roc_curve(label.cpu().detach().numpy(), link_pred.cpu().detach().numpy(), pos_label=4)

            # auc = roc_auc_score(label.cpu().detach().numpy(), link_pred.cpu().detach().numpy())

            # print(loss)
        loss_epoch += loss.item()

    return loss_epoch/ len(valid_loader), tpr, fpr


def train_filtering(train_loader, valid_loader):
    model = Filter_MLP()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    train_losses = []
    val_losses = []

    last_loss = 1e9
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plt.ion()
    # plt.show()
    for epoch in range(30000):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer)

        val_loss, tpr, fpr = eval_one_epoch(model, valid_loader)
        print(f"Epoch {epoch} train loss: {train_loss} val loss: {val_loss}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ax.clear()
        ax2.clear()

        ax.plot(train_losses, '.-r', label="train")
        ax.plot(val_losses, '.-b', label="val")
        for i in range(8):
            ax2.plot(fpr[i], tpr[i], '.-', label=str(i+1) + "hit, auc=" + str(np.round(auc(fpr[i], tpr[i]), 4)))

        ax2.plot(fpr[8], tpr[8], '.-', label="micro, auc=" + str(np.round(auc(fpr[8], tpr[8]), 4)))

        ax2.legend()

        # fig.tight_layout()
        # plt.pause(0.001)
        if val_loss < last_loss:
            # visualize_filtering(model, valid_loader, train_losses, val_losses)
            torch.save(model.state_dict(), os.path.join(workdir, "Model", "filtering_model.pth"))
            fig.savefig(os.path.join(workdir, "Model", "filtering_model.png"), dpi=300)
            last_loss = val_loss

    fig.savefig(os.path.join(workdir, "Model", "filtering_model.png"), dpi=300)

    # 读取数据
    # visualize_filtering(model, valid_loader, train_losses, val_losses)


def main(force=False):
    train_data = np.load(os.path.join(workdir, "Preprocess", "sample", "train.npy"))
    test_dataset = np.load(os.path.join(workdir, "Preprocess", "sample", "test.npy"))

    # train_data = train_data[0:500000, :]
    # test_dataset = test_dataset[0:10000, :]

    train_data = FilteringData(train_data)
    test_dataset = FilteringData(test_dataset)

    train_dataloader = DataLoader(train_data, batch_size=8192*2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8192*2, shuffle=True)

    train_filtering(train_dataloader, test_dataloader)

    pass


if __name__ == "__main__":
    main()
