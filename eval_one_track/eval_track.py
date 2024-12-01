import os

import matplotlib.pyplot as plt

from dataset import FilteringData
from Net import Filter_MLP
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from utils import *

from sklearn.metrics import roc_curve


def eval_one_track(path):
    df = pd.read_csv(path, index_col=0)
    # print(df)

    df_l0 = df[df["layer"] == 0]
    df_l1 = df[df["layer"] == 1]
    df_l2 = df[df["layer"] == 2]
    df_l3 = df[df["layer"] == 3]

    if_hit = [len(df_l0) > 0, len(df_l1) > 0, len(df_l2) > 0, len(df_l3) > 0]
    if if_hit.count(True) < 4:
        print("Not enough hits, maximum 3 layers with hits")
        return 0

    hits_len = [len(df_l0), len(df_l1), len(df_l2), len(df_l3)]
    max_hits = np.max([len(df_l0), len(df_l1), len(df_l2), len(df_l3)])

    pairs = []
    pairs_true = []

    for hit_nb in tqdm(range(1000*max_hits)):
        hit_l0 = df_l0.iloc[hit_nb % hits_len[0]]
        hit_l1 = df_l1.iloc[hit_nb % hits_len[1]]
        hit_l2 = df_l2.iloc[hit_nb % hits_len[2]]
        hit_l3 = df_l3.iloc[hit_nb % hits_len[3]]
        pairs.append([hit_l0["hit_id"], hit_l1["hit_id"], hit_l2["hit_id"], hit_l3["hit_id"]])
        pairs_true.append([hit_l0["mcparticleID"], hit_l1["mcparticleID"], hit_l2["mcparticleID"], hit_l3["mcparticleID"]])


    pairs = np.array(pairs)
    pairs_true = np.array(pairs_true)


    # l0_sample = df_l0.sample(n=1, replace=True)
    # l1_sample = df_l1.sample(n=max_hits, replace=True)
    # l2_sample = df_l2.sample(n=max_hits, replace=True)
    # l3_sample = df_l3.sample(n=max_hits, replace=True)
    #
    # # print(l0_sample, l1_sample, l2_sample, l3_sample)
    #
    # id_l0 = l0_sample["hit_id"].values * np.ones_like(l1_sample["hit_id"].values)
    # id_l1 = l1_sample["hit_id"].values
    # id_l2 = l2_sample["hit_id"].values
    # id_l3 = l3_sample["hit_id"].values
    #
    # pid0 = l0_sample["mcparticleID"].values * np.ones_like(l1_sample["mcparticleID"].values)
    # pid1 = l1_sample["mcparticleID"].values
    # pid2 = l2_sample["mcparticleID"].values
    # pid3 = l3_sample["mcparticleID"].values

    # print(id_l0, id_l1, id_l2, id_l3)

    # pairs = np.concatenate([id_l0.reshape(-1, 1), id_l1.reshape(-1, 1), id_l2.reshape(-1, 1), id_l3.reshape(-1, 1)],
    #                        axis=1)
    # pairs_true = np.concatenate([pid0.reshape(-1, 1), pid1.reshape(-1, 1), pid2.reshape(-1, 1), pid3.reshape(-1, 1)],
    #                             axis=1)

    kind_pairs = 4 * (pairs_true[:, 1] == pairs_true[:, 0]) + 2 * (pairs_true[:, 2] == pairs_true[:, 0]) + 1 * (
                pairs_true[:, 3] == pairs_true[:, 0])

    trans_kind = {0: 0,
                  4: 1,
                  2: 2,
                  1: 3,
                  6: 4,
                  5: 5,
                  3: 6,
                  7: 7}

    # transform kind_pairs with trans_kind
    # print(kind_pairs)
    kind_pairs = np.vectorize(trans_kind.get)(kind_pairs)
    # print(kind_pairs)



    # print(pairs)
    print(pairs_true)
    # print(kind_pairs)

    sample = np.concatenate([pairs, kind_pairs.reshape(-1, 1)], axis=1)
    print(sample)

    data = construct_sample_var(sample, df)
    print(data)

    return data





def filtering(data):
    model = Filter_MLP()
    dict_path = os.path.join(workdir, "Model", "filtering_model.pth")
    model.load_state_dict(torch.load(dict_path))
    model.to(device)

    plt.figure(figsize=(10, 8))
    ax2 = plt.subplot(1, 1, 1)

    data = FilteringData(data)
    loader = DataLoader(data, batch_size=8192, shuffle=False)
    val_loss, tpr, fpr = eval_one_epoch(model, loader)

    for i in range(8):
        ax2.plot(fpr[i], tpr[i], '.-', label=str(i + 1) + "hit, auc=" + str(np.round(auc(fpr[i], tpr[i]), 4)))

    ax2.plot(fpr[8], tpr[8], '.-', label="micro, auc=" + str(np.round(auc(fpr[8], tpr[8]), 4)))

    ax2.legend()
    plt.show()


def eval_one_epoch(model, valid_loader):
    model.eval()
    loss_epoch = 0
    # print(len(valid_loader))
    link_preds = []
    labels = []
    for embed_link, label in tqdm(valid_loader):
        # print(batch_hit1.shape, batch_hit2.shape, true_dist.shape)
        link_pred = model(embed_link)
        link_preds.append(link_pred)
        labels.append(label)

    link_preds = torch.cat(link_preds, dim=0)
    labels = torch.cat(labels, dim=0)


    criterion = nn.CrossEntropyLoss()
    loss = criterion(link_preds, labels[:, 0].long())

    fpr = dict()
    tpr = dict()
    # roc_auc = dict()

    y = label_binarize(labels[:, 0].long().cpu(), classes=[0, 1, 2, 3, 4, 5, 6, 7])  # 三个类别
    # print(y)

    link_pred = nn.functional.softmax(link_preds, dim=1)
    link_pred = link_pred.cpu().detach().numpy()

    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], link_pred[:, i])

    fpr[8], tpr[8], _ = roc_curve(y.ravel(), link_pred.ravel())

    # roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr, tpr, thresholds = roc_curve(label.cpu().detach().numpy(), link_pred.cpu().detach().numpy(), pos_label=4)

    # auc = roc_auc_score(label.cpu().detach().numpy(), link_pred.cpu().detach().numpy())

    # print(loss)
    loss_epoch += loss.item()

    return loss_epoch / len(valid_loader), tpr, fpr


def main():
    path = os.path.join(workdir, "PreProcess", "degen_csv", "evt_8", "hit_9328.csv")
    # path = r"D:\files\pyproj\GNN\4hitv2\work_dir\PreProcess\degen_csv\evt_8\hit_9328.csv"

    data = eval_one_track(path)
    filtering(data)
    pass


if __name__ == "__main__":
    main()
