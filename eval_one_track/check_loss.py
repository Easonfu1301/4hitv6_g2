import numpy as np

from utils import *
from Net import Filter_MLP
from dataset import FilteringData
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import  roc_curve, auc

model = Filter_MLP()
dict_path = os.path.join(workdir, "Model", "filtering_model.pth")
model.load_state_dict(torch.load(dict_path))
model.to(device)






def check_loss(model, pairs):
    pairs  = FilteringData(pairs)
    loader = DataLoader(pairs, batch_size=512, shuffle=False, num_workers=0)

    preds = []
    labels = []

    for embed_link, label in tqdm(loader):
        link_pred = model(embed_link)
        preds.append(link_pred)
        labels.append(label)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    # softmax

    preds = F.softmax(preds, dim=1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(preds, labels[:, 0].long())

    y = label_binarize(labels[:, 0].long().cpu(), classes=[0, 1, 2, 3, 4, 5, 6, 7])  # 三个类别
    # print(y)

    link_pred = nn.functional.softmax(preds, dim=1)
    link_pred = link_pred.cpu().detach().numpy()
    fpr = dict()
    tpr = dict()
    for i in range(8):

        fpr[i], tpr[i], _ = roc_curve(y[:, i], link_pred[:, i])

    fpr[8], tpr[8], _ = roc_curve(y.ravel(), link_pred.ravel())

    for i in range(8):
        plt.plot(fpr[i], tpr[i], label=f"Class {i}, AUC: {auc(fpr[i], tpr[i]):.3f}, sum: {np.sum(y[:, i])}")
    plt.plot(fpr[8], tpr[8], label=f"Total, AUC: {auc(fpr[8], tpr[8]):.3f}")
    plt.legend()
    plt.show()






def main():
    pairs = []
    dfs = []

    for r in range(901, 911):
        pairs.append(np.load(os.path.join(workdir, "Eval", "evt_%i.npy"%r)))
        files = os.listdir(os.path.join(workdir, "Eval", "Pre", "evt_%i"%r))
        for file in files:
            # print(file)
            dfs.append(pd.read_csv(os.path.join(workdir, "Eval", "Pre", "evt_%i"%r, file), index_col=0))

    # print(dfs)

    dfs = pd.concat(dfs)
    dfs.drop_duplicates(inplace=True)
    pairs = np.concatenate(pairs)

    # print(dfs)
    pairs_var = construct_sample_var(pairs, dfs)
    # print(pairs_var)
    check_loss(model, pairs_var)





if  __name__ == "__main__":
    main()
