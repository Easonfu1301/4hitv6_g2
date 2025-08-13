import os.path
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

from dataset import FilteringData
from Net import Filter_MLP
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from utils import *
import torch.nn.functional as F
import matplotlib.ticker as mtick

from sklearn.metrics import roc_curve

model = Filter_MLP()
dict_path = os.path.join(workdir, "Model", "filtering_model.pth")
model.load_state_dict(torch.load(dict_path))
model.to(device)

P_MIN = 0

def eval_one_track(path, df_evt):
    df = pd.read_csv(path, index_col=0)
    # print(df)

    df_l0 = df[df["layer"] == 0]
    df_l1 = df[df["layer"] == 1]
    df_l2 = df[df["layer"] == 2]
    df_l3 = df[df["layer"] == 3]

    if_hit = [len(df_l0) > 0, len(df_l1) > 0, len(df_l2) > 0, len(df_l3) > 0]

    df_pid = df_evt[df_evt["mcparticleID"] == df["mcparticleID"].values[-1]]
    if df_pid["p"].values[-1] < P_MIN:
        # print("Low momentum")
        return None


    if if_hit.count(True) < 4:
        # print("Not enough hits, maximum 3 layers with hits")
        return None

    weight_dict = [{}, {}, {}, {}]

    for iter in range(5):
        max_hits = np.max([len(df_l0), len(df_l1), len(df_l2), len(df_l3)])

        hits_len = [len(df_l0), len(df_l1), len(df_l2), len(df_l3)]

        pairs = []
        pairs_true = []

        for hit_nb in range(2 * max_hits):
            hit_l0 = df_l0.iloc[hit_nb % hits_len[0]]
            hit_l1 = df_l1.iloc[hit_nb % hits_len[1]]
            hit_l2 = df_l2.iloc[hit_nb % hits_len[2]]
            hit_l3 = df_l3.iloc[hit_nb % hits_len[3]]
            pairs.append([hit_l0["hit_id"], hit_l1["hit_id"], hit_l2["hit_id"], hit_l3["hit_id"]])
            pairs_true.append(
                [hit_l0["mcparticleID"], hit_l1["mcparticleID"], hit_l2["mcparticleID"], hit_l3["mcparticleID"]])
        pairs = np.array(pairs)
        pairs_true = np.array(pairs_true)

        data = construct_sample(pairs, pairs_true, df)
        pred = predict(data)
        weight_dict = get_weight(pairs, pairs_true, pred, df)

        hits = [df_l0, df_l1, df_l2, df_l3]

        for i in range(1, 4):
            # get last keys
            hit_id_cut = list(weight_dict[i].keys())[:1 + int(0.3 * len(weight_dict[i]))]
            # print(hit_id_cut)
            hits[i] = hits[i][hits[i]["hit_id"].isin(hit_id_cut)]

        df_l0, df_l1, df_l2, df_l3 = hits

    # for i in range(4):
    #     print(f"layer {i}, weight by layer, {weight_dict[i]}")

    # df_all = pd.read_csv(r"D:\files\pyproj\GNN\4hitv2\work_dir\RawData\mini15_allDets_hits_1000eV_noCorrect.csv")
    # df_all = df_all[df_all["eventID"] == df["eventID"].values[0]]
    # print(f"eventID: {df}")

    # print(df_pid)
    # print(df)
    # print(df_pid)

    found = [0, 0, 0, 1]
    for i in range(0, 3):
        if weight_dict[i][list(weight_dict[i].keys())[0]][1] == weight_dict[3][list(weight_dict[3].keys())[0]][1]:
            found[i] = 1

    found_sum = np.sum(found)
    hit_keys = [int(list(weight_dict[i].keys())[0]) for i in range(4)]
    type_keys = [int(weight_dict[i][list(weight_dict[i].keys())[0]][1] == weight_dict[3][list(weight_dict[3].keys())[0]][1]) for i in range(4)]
    trans_type = {0: [0, 0, 0, 1],
                  1: [1, 0, 0, 1],
                  2: [0, 1, 0, 1],
                  3: [0, 0, 1, 1],
                  4: [1, 1, 0, 1],
                  5: [1, 0, 1, 1],
                  6: [0, 1, 1, 1],
                  7: [1, 1, 1, 1]}

    for key in trans_type.keys():
        if type_keys == trans_type[key]:
            type_keys = key
            break

    hit_keys.append(type_keys)
    hit_keys.append(int(found_sum))
    hit_keys.append(len(df_pid))
    hit_keys.append(df_pid["p"].values[-1])
    hit_keys.append(df_pid["mcparticleID"].values[-1])

    # print(hit_keys)




    # print(hit_keys)

    # print(f"found {found_sum} out of {len(df_pid)} hits")

    return int(found_sum), len(df_pid), hit_keys


def construct_sample(pairs, pairs_true, df):
    kind_pairs = 4 * (pairs_true[:, 0] == pairs_true[:, 3]) + 2 * (pairs_true[:, 1] == pairs_true[:, 3]) + 1 * (
            pairs_true[:, 2] == pairs_true[:, 3])

    trans_kind = {0: 0,
                  4: 1,
                  2: 2,
                  1: 3,
                  6: 4,
                  5: 5,
                  3: 6,
                  7: 7}
    kind_pairs = np.vectorize(trans_kind.get)(kind_pairs)

    sample = np.concatenate([pairs, kind_pairs.reshape(-1, 1)], axis=1)

    data = construct_sample_var(sample, df)

    return data





def predict(data, model=model):
    data = FilteringData(data)
    data = DataLoader(data, batch_size=int(np.min([len(data), 1024])), shuffle=False)

    preds = []

    for embed_link, label in data:
        link_pred = model(embed_link)
        preds.append(link_pred)

    preds = torch.cat(preds, dim=0)

    # softmax

    preds = F.softmax(preds, dim=1)

    return preds


def get_weight(pair, pair_true, pred, df):
    weight_dict = {}
    trans_kind = {0: [0, 0, 0, 1],
                  1: [1, 0, 0, 1],
                  2: [0, 1, 0, 1],
                  3: [0, 0, 1, 1],
                  4: [1, 1, 0, 1],
                  5: [1, 0, 1, 1],
                  6: [0, 1, 1, 1],
                  7: [1, 1, 1, 1]}

    pred = pred.cpu().detach().numpy()

    for i in range(pair.shape[0]):
        for kind in range(8):
            kind_transed = trans_kind[kind]
            for j in range(4):
                # if kind_transed[j - 1] == 1:
                try:
                    weight_dict[pair[i, j]][0] += pred[i, kind] * kind_transed[j]
                    weight_dict[pair[i, j]][1] += 1 * kind_transed[j]
                except Exception as e:
                    weight_dict[pair[i, j]] = [pred[i, kind], 1, pair_true[i, j]]

    for key in weight_dict.keys():
        weight_dict[key] = [weight_dict[key][0] / weight_dict[key][1], weight_dict[key][2]]

    weight_by_layer = [{}, {}, {}, {}]
    for key in weight_dict.keys():
        layer = int(df.loc[key, "layer"])
        weight_by_layer[layer][key] = weight_dict[key]
        # sort by value\
        weight_by_layer[layer] = dict(sorted(weight_by_layer[layer].items(), key=lambda x: x[1][0], reverse=True))

    # for i in range(4):
    #     print(f"layer {i}, weight by layer, {weight_by_layer[i]}")

    return weight_by_layer



def process_one_evt(df, evt):
    hit_keys = []
    map_found = {}
    count_key = {}

    paths = os.path.join(workdir, "Eval", "Pre", f"evt_{evt}")
    df_evt = df[df["eventID"] == evt]
    count = df_evt["mcparticleID"].value_counts()
    # print(count)
    for key in count.keys():
        hits = df_evt[df_evt["mcparticleID"] == key]
        if hits["p"].values[0] < P_MIN:
            continue
        len_l3 = len(hits[hits["layer"] == 3])
        if len_l3 == 0:
            len_l3 = 1
        try:
            count_key[int(count[key])] += 1 * len_l3
        except Exception as e:
            count_key[int(count[key])] = 1 * len_l3

    files = os.listdir(paths)

    for file in tqdm(files):
        path = os.path.join(paths, file)

        try:
            found, total, hit_key = eval_one_track(path, df_evt)
            hit_keys.append(hit_key)
        except Exception as e:
            # print(e)
            continue

        try:
            map_found[total].append(found)
        except Exception as e:
            map_found[total] = [found]

    # print(np.array(hit_keys))
    np.save(os.path.join(workdir, "Eval", "evt_%i.npy" % evt), np.array(hit_keys))
    return map_found, count_key


def main():
    # path = r"D:\files\pyproj\GNN\4hitv2\work_dir\PreProcess\degen_csv\evt_8\hit_9026.csv"
    path = os.path.join(workdir, "RawData", "mini15_allDets_hits_1000eV_noCorrect.csv")
    df = pd.read_csv(path, index_col=0)

    map_found = {}
    count_key = {}


    # for evt in range(901, 911):
    #     map_found0, count_key0 = process_one_evt(df, evt)

    with Pool(8) as p:
        res = [p.apply_async(process_one_evt, args=(df, evt)) for evt in range(201, 900)]
        res = [r.get() for r in res]
        for map_found0, count_key0 in res:
            for key in map_found0.keys():
                try:
                    map_found[key] += map_found0[key]
                except Exception as e:
                    map_found[key] = map_found0[key]

            for key in count_key0.keys():
                try:
                    count_key[key] += count_key0[key]
                except Exception as e:
                    count_key[key] = count_key0[key]


    print(map_found)



    print(count_key)
    plt.figure()
    bins = 40  # 分组数量
    bar_width = 0.11  # 每个柱的宽度
    for idx, key in enumerate(map_found.keys()):
        if key == 1 or key > 5:
            continue
        # height of bar is the percentage of the total number of tracks
        hist, bin_edges = np.histogram(map_found[key], bins=bins, range=(0.5, 4.5))
        # plt.bar(bin_edges[:-1] + (idx - 2) * bar_width, hist / len(count_key[key]), bar_width,
        plt.bar(bin_edges[:-1] + (idx - 2) * bar_width, hist / count_key[key], bar_width,
                        label=f"find n with {key} hit, all {count_key[key]}, constructable {len(map_found[key])} tracks")
        # formatter = mtick.PercentFormatter(xmax=len(map_found[key]))

    # set minorticks
    plt.minorticks_on()

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))

    plt.grid(which='minor', linestyle=':', linewidth='0.3', color='gray')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.xlabel("number of found tracks")
    plt.ylabel("percentage of total tracks")


    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(workdir, "Eval", "eval.png"), dpi=300)
    plt.show()
    # print(weight_dict)

    # df = pd.read_csv(path)
    # print(df)

    return map_found


if __name__ == "__main__":
    main()
