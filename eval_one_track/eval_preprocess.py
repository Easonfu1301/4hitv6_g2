from utils import *
from sklearn.neighbors import KDTree
import warnings

warnings.filterwarnings("ignore")
import natsort


def cut_more_than_5_hits(hit_df):
    # count = hit_df["mcparticleID"].value_counts()
    # count = count[(count > 4) | (count < 3)]
    # count = count[count != 4]
    # count = count[count > 5]

    # for pid in count.index:
    #     hit_df = hit_df[hit_df["mcparticleID"] != pid]


    # for pid in hit_df["mcparticleID"].unique():
    #     hit_df_pid = hit_df[hit_df["mcparticleID"] == pid]
    #     layer_count = hit_df_pid["layer"].value_counts()
    #     # print(layer_c ount,len(layer_count))
    #     if len(layer_count) != 4:
    #         hit_df = hit_df[hit_df["mcparticleID"] != pid]


    # print(hit_df["mcparticleID"].value_counts())
    return hit_df


def find_nearest_hit(hit_df):
    # xyz = hit_df[["x", "y", "z"]].values

    layer_hit_frame = []
    for layer in range(4):
        layer_hits = hit_df[hit_df["layer"] == layer]
        layer_xyz = layer_hits[["x", "y", "z"]].values

        tree = KDTree(layer_xyz)
        dist, ind = tree.query(layer_xyz, k=2)

        layer_hits["nearest_dist"] = dist[:, 1]

        layer_hit_frame.append(layer_hits)

    layer_hit_frame = pd.concat(layer_hit_frame)
    # print(layer_hit_frame)
    layer_hit_frame.sort_values("hit_id", inplace=True)
    return layer_hit_frame


def cut_range(hit_df, hit_index):
    hit = hit_df[hit_df["hit_id"] == hit_index]
    # print(hit)
    xyz_hit = hit[["x", "y", "z"]].values[0]
    hit_df_123 = hit_df[hit_df["layer"] != 3]
    # print(hit_df_123)

    xyz_all = hit_df_123[["x", "y", "z"]].values
    id_all = hit_df_123["hit_id"].values

    cos_theta = np.dot(xyz_all, xyz_hit) / (np.linalg.norm(xyz_all, axis=1) * np.linalg.norm(xyz_hit))
    cos_theta[cos_theta > 1] = 1
    theta = np.arccos(cos_theta)

    index = np.where(theta < 0.03)
    # print(index, theta.shape, index[0].shape)

    hit_df_123 = hit_df_123.loc[id_all[index]]
    hit_df_0123 = pd.concat([hit, hit_df_123])
    hit_df_0123.sort_values("hit_id", inplace=True)
    # print(hit_df)

    return hit_df_0123


def process_one_evt(path, store_path):
    df = pd.read_csv(path, index_col=0)
    df = cut_more_than_5_hits(df)
    df = find_nearest_hit(df)

    df_l3 = df[df["layer"] == 3]

    for i in tqdm(df_l3["hit_id"].values):
        cut_df = cut_range(df, i)
        cut_df.to_csv(os.path.join(store_path, f"hit_{i}.csv"))
        # break

    return 1

    # print(df)
    # print(df)


def main():
    file_list = os.listdir(os.path.join(workdir, "RawData", "processed_csv"))
    file_list = natsort.natsorted(file_list)

    re = []
    with Pool(16) as p:
        for file in file_list[900:950]:
            store_path = os.path.join(workdir, "Eval", "Pre", file.split(".csv")[0])
            os.makedirs(store_path, exist_ok=True)
            re.append(p.apply_async(process_one_evt, args=(os.path.join(workdir, "RawData", "processed_csv", file), store_path)))
        re = [r.get() for r in re]



    # for file in file_list[:100]:
    #     store_path = os.path.join(workdir, "PreProcess", "degen_csv", file.split(".csv")[0])
    #     os.makedirs(store_path, exist_ok=True)
    #     process_one_evt(os.path.join(workdir, "RawData", "processed_csv", file), store_path)
    #     # break


if __name__ == "__main__":
    main()
