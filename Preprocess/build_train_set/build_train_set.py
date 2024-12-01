import numpy as np

from utils import *
import natsort


def sample_kind_n(df_l0, df_l1, df_l2, df_l3, kind_n):
    # one hot encoding by kind n
    trans_kind = {0: [0, 0, 0],
                  1: [1, 0, 0],
                  2: [0, 1, 0],
                  3: [0, 0, 1],
                  4: [1, 1, 0],
                  5: [1, 0, 1],
                  6: [0, 1, 1],
                  7: [1, 1, 1]}

    kind = trans_kind[kind_n]
    # print(kind_n)

    main_pid = df_l3["mcparticleID"].values[0]
    # print(main_pid  )
    select_hits = [-1, -1, -1, df_l3["hit_id"].values[0], kind_n]

    try:
        if kind[0]:
            df_l0 = df_l0[df_l0["mcparticleID"] == main_pid]
            df_l0 = df_l0.sample(1)
            select_hits[0] = df_l0["hit_id"].values[0]
        else:
            df_l0 = df_l0[df_l0["mcparticleID"] != main_pid]
            df_l0 = df_l0.sample(1)
            select_hits[0] = df_l0["hit_id"].values[0]

        if kind[1]:
            df_l1 = df_l1[df_l1["mcparticleID"] == main_pid]
            df_l1 = df_l1.sample(1)
            select_hits[1] = df_l1["hit_id"].values[0]
        else:
            df_l1 = df_l1[df_l1["mcparticleID"] != main_pid]
            df_l1 = df_l1.sample(1)
            select_hits[1] = df_l1["hit_id"].values[0]

        if kind[2]:
            df_l2 = df_l2[df_l2["mcparticleID"] == main_pid]
            df_l2 = df_l2.sample(1)
            select_hits[2] = df_l2["hit_id"].values[0]
        else:
            df_l2 = df_l2[df_l2["mcparticleID"] != main_pid]
            df_l2 = df_l2.sample(1)
            select_hits[2] = df_l2["hit_id"].values[0]
    except ValueError as e:
        # print(e)
        return 0

    # print(select_hits)
    select_hits = [int(i) for i in select_hits]
    return select_hits


def construct_sample(hit_df):
    hit_l0 = hit_df[hit_df["layer"] == 0]
    hit_l1 = hit_df[hit_df["layer"] == 1]
    hit_l2 = hit_df[hit_df["layer"] == 2]
    hit_l3 = hit_df[hit_df["layer"] == 3]

    if_hit = [len(hit_l0) > 0, len(hit_l1) > 0, len(hit_l2) > 0, len(hit_l3) > 0]

    if if_hit.count(True) < 4:
        # print("Not enough hits, maximum 2 layers with hits")
        return 0

    sample_one_hit = []

    for i in range(8):
        sample_n_kind = sample_kind_n(hit_l0, hit_l1, hit_l2, hit_l3, i)

        sample_one_hit.append(sample_n_kind)


    # print(sample_one_hit)

    return sample_one_hit


def process_one_evt(evt_path):
    hit_paths = os.listdir(evt_path)

    sample_n = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: []
    }

    for hit_path in tqdm(hit_paths):
        hit_df = pd.read_csv(os.path.join(evt_path, hit_path), index_col=0)

        sample = construct_sample(hit_df)
        if sample == 0:
            continue

        for i in range(8):
            # print(sample[i])
            if sample[i] != 0:
                sample_n[i].append(sample[i])

    sample_index = []

    for i in range(8):
        sample_n[i] = np.array(sample_n[i], dtype=int)
        sample_index.append(sample_n[i])
    # print(sample_n)

    sample_index = np.concatenate(sample_index, axis=0)
    # print(sample_index)

    return sample_index

    # plt.hist(sample_index[:, 4], bins=8)
    # plt.show()

    # construct_sample_var(sample_index, hit_df)





def main():
    file_list = os.listdir(os.path.join(workdir, "PreProcess", "degen_csv"))
    file_list = natsort.natsorted(file_list)

    samples = []

    with Pool(16) as p:
        for file in tqdm(file_list):
            evt_path = os.path.join(workdir, "PreProcess", "degen_csv", file)

            samples.append(p.apply_async(process_one_evt, args=(evt_path,)))


        samples = [sample.get() for sample in samples]





    # for file in file_list:
    #     evt_path = os.path.join(workdir, "PreProcess", "degen_csv", file)
    #     # df_path = os.path.join(workdir, "RawData", "processed_csv", file+".csv")
    #     # df = pd.read_csv(df_path, index_col=0)
    #     # hit_df.append(df)
    #     # for times in range(100):
    #     sample = process_one_evt(evt_path)
    #     samples.append(sample)

    hit_df = []



    for file in tqdm(file_list):
        paths = os.listdir(os.path.join(workdir, "PreProcess", "degen_csv", file))
        for path in paths:
            path = os.path.join(workdir, "PreProcess", "degen_csv", file, path)
            # print(path)
            hit_df.append(pd.read_csv(path, index_col=0))

    samples = np.concatenate(samples, axis=0)
    hit_df = pd.concat(hit_df, axis=0)
    hit_df = hit_df.drop_duplicates()
    print(hit_df)

    data = construct_sample_var(samples, hit_df)
    np.random.shuffle(data)

    train_data, test_data = split_data(data)

    np.save(os.path.join(workdir, "PreProcess", "sample", "train.npy"), train_data)
    np.save(os.path.join(workdir, "PreProcess", "sample", "test.npy"), test_data)

    # break


def split_data(data):
    np.random.shuffle(data)
    train_data = data[:int(data.shape[0] * 0.8)]
    test_data = data[int(data.shape[0] * 0.8):]

    return train_data, test_data


if __name__ == "__main__":
    main()
