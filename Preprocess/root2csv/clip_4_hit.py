import os.path
import pandas as pd

csv_path = os.path.join(workdir, r"raw_hit\mini15_allDets_hits_1000eV_noCorrect.csv")


def clip_4_hit(df, evt):
    # print(evt)
    df_evt = df[df["eventID"] == evt]
    hit_num = get_hit_num(df_evt)
    df_evt_reduce = df_evt[df_evt["mcparticleID"].isin(hit_num.keys().values[hit_num.values == 4])]
    # print(df_evt_reduce.shape[0])
    del df_evt

    return df_evt_reduce


def get_hit_num(df):
    hit_num = df["mcparticleID"].value_counts()

    return hit_num


def main():
    # df_evt_reduce = []
    df = pd.read_csv(csv_path)

    # with Pool(8) as pool:
    #     results = [pool.apply_async(clip_4_hit, (df.copy(), evt,)) for evt in tqdm(range(1, 1001))]
    #     df_evt_reduce = [result.get() for result in tqdm(results)]

    df_evt_reduce = []

    for i in tqdm(range(1, 1001)):
        df_evt_reduce.append(clip_4_hit(df, i))


    df_evt_reduce = pd.concat(df_evt_reduce)
    df_evt_reduce = df_evt_reduce.sort_values(by=["hit_id"])
    print(df_evt_reduce)

    df_evt_reduce.to_csv(os.path.join(workdir, r"raw_hit\mini15_allDets_hits_1000eV_noCorrect_cliped.csv"),
                         columns=["hit_id", "x", "y", "z", "layer_id", "mcparticleID", "eventID", "layer"])


if __name__ == "__main__":
    main()
