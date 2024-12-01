import numpy as np

from utils import *
import natsort




def post_process(file_path, evt_df):
    restored_trk = np.load(file_path)
    # print(restored_trk.shape)
    trk_df = build_trk_df(restored_trk)

    recon_df = get_trk_num(evt_df)

    stas_df = stastic_count(trk_df, recon_df)








def build_trk_df(restored_trk):
    trk_df = pd.DataFrame(restored_trk, columns=["hit1", "hit2", "hit3", "hit4", "kind", "found", "total", "p", "trk_pid"])
    trk_df = trk_df.astype(int)
    # trk_df.sort_values(by="found", ascending=False, inplace=True)
    # print(trk_df)
    # print(trk_df[trk_df["p"] >1500])
    # print(trk_df[(trk_df["p"] >1500) & (trk_df["found"] >2)])

    ##################   store to be done   ##################

    ########################################################

    return trk_df


def get_trk_num(evt_df):
    pids = evt_df["mcparticleID"].value_counts()

    pids = pids[pids >1]
    recon = np.zeros((len(pids), 3))

    for i, pid in enumerate(pids.index):
        recon[i, 0] = pid
        recon[i, 1] = evt_df[evt_df["mcparticleID"] == pid]["p"].values[0]
        recon[i, 2] = pids[pid]




    recon_df = pd.DataFrame(recon, columns=["trk_pid", "p", "total"])
    # print(recon_df)
    return recon_df


def stastic_count(trk_df, recon_df):

    # print(recon_df)
    # print(trk_df)

    stas_df = pd.DataFrame(columns=["trk_pid", "p", "total", "found_min", "found_max", "if_recon", "recon_count"])

    pids = recon_df["trk_pid"].values

    for idx, pid in enumerate(pids):
        stas_df.loc[idx, "trk_pid"] = pid
        stas_df.loc[idx, "p"] = recon_df.loc[idx, "p"]
        stas_df.loc[idx, "total"] = recon_df.loc[idx, "total"]

        found  = trk_df[trk_df["trk_pid"] == pid]

        stas_df.loc[idx, "found_min"] = found["found"].min() if len(found) > 0 else 0
        stas_df.loc[idx, "found_max"] = found["found"].max() if len(found) > 0 else 0
        stas_df.loc[idx, "if_recon"] = 1 if found["found"].max() > 2 else 0
        stas_df.loc[idx, "recon_count"] = len(found)

    # print(stas_df)

    recon_able = stas_df[stas_df["total"] >=3]
    recon_able = recon_able[recon_able["p"] > 1000]
    cc = recon_able["if_recon"].value_counts()
    print(cc)
    print(recon_able)

    return stas_df














    pass





def main():

    path = os.path.join(workdir, "Eval")
    # list all the files in the directory end with .npy
    file_list = [f for f in os.listdir(path) if f.endswith('.npy')]
    file_list = natsort.natsorted(file_list)




    for file in file_list[:1]:
        evt = file.split(".")[0]
        evt_df = pd.read_csv(os.path.join(workdir, "RawData", "processed_csv", f"{evt}.csv"))
        post_process(os.path.join(path, file), evt_df)





if __name__ == "__main__":
    main()