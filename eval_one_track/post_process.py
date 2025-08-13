import matplotlib.pyplot as plt
import numpy as np

from utils import *
import natsort
from scipy.stats import binom



def post_process(file_path, evt_df):
    restored_trk = np.load(file_path)
    # print(restored_trk.shape)
    trk_df = build_trk_df(restored_trk)

    recon_df = get_trk_num(evt_df)

    stas_df = stastic_count(trk_df, recon_df)
    # plot_efficiency(stas_df, 100)

    return stas_df








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

    # print(stas_df[20000 < stas_df["p"] < 30000])

    # df_20_30 = stas_df[(stas_df["p"] > 20000) & (stas_df["p"] < 30000)]
    # print(df_20_30, len(df_20_30))
    # df_20_30_cut2 = df_20_30[df_20_30["total"] >= 3]
    # print(df_20_30_cut2, len(df_20_30_cut2))



    # counts = df_20_30
    # print(counts)

    # recon_able = stas_df[stas_df["total"] >=3]
    # recon_able = recon_able[recon_able["p"] > 1000]
    # cc = recon_able["if_recon"].value_counts()
    # print(cc)
    # print(recon_able)

    return stas_df














    pass



def plot_efficiency(stas_df, bin):
    stas_df0 = stas_df.copy()

    stas_df = stas_df[stas_df["total"] >=3]

    p_min = stas_df["p"].min()
    p_max = 50*1000
    # p_max = stas_df["p"].max()

    p_range = np.linspace(p_min, p_max, bin)



    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    # hist, bins = np.histogram(stas_df["p"], bins=bin, range=(p_min, p_max))

    ax.hist(stas_df["p"]/1000, bins=bin, range=(p_min/1000, p_max/1000), histtype="step", color="b", label="p distribution", log=True)



    eff = np.zeros((bin-1, 8))

    for p in range(bin-1):
        eff[p, 0] = p_range[p] / 1000
        eff[p, 1] = stas_df[(stas_df["p"] >= p_range[p]) & (stas_df["p"] < p_range[p+1])]["if_recon"].mean()
        eff[p, 2] = stas_df[stas_df["p"] >= p_range[p]]["if_recon"].mean()


        alpha = 0.05

        n_frac = len(stas_df[(stas_df["p"] >= p_range[p]) & (stas_df["p"] < p_range[p+1])])
        n_all = len(stas_df[stas_df["p"] >= p_range[p]])
        p_hat_frac = eff[p, 1]
        p_hat_all = eff[p, 2]


        lower_bound_frac = binom.ppf(alpha / 2, n_frac, p_hat_frac) / n_frac
        upper_bound_frac = binom.ppf(1 - alpha / 2, n_frac, p_hat_frac) / n_frac

        lower_bound_all = binom.ppf(alpha / 2, n_all, p_hat_all) / n_all
        upper_bound_all = binom.ppf(1 - alpha / 2, n_all, p_hat_all) / n_all


        eff[p, 3] = lower_bound_frac
        eff[p, 4] = upper_bound_frac

        eff[p, 5] = lower_bound_all
        eff[p, 6] = upper_bound_all


        n_frac = stas_df0[(stas_df0["p"] >= p_range[p]) & (stas_df0["p"] < p_range[p+1])]
        # n_frac = stas_df0[stas_df0["p"] >= p_range[p]]

        # print(n_frac[n_frac["if_recon"] == 0])
        # print(n_frac[n_frac["if_recon"] == 1])

        flase_num = n_frac[n_frac["if_recon"] == 0]["recon_count"].sum()
        true_num = n_frac[n_frac["if_recon"] == 1]["recon_count"].sum()

        try:
            eff[p, 7] = flase_num / true_num
        except ZeroDivisionError:
            eff[p, 7] = np.nan
    # ax2.plot(eff[:, 0], eff[:, 1], "r", label="Efficiency")
    # error bar
    ax2.errorbar(eff[:, 0], eff[:, 1], yerr=[eff[:, 1] - eff[:, 3], eff[:, 4] - eff[:, 1]], fmt='.-', color='r', ecolor='b', elinewidth=2, capsize=4, label="Efficiency of p in bin")
    ax2.errorbar(eff[:, 0], eff[:, 2], yerr=[eff[:, 2] - eff[:, 5], eff[:, 6] - eff[:, 2]], fmt='.-', color='y', ecolor='gray', elinewidth=2, capsize=4, label="Efficiency of p above bin")
    ax2.plot(eff[:, 0], eff[:, 7], "g", label="False recon count / right recon count")


    # minor_ticks on
    ax2.minorticks_on()
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')

    ax.set_xlabel("Momentum [GeV]")
    ax.set_ylabel("Counts")
    ax2.set_ylabel("Efficiency")

    ax2.legend()
    ax.legend()
    fig.savefig(os.path.join(workdir, "Eval", "Efficiency1.png"))
    plt.show()







    pass



def main():

    path = os.path.join(workdir, "Eval")
    # list all the files in the directory end with .npy
    file_list = [f for f in os.listdir(path) if f.endswith('.npy')]
    file_list = natsort.natsorted(file_list)



    stas_dfs = []
    for file in tqdm(file_list[:100]):
        # print(file)
        evt = file.split(".")[0]
        evt_df = pd.read_csv(os.path.join(workdir, "RawData", "processed_csv", f"{evt}.csv"))
        stas_df = post_process(os.path.join(path, file), evt_df)
        stas_dfs.append(stas_df)

    stas_dfs = pd.concat(stas_dfs)



    plot_efficiency(stas_dfs, 100)







if __name__ == "__main__":
    main()