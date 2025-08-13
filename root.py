import numpy as np

from utils import *



def vis_l0_hit_ratio(df, p):
    df = df[df["p"] > p]

    pids = df["mcparticleID"].unique()

    count = 0
    count_4 = 0

    for pid in pids:
        trk = df[df["mcparticleID"] == pid]
        if len(trk) == 4:
            count_4 += 1

            if len(trk[trk["layer"] == 0]) > 0 :
                count += 1


    print(p, "\thitl1", count/ count_4, "\t4hit", count_4 / len(pids))


    return count / count_4, count_4 / len(pids)


















if __name__ == "__main__":


    df = pd.read_csv(os.path.join(workdir, "RawData", "mini15_allDets_hits_1000eV_noCorrect.csv"), index_col=0)
    df = df[df["eventID"] == 1]
    p = 2500

    print(df)

    ratios = np.ones_like(np.linspace(0, 50*1e3, 20))
    fracs = np.ones_like(np.linspace(0, 50*1e3, 20))

    for idx, p in enumerate(np.linspace(0, 50*1e3, 20)):

        r, f = vis_l0_hit_ratio(df, p)

        ratios[idx] = r
        fracs[idx] = f

    plt.plot(np.linspace(0, 50*1e3, 20), ratios)
    plt.plot(np.linspace(0, 50*1e3, 20), fracs)
    plt.show()