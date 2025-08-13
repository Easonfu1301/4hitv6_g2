from utils import *


def vis_one_trk(df, hit_array, ax):
    # print(df)
    # print(hit_array)

    # print(df)
    # print(hit_array)

    xyz = df.loc[hit_array, ["x", "y", "z", "mcparticleID"]].values
    true_id = xyz[-1, 3]

    df_true = df[df["mcparticleID"] == true_id]
    xyz_true = df_true[["x", "y", "z"]].values

    if list(xyz[:, 3] == xyz[-1, 3]).count(True) == 2 and len(xyz_true) == 4:
        # return

        # print(xyz)
        ax.plot(xyz_true[:, 0], xyz_true[:, 1], xyz_true[:, 2], marker='.', markersize=5, linestyle='-', color='g',
                alpha=0.5)

        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', markersize=5, linestyle='-', color='r', alpha=0.3)
        # plt.show()


def vis_all_trk(df, hit_array):
    # print(df)
    # print(hit_array)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 2400, marker='o', color='b')

    for i in tqdm(range(0, hit_array.shape[0])):
        vis_one_trk(df, hit_array[i, :-5], ax)
        # break

    # equal axis
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(os.path.join(workdir, "Eval", "vis_all_trk.png"), dpi=1000)
    import pickle
    pickle.dump(fig, open(os.path.join(workdir, "Eval", "vis_all_trk.pkl"), "wb"))
    plt.show()


def main():
    evt = 211
    df = pd.read_csv(os.path.join(workdir, "RawData", "processed_csv", "evt_%i.csv" % evt), index_col=0)
    # print(df)
    hit_array = np.load(os.path.join(workdir, "Eval", "evt_%i.npy" % evt))
    # print(hit_array)

    vis_all_trk(df, hit_array)


if __name__ == "__main__":
    main()
