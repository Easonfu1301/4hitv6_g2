import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from .print_deco import only_check

matplotlib.use('tkAgg')

@only_check
def plot_hits(hit_df):
    x = hit_df["x"].values
    y = hit_df["y"].values
    z = hit_df["z"].values

    ids = hit_df["mcparticleID"].unique()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 2000, c='r', marker='o')

    for id in tqdm(ids, desc="plotting hits"):
        hit = hit_df[hit_df["mcparticleID"] == id]
        x = hit["x"].values
        y = hit["y"].values
        z = hit["z"].values
        plt.plot(x, y, z, '.-', label=id)


    # equal xy axis, but the z
    ax.set_aspect('equal')




    plt.show()



