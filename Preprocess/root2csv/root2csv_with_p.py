import os.path
import numpy as np
import uproot as uproot
import pandas as pd
from utils import *


def root2df(file_path=rootpath):
    """
    Convert root file to pandas dataframe
    """
    # Open the root file
    root_file = uproot.open(file_path)
    dfs = []
    # Get the tree
    for i in tqdm(range(4)):
        tree = root_file["layer%i" % i]
        df_temp = pd.DataFrame()

        df_temp["x"] = tree["layer%i_x" % i].array(library="np")
        df_temp["y"] = tree["layer%i_y" % i].array(library="np")
        df_temp["z"] = tree["layer%i_z" % i].array(library="np")
        df_temp["layer_id"] = tree["layer%i_id" % i].array(library="np")
        df_temp["mcparticleID"] = tree["mcparticleID_L%i" % i].array(library="np")
        # df_temp["mcparticleID"] = tree["layer%i_id" % i].array(library="np")
        df_temp["eventID"] = tree["eventID_L%i" % i].array(library="np")
        df_temp["p"] = tree["p_L%i" % i].array(library="np")
        df_temp["isPrim"] = tree["isPrim_L%i" % i].array(library="np")
        df_temp["isDecay"] = tree["isDecay_L%i" % i].array(library="np")

        df_temp["vertex_x"] = tree["layer%i_vertex_x" % i].array(library="np")
        df_temp["vertex_y"] = tree["layer%i_vertex_y" % i].array(library="np")
        df_temp["vertex_z"] = tree["layer%i_vertex_z" % i].array(library="np")

        df_temp["layer"] = i * np.ones(df_temp.shape[0])

        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df["hit_id"] = np.arange(df.shape[0])

    export_path = os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv")
    df = df[df["eventID"] <= 1000]
    print(df)
    df.to_csv(export_path,
              columns=["hit_id", "x", "y", "z", "p", "isPrim", "isDecay", "vertex_x", "vertex_y", "vertex_z",
                       "layer_id", "mcparticleID", "eventID", "layer"])

    return df


def main(force=True):
    # if os.path.exists(os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv")) and not force:
    #     print("File already exists")
    # else:
    root2df()


if __name__ == "__main__":
    # main(force=False)
    df = pd.read_csv(os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect_10000.csv"), index_col=0)
    print(df)
    # print(df.head())
    # print(df)

    evt_id = df["eventID"].unique()

    evt_id = evt_id[evt_id <= 1000]
    # print(evt_id)
    #
    df = df[df["eventID"].isin(evt_id)]
    print(df)
    #
    df.to_csv(os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv"))

    # rt = uproot.open(
    #     r"D:\files\pyproj\GNN\4hitv2\Preprocess\root2csv\mini15_allDets_hits_10000eV_noCorrect_moreInfo.root")
    # print(rt["layer0"].keys())
    # print(np.unique(rt["layer0"]["eventID_L%i" % 0].array(library="np")))
    # x = rt["layer0"]["p_L0"].array(library="np")
    # # x=x[x>5000]
    # plt.hist(x, bins=1000, log=True)
    # plt.show()
    pass
