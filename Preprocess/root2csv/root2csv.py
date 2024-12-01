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
    for i in range(4):
        tree = root_file["layer%i" % i]
        df_temp = pd.DataFrame()

        df_temp["x"] = tree["layer%i_x" % i].array(library="np")
        df_temp["y"] = tree["layer%i_y" % i].array(library="np")
        df_temp["z"] = tree["layer%i_z" % i].array(library="np")
        df_temp["layer_id"] = tree["layer%i_id" % i].array(library="np")
        df_temp["mcparticleID"] = tree["mcparticleID_L%i" % i].array(library="np")
        # df_temp["mcparticleID"] = tree["layer%i_id" % i].array(library="np")
        df_temp["eventID"] = tree["eventID_L%i" % i].array(library="np")

        df_temp["layer"] = i * np.ones(df_temp.shape[0])

        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df["hit_id"] = np.arange(df.shape[0])

    export_path = os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv")
    df.to_csv(export_path, index=False,
              columns=["hit_id", "x", "y", "z", "layer_id", "mcparticleID", "eventID", "layer"])

    return df


def main(force=False):
    if os.path.exists(os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv")) and not force:
        print("File already exists")
    else:
        root2df()







if __name__ == "__main__":
    # main(force=False)
    # df = pd.read_csv(os.path.join(workdir, r"RawData\mini15_allDets_hits_10000eV_noCorrect.csv"))
    path = os.path.join(workdir, "Preprocess", "root2csv", "mini15_allDets_hits_10000eV_noCorrect_moreInfo.root")
    rt = uproot.open(path)
    print(rt["layer0"].keys())
    print(np.unique(rt["layer0"]["eventID_L%i" % 0].array(library="np")))
    x = rt["layer0"]["p_L0"].array(library="np")
    # x=x[x>5000]
    plt.hist(x, bins=1000, log=True)
    plt.show()
    pass
