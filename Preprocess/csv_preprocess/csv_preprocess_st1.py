from utils import *
import pandas as pd





def process_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    for i in tqdm(range(1, 1001)):
        df_evt = df[df["eventID"] == i]
        # print(df_evt)
        df_evt.to_csv(os.path.join(workdir, r"RawData\processed_csv\evt_%i.csv" % i))
        df_evt = norm_hits(df_evt)
        df_evt.to_csv(os.path.join(workdir, r"RawData\processed_normed_csv\evt_%i.csv" % i))



def norm_hits(hit_df):
    return hit_df




def main():
    path = os.path.join(workdir, r"RawData\mini15_allDets_hits_1000eV_noCorrect.csv")
    process_csv(path)
    pass





if __name__ == "__main__":
    main()