from utils import *
from Preprocess.root2csv import root2csv_with_p
from Preprocess.csv_preprocess import csv_preprocess_st1, csv_preprocess_st2
from Preprocess.build_train_set import build_train_set





def main(force=False):
    pass
    # root2csv_with_p.main()
    # csv_preprocess_st1.main()
    csv_preprocess_st2.main()
    build_train_set.main()













if __name__ == "__main__":
    main()