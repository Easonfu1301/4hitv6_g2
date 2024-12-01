from utils import *





def check():
    path_list = os.path.join(workdir, "PreProcess", "degen_csv", "evt_1")
    files = os.listdir(path_list)

    # fig = plt.figure()
    plt.ion()
    # plt.show()
    for file in files:
        print(file)

        file = os.path.join(path_list, file)
        df = pd.read_csv(file)
        plot_hits(df)
        plt.show()
        plt.pause(1)
        plt.close("all")
        # break
    plt.ioff()
    plt.show()




if __name__ == "__main__":
    check()
    pass