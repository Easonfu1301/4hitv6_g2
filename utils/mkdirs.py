from .setting import *
import os






dirs  = [
    "RawData",
    os.path.join("RawData", "processed_csv"),
    os.path.join("RawData", "processed_normed_csv"),
    os.path.join("RawData", "propressed_with_near"),
    "PreProcess",
    os.path.join("PreProcess", "degen_csv"),
    os.path.join("PreProcess", "sample"),
    "Model",
    "Eval",
]


def mkdirs():
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)


    for dir in dirs:
        if not os.path.exists(os.path.join(workdir, dir)):
            os.makedirs(os.path.join(workdir, dir), exist_ok=True)