import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from .mkdirs import mkdirs
import matplotlib
matplotlib.use('tkAgg')
import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils.setting import *
from .print_deco import *
from .plot_hits import *
from .construct_sample_var import construct_sample_var


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mkdirs()
