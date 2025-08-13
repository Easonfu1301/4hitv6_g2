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
matplotlib.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman'],
    'mathtext.fontset' : 'cm',           # 数学符号继续用 Computer Modern
    'mathtext.rm'      : 'Times New Roman',
    'axes.titlesize': 18,  # 主标题 plt.title
    'axes.labelsize': 14,  # x/y 轴标题 plt.xlabel / plt.ylabel
    'xtick.labelsize': 12,  # x 轴刻度
    'ytick.labelsize': 12,  # y 轴刻度
    'legend.fontsize': 12,  # 图例
    # === 小刻度可见性（Matplotlib ≥3.4） ===
    'xtick.minor.visible': True,  # 开启 x 轴 minor ticks :contentReference[oaicite:0]{index=0}
    'ytick.minor.visible': True,  # 开启 y 轴 minor ticks

    # === 网格 ===
    'axes.grid': True,  # 缺省就画网格 :contentReference[oaicite:1]{index=1}
    'axes.grid.which': 'both',  # 对主、次刻度都画（>=3.5 支持）
    'axes.grid.axis': 'both',  # x、y 都画

    # —— 网格样式：自己喜欢的风格 ——
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.8,
})