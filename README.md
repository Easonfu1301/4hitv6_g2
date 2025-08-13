### 4hitv6_g2

基于 4 层探测器命中的数据处理与过滤模型训练管线。项目提供从 ROOT 数据到 CSV、样本构建到 MLP 训练的一键式流水线，入口为 `pipeline_v4.py`。

---

## 安装

- 推荐环境：
  - Windows 10/11，Python 3.10 或 3.11
  - NVIDIA GPU，已安装 CUDA 11.8（对应 PyTorch 2.4.1 + cu118）

在开始前，请先从 [CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads) 安装与显卡驱动匹配的 CUDA 11.8。

1) 创建虚拟环境（venv，Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```



2) 安装依赖：

```bash
pip install -r requirements.txt
```

3) 安装 PyTorch（CUDA 11.8）：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

提示：`requirements.txt` 中的 `torch` 可能无法直接从 PyPI 安装，建议按上面命令从官方索引安装。

---

## 路径配置

在运行前请设置原始 ROOT 文件路径与工作目录。在 `utils/setting.py` 中修改：

```python
rootpath = r"D:\files\pyproj\GNN\4hitv6_g2\Preprocess\root2csv\mini15_allDets_hits_10000eV_noCorrect_moreInfo.root"
workdir  = r"D:\files\pyproj\GNN\4hitv6_g2\work_dir2"
```

- `rootpath`：你的输入 ROOT 文件路径
- `workdir`：所有中间产物与模型输出的根目录（会自动创建子目录）

工作目录下会自动创建如下结构：

- `RawData/`：原始与预处理 CSV
- `PreProcess/degen_csv/`：按事件与候选命中切割的数据
- `PreProcess/sample/`：训练/测试样本 `train.npy`、`test.npy`
- `Model/`：模型与训练曲线图
- `Eval/`：评估相关输出（若使用）

---

## 一键运行（推荐）

在项目根目录执行：

```bash
python pipeline_v4.py
```

流水线包含以下阶段：
- Root → CSV：`Preprocess/root2csv/root2csv_with_p.py`
- 事件级 CSV 预处理（拆分/归一化占位）：`Preprocess/csv_preprocess/csv_preprocess_st1.py`、`csv_preprocess_st2.py`
- 样本构建：`Preprocess/build_train_set/build_train_set.py`
- 训练过滤 MLP：`Trainning/train_filtering.py`

输出结果：
- 训练好的权重：`{workdir}/Model/filtering_model.pth`
- 训练/验证曲线与 ROC：`{workdir}/Model/filtering_model.png`
- 训练/测试数据：`{workdir}/PreProcess/sample/train.npy`、`test.npy`

注意：
- 当前代码会覆盖已有中间结果；如需保留历史结果，请先备份 `workdir`。
- 一次性步骤：以下三步在同一 `workdir` 下仅需运行一次，重复运行可能覆盖。首次跑完后可在 `pipeline_v4.py` 中将其注释掉：

```python
    # root2csv_with_p.main()
    # csv_preprocess_st1.main()
    # csv_preprocess_st2.main()
```

---

## 分阶段运行（可选）

如需单独执行每个阶段，可在项目根目录运行下列命令（示例）：

```bash
# 1) ROOT → CSV（写入 RawData 下）
python -c "from Preprocess.root2csv import root2csv_with_p; root2csv_with_p.main()"

# 2) 事件级 CSV 处理（写入 RawData/processed_csv 与 RawData/processed_normed_csv）
python -c "from Preprocess.csv_preprocess import csv_preprocess_st1; csv_preprocess_st1.main()"
python -c "from Preprocess.csv_preprocess import csv_preprocess_st2; csv_preprocess_st2.main()"

# 3) 样本构建（写入 PreProcess/sample/train.npy 与 test.npy）
python -c "from Preprocess.build_train_set import build_train_set; build_train_set.main()"

# 4) 训练过滤模型（写入 Model/filtering_model.pth 与 .png）
python -c "from Trainning import train_filtering; train_filtering.main()"
```

---

## 关键参数与建议

- 批大小：`Trainning/train_filtering.py` 中默认 `batch_size=8192*2`，若显存不足请下调（如 4096 或 1024）。
- 并行：构建数据阶段使用 `multiprocessing.Pool(16)`，可根据 CPU 核心数调整。
- 设备：自动检测 GPU（`utils/__init__.py` 中的 `device`）。

---

## 常见问题

- 安装 PyTorch 报错：请使用上文提供的官方索引安装命令，确保 CUDA 版本匹配；仅 CPU 环境使用 `+cpu` 版本。
- Matplotlib 后端问题：项目强制使用 `tkAgg`。若无 GUI 或报错，可将 `utils/__init__.py` 中的 
  `matplotlib.use('tkAgg')` 替换为 `matplotlib.use('Agg')` 再运行。
- 数据路径错误：确认 `utils/setting.py` 中的 `rootpath` 指向存在的 ROOT 文件。

---

## 快速开始

```bash
git clone <your-repo>
cd 4hitv6_g2
# venv（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1 # 如果失败则先 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# 编辑 utils/setting.py，设置 rootpath 与 workdir ！！！！！
python pipeline_v4.py
```

训练完成后在 `{workdir}/Model` 下获得 `filtering_model.pth` 与 `filtering_model.png`。

