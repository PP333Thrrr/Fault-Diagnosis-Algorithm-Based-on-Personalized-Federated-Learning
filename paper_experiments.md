# 论文实验建议（基于本项目）

这份清单的目标是：用**最少的实验组**把“联邦学习的优势”在你的项目里跑出来，并且能直接写进论文（表格 + 曲线 + 统计）。

## 你这个项目里，什么算“联邦学习优势”

在本仓库里（`main.py` / `backend/app.py`），你能可靠地展示 4 件事：

1. **不共享原始数据**：训练只在各客户端本地进行，服务器只聚合模型参数（适合写“隐私/数据孤岛”动机）。
2. **比本地训练更好**：`base`（无通信）vs `fedavg`/`fedprox`/`metafed`（有通信）在同一划分下的性能差距。
3. **越 Non-IID 越体现价值**：通过 `--non_iid_alpha`（Dirichlet 标签偏移）控制异质性，比较不同 alpha 下的提升幅度。
4. **通信-效果折中**：通过 `--wk_iters`（本地训练轮数）和 `--iters`（通信轮数）展示“更少通信也能接近效果”的趋势。

> 对故障诊断数据集（`cwru`/`seu`），建议同时报告 `Accuracy + Macro-F1`（项目里已实现）。

## 你的项目里，推荐用哪些数据集做论文主实验

建议主打你已经集成并且指标齐全的两套故障诊断数据：

- `cwru`：`x.npy` 形状约 `(1898, 1024)`，类别数 13（样本较少，别把客户端数设太大）
- `seu`：`x.npy` 形状约 `(10230, 1024)`，类别数 5（更适合 20 客户端）

## 建议你最终在论文里跑出来的“最小实验组”（4 组足够写完）

下面每一组都建议 **3 个随机种子**（`--seed 0/1/2`），最后在表格里写 `mean±std`。

### 实验组 A：核心对比（证明“联邦 > 本地”）

固定一个中等异质性：`alpha=0.1`，对每个数据集跑：

- `base`（无通信，本地训练）
- `fedavg`（FedAvg）

论文写法：强调“在不上传原始数据的前提下，联合训练显著提升平均准确率 / F1”。

### 实验组 B：异质性扫描（证明“越 Non-IID 越需要 FL/个性化”）

对每个数据集固定客户端数，跑三种 alpha：

- `alpha ∈ {0.6, 0.1, 0.01}`
- 算法至少跑 `base` 和 `fedavg`

可选加一条个性化方法（推荐 `metafed`），更容易写出“在极端 Non-IID 下优势更明显”。

### 实验组 C：算法对比（证明“个性化方法更强”）

固定最难设置：`alpha=0.01`，对每个数据集跑：

- `fedavg`
- `fedprox`（建议 `mu ∈ {1e-3, 1e-2, 1e-1}` 简单网格挑最好）
- `metafed`（阈值/lam 用默认即可，论文里说明采用默认或少量调参）

输出表格：`Accuracy`、`Macro-F1`（`cwru/seu`），再加一个 “Worst-client Acc”（建议你从前端曲线或日志里额外统计）。

### 实验组 D：通信代价（证明“通信更少也能接近效果”）

固定数据集 + 算法（推荐 `fedavg`），保持**总本地训练量**大致相同：

- 方案 1：`iters=300, wk_iters=1`
- 方案 2：`iters=150, wk_iters=2`
- 方案 3：`iters=60, wk_iters=5`

画图：横轴通信轮数（iters），纵轴 accuracy/F1；并在表里补充训练耗时（后端已有 `training_duration_seconds`）。

## 直接可跑的命令模板（CLI）

建议用仓库自带虚拟环境（如果你一直在用 `./venv`）：

```
./venv/bin/python main.py --dataset cwru --partition_data non_iid_dirichlet --batch 8 --n_clients 10 --iters 150 --wk_iters 2 --eval_every 5 --non_iid_alpha 0.1 --seed 0 --split_seed 1 --alg fedavg
```

对 `seu` 通常可以：

```
./venv/bin/python main.py --dataset seu --partition_data non_iid_dirichlet --batch 8 --n_clients 20 --iters 150 --wk_iters 2 --eval_every 5 --non_iid_alpha 0.1 --seed 0 --split_seed 1 --alg fedavg
```

说明：
- `--seed`：影响模型初始化/训练随机性
- `--split_seed`：影响客户端数据划分随机性（Dirichlet/打乱）；默认是 1，论文做稳健性建议换成 0/1/2
- `--eval_every 5`：加速（每 5 轮才评估一次）

## 后端 UI 跑完后，如何导出“论文表格数据”

后端会把每次训练的摘要写到 `backend/training_history.json`。

你可以用这个脚本把历史导出成 CSV 或直接输出 Markdown 表格：

```
./venv/bin/python tools/export_training_history.py --out_csv results/history.csv
./venv/bin/python tools/export_training_history.py --aggregate
```

如果你做了多 seed 重复实验，`--aggregate` 输出的 `mean±std` 可以直接粘进论文（再稍微排版）。

