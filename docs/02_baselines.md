# Baseline 方法清单

## Task A: 早期寿命预测

| ID | 方法 | 出处 | 代码源 | 集成难度 |
|---|---|---|---|---|
| B0a | **Severson Elastic Net** (ΔQ100−10 + 6 features) | Nature Energy 2019 | 自实现 (≈ 200 行 Python) | ★☆☆ |
| B0b | **BEEP** (Bayesian early prediction) | TRI 2020 | `pip install beep` | ★★☆ |
| B1a | **Vanilla LSTM** | 经典 | 自实现 | ★☆☆ |
| B1b | **Vanilla Transformer** (8 层) | 经典 | 自实现 | ★☆☆ |
| B1c | **LSTM + Attention** | Front. Electron. 2025 | 自实现 | ★☆☆ |
| B2a | **BatteryGPT** | Nat. Commun. 2025 | 论文官方仓库 (若开放) | ★★★ |
| B2b | **PBT (Pretrained Battery Transformer)** | arXiv 2512.16334 | 官方仓库 | ★★★ |
| B2c | **Bat-T-GNN** (cycle-aware + PINN) | Sci. Rep. 2025 | 论文仓库 | ★★☆ |
| B2d | **DGAT** | Energy 2025 | 论文仓库 | ★★☆ |

## Task B: 快充协议优化

| ID | 方法 | 出处 | 代码源 | 集成难度 |
|---|---|---|---|---|
| BB1 | **CLO-style closed-loop optimization** | Attia et al. Nature 2020 | 自实现 (BO + early stopping) | ★★☆ |
| BB2 | **Deep BO + BRNN** | IEEE TASE 2025 | 自实现 | ★★☆ |
| BB3 | **TD3 Health-aware RL** | arXiv 2505.11061 | `stable-baselines3` 二次开发 | ★★★ |
| BB4 | **DSAC-CAL (safety-constrained RL)** | IEEE TVT 2025 | 论文仓库 | ★★★ |

## 实施优先级

> 假设主任务选 (A+B) + MIT 主数据集，2 周内目标：
>
> - W1: B0a (Severson) + B1b (Transformer) + Ours-noGame
> - W2: B2a/B2c 二选一 (取决于代码可用性) + BB1 (CLO BO) + Ours-full

## 共用评测协议

- **MIT 划分**: Train 41 / Primary Test 43 / Secondary Test 40 (沿用 Severson 原始)
- **早期预测**: 输入前 100 cycle (或 5/10/30/50/100 ablation)
- **零样本协议**: 把 72 个协议按一定比例（10/25/50%）留出，cell 全部在训练中只出现训练协议下的，测试在未见过的协议
- 重复 5 次随机种子取均值
