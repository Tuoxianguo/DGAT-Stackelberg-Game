# 方法设计 (HSMM-GraphGame)

## 0. 符号约定

- $c \in \{1, \dots, N_c\}$: 电池索引；$p \in \{1, \dots, N_p\}$: 协议索引
- $t = 1, \dots, T_c$: cycle 编号
- $x_t^{(c)} \in \mathbb{R}^{F \times L}$: 第 $t$ 个 cycle 的 $F$ 个测量序列，长度 $L$（重采样后）
- $y^{(c)} \in \mathbb{N}$: 电池 $c$ 的最终循环寿命（到 80% SOH）
- $s_t \in \{1, \dots, K\}$: cycle $t$ 所处的隐藏退化阶段（$K=4$：健康/线性老化/拐点过渡/快速衰退）

## 1. Module 1: Sequence Encoder

输入 → cycle 嵌入 $z_t \in \mathbb{R}^d$。

实现选项：
1. **从零自训的 Transformer Encoder** (8 层 × 4 head × 256 d, cycle-aware patching, RoPE)
2. **BatteryGPT/PBT 预训练主干** + 线性 adapter（如代码可用，效果更好）

`src/battery_paper/models/proposed/encoder.py`

## 2. Module 2: HSMM Stage Layer ★创新点 1★

### 2.1 模型

K 个状态的 Explicit-Duration HSMM:

- 转移：$A_{ij}$，约束为左到右无回退（$A_{ij}=0$ for $j<i$）
- 状态停留时间：$d_i \sim \text{Weibull}(\alpha_i, \beta_i)$（覆盖 1 ~ 数百 cycle）
- 观测：$z_t \mid s_t = i \sim \mathcal{N}(\mu_i, \Sigma_i)$（高斯发射，亦可 GMM）

为了端到端可微，采用 **可微分前向-后向算法** (cf. Yu, *Hidden Semi-Markov Models*, 2016)，
log-space 实现。最终损失：

$$
\mathcal{L}_{\text{HSMM}} = -\sum_c \log p_\theta(z_{1:T_c}^{(c)}) + \lambda_{\text{rul}} \cdot \text{MAE}(\hat y, y)
$$

其中 $\hat y$ 由 HSMM 的"剩余预期停留时间"求和得到（解析）。

### 2.2 物理对应

| 阶段 $s_t$ | 物理含义 | 典型 cycle 数 |
|---|---|---|
| 1 | 健康期 (初始 SEI 稳定) | 10-300 |
| 2 | 线性老化 (SEI 生长) | 200-1500 |
| 3 | 拐点过渡 (析锂 + 主动锂损失加速) | 30-150 |
| 4 | 快速衰退 (析锂主导, 多重副反应) | 20-100 |

→ HSMM 显式学到的状态序列可与电化学解释对齐，是论文 Discussion 的亮点。

`src/battery_paper/models/proposed/hsmm.py`

## 3. Module 3: 协议-电池异构 GNN ★创新点 2★

### 3.1 图构造

异构图 $\mathcal{G} = (\mathcal{V}_p \cup \mathcal{V}_c, \mathcal{E}_{p \leftrightarrow c} \cup \mathcal{E}_{c \leftrightarrow c})$

- 协议节点 $v_p$: 特征 = [CC1, Q1, CC2, cutoff_V, T_ambient] (MIT 5-dim)
- 电池节点 $v_c$: 特征 = [初始容量, 早期 cycle 嵌入 $z_1, ..., z_{100}$, HSMM 阶段后验]
- 协议-电池边: 若 $c$ 用 $p$ 充电则连边
- 电池-电池边: $\mathrm{cos}(z^{(c_1)}, z^{(c_2)}) > \tau$ 的 KNN 图

### 3.2 传播

R-GCN / HGT 两层；输出 $u_p, v_c$ 用于：
- 协议外推：仅靠 $u_p$ 预测在新协议下任意电池的预期寿命
- 电池表征：$v_c$ 替换或拼接 HSMM 嵌入提升预测精度

`src/battery_paper/models/proposed/hetero_graph.py`

## 4. Module 4: Stackelberg 博弈优化 ★创新点 3★

### 4.1 形式化

记协议参数 $p = (CC_1, Q_1, CC_2, V_{\text{cut}})$，电池寿命 $L(p) = f_\theta(p)$ (由 HSMM-GNN 给出)，
充电时间 $T(p)$（解析可得）。**Stackelberg game**:

$$
\begin{aligned}
&\text{Leader (寿命):} \quad \max_p \; L(p) \\
&\text{Follower (用户体验):} \quad \min_p \; T(p) \\
&\text{s.t.} \quad T(p) \le T_{\max}, \;\; V_{\text{anode}}(p) \ge V_{\text{Li}}^{\text{plating}}
\end{aligned}
$$

转化为双层优化 (BiLO):

$$
p^* = \arg\max_p L(p) \; \text{s.t.} \; p \in \arg\min_{p'} T(p') + \mu \cdot g_{\text{safety}}(p')
$$

通过 **隐式微分** (implicit differentiation) 端到端可训练；
扫描 $\mu$ 得到 Pareto 前沿。

### 4.2 与 RL/BO 比较

- BO 黑箱式只能给出有限点，本方法给出**闭式**前沿；
- RL 需要采样、长时间训练，本方法纯梯度，秒级出解；
- Stackelberg 解严格优于线性加权（理论上可证）。

`src/battery_paper/games/stackelberg.py`

## 5. 端到端训练流程

```python
# pseudo-code
for batch in loader:
    z = encoder(batch.x)                       # Module 1
    hsmm_loss, post, rul_hat = hsmm(z, batch.y)  # Module 2
    u_p, v_c = gnn(z, post, batch.graph)       # Module 3
    rul_hat2 = head([v_c])
    loss_pred = mse(rul_hat2, batch.y)
    if game_on:
        p_star = stackelberg(u_p)              # Module 4 (隐式微分)
        loss_game = -mean(L_hat(p_star)) + reg
    else:
        loss_game = 0
    loss = hsmm_loss + α * loss_pred + β * loss_game
    loss.backward(); opt.step()
```

## 6. 消融顺序

| 编号 | 配置 |
|---|---|
| A0 | Encoder only (Transformer) |
| A1 | + HSMM (创新点 1) |
| A2 | + GNN (创新点 2, 关 HSMM) |
| A3 | + HSMM + GNN (无 Game) |
| A4 | + HSMM + GNN + Game (Ours-Full) |
