# 数据集清单与下载说明

## 主数据集 (必选)

### 1) MIT / Stanford / Toyota (Severson 2019)

- **论文**: Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4 (2019).
- **规模**: 124 节 LFP/石墨 18650 电池，72 种 CC-CC 多段快充协议，循环至 80% SOH。
- **批次**:
  - Batch 1: 2017-05-12, 41 cells
  - Batch 2: 2017-06-30, 43 cells
  - Batch 3: 2018-04-12, 40 cells
- **官方下载**: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
  - 三个 `.mat` 文件 (`2017-05-12_batchdata_updated_struct_errorcorrect.mat` 等)
- **预处理**: 推荐使用 [BEEP](https://github.com/TRI-AMDD/beep) 或社区脚本
  [`rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation`](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- **关键字段**: cycle-level (`V`, `I`, `T`, `Qc`, `Qd`, `dQdV`), summary (`cycle_life`, `IR`, `Tavg`)

### 2) HUST (Tian 2022)

- **论文**: Tian et al., *Capacity attenuation mechanism modeling and health assessment of lithium-ion batteries*, Energy 244 (2022).
- **规模**: 77 节，多种放电协议。
- **下载**: Zenodo: https://doi.org/10.5281/zenodo.6405084 (检查最新 DOI)

### 3) CALCE CS2 / CX2

- **机构**: U. Maryland CALCE
- **下载**: https://web.calce.umd.edu/batteries/data.htm

### 4) NASA PCoE (B5/B6/B7/B18)

- **下载**: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### 5) Oxford Battery Degradation Dataset

- **作者**: Birkl, Howey 等 (Oxford)
- **下载**: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac

## 可选 / 加分项

### 6) Stanford-Toyota 2024 Extension

- 在 https://data.matr.io 检索 "fast charging 2024"（如未公开则跳过）

### 7) ByteDance / 国内公开数据集

- 字节跳动 BEEP-PINN 数据 (若 release，需在 GitHub 跟踪)
- 清华 LiBattGNN 数据 (若 release)

## 下载策略

- 全部在云服务器 `data/raw/` 下下载（10 PB NAS，无空间压力）
- 每个数据集一个子目录：`data/raw/mit/`, `data/raw/hust/`, `data/raw/calce/`, ...
- 下载完成后写入 `data/raw/<dataset>/MANIFEST.json` 记录文件 sha256 + 来源 URL，方便复现

## 协议表

MIT 数据集 72 个协议的结构化表见 `data/processed/mit_protocols.csv`（在预处理后生成）。
