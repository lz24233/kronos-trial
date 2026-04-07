---
name: bpqp-quant
description: |
  A股量化投资实验技能，专为 BPQP (NeurIPS 2024) 项目设计。
  当用户提到以下任何内容时立即使用此技能：
  - 运行/修改 BPQP 实验、回测、预测
  - 修改因子库（添加/删除/替换因子）
  - 调整 GRU 训练参数（hidden_size/epochs/lr/dropout）
  - 切换数据源（mock → Qlib 真实数据）
  - 分析回测结果（IC/Sharpe/最大回撤/超额收益）
  - 生成2026年预测结果
  - 调试 Qlib 安装和数据问题
  - 修改 BPQP 优化层参数（gamma/交易成本/仓位约束）
  覆盖所有量化相关任务，即使用户只说"帮我跑一下"也应触发。
---

# BPQP A股量化实验技能

## 项目概览

基于 NeurIPS 2024 论文《BPQP: A Differentiable Convex Optimization Framework》的A股量化复现项目。

```
项目结构:
src/bpqp_main.py     ← 主程序（数据/因子/GRU/BPQP/回测/预测全在这里）
configs/baseline.yaml ← 超参数配置
results/             ← 输出目录（模型/结果JSON）
skills/bpqp-quant/   ← 本技能目录
```

---

## 快速运行

```bash
# 模拟数据（开箱即用，无需安装Qlib）
python src/bpqp_main.py --mode mock

# 真实Qlib数据
python src/bpqp_main.py --mode qlib --qlib_dir ~/.qlib/qlib_data/cn_data

# 快速测试（少epoch）
python src/bpqp_main.py --mode mock --epochs 10 --hidden 16

# 加载已有模型，只做预测
python src/bpqp_main.py --mode predict --load_model results/best_model.npz
```

---

## 模块速查

### A. 添加新因子

在 `FactorLibrary.build()` 方法中添加，遵循统一格式：

```python
# 在现有因子后追加
f = np.zeros((T, N))
for t in range(w, T):
    f[t] = <你的因子计算逻辑>   # shape: (N,)
factors.append(f)
names.append("MY_FACTOR")
# ↑ 截面排名标准化会自动应用，无需手动处理
```

**已有21个研报因子（修改前确认不重复）：**
| 因子名 | 来源研报 | 方向 |
|--------|---------|------|
| MOM_1/3/6/12m | 华泰AI量价 | 动量→正 |
| REV_1m | A股反转效应 | 反转→正 |
| NVOL_3/6/12m | 华安低波溢价 | 低波→正 |
| TVOL_3/6m | BigQuant 2024 | 稳定→正 |
| PV_CORR_3/6m | 华泰全频段 | 量价同向→正 |
| LOWVOL_F | 知乎2024低位放量 | 主力建仓→正 |
| VOL_SURGE | 方正2022成交量激增 | 量增价涨→正 |
| CROWD | BigQuant 2024拥挤度 | 拥挤→负 |
| ANALYST_COV | 华泰2024分析师 | 覆盖增加→正 |
| MA_3/6/12m | 均线偏离 | 均线之上→正 |
| SIZE | 2025最有效因子 | 小市值→正 |
| MKT_MOM | 宏观趋势 | 顺势→正 |

### B. 修改GRU超参数

直接修改 `DEFAULT_CONFIG` 字典或通过命令行参数：

```python
# 在 DEFAULT_CONFIG 中：
"hidden_size": 32,   # 推荐范围: 16-128，越大越慢
"seq_len":      6,   # 输入序列长度（月），推荐: 3-12
"epochs":      60,   # 训练轮数，Qlib真实数据建议100+
"lr":       0.002,   # 学习率，Adam优化器
"dropout":   0.10,   # Dropout比例，防过拟合
"grad_clip":  5.0,   # 梯度裁剪阈值
```

### C. 修改BPQP优化层

```python
# 在 BPQPOptimizer 初始化处：
gamma       = 1.0    # 风险厌恶系数：越大越保守
tc          = 0.002  # 单边交易成本（0.2% = 印花税0.1% + 手续费0.1%）
max_weight  = 0.15   # 单只股票最大仓位，监管要求通常≤20%
lw_shrinkage= 0.10   # Ledoit-Wolf收缩，样本不足时调大（0.1-0.3）
```

### D. 切换到Qlib真实数据

**Step 1: 安装Qlib**
```bash
# Linux/Mac
pip install qlib

# Windows（需要先装 Microsoft C++ Build Tools）
pip install git+https://github.com/microsoft/qlib.git
```

**Step 2: 下载CSI500数据**
```bash
python scripts/get_data.py qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn --version v2
# 约需20-30分钟，数据约2GB
```

**Step 3: 运行**
```bash
python src/bpqp_main.py --mode qlib
```

**Step 4: 调试Qlib连接**
```python
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
from qlib.data import D
df = D.features(["600036.SH"], ["$close"], start_time="2022-01-01", end_time="2022-03-31")
print(df)  # 能正常打印说明数据正常
```

---

## 结果解读指南

读取 `results/bpqp_results.json` 后按以下标准判断：

| 指标 | 差 | 可接受 | 优秀 |
|------|----|--------|------|
| IC均值 | < 0 | 0.02-0.04 | > 0.05 |
| ICIR | < 0.3 | 0.3-0.8 | > 0.8 |
| 年化超额 | < 0% | 2-5% | > 8% |
| Sharpe | < 0.5 | 0.5-1.0 | > 1.0 |
| 最大回撤 | > -30% | -15% ~ -30% | < -15% |

**IC为负的最常见原因及修复：**
1. 训练样本太少（< 1000条）→ 扩大股票池或用Qlib真实数据
2. 因子方向错误 → 检查因子符号（如反转因子应取负号）
3. 过拟合 → 增大 dropout，减小 hidden_size
4. 截面效应被风格污染 → 加行业中性化预处理

---

## 常见任务模板

### 任务1：提升IC，用户说"效果不好"
```
1. 检查 results/bpqp_results.json 中 ic_mean
2. 如果 < 0：先确认因子方向（rank_normalize后正向还是负向）
3. 增加训练数据：--mode qlib 或扩大 STOCKS 列表
4. 调整正则化：增大 dropout 到 0.2，减小 hidden 到 16
5. 加行业中性化：在 FactorLibrary.build() 末尾加截面回归残差
```

### 任务2：用户想加某个研报因子
```
1. 确认因子逻辑（预测方向：正/负）
2. 在 FactorLibrary.build() 末尾追加代码块
3. 运行 python src/bpqp_main.py --mode mock --epochs 5 快速验证
4. 检查 results/bpqp_results.json 中 ic_mean 变化
```

### 任务3：2026预测结果不合理（置信度太低/推荐全一样）
```
1. 检查 prev_w 是否传入正确（最后一个回测期的权重）
2. 调整 decay 衰减系数（默认每月-5%）
3. 调整 noise_scale（默认0.02*mo，越远越随机）
4. 调整 BPQP gamma（预测月份越远越保守，建议gamma=1.5）
```

### 任务4：Qlib数据加载报错
```bash
# 错误1: ModuleNotFoundError
pip install qlib --break-system-packages

# 错误2: 数据文件不存在
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 错误3: 股票代码格式不匹配
# Qlib使用 600036.SH 格式，检查 STOCKS 列表中的后缀是否正确

# 错误4: Windows编译失败
pip install git+https://github.com/microsoft/qlib.git
```

---

## 架构说明（修改代码前必读）

```
DataLoader
  └── _load_mock()   → 基于真实统计特征生成模拟数据
  └── _load_qlib()   → 接入Qlib真实Alpha158数据

FactorLibrary
  └── build()        → 21个研报因子，自动截面排名标准化

GRU
  └── forward()      → 前向传播（支持dropout）
  └── backward()     → BPTT完整反向传播
  └── adam_step()    → Adam优化器
  └── save/load()    → 模型持久化

BPQPOptimizer
  └── estimate_cov() → Ledoit-Wolf收缩协方差
  └── optimize()     → 投影梯度求解均值-方差优化

Trainer
  └── train()        → 端到端训练，Early stopping by Val IC

Backtester
  └── run()          → 月度再平衡回测，含交易成本
  └── _compute_metrics() → IC/Sharpe/回撤等指标

Forecaster
  └── forecast_2026() → 逐月预测，置信度衰减
```

---

## 接入PyTorch版本（有GPU时使用）

如果环境有PyTorch，可以将GRU替换为更高性能版本：

```python
# 在 src/bpqp_main.py 顶部加入
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 替换 GRU 类的 forward/backward 为 torch.nn.GRU
# 具体实现见 src/gru_torch.py（待创建）
```

---

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `results/bpqp_results.json` | 完整实验结果（回测+预测+配置） |
| `results/best_model.npz` | 最优GRU权重（可用 --load_model 加载） |

JSON结构：
```json
{
  "backtest": { "ann_return": 8.6, "sharpe": 0.436, ... },
  "monthly":  [{ "date": "2022-07", "port_ret": -0.013, "ic": 0.236, ... }],
  "forecast_2026": [{ "month": "2026-01", "confidence": 0.85, "top_picks": [...] }]
}
```
