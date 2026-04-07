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
  - GitHub 推送、token 配置、分支操作
  覆盖所有量化相关任务，即使用户只说"帮我跑一下"也应触发。
---

# BPQP A股量化实验技能

## 项目结构

```
quant/
├── src/
│   └── bpqp_main.py          ← 全部核心代码（唯一主文件）
├── results/
│   ├── bpqp_results.json     ← 回测+预测结果
│   └── best_model.npz        ← 已训练GRU权重
├── skills/bpqp-quant/
│   └── SKILL.md              ← 本文件
├── .env                      ← GitHub Token（不提交git）
├── .gitignore                ← 必须包含 .env 和 results/
├── requirements.txt
└── README.md
```

---

## ① GitHub Token 配置（Claude Code 推送代码用）

**Token 写在哪里：项目根目录的 `.env` 文件**

```bash
# 在项目根目录创建 .env 文件
echo "GITHUB_TOKEN=ghp_你的token" > .env
```

`.env` 文件内容：
```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_REPO=https://github.com/lz24233/quant.git
```

**让 git 使用这个 token：**
```bash
# 方法1：直接写入 git remote（推荐）
git remote set-url origin https://ghp_你的token@github.com/lz24233/quant.git

# 方法2：通过环境变量（每次终端会话需要重新 source）
source .env
git remote set-url origin https://$GITHUB_TOKEN@github.com/lz24233/quant.git
```

**必须把 `.env` 加入 `.gitignore`，防止 token 泄露：**
```bash
echo ".env" >> .gitignore
echo "results/" >> .gitignore    # 模型权重也不建议提交
```

**Token 权限设置（最小够用）：**
- 打开 https://github.com/settings/tokens → Generate new token (classic)
- 勾选 `repo`（全选）即可
- 有 GitHub Actions 需求再加 `workflow`

---

## ② 切换真实 Qlib 数据（最高优先级改进）

当前模型用模拟数据训练，IC = -0.026，**根本原因是训练样本只有 750 条**。
接入真实数据后样本量提升 20 倍（500只×30个月=15000条），预计 IC 改善至 0.04-0.07。

### Step 1：安装 Qlib

```bash
# Linux / Mac
pip install qlib

# Windows（需先装 Microsoft C++ Build Tools）
# 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install git+https://github.com/microsoft/qlib.git
```

### Step 2：下载 CSI500 数据（约 2GB，20-30 分钟）

```bash
python scripts/get_data.py qlib_data \
  --target_dir ~/.qlib/qlib_data/cn_data \
  --region cn --version v2
```

### Step 3：运行真实数据实验

```bash
python src/bpqp_main.py --mode qlib --qlib_dir ~/.qlib/qlib_data/cn_data
```

### Step 4：验证 Qlib 是否正常

```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
df = D.features(["600036.SH"], ["$close"], start_time="2022-01-01", end_time="2022-03-31")
print(df)   # 能打印出来就说明数据正常
```

### 常见 Qlib 报错修复

```bash
# ModuleNotFoundError: No module named 'qlib'
pip install qlib

# FileNotFoundError: 数据目录不存在
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 股票代码找不到 → 检查后缀格式
# 正确格式: 600036.SH  000858.SZ  300750.SZ
# 错误格式: SH600036   sh.600036

# Windows 编译报错 → 改用 git 安装
pip install git+https://github.com/microsoft/qlib.git
```

---

## ③ 可改进方向（按优先级排序）

### 🔴 高优先级（直接影响 IC）

**1. 接入真实 Qlib 数据**
- 当前: 750条训练样本，IC = -0.026
- 改后: ~15000条，预计 IC → 0.04-0.07
- 操作: 见上方 "切换真实 Qlib 数据" 部分

**2. 行业+市值中性化**
- 当前信号含行业风格暴露，ICIR 被系统性偏差污染
- 在 `FactorLibrary.build()` 末尾加截面回归残差：
```python
# 行业中性化（在截面排名标准化之前插入）
from sklearn.linear_model import LinearRegression
for t in range(T):
    X_dummy = pd.get_dummies(sectors).values.astype(float)  # 行业哑变量
    for fi in range(F):
        y = factors_arr[t, :, fi]
        if y.std() > 1e-8:
            resid = y - X_dummy @ np.linalg.lstsq(X_dummy, y, rcond=None)[0]
            factors_arr[t, :, fi] = resid
```

**3. 扩大股票池（25 → 全量 CSI500）**
- 当前仅 25 只，截面分散度不足
- 接入 Qlib 后将 `STOCKS` 替换为完整 CSI500 成分（约 500 只）
- 样本量: 25×30 = 750 → 500×30 = 15000，×20 倍

---

### 🟡 中优先级（提升模型质量）

**4. 自适应 β 调度（BPQP 论文核心贡献）**
- 固定 `beta=0.1` 导致预测 loss 与决策 loss 竞争
- 在 `Trainer.train()` 中实现动态调度：
```python
# 前 20 epoch: 加大预测权重帮助收敛
# 后 40 epoch: 偏向决策 loss 优化组合目标
beta = 0.5 if epoch < 20 else 0.05
loss = beta * pred_loss + (1 - beta) * decision_loss
```

**5. 增加训练 epoch + Early Stopping 改进**
- 当前: 60 epoch，Qlib 真实数据建议 ≥ 150 epoch
- 将 Early Stopping 改为连续 20 epoch Val IC 不提升才停止：
```bash
python src/bpqp_main.py --mode qlib --epochs 150
```

**6. Transformer 替换 GRU**
- 华泰 2024 研报显示 Transformer 在量价序列上效果优于 RNN
- 预计 IC 提升 0.005-0.01
- 需要 PyTorch：`pip install torch`
- 在代码顶部加：
```python
try:
    import torch; import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

**7. Ledoit-Wolf 协方差改进**
- 当前收缩系数固定 `lw_shrinkage=0.10`
- 改为 Oracle Approximating Shrinkage (OAS)，对小样本更稳健：
```python
from sklearn.covariance import OAS
oas = OAS().fit(ret_train)
Sigma = oas.covariance_ + np.eye(N) * 1e-4
```

---

### 🟢 低优先级（工程完善）

**8. A股特殊约束**
- 涨停股不可买（当日已涨 10% 的股票在 BPQP 中权重强制置 0）
- 跌停股不可卖（对应权重下界设为当前持仓）
- 在 `BPQPOptimizer.optimize()` 中加入可交易性掩码：
```python
# tradable_mask: 0=不可交易，1=可交易
w_new = w_new * tradable_mask
w_new = w_new / w_new.sum() if w_new.sum() > 0 else prev_w
```

**9. 换手率约束**
- 防止月换手率超过 100%（实盘冲击成本过大）
- 在 BPQP 优化层加总换手约束：`sum(|w - w_prev|) ≤ 0.5`

**10. 结果可视化脚本**
- 当前结果在 JSON 文件中，添加 `python src/plot_results.py` 生成图表
- 依赖: `pip install matplotlib`

---

## 模块速查

### 添加新因子（研报因子库扩展）

在 `FactorLibrary.build()` 末尾追加，格式统一：

```python
f = np.zeros((T, N))
for t in range(w, T):
    f[t] = <你的因子计算，结果 shape=(N,)>
factors.append(f)
names.append("MY_FACTOR_NAME")
# 截面排名标准化自动应用，无需手动处理
```

**已有 21 个因子（添加前确认不重复）：**

| 因子名 | 来源研报 | 预测方向 |
|--------|---------|---------|
| MOM_1/3/6/12m | 华泰 AI 量价 | 动量 → 正 |
| REV_1m | A 股反转效应 | 反转 → 正 |
| NVOL_3/6/12m | 华安低波溢价 | 低波 → 正 |
| TVOL_3/6m | BigQuant 2024 换手稳定 | 稳定 → 正 |
| PV_CORR_3/6m | 华泰全频段量价 | 量价同向 → 正 |
| LOWVOL_F | 知乎 2024 低位放量 | 主力建仓 → 正 |
| VOL_SURGE | 方正 2022 成交量激增 | 量增价涨 → 正 |
| CROWD | BigQuant 2024 拥挤度 | 拥挤 → 负 |
| ANALYST_COV | 华泰 2024 分析师覆盖 | 覆盖增 → 正 |
| MA_3/6/12m | 均线偏离 | 均线之上 → 正 |
| SIZE | 2025 最有效因子 | 小市值 → 正 |
| MKT_MOM | 宏观趋势 | 顺势 → 正 |

### 修改 GRU 超参数

```python
# DEFAULT_CONFIG 中修改，或通过命令行 --epochs / --hidden / --lr
"hidden_size": 32,    # 推荐: 16-128，接入真实数据后可调到 64
"seq_len":      6,    # 输入月数，推荐: 3-12
"epochs":      60,    # 真实数据建议: 100-150
"lr":       0.002,    # Adam 学习率
"dropout":   0.10,    # 样本少时加大到 0.2-0.3
"grad_clip":  5.0,    # 梯度裁剪，一般不需要改
```

### 修改 BPQP 优化层

```python
gamma        = 1.0    # 风险厌恶：越大持仓越分散
tc           = 0.002  # 交易成本：印花税 0.1% + 手续费 ~0.1%
max_weight   = 0.15   # 单股上限，实盘通常 ≤ 10%
lw_shrinkage = 0.10   # 协方差收缩：样本少时调大到 0.2-0.3
```

---

## 结果解读标准

| 指标 | 差 ❌ | 可接受 ⚠️ | 优秀 ✅ |
|------|-------|-----------|--------|
| IC 均值 | < 0 | 0.02-0.04 | > 0.05 |
| ICIR | < 0.3 | 0.3-0.8 | > 0.8 |
| 年化超额 | < 0% | 2-5% | > 8% |
| Sharpe | < 0.5 | 0.5-1.0 | > 1.0 |
| 最大回撤 | > -30% | -15%~-30% | < -15% |

**IC 为负的修复流程：**
1. 样本不足 → 接入 Qlib 真实数据，扩大股票池
2. 因子方向错 → 检查 `factors.append(-f)` 还是 `factors.append(f)`
3. 过拟合 → dropout 从 0.1 → 0.2，hidden 从 32 → 16
4. 风格污染 → 加行业中性化（见改进方向 #2）

---

## 快速命令参考

```bash
# 模拟数据快速跑通（5 epoch 验证流程）
python src/bpqp_main.py --mode mock --epochs 5 --hidden 16

# 真实数据完整训练
python src/bpqp_main.py --mode qlib --epochs 100 --hidden 64

# 加载已有模型只做预测
python src/bpqp_main.py --mode predict --load_model results/best_model.npz

# 推送到 GitHub
git add src/bpqp_main.py results/bpqp_results.json
git commit -m "update: 接入Qlib真实数据，IC提升至0.05"
git push -u origin main
```
