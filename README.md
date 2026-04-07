# BPQP A股量化投资研究 v3.0

基于 NeurIPS 2024 论文《BPQP: A Differentiable Convex Optimization Framework for Efficient End-to-End Learning》

## 快速开始

```bash
pip install numpy pandas scikit-learn scipy

# 模拟数据（立即可运行）
python src/bpqp_main.py --mode mock

# Qlib真实数据
python src/bpqp_main.py --mode qlib
```

## 架构

```
数据(Qlib/模拟) → 研报因子库(21个) → GRU(BPTT端到端) → BPQP优化层 → 回测+预测
```

| 组件 | 说明 |
|------|------|
| **DataLoader** | 支持模拟数据（真实统计特征）和Qlib真实Alpha158 |
| **FactorLibrary** | 21个因子，来自华泰/方正/BigQuant/知乎等2024研报 |
| **GRU** | 纯numpy实现，完整BPTT反向传播，Adam优化器 |
| **BPQPOptimizer** | 均值-方差优化，Ledoit-Wolf协方差收缩，含交易成本 |
| **Backtester** | 2022-07~2026-03月度再平衡回测 |
| **Forecaster** | 2026全年逐月预测，置信度衰减 |

## 结果（模拟数据基线）

| 指标 | 数值 |
|------|------|
| 年化收益 | +8.60% |
| Sharpe | 0.436 |
| 最大回撤 | -24.03% |
| IC均值 | -0.026 |

> ⚠️ IC为负主要因为训练样本仅750条。接入Qlib真实数据（500只股票×30个月=15000条）后预计IC提升至0.04-0.07。

## 在 Claude Code 中使用

将 `skills/bpqp-quant/SKILL.md` 放入 Claude Code 技能目录，之后只需说：
- "帮我把因子改成日频"
- "把hidden_size调到64重新训练"
- "2026年预测结果是什么"
- "Qlib安装失败怎么办"

Claude Code 会自动读取技能说明并给出精确操作。

## 改进路线图

1. **接入Qlib** → IC从-0.026提升至0.04-0.07（最高优先级）
2. **行业中性化** → 消除风格暴露，提升ICIR
3. **扩大股票池** → 25→500只，样本量×20
4. **自适应β** → 参考configs/adaptive_beta.yaml
5. **PyTorch替换GRU** → GPU加速，支持更大hidden_size

## 论文参考

- **BPQP**: https://arxiv.org/abs/2411.19285 (NeurIPS 2024)
- **Qlib**: https://github.com/microsoft/qlib
