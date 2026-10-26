# 来源与公式审计

## 论文

- Xiaodi Shang, Chuanyu Cui, Xudong Sun, Xiaopeng Wang, Jiahua Zhang.
- *Classification Task-Driven Hyperspectral Band Selection via Interpretability From XGBoost*.
- IEEE JSTARS, Vol.18, 2025, 13733-13754.
- DOI: 10.1109/JSTARS.2025.3572278.
- 本地 PDF SHA256:
  `a97015cbc54a152bcac4760ac3169a1df18792e92533e9321b906a17554452e8`。
- PDF 为 22 页，首页明确标注 CC BY 4.0。

## 官方代码

- https://github.com/FunctionMayBeStatic/XGBS
- 固定提交：`abd1a0d8717cf3546193db5e093cd513e07596a8`。
- `official_code/` 保持该提交的源码文件未修改，并以 `SOURCE_COMMIT.txt` 固定提交号；
  `adapter_source/` 保存回归适配。

## 已核实差异

1. 原文 Eq.14 为 `max(abs(corr) * Gaussian, mu)`，delta=20、mu=0.1；
   官方代码先把相关系数截断到 [0.9,1]，再乘归一化 Gaussian。
2. 原文 Eq.17 直接使用 q 与 R；官方代码对中间量增加了额外 Min-Max。
3. 原文 Eq.18 对顺序依赖、互补依赖分别求和归一化，再以 beta 组合；
   官方代码先合并矩阵且顺序/互补的 beta 位置相反。
4. 原文是分类；本实验把 XGBClassifier 改为 XGBRegressor，并标准化连续 y。
   这是明确披露的任务适配，不是原论文原任务复现。
5. 原文用 SVM 评价分类准确率；本项目为与其他选带对比统一，使用带截距 OLS
   评价连续温度。OLS 不参与 XGBBoost 树拟合，仅用于 alpha/beta 的 Validation
   冻结和最终温度指标计算。
