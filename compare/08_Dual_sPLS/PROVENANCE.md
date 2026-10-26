# 来源与版本记录

## 论文

- DOI：`10.1016/j.chemolab.2023.104813`
- 公开全文：arXiv `2301.07206`
- 本地文件：`paper_Dual_sPLS_Chemometrics_2023_arXiv.pdf`
- 版本性质：作者公开预印本，不宣称为出版社排版终稿。
- 页数：41 页。
- SHA-256：`2e3cb818151981cb6f815631804f4391dacd98d780ec84afd7069ae1ed2debc0`

## 官方代码

- 来源：CRAN 作者包 `dual.spls`。
- 版本：`0.1.4`。
- 归档文件：`official_source/dual.spls_0.1.4.tar.gz`。
- SHA-256：`641e0f125ee0a59d1e9f44e2059900bc8fbf2afdc344290b3ed94dc59ee5e4aa`。
- 许可证：MIT（以包内 `DESCRIPTION` 和 `LICENSE` 为准）。
- 解压源码：`official_source/dual.spls/`。

论文作者与 CRAN 包作者信息相符。源码仅作为后续公式核对和忠实复现依据；目前未修改官方源码。

## 复现边界

- 已依据论文 Algorithm 2 与作者 R 函数 `d.spls.lasso`、`d.spls.cv`、`d.spls.errorcv` 完成独立 Python 实现。
- `implementation_started=true`。
- `selection_started=false`。
- `regression_started=false`。
- 不把预印本误标为出版社正式 PDF，不把尚未运行的流程描述为实验结果。

论文明确 NIR 实验使用 `ppnu=0.99`，比较1至10个潜变量。作者包默认使用30次、70%校准样本的内部验证；本实现将两者组合为正式配置。固定随机种子属于可复现性补全约定，因为作者 R 接口本身未固定种子。
