# Provenance

- 论文 DOI：`10.1016/S0169-7439(01)00119-8`
- 论文元数据来源：Elsevier/ScienceDirect。
- 作者官方主页：https://www.ele.ita.br/~kawakami/spa/
- 官方材料：`SPA_GUI.p`、`gui_spa.zip`、`SPA_GUI_Manual_v3p3.pdf`、`Extract_corn_dataset.m`。
- 官方手册 SHA-256：`20807eabc9d2adb8bb863311174995e24b3fd687b1376ffc1e2cebe946cbaab9`

## 官方源码

作者压缩包 `gui_spa.zip` 包含以下可读源码，已经原样展开到 `official_materials/source_2007/`：

- `spa.m`：`fd33d143035dad5f0a93334b79ea18501321617ee5600fa230f1d0b7e1bdb173`
- `projections_qr.m`：`0c0ae9345d51c9bd48e1c0682bcf98ca8216616a28cfcbab9a8bdd22a030e99e`
- `validation.m`：`4f33ac897bf907070bc7e927933ae2dcc02f3aed76bdea8fea91572af52fa7ba`
- `validation_metrics.m`：`cb2a4610d90175ae06a69d9831d8f24c8babb3a1bbc49eb8f79fb76f4173ed23`
- `statistical_prediction_error.m`：`bde3110753d28b195b071b982192415507c915cf1c718463c593178ee522f0fa`

## 方法边界

Python 实现逐语义对齐 `spa.m` 的 Phase 1–2、`projections_qr.m` 与 `validation.m`。MATLAB 三输出 QR 对应 SciPy pivoted QR，MATLAB 样本标准差对应 `ddof=1`，含截距反斜杠最小二乘使用 SciPy GELSY。

`spa.m` 文件日期为2007年，其 Phase 3 是作者后续 SPA-MLR 流程的一部分；本项目不把它追加到2001原始 SPA。由于合法开放的2001全文仍未取得，当前状态准确标记为“作者源码对齐复现”，而非“原文逐页核验”。

服务器没有 MATLAB/Octave，因此尚不能生成跨语言黄金输出；Python公式、形状、并列、验证及确定性测试已经完成。
