# Provenance

- DOI：`10.1109/LGRS.2025.3564478`
- IEEE 正式发表日期：2025-04-25
- 论文来源：Tampere University 机构仓库中的 CC BY 4.0 正式版本。
- 页数：5
- 论文 SHA-256：`cdf1fd0daba668c668199de5acbc99299d7efbfcc9ffe6b0533322465f055f69`
- 作者官方代码：https://github.com/LeiXuAI/Hyperspectral
- 固定提交：`287cff12c311c27532d783f9921242752662fd06`

论文明确参数：隐藏层 128，Adam `lr=0.001`、`beta1=0.9`、`beta2=0.999`，学习率里程碑 15/30、`gamma=0.1`，`lambda=0.005`。退火配置为 T1 `(1,0.001,40,1)`、T2 `(1,0.001,200,256)`、T3 `(1,0.01,200,32)`。

## 论文优先规则

- 学习率里程碑使用正文的 15/30，不采用官方代码的 10/25/35。
- 重建损失直接实现正文式(5)，不采用官方代码对连续目标与 sigmoid 输出调用 `CrossEntropyLoss` 的行为。
- 硬 Top-K、最低 `theta` 选择、逐样本 Concrete 掩码正则和 `drop_last=True` 与正文伪代码及官方实现一致。
- 硬索引切断重建项到门参数的梯度是原文/官方实现本身的边界。本复现不加入论文未声明的可微近似，并在测试中固定验证这一事实。

独立实现位于 `dropout_cae_reproduction/`，不修改 `official_code/`。合成运行仅验证实现和 CUDA 流程，不作为选带有效性或温度预测性能证据。
