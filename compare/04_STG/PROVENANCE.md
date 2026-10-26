# Provenance

- 论文：Yamada, Lindenbaum, Negahban, Kluger, *Feature Selection using Stochastic Gates*, ICML 2020。
- 正文来源：PMLR 官方页面。
- 正文 SHA-256：`e6bbe085e17b4b2a864d5b852f3be19364ec652ac3b0d596c48c0ef2fc8f456a`。
- 补充材料 SHA-256：`eda8995691808994af6c22f2a58817738e31f2592c88ff9c55c2b9c041ea7c65`。
- 官方代码：https://github.com/runopti/stg
- 固定提交：`8b60e349a70d2f392b13d2505c316cdd8868a067`。

论文正文以 `z=clip(mu+epsilon,0,1)` 和 `Phi(mu/sigma)` 表述随机门，并在
Algorithm 1 中令 `mu=0.5`。作者代码把可训练参数写在零附近，再在 hard-sigmoid
和正则器中整体加 0.5；两种参数化等价。本复现直接使用论文符号并从 0.5 初始化。

正文公式写成概率之和；补充材料明确说明实际实现会除以特征数 D，使不同维度下的
超参数搜索尺度相近。正式复现采用补充材料和官方源码一致的概率均值。

论文对不同数据集采用不同网络和经验证集搜索得到的超参数，没有给出面向任意新光谱
数据的唯一配置。网络、SGD、学习率、训练轮数和权重衰减来自作者官方回归示例；lambda
按论文补充材料的范围在 Validation 上选择，不宣称等同于论文任一数据集的最终设置。
