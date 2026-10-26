# BS-Nets paper acquisition

Acquired and verified on 2026-09-08.

## Identity

- Title: *BS-Nets: An End-to-End Framework for Band Selection of Hyperspectral Image*
- Authors: Yaoming Cai, Xiaobo Liu, Zhihua Cai
- Published article: IEEE Transactions on Geoscience and Remote Sensing, 58(3), 1969-1984 (2020)
- Published article DOI: `10.1109/TGRS.2019.2951433`
- IEEE record: https://ieeexplore.ieee.org/document/8907858/
- Open manuscript record: https://arxiv.org/abs/1904.08269
- Download source: https://arxiv.org/pdf/1904.08269

## Local artifact

- File: `paper_BS_Nets_TGRS_2020.pdf`
- Artifact type: author-submitted arXiv v1 manuscript, submitted 2019-04-17
- Size: 2,103,968 bytes
- Pages: 14
- PDF version: 1.5; unencrypted
- SHA-256: `0370ba6b741819ddd41ade260f52820978021855a6d92715a8c056740498387d`

The PDF title and authors were verified from extracted text. Pages 1, 5, and 14 were rendered and visually checked for completeness and legibility. The arXiv manuscript is not the final IEEE typeset version: the published pagination and DOI are external bibliographic metadata and are not embedded in this PDF. Do not describe this artifact as an IEEE publisher PDF.

At the paper-only acquisition step, no source code had yet been downloaded or modified.

## Official source acquisition

- Repository: https://github.com/AngryCai/BS-Nets
- Preserved directory: `official_repository/`
- Fixed commit: `bbe8dc258b70fbf727bd3d577ea4d516d095bd35`
- Git tree: `1f44bd0e368c2890128e0ad55266c62fe1a93bac`
- `BS_Net_FC.py` SHA-256: `b96f130b80a57756fb83f4fbc18114cca6a2d88a682e85baee5705f9d103e305`
- `BS_Net_Conv.py` SHA-256: `8dfae355d7b3c1970cf798c856cee13f28d7d52c45ed829919b202bb7428ca31`

The repository was cloned without line-ending conversion and its worktree was verified clean.
It includes the two author implementations, utilities, schematics, and Indian Pines data.
No license file is present, so no redistribution permission is inferred.

## Reproduction decisions

The new `bsnets/` package is independently written and leaves the official repository unchanged.
It is a method-level, non-official PyTorch adaptation, not execution of TensorFlow 1.6.

| Item | Locked source or convention |
| --- | --- |
| Variant | BS-Net-FC only; tabular samples do not have the spatial patches required by BS-Net-Conv |
| Architecture | Paper Table III plus the official source's BatchNorm placement and BAM double-ReLU ordering |
| Reconstruction objective | Paper Eq. 10, not the official code's elementwise-mean MSE |
| Learning rate | Paper Table II value `2e-3`; official source and its history use the conflicting `2e-5` |
| Lambda and maximum iteration | Paper values `1e-2` and 100, with 100 interpreted as epochs as in official source |
| Batch size | Official source value64, absent from paper |
| Optimizer | Adam, betas(0.9,0.999), epsilon1e-8, no weight decay |
| Initialization | Xavier uniform kernels and zero biases, matching official TensorFlow defaults |
| BatchNorm | TensorFlow 1.6 epsilon1e-3, decay0.99 and population-variance moving statistics reproduced explicitly |
| Preprocessing | Per-band min-max fitted only on outer `X_train`; official code fits it to the full HSI |
| Constant bands | Excluded using `X_train` only and mapped back to original zero-based indices |
| Batching | Official rolling shuffled full batches, `floor(N/64)` updates per epoch; require N>=64 |
| Ranking | Paper Eq. 11 on complete `X_train`, final epoch, smaller original index breaks exact ties |
| K | One ranking per outer split; nested prefixes for4,5,6 because K never enters training |
| Randomness | One run per outer split using its shared seed; no restart selection |
| Degeneracy | Score span <=1e-12 is flagged; deterministic output retained without retry or parameter fallback |

The formal profile does not use temperature labels, validation/test features, downstream metrics,
early stopping, source-code learning-rate fallback, or BS-Net-Conv operations. The quick profile
exists only for automated tests. Current blackbody data remains inspection-only.
