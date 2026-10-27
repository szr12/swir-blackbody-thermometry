# SWIR blackbody thermometry and spectral datasets

Open research code for task-driven 12-band selection, expert-specific Helmert
color-ratio features and regional temperature regression. The repository also
documents the dual-channel blackbody dataset and ten real-object combustion scenes.

## Dataset downloads and access

| Dataset or resource | Link | Access status (26 October 2026) |
|---|---|---|
| Processed blackbody benchmark, background-subtracted (401 × 264) | [Download NPZ](preprocessing_validation/preprocessed_swir_background_corrected_v1/swir_roi_background_corrected.npz) | Public in this repository; CC BY 4.0 |
| Paired blackbody spectra without background subtraction | [Download NPZ](preprocessing_validation/preprocessed_swir_no_background/swir_roi_no_background.npz) | Public in this repository; CC BY 4.0 |
| Original dual-channel blackbody scans | [Reserved ScienceDB DOI](https://doi.org/10.57760/sciencedb.016aq) | Submitted for curation; DOI registration and public download are pending |
| Ten real-object combustion scenes: 10 types, 4 distances, 2 channels | [Inventory, metadata and SHA-256 checksums](data/metadata/real_objects/) | Raw archives are uploading to a separate ScienceDB deposit; its record URL and DOI are pending |
| Saved benchmark predictions and evaluation results | [Results](results/) | Public in this repository; CC BY 4.0 |
| Frozen model weights | [Download model archive](https://github.com/szr12/swir-blackbody-thermometry/releases/download/v0.1.0/frozen_models_seed4242.zip) | Public release asset |

See [data access details](docs/DATA_ACCESS.md) for coverage, licensing and the
distinction between measured spectra and model predictions. The real-object
ScienceDB link will be added here once the new record has an identifier; the
blackbody DOI belongs only to the blackbody collection.

## Quick verification

```bash
python tools/verify_public_results.py
```

This uses only the Python standard library. It recomputes reported RMSE, MAE,
R² and MARD from saved scan-level predictions without training a model.

For model and preprocessing work:

```bash
python -m pip install -r requirements.txt
python tools/reproduce_pipeline.py --protocol random --method ours --prepare-only
```

Run training only when explicitly requested by removing `--prepare-only`.
GPU/CUDA settings affect reproducibility; the archived evaluation used Python
3.12.14, NumPy 2.0.2, PyTorch 2.12.1+cu130, one CPU thread and TF32 disabled.

## Current experiment protocol

- Fixed random split: seed 42, 280/60/61 Train/Validation/Test scans.
- Ordered split: Train 50–329 °C, Validation 330–389 °C, Test 390–450 °C.
- Five complete runs: model seeds 4242–4246; selector-only baseline seeds 42–46.
- Six selector-only methods use the identical native regional MLP, with each
  expert fitting its own feature transform. SPA–MLR, CARS–PLS, STG and Dual-sPLS
  retain their original prediction pipelines.
- K=12, redundancy coefficient 0.04, regional overlap 55 °C, routing width 15 °C.
- Adaptive-weight training; equal-weight Validation selection and reported metrics.

| Complete pipeline | Random split | High-temperature extrapolation |
|---|---:|---:|
| RMSE (°C), mean ± sample SD | 7.641 ± 1.274 | 4.211 ± 1.821 |
| MARD (%) | 3.243 ± 0.524 | 0.811 ± 0.407 |

These are same-batch development holdouts, not independent acquisition batches.
The random no-background ablation converged in 3/5 attempts; its two failed runs
are retained separately and excluded from the reported converged-run means.
Table 5 compares complete pipelines, not decoder-only repeats of one fixed subset.

## Training–validation diagnostics

[Five-seed learning curves and reproducible source data](results/training_validation_diagnostic_20261027/) cover the native global model and four regional experts. Train-only fused RMSE was 3.530 ± 0.206 °C, versus 9.233 ± 4.036 °C on Validation. Some trajectories show overfitting concerns; this does not establish optimal network capacity. Test was excluded and the final benchmark tables are unchanged.

## Layout

| Directory | Contents |
|---|---|
| `optimization_experiment/` | Frozen model implementation and experiment code |
| `main_experiment/src/blackbody_hsic/` | Shared spectral feature and selection utilities |
| `compare/` | Ten method reproductions, provenance and XGBS MIT notice |
| `preprocessing_validation/` | Same-frame spatial/background processing and small matrices |
| `real_object_thermometry/` | Frozen-model spatial inference and diagnostics |
| `configs/` | Archived benchmark and selector settings |
| `results/` | Current tables, predictions, splits and selection outputs |
| `data/metadata/` | Dataset inventory and checksums |
| `docs/DATA_ACCESS.md` | Data coverage, access state and interpretation boundaries |

The legacy orchestration scripts record historical output-directory conventions.
Use the portable `tools/reproduce_pipeline.py` entry point for new benchmark runs.
Raw real-object inference requires the original cubes and released frozen weights.
Download `frozen_models_seed4242.zip` from
https://github.com/szr12/swir-blackbody-thermometry/releases/tag/v0.1.0
and extract it into the repository root to create `checkpoints/`.
Update the local raw-data paths in the supplied config; never choose parameters
using the Test set or real-object reference-temperature descriptions.

## Licensing and citation

Author-owned code: MIT. Data and numerical results: CC BY 4.0.
Third-party conditions are retained in `THIRD_PARTY_NOTICES.md` and method provenance.
The manuscript is in preparation; no publication DOI is invented. Cite the
versioned code release and, once registered/public, the ScienceDB data record.

## 中文说明

本仓库公开当前论文的核心实现、十种对比方法复现、冻结配置和逐扫描结果。
表5六种仅选带基线接入与本文相同的原生区域MLP，每位专家独立估计特征参数。
实物数据使用独立ScienceDB记录，共10类、4个距离、2个通道，原始文件240个。
上方提供处理后数据的直接下载入口；黑体原始数据仍待审核，实物原始数据正在上传，
新实物数据集的ScienceDB链接和DOI尚未取得，暂不宣称原始数据已公开下载。
已有实物二维图属于模型迁移输出，没有配准真值，不作为实物测温精度证明。
数据上传、远端完整校验和公开发布是分别记录的状态。
