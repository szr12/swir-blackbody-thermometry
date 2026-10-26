# Provenance

## Paper

- Title: *Stochastic gate-based autoencoder for unsupervised hyperspectral band selection*
- Authors: He Sun, Lei Zhang, Lizhi Wang, Hua Huang
- Journal: Pattern Recognition 132 (2022), article 108969
- DOI: `10.1016/j.patcog.2022.108969`
- Publisher: Elsevier
- Copyright: © 2022 Elsevier Ltd. All rights reserved
- Local research-copy filename: `paper_SGAE_PatternRecognition_2022.pdf`
- Pages: 11
- SHA-256: `1C3279569516520CBFE18872F30354EEE0CE6177AFC53465FF0E3D625F7CCAD9`
- Source: privately supplied by the researcher after institutional access was unavailable
- Distribution: the PDF is an authorized/private research copy and must not be redistributed

The title page, method pages 3-5, and final page were rendered and visually
checked. The file is unencrypted, tagged, reports no PDF structural suspects,
and its title and DOI metadata match the article.

## Code status

No official author code was found. This repository is an independent,
non-official adaptation for two-dimensional temperature spectra. The paper is
the source for the hard Top-K stochastic gate, three-layer decoder, Gaussian
CDF regularizer, learning rate, stopping threshold, maximum epochs, and
dataset-specific sigma discussion.

The following necessary details are not disclosed or are internally ambiguous:

1. gradient propagation through the binary Top-K gate;
2. optimizer and weight initialization;
3. Eq. 6/7 prints a sum/mean inconsistency, contradicted by Fig. 2's initial
   regularizer value of 0.5;
4. sigma is tuned by dataset and has no paper value for temperature tables;
5. input scaling is not reported.

This code uses straight-through Top-K, Adam with PyTorch initialization,
`mean(Phi(mu/sigma))`, and sigma 0.01. The earlier training-fold Z-score was
removed because the paper does not report it. All assumptions are
machine-recorded in result JSON. The final subset now uses the stochastic Top-K
gate from the stopping epoch, matching Algorithm 1, instead of recomputing a
deterministic `mu` Top-K after training.

## Server validation

- Server user/host: `ubuntu@192.168.1.105`
- Project path: `/media/ubuntu/SSD/share/work2/compare/07_SGAE`
- Shared environment: `/media/ubuntu/SSD/share/work2/compare/.venv-linux`
- Python: 3.12.14
- PyTorch: 2.12.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- GPU: NVIDIA GeForce RTX 3080
- Test result: 18 passed
- CUDA synthetic validation: K=4, 5, and 6 completed with the paper profile;
  all three runs reached the 50-epoch cap, and repeated selections matched.
- Machine-readable result: `outputs/synthetic_demo_cuda.json`

The synthetic result validates implementation flow and determinism only. It
is not evidence of temperature-prediction performance.

## Provisional blackbody data adapter

The adapter was validated on the researcher-supplied provisional directory
`黑体温度高光谱数据` on 2026-09-07. It parses paired ENVI HDR/SPE files, validates
the exact binary size, and exposes a logical `[lines, samples, bands]` memmap
without modifying source data. The observed source format is BIL, ENVI data
type 12 (unsigned 16-bit), byte order 0.

VNIR and SWIR remain separate wavelength grids. Cube groups and original line
indices are preserved. Because the acquisitions share line start 0 but have
different line counts, the default tabular conversion uses a common-prefix
alignment. No SGAE or regression training was run against this provisional
dataset during adapter validation.

## Formal v3 comparison (2026-09-22)

- Dataset SHA-256: `311be63238aa1330cb63350c6ce24c806573903840a63ffdb5caf9e97c5f8b36`.
- Shared seed-42 split: Train/Validation/Test 280/60/61; SGAE sees Train X only.
- K=14, lr=1e-4, sigma=0.01, tau=1e-3 and maximum 50 epochs were fixed before
  evaluation. Test does not select configuration or stopping time.
- The paper has no continuous-temperature regressor; frozen bands use the same
  Train+Validation OLS adapter as other unsupervised comparisons.
- The run reached the 50-epoch cap. Mean gate probability remained about 0.447
  and did not meet tau, so no convergence claim is allowed.
- Raw v3 scale makes reconstruction loss dominate the regularizer. Because the
  paper and unavailable author code do not disclose input scaling, this is
  reported as a reproduction limitation rather than silently tuned away.
