# Provenance and reproduction boundary

## Paper

- He Sun, Lei Zhang, Jinchang Ren, Hua Huang, *Novel hyperbolic
  clustering-based band hierarchy (HCBH) for effective unsupervised band
  selection of hyperspectral images*, Pattern Recognition 130 (2022), 108788.
- DOI: `10.1016/j.patcog.2022.108788`.
- Project PDF: `paper_HCBH_PatternRecognition_2022.pdf`.
- Source: Robert Gordon University institutional repository author manuscript.
  The live Worktribe download was blocked by Cloudflare verification during
  retrieval, so the same public AAM URL was recovered from the Internet Archive
  snapshot dated 2024-04-16. OpenAlex, Semantic Scholar and Unpaywall identify
  the repository copy as green open access under CC BY-NC-ND. The file is
  retained as a research copy and is not redistributed from this project.
- Validated PDF: 36 pages, 6,622,700 bytes, SHA-256
  `f1c11aa6c2fe01d8ed9c1567a31a2835d820f9e464c2d13ca9f9330602edac6d`.

No trustworthy author code repository was found. The paper reports MATLAB 2020b
experiments but does not provide a source URL. This implementation is independent
and must not be described as official or numerically identical to author code.

## Direct paper transcription

- Each band is one initial node; only neighbouring contiguous clusters merge.
- New cluster spectra are arithmetic means of member bands.
- Origin distance, hyperbolic distance, cluster representation and adaptive
  distance follow Eqs. 1, 3, 4 and 5.
- Eq. 3 uses the paper's additive `epsilon_2`; the subscript is not interpreted
  as exponentiation.
- Mutual nearest neighbours are preferred; otherwise the lightest edge merges.
- The representative of each target cluster is the band with minimum origin
  distance. HCBH uses neither labels nor learned model parameters.

## Locked implementation conventions not disclosed by the paper

- Training-fold global Min-Max is interpreted literally over the entire input,
  not separately per band or acquisition.
- A single common scale maps the maximum initial band L2 norm to `1-1e-12`,
  resolving the paper's conflict between elementwise `(0,1)` values and the open
  Poincare-ball domain. `epsilon_1=epsilon_2=1e-12`.
- Multiple training acquisitions are concatenated conceptually after common-line
  truncation. Exact Gram sufficient statistics avoid materializing that matrix.
- Mutual pairs are ordered by adaptive distance then left original boundary;
  equal representative scores choose the smaller original band index.
- CPU NumPy float64 is canonical. These choices are fixed, not tuned on
  temperature prediction.

## Formal v3 comparison (2026-09-22)

- Dataset: `preprocessed_swir_v3/swir_preprocessed_foldsafe_v3.npz`, SHA-256
  `311be63238aa1330cb63350c6ce24c806573903840a63ffdb5caf9e97c5f8b36`.
- Shared split: seed 42, temperature-stratified 70/15/15, Train/Validation/Test
  280/60/61. Selection sees Train features only.
- Each Train spectrum is treated as one observation row of the paper's reshaped
  HSI matrix. This is a data-shape adaptation, not author-code numerical parity.
- K=14 is fixed to match the main experiment. The paper states HCBH accepts any
  desired K and introduces no tunable method parameter beyond K.
- Frozen bands are evaluated with the same Train+Validation OLS adapter used by
  the other unsupervised comparisons. The adapter is not part of HCBH.
- No trustworthy author implementation was found, so formula-level fidelity and
  disclosed numerical conventions are the strongest supportable claim.
