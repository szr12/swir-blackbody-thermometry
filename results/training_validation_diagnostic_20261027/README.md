# Training–validation diagnostics

Five archived seed-specific 12-band subsets are reused without selection or hyperparameter search. The native 256–256–128 global model and four regional experts are fitted on Train only. The original split has 280 Train, 60 Validation and 61 Test scans. Test is excluded from fitting, monitoring and prediction. There are 25 scientific fits.

An in-memory AST transformation inserts one observer after the native training-trace append. Removing that call recovers the original fit AST. Synthetic parity checks show that monitoring, including large changes to validation labels, leaves training weights and original trace values bitwise unchanged. Frozen sources are in `payload/runtime/`.

Fused training RMSE is 3.530 ± 0.206 °C and validation RMSE is 9.233 ± 4.036 °C (mean ± sample SD, five seeds). Some validation curves plateau or rise while training error falls. These diagnostics show seed-dependent overfitting concerns, not optimal capacity or absence of overfitting. Validation had already informed development; this is not an independent generalization estimate. Returned validation predictions exactly reproduce all five archived Train-only fits. The 340-scan final fits and main result tables remain unchanged.

## Verify without training

From the repository root:

```bash
python tools/verify_training_validation_diagnostic.py results/training_validation_diagnostic_20261027
```

## Reproduce

Use the documented model environment, including CUDA, and a new output directory:

```bash
python optimization_experiment/run_training_validation_diagnostic.py --project . --payload results/training_validation_diagnostic_20261027/payload --output outputs/training_validation_diagnostic_replay --device cuda
```

The observer self-check uses synthetic data only to test isolation. Expert validation masks follow the predeclared overlapping intervals for diagnosis; fused routing uses coarse predictions. Validation does not change gradients, stopping or checkpoint selection.

## Figure and source data

`figures/learning_curves.pdf` and `.svg` are vector exports; `.png` is a 600 dpi preview. Each model has five individual trajectories without smoothing or extension past stopping. Circles mark returned checkpoints. Alignment, text-size and collision checks passed. All curves and endpoint predictions are in `results/`. The existing supplementary information reports these results as Fig. S1.
