# DRL4BS source acquisition (2026-09-07)

## Paper: downloaded and verified

- Title: Deep Reinforcement Learning for Band Selection in Hyperspectral Image Classification.
- DOI: 10.1109/TGRS.2021.3067096; IEEE TGRS volume 60 (2022).
- Source: https://elib.dlr.de/146194/
- Download: https://elib.dlr.de/146194/1/Deep_Reinforcement_Learning_for_Band_Selection_in_Hyperspectral_Image_Classification.pdf
- File: `paper_DRL4BS_TGRS_2022.pdf`; 9,666,588 bytes; 14 pages; unencrypted PDF 1.4.
- SHA-256: `345f10b479a8844e01513b316b964b1a8f20cf0994986c6cc7575d5c490e4c1b`.
- Title/DOI verified with PDF metadata and extracted text. First and last pages rendered and visually checked. DLR labels this a postprint/accepted manuscript, not the final paginated journal version.
- First page states copyright IEEE 2021, personal use permitted; republication/redistribution requires IEEE permission. No CC license asserted.
- Retrieved via local network directly onto the server share. An incomplete server-network download was replaced with the complete verified copy; no local download copy remains.

## Official repository and historical source: acquired, incomplete for training

- Repository: https://github.com/lcmou/DRL4BS
- `official_repository/`: full Git clone, HEAD `c952228074cddbbefb9f0c60c5026b54644acc50`. HEAD contains README only. Git object integrity check passed.
- Historical snapshot: `9ab5c9ce7bb03f58559ebbc91e0f7ad50f376130`, exported without changes to `official_historical_9ab5c9c.zip` and `official_historical_source/`.
- ZIP SHA-256: `63c061b30bf3ccfc1c0d8ad03748a33b576204c763c3471f56189e103742710d`; all archive entries passed integrity testing.
- Snapshot contains two Python demos, two MATLAB classification scripts, README, and two classification MAT files. Files originate from the author's Git history, not a third-party implementation.
- Python demos define a Q-network/agent and load pretrained weights for inference. Required `models/*.h5` and `data4drl/*.mat` are absent. No complete training loop is present in this snapshot.
- No explicit license file found in the snapshot. Do not assume an open-source redistribution license.

## External full-code link: not acquired

- README links https://drive.google.com/file/d/1-w9wD8md0QlYDGbnsqtWw2UpgygDliB2/view?usp=sharing
- Local unauthenticated request to the file page returned HTTP 404. The drive.usercontent.google.com direct-download endpoint also returned 404. Server request to Drive timed out after 20 seconds.
- These observations establish download failure through tested routes, not definitive proof that the author deleted the file or that authenticated access is impossible.
- Full external archive, pretrained weights, and complete training code remain unverified/unavailable. Do not describe this acquisition as a complete runnable official release.

## Initial source-inspection caveats

- Historical demo uses SciPy `entropy(x[:, i])`, then global min-max normalization, not an explicitly coded histogram entropy estimator.
- Replay target contains `reward + gamma * np.argmax(next_prob)`; this returns an action index rather than the maximum Q value appearing in the paper. Preserve the original file; resolve this discrepancy before implementation.
- The source uses legacy TensorFlow/Keras APIs and Python 2 print syntax. No source was modified or executed.
- At acquisition time no adaptation or training had been performed. The implementation stage below supersedes that initial status; the provisional data remains inspection-only.

## Implemented adaptation (2026-09-07)

New drl4bs modules independently implement the paper-first Tabular adaptation approved by the user.
The paper, official repository and exported historical source remain unchanged.

| Behavior | Evidence or explicit convention |
| --- | --- |
| Multihot state, valid unselected actions, K-step episodes | Paper Section II / Algorithms 1-2 |
| Mean entropy difference reward, log base 2 | Paper Eqs. 2-3 |
| Observations divided by band sum; global entropy min-max | Official historical demo calling scipy.stats.entropy and rescaling IEs; paper does not specify a density estimator |
| B-2B-2B-B network, ReLU, replay 50,000, batch 100, lr1e-4, Nadam betas | Paper Section II-C |
| Uniform Xavier weights and zero bias | Explicit implementation of paper initialization reference; no author initialization seed is available |
| gamma0.99; terminal target only reward; valid-action target mask | Historical demo |
| Max Q value, not argmax index; loss on chosen action only | Paper Eqs. 7-8, correcting historical replay expression and replacing full-vector Keras loss |
| epsilon decay0.95 rather than historical0.995 | Paper Section II-C prioritized |
| Decay after each successful replay, lower bound clamped0.01 | Historical update placement plus explicit boundary handling |
| One minibatch/one gradient step per episode after warmup | Paper Algorithm 1; not Keras default splitting into minibatches32 |
| Shared K=6 policy and prefix K4/5/6 | User-approved adaptation; no evidence all paper experiments used this strategy |
| Fixed10,000 episodes; quick50 | User-approved budget, not paper convergence criterion |
| Zero-column removal, positive constants retained, entropy span<=1e-12 bits gives zero rewards | User-approved safeguards; no claim reward is meaningful in degenerate cases |
| PyTorch NAdam eps1e-8, momentum_decay0.004, no decay, foreachFalse | Locked framework defaults/conventions; no numerical equivalence to legacy Keras asserted |
| Deterministic float32 network, float64 entropy; no z-score/negative shift | Reproducibility/input conventions for this adaptation |

No target network, Double DQN, temperature rewards, histogram entropy or reward optimization were added.
The PyTorch code is method-level reproduction, not execution of the original author's complete training pipeline.
The old source depends on unavailable model/data files; the external full archive remains unacquired.
Data adapter is copied byte-for-byte from the shared earlier comparisons; evaluation helper is reused with
additional prediction validation and explanations for undefined metrics.

## Verification artifacts

See outputs/acceptance.json and outputs/tests.xml for measured verification.
See outputs/cuda_paper/result.json for the actual device, budget, losses, selected bands, data/configuration
and source identity; model.pt contains final weights only, not a resumable optimizer/replay checkpoint.
See outputs/blackbody_format_inspection.json for inspection-only provisional data metadata.
The paper copyright and lack of a discovered official source license remain as recorded above;
do not publicly redistribute the original paper, models or archived code under an assumed permissive license.

## Formal v3 comparison (2026-09-22)

- Dataset: `preprocessed_swir_v3/swir_preprocessed_foldsafe_v3.npz`, SHA-256
  `311be63238aa1330cb63350c6ce24c806573903840a63ffdb5caf9e97c5f8b36`.
- Shared split: seed 42, temperature-stratified 70/15/15, Train/Validation/Test 280/60/61.
- Paper permits variable K (experiments use 5 to 60); K=14 is fixed before training to match the main experiment.
- The old 4/5/6 API restriction was an adaptation, not a paper rule, and has been removed.
- Selection sees Train spectra only. The 1451.76 nm band contains negative Train values and is excluded because
  the paper's entropy probability is undefined for negative intensities. No shift, clipping or label is used.
- The paper has no continuous-temperature predictor. Frozen bands are evaluated by the same Train+Validation OLS
  adapter used for the BS-Net unsupervised comparison; this adapter is explicitly outside DRL4BS.
- The 10,000-episode maximum is a reproduction convention because the paper gives no numerical convergence test.
  The formal run exhausted this budget and its final Bellman loss was large, so no convergence claim is made.
