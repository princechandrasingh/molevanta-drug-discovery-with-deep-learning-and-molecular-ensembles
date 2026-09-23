# Shared-split reference comparison v1.1

Written 2026-09-23 before this experiment's model fitting or test scores. This is a follow-up to an already observed pilot, not a prospectively registered study. The run records this document's hash. Earlier pilot results remain unchanged.

## Question and fixed design

Does the project's compact D-MPNN improve average precision over a pinned Chemprop D-MPNN plus molecular descriptors when trained and evaluated under one shared protocol? How does a fixed random forest compare?

- Use the same verified, curated Table S1B: 2,292 compounds, 119 active. Do not change standardization for this comparison.
- Fix seeds **101, 202, 303, 404, 505**, with the v0.2 connectivity-grouped random and scaffold-grouped allocator. Preflight all ten partitions before fitting. Validation and test must each contain at least 150 observations and 5 positives. Do not replace a seed because of its score.
- All models receive identical train, validation and test membership for each seed and split type. Save membership and hashes. Additional seeds do not make the molecules independent of the pilot or of one another.
- One fixed configuration per model family. No architecture search, model ensembling, score-based protocol changes, or refitting on validation/test observations.
- Primary endpoint: test average precision (scikit-learn AP). Secondary endpoints: ROC-AUC, tie-averaged precision@20, Brier score, training runtime and parameter count.
- Compare models using **paired per-seed differences** on identical test sets. Report mean, sample SD, individual values, and number of positive differences. These are descriptive comparisons; no significance claim, independent-sample confidence interval or bootstrap pretending repeated molecules are independent.

## Models

1. Class-prior reference: training positive fraction.
2. Random forest: 500 trees, sklearn defaults otherwise, random state equal to the split seed, two CPU workers, radius-2/2,048-bit Morgan fingerprints without chirality. This configuration differs from the tuned/class-weighted pilot forest.
3. Project compact D-MPNN: existing independently written graph implementation, hidden size 64, depth 3, mean pooling, dropout 0.1, no descriptors.
4. Chemprop **1.6.1**: use the installed third-party `MoleculeModel` and molecular featurization APIs; directed bond messages, hidden size 300, depth 3, mean aggregation, two feed-forward layers of width 300, dropout 0, no message-layer bias. Concatenate the 200 `RDKit2DNormalized` descriptors from descriptastorus **2.8.0**. The descriptor CDF transforms are supplied by that library, not estimated on this experiment's test data. Do not scale them again. Apply the reference library's `MoleculeDatapoint` NaN-to-zero policy and audit every replacement. No fitted imputation. Fail if calculation fails, dimensions differ, or any infinite value remains.

Both neural networks use the same independently written training harness: 30 complete epochs, batch size 50 including the final partial batch, unweighted binary cross-entropy, Adam without weight decay, gradient norm cap 5, and a per-step learning-rate schedule with 2 warmup epochs (0.0001 to 0.001) then exponential decay to 0.0001 at epoch 30. The best validation AP epoch is retained; ties retain the earlier checkpoint. No early stopping. Seed Python, NumPy and Torch; use deterministic CPU Torch operations with two threads. This changes the compact model's training recipe from the pilot and is recorded before the new results.

Equal configuration counts and training epochs do not imply equal parameter counts, FLOPs, optimization difficulty or fully optimized model families. The reference library performs official graph/model computations, while the local harness defines training and evaluation. It is not the paper's original training program.

## Interpretation boundary

The paper reports random-split ROC-AUC around 0.896 on a different, uncurated sample and a different experimental setup. Its numerical result is **not a matched baseline** for this experiment. Chemprop 1.6.1 postdates the paper, RDKit versions differ, architecture optimization is omitted, and validation AP is used here. Winning this benchmark would support only a narrowly stated internal comparison, not an improvement over the published study or experimental drug discovery.

The label-aware group allocator, its treatment of all acyclic compounds as one group, and the tendency to place large groups into training remain limitations. No external validation is added by this experiment.

## Deferred validation, before broader claims

Standardization sensitivity requires a separate protocol: compare parent/neutralized versus salt-preserving representations using stable source-linked groups and the same source-level held-out assignments; deduplication and contradictory labels must not permit cross-partition leakage. Report sample changes separately from representation effects.

An external benchmark must measure a compatible assay and retain all eligible compounds, with overlap checks performed before scoring. The original paper's selected top-ranked hits are unsuitable as an unbiased external set. No compatible untouched external cohort has yet been established; this remains required for a generalization claim.

## Pre-training correction

**v1.0, 2026-09-23:** descriptor validation rejected NaN values before any model fitting or test scoring. The failed run and original source/protocol snapshot remain in `runs/reference-v1/`.

**v1.1, 2026-09-23:** corrected the adapter to use Chemprop's own `MoleculeDatapoint` policy, which replaces NaNs with zero. This is a compatibility correction based on reference-library behavior, not a response to model performance. Infinite descriptors still fail. Seeds, partitions, all model configurations, epoch budgets and evaluation rules are unchanged. Tests cover missing-value handling and checkpoint selection before restarting in `runs/reference-v1-1/`.
