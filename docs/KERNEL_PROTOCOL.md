# Fingerprint kernels and fixed ensembles: protocol v1

Specified before fitting or scoring the new candidates, 2026-09-23. This is a bounded exploratory follow-up on the previously inspected Stokes Table S1B benchmark. It is not an independent confirmatory experiment or an exact reproduction of any cited paper.

## Research basis and decisions

- [Yang et al. (2019), Analyzing Learned Molecular Representations for Property Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727618/) discusses sparse-data limitations, molecular descriptors, hyperparameter optimization and equal-weight neural ensembles. We retain the existing descriptor-augmented graph checkpoint. Our heterogeneous ensembles below are our own experimental choices, not their published architecture or claimed result.
- [Tripp et al. (NeurIPS 2023), Tanimoto Random Features for Scalable Molecular Machine Learning](https://arxiv.org/abs/2306.14809) studies Tanimoto kernels on fingerprints and approximations for scale. At 2,292 compounds, an exact kernel is practical: we use the established exact binary Tanimoto coefficient, not their random-feature implementation or proposed approximation.
- [Cawley and Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html) motivates limiting selection complexity and separating selection from evaluation. Our kernel selection uses three inner folds restricted to outer training data. This does not undo earlier inspection of the outer test benchmark.

These sources motivate experiments; they do not establish that these methods will improve this antibacterial assay. No paper/repository implementation is copied. Existing licensed RDKit and scikit-learn implementations provide fingerprints and estimators.

## Fixed data and comparisons

Reuse the curated 2,292 compounds (119 positives), chemistry policy, and all ten `reference-v1-1` partitions: random/scaffold, seeds 101, 202, 303, 404, 505. No new seeds or changed labels, exclusions, split allocation or descriptor normalization. Retain all frozen scores from `improvement-v1`, including negative outcomes. Train only on each original training partition (1,834 compounds/95 positives). Outer validation and test each contain 229 compounds/12 positives.

New candidates, all reported:

1. **kernel_r2:** SVM on the exact Tanimoto kernel of radius-2, 2,048-bit Morgan fingerprints, without chirality.
2. **kernel_multiscale:** SVM on the arithmetic mean of radius-2 and radius-3 Tanimoto kernels, both 2,048 bits without chirality. This fixed combination of valid kernels has no learned mixing weight.
3. **kernel_fusion_forest:** equal-weight mean of kernel_multiscale, the frozen custom fusion_60 graph model, and the frozen 500-tree forest.
4. **kernel_chemprop_forest:** equal-weight mean of kernel_multiscale, the frozen Chemprop reference, and the frozen forest. This explicitly uses the attributed Chemprop implementation.
5. **chemprop_forest:** equal-weight mean of frozen Chemprop and forest, a control to measure whether the new kernel adds value to that ensemble.

Primary standalone comparison: kernel_multiscale versus the fixed forest on scaffold average precision (AP). Also report every candidate versus forest and Chemprop on both split types, kernel_multiscale versus kernel_r2, kernel_fusion_forest versus the previous selected_blend, and kernel_chemprop_forest versus chemprop_forest. Ensemble and tuned-model compute/selection budgets exceed those of individual fixed baselines; do not claim an isolated architecture gain.

## Training-only selection and calibration

For each outer partition, construct exactly three `StratifiedGroupKFold` folds with shuffle enabled and its recorded seed. Group by connectivity for random partitions and scaffold for scaffold partitions. Preserve all groups including the acyclic group. Check both classes, group disjointness and complete exactly-once out-of-fold coverage; fail rather than reroll any invalid split. Save fold membership and class counts.

For each kernel family, evaluate the same eight settings in this exact order: C = 0.1, 1, 10, 100, each with class_weight = None then balanced. SVC uses a precomputed kernel, probability=False and other library defaults. Select highest arithmetic mean inner-fold AP on unbounded decision scores; exact ties prefer earlier settings. Save every fold score and every out-of-fold decision. No outer validation/test scores enter kernel selection.

Refit the chosen SVC on all outer training rows. Fit a regularized sigmoid calibration (`LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`) on its selected out-of-fold decisions and training labels only. This is regularized logistic calibration, not a reproduction of libsvm's Platt implementation. Reject a non-positive calibration coefficient instead of silently reversing rankings. Internal hyperparameter selection and calibration reuse the inner folds, so internal calibration/selection estimates are optimistic and are not reported as independent performance estimates.

The neural and forest components retain their existing training/checkpoint choices. No graph retraining, weights fitted on validation, or additional ensemble candidates. Arithmetic means use float64 component scores. An SVM calibration fitted on training data does not establish externally calibrated probabilities; all outputs remain experimental assay scores.

## Two-phase evaluation and verification

The train command never calculates test predictions. It records source/protocol/environment and baseline artifact hashes, the 20 kernel models, all cross-validation histories, fixed ensemble definitions, and diagnostic outer-validation predictions. Every kernel checkpoint must reload with identical validation predictions. All ten decisions are hashed into a selection lock before a separate evaluate command can run.

Evaluation verifies that lock, the prior runs and the current executing source against its snapshot; then scores all five candidates once. Original metrics/predictions are appended unchanged. The report checks exact test membership/labels, all metrics, ensemble arithmetic, fold isolation, selection decisions, and saved artifact hashes. Save paired differences, sample SDs, per-seed values and a comparison figure, including unfavorable outcomes. No changes in response to test results in this run.

The reused benchmark, overlapping repeated test cohorts, 12 positives per test, greedy scaffold allocation and externally supplied descriptor CDFs limit conclusions. Five-seed SD is not a confidence interval; there is no statistical superiority claim. A compatible untouched external assay remains necessary. Do not claim a discovered drug, validated efficacy/safety, or superiority over Stokes' published result.
