# Count fingerprints: protocol v1

Specified on 2026-09-24 before fitting or scoring these candidates. This is an exploratory follow-up on the already inspected Stokes Table S1B benchmark. No result from it constitutes independent confirmation.

## Motivation and fixed candidates

[Tripp et al., NeurIPS 2023](https://arxiv.org/html/2306.14809v2) distinguishes binary and count Morgan fingerprints and dot-product versus min/max Tanimoto kernels. Counts preserve the multiplicity of molecular environments. We test the established exact dot-product kernel, not their random-feature approximation, GP models, or code. Their results do not establish an advantage on this antibacterial classification task.

RDKit supplies 2,048-dimensional hashed Morgan counts without chirality. For nonnegative vectors, use `x.y / (||x||² + ||y||² - x.y)`, defining two zero vectors as similarity 1. This is not RDKit sparse-count min/max similarity. Arithmetic uses float64 before products. Counts still have hash collisions and do not distinguish stereoisomers.

Four candidates, all retained:

1. `count_r2`: radius-2 count Tanimoto SVM.
2. `count_multiscale`: SVM on the equal mean of radius-2 and radius-3 count kernels.
3. `count_fusion_forest`: equal mean of count_multiscale, the frozen fusion_60 model, and forest.
4. `count_chemprop_forest`: equal mean of count_multiscale, the frozen Chemprop model, and forest.

Primary comparison: **count_multiscale versus binary kernel_multiscale on scaffold average precision**. Secondary comparisons: count_r2 versus kernel_r2, count_fusion_forest versus kernel_fusion_forest, and count_chemprop_forest versus kernel_chemprop_forest, on both split types. Preserve every prior model in the aggregate table. No choice of a new default based on the most favorable test comparison.

## Matched experimental budget

Reuse the original 2,292 curated compounds, ten outer partitions, and exact three-fold training assignments from kernel-v1. Seeds remain 101, 202, 303, 404, 505 for random and scaffold splits. No additional training rows, tuning parameters, revised labels, or neural retraining.

Each count family uses the existing eight C/class-weight candidates in the original order: C 0.1, 1, 10, 100, each unweighted then balanced. Choose highest mean inner-fold AP; exact ties choose the first candidate. Reuse group-preserving folds, validation, and the training-only logistic sigmoid calibration from kernel-v1. Calibration and hyperparameter selection share folds and are not independent estimates. Outer validation is a diagnostic only. Every count checkpoint must reload with identical validation scores. Existing binary model reloads must also reproduce their recorded validation scores after the shared-code extension.

## Lock and evaluation

Train and evaluate are separate commands. Hash the data, base run, source, protocol, environment lock, split membership, fold membership, all CV decisions, models, and validation outputs. Lock all 20 count models and fixed ensemble definitions before new test scores can be calculated. Reject changed artifacts or executing source. Do not overwrite existing experiment/report directories.

Evaluation reuses checksum-verified, previously published component predictions in exactly the same test-molecule order. It never fits ensemble weights. Save every candidate's AP, ROC-AUC, expected precision@20 under tied scores, and Brier score. Verify exact test IDs/labels, ensemble arithmetic, complete coverage, checkpoint reloads, and metric reconstruction before generating aggregate reports. Publish aggregate results only; keep molecular data and checkpoints local.

## Limits

Tuning and calibration are matched to the binary kernels; both ensembles cost more than their individual components. The benchmark has already guided earlier development, five seed cohorts overlap, and each test has only 12 positives. Sample SDs are descriptive, not confidence intervals. All scores remain experimental assay scores. External validation is still required; do not claim paper superiority, validated probabilities, therapeutic benefit, or a discovered drug.
