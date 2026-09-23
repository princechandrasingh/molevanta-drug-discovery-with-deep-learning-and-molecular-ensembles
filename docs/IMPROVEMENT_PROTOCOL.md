# Compact-model improvement experiment v1

Written before fitting the new candidates. This is explicitly **exploratory development on an already inspected benchmark**. Prior test results motivated descriptor augmentation; reusing those partitions is not independent confirmation. No improvement over the published paper is claimed.

## Fixed design

Retain the curated 2,292-molecule dataset, ten partitions (random/scaffold; seeds 101, 202, 303, 404, 505), and pinned reference environment from `reference-v1-1`. Verify dataset, descriptor and split checksums. Do not reroll splits, remove compounds, or change the completed benchmark. Reuse the fixed 500-tree forests, with their artifact hashes recorded.

Primary comparison: **fusion_30 versus the old compact model**, paired on identical test compounds. The graph backbone, width 64, depth 3, pooling, batch size 50, dropout 0.1, loss, Adam schedule, 30 epochs and validation checkpoint rule remain the same. The new head concatenates 200 fixed normalized descriptors to the 64-dimensional graph embedding, followed by a 64-unit hidden layer and one output. Backbone initialization follows the old constructor; the new head is newly initialized. Thus the primary change is descriptor augmentation of the head, not a larger graph backbone or more epochs.

Prespecified candidates:

| Name | Representation | Training |
| --- | --- | --- |
| descriptor_mlp | 200 descriptors -> 64-unit hidden layer, ReLU, dropout 0.1 -> one logit | 30 epochs; existing 2-epoch warmup/exponential-decay schedule |
| fusion_30 | Existing compact graph embedding + the 200 descriptors -> same 64-unit head | Same 30-epoch schedule as the reference comparison |
| fusion_60 | Same architecture as fusion_30 | 60 epochs; constant Adam learning rate 0.001, weight decay 0.00001 |

All models use unweighted binary cross-entropy, batch size 50 including the partial batch, gradient norm cap 5, deterministic CPU PyTorch with two threads, and the split seed for initialization and shuffling. Each model retains the highest validation-AP epoch; ties choose the earlier epoch. No early stopping. The two 30-epoch candidates have no weight decay. The longer candidate jointly changes training duration and schedule; it is not an isolated epoch-count ablation.

The descriptors and Chemprop-compatible missing-value rule are unchanged. Use the verified descriptor cache produced in the reference run; the normalization transforms were supplied by descriptastorus, not fitted to this dataset. No extra feature scaling, imputation fitting or probability calibration.

## Validation-only selection and combination

For each partition, select the neural candidate with highest validation AP, breaking ties in the table order. Then select a convex score combination with the existing forest: `score = weight * neural + (1 - weight) * forest`, for weights **0, 0.25, 0.5, 0.75, 1**. Break ties in this order, preferring less neural weight. Preserve the selected candidate, weight, checkpoint epoch and every validation score.

The selected combination is secondary and has a larger selection budget than the old fixed models. Only 12 validation positives are available per partition, so selection may be unstable. A selected weight of zero is exactly the existing forest, not an improved neural model. Report the standalone models separately.

## Two-phase evaluation

1. Train all 30 candidate fits using only training labels and validation checkpoint scores. Record validation predictions and lock **all ten** candidate/weight decisions before calculating any new test prediction or metric. Hash this selection lock, every model artifact, protocol, source snapshot and environment.
2. Evaluate once after the lock. Report every candidate and the selected combination, including failures and negative differences. Compare with frozen compact, forest and Chemprop results on precisely matched test IDs/labels. Do not change settings after viewing the new test results.

Primary metric: average precision. Secondary metrics: ROC-AUC, tie-averaged precision@20, Brier score, neural parameter count and training time. Report mean, sample SD, paired per-seed differences and positive-difference counts. Do not treat repeated compounds as independent cohorts or use significance language.

## Delivery and limits

Provide verified saved models, a selected-pipeline prediction command, aggregate results/figure, and an updated source-only bundle. Retain licensing/provenance notes: the new fusion head and orchestration are project-specific AI-assisted code; descriptors, chemistry, forests and tensor operations use attributed third-party libraries. Descriptor augmentation, score averaging and D-MPNNs are established methods, not new algorithmic inventions.

External evaluation, dataset redistribution rights, outbound project licensing, standardization sensitivity and clinical/experimental validation remain unresolved. Prediction scores are uncalibrated assay scores. A stronger internal score does not establish a drug discovery or superiority to the paper's original optimized model.
