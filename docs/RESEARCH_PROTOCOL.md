# Research protocol v0.2

## Question

How does held-out antibacterial-activity ranking change when evaluation separates molecules by core chemical scaffold, and how do fingerprint baselines compare with a small directed graph model?

This is an exploratory computational extension of Stokes et al., not a preregistered confirmatory study or an exact replication. Initial seeds and model configurations were fixed before the first scores were observed. The split allocation was revised after a diagnostic run exposed inadequate partition counts, as documented below. This is not a prospectively preregistered analysis.

## Prespecified pilot

- Publisher Table S1B only, curated as described in DATA_PROVENANCE.md.
- Seeds 11, 22, 33; random/connectivity and scaffold grouping.
- Each split uses one greedy group allocation minimizing normalized squared error against 80/10/10 row and positive-count targets. Groups are prioritized by their fraction of total rows or positives (whichever is larger), with seeded Uniform(0.8, 1.2) multiplicative perturbation. Ties are deterministic. This uses labels for stratification, not for model fitting or score-based partition selection.
- Preflight all partitions before any fitting: each validation/test set must contain at least 150 rows and 5 positives. No repeated seed search to obtain a preferred score. The v0.2 pilot has 229 rows / 12 positives in each validation/test set and 1,834 rows / 95 positives in training.
- Models: prior, Logistic Regression, Random Forest, independently implemented compact D-MPNN.
- Primary metric: average precision. Secondary metrics: ROC-AUC, tie-averaged precision@20, Brier score.
- Hyperparameter selection: validation AP. Graph training: at most 30 epochs; patience 7; fixed architecture. Early stopping/checkpointing uses validation only.
- Seeds drive partition generation and model random state. CPU graph training uses deterministic PyTorch algorithms and 2 threads.

## Interpretation checks

Inspect actual partition sizes and positive counts, graph epoch history, and nearest-training similarities. Random and scaffold AP can change simply because positive prevalence changes. A no-skill class-prior model provides the AP baseline for each actual test set. Do not compare small AP differences as statistically conclusive.

Use maximum training similarity to describe failure patterns, not as calibrated predictive uncertainty. Fingerprints can collide and their similarity is not proof of biological similarity. Scaffold disjointness is not proof of complete absence of chemical overlap or every kind of leakage.

## Boundaries

- The project trains on a single published phenotypic assay. It does not evaluate human treatment outcomes or compound safety.
- It does not generate or experimentally test molecules.
- No study of full chemical libraries is performed in the initial milestone.
- Standardization and smaller model choices mean the original paper's numerical result is not a reproduction target.
- The graph model preserves chirality in atom features; Morgan fingerprints ignore chirality. A later representation ablation should address this difference.
- Different model tuning budgets and model sizes limit architecture comparisons.
- Repeated seeds share molecules; they measure split/training variability, not independent external validation.
- The allocator places large/difficult groups early and may consistently assign them to training. Thus the results characterize this particular holdout regime, not performance over every possible unseen scaffold. In particular, acyclic compounds form one large group.

## Change log

**v0.1 diagnostic run, 2026-09-23:** used `StratifiedGroupKFold(10, shuffle=True)` and selected the first two folds for test/validation. It completed, but scaffold test sets had 47/51/57 rows and only 1/1/5 positives. Its metrics and manifests are preserved locally in `runs/pilot-v1/` and should not be used for model/generalization claims. No model parameters were changed in response to those test scores.

**v0.2 pilot, 2026-09-23:** replaced the inadequate fold allocation with the single-pass count-balancing objective above; added minimum partition-size/class-count gates and a large-group regression test. Verified all six partition compositions before training. Kept the original seed list and every model's settings unchanged. This fixes a study-design problem; it does not convert the exploratory pilot into independent confirmation.

## Next study, before further model tuning

Write a new version of this protocol with matched tuning budgets, additional fixed seeds, a defined standardization sensitivity experiment, and a compatible untouched external set. Keep the pilot results as a frozen reference and separate development results from the new final evaluation. Record the reason for every methodological change.
