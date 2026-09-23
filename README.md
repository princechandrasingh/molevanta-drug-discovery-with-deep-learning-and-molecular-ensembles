# Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles

**Do molecular models still predict antibacterial activity when the test molecules have unfamiliar core structures?**

Molevanta is a reproducible computational drug-discovery project combining deep learning with molecular ensembles. Its first application is early-stage antibiotic candidate screening, inspired by [Stokes et al., *A Deep Learning Approach to Antibiotic Discovery*, Cell (2020)](https://doi.org/10.1016/j.cell.2020.01.021). It evaluates molecular-fingerprint models, directed message-passing neural networks, and their ensembles on the paper's E. coli screening dataset.

The project includes data curation, fixed random and scaffold partitions, validation-only model selection, repeated-seed evaluation, saved predictions, chemical-similarity error analysis, and a command-line prediction demo. It is a new computational extension, not an exact reproduction of the paper or evidence of a newly discovered drug.

## Quick start

Use Python 3.10 or newer. Run commands from this project directory.

GitHub repository: [`molevanta-drug-discovery-with-deep-learning-and-molecular-ensembles`](https://github.com/princechandrasingh/molevanta-drug-discovery-with-deep-learning-and-molecular-ensembles).

The existing Python distribution identifier `molecular-generalization`, module `molstudy`, and CLI command `molstudy` are retained for compatibility with the saved experiments and environments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
molstudy download
molstudy prepare
pytest -q
molstudy run --output runs/my-pilot --seeds 11 22 33 --epochs 30
```

For the exact dependency versions used in the initial macOS/Python 3.10 run, install `requirements-lock.txt` before installing this project with `python -m pip install -e . --no-deps`. The lock lists versions, not platform-specific wheel hashes; other Python/platform combinations may require changes.

A baseline-only run avoids graph training:

```bash
molstudy run --output runs/baselines --seeds 11 22 33 --models prior logistic forest
```

Existing run directories are never overwritten. Outputs include `REPORT.md`, `model_comparison.png`, `metrics.csv`, `predictions.csv`, `summary.csv`, and a provenance/configuration manifest.

## Latest improvement: fingerprint kernels and fixed ensembles

The [kernel study report](reports/kernel-v1/REPORT.md) adds two Tanimoto-kernel SVMs, training-only grouped cross-validation, sigmoid calibration, and three fixed equal-weight ensembles. Mean average precision over the same five seeds (higher is better):

| Model | Random | Scaffold |
| --- | ---: | ---: |
| Previous validation-selected blend | 0.699 | 0.731 |
| Existing fixed forest | 0.716 | 0.746 |
| Existing Chemprop reference | 0.723 | 0.717 |
| New multiscale kernel alone | 0.750 | 0.737 |
| New kernel + custom graph + forest | **0.769** | **0.758** |
| New kernel + Chemprop + forest | **0.770** | **0.760** |
| New Chemprop + forest control | 0.759 | 0.750 |

The custom graph ensemble improves over the previous blend in **9/10 paired comparisons**, including all five scaffold seeds. The standalone multiscale kernel falls below the fixed forest on mean scaffold AP, so the predeclared primary standalone hypothesis was not supported. The reference ensemble adds about 0.010 AP over the two-component ensemble in each split type. These are descriptive internal results on reused tests, not independent validation or an improvement claim over the published paper. All candidates, per-seed differences, sample SDs and unfavorable outcomes are retained. Ensembles and tuning use larger budgets than individual fixed baselines.

The [protocol and literature analysis](docs/KERNEL_PROTOCOL.md) explains the use of [Tanimoto kernels](https://arxiv.org/abs/2306.14809), [molecular representations and ensembles](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727618/), and [model-selection safeguards](https://www.jmlr.org/papers/v11/cawley10a.html). Each kernel's C/class-weight setting is chosen from eight candidates using three training-only folds that preserve chemical groups. All 20 settings and the fixed ensemble rules are locked before test scoring. No new dependencies were installed; the already-present threadpoolctl dependency is now explicitly declared for kernel computation thread limits.

```bash
# Requires the completed reference and descriptor-improvement runs.
PYTHONPATH=src .reference-venv/bin/python -m molstudy.kernel_study train --reference runs/my-reference --previous runs/my-improvement --output runs/my-kernel
PYTHONPATH=src .reference-venv/bin/python -m molstudy.kernel_study evaluate runs/my-kernel
PYTHONPATH=src .reference-venv/bin/python -m molstudy.kernel_report runs/my-kernel --output reports/my-kernel

# Return every component and ensemble score from saved checkpoints.
PYTHONPATH=src .reference-venv/bin/python -m molstudy.kernel_study predict runs/my-kernel --split scaffold --seed 202 --smiles 'CCO'
```

The repository includes aggregate reports; datasets, checkpoints and historical `runs/` directories remain local. Run the training commands above to produce your own artifacts. If you have the original local `runs/kernel-v1` experiment, whose source snapshot predates the readability cleanup, use the verified historical launcher:

```bash
.reference-venv/bin/python scripts/run_frozen_kernel.py runs/kernel-v1 predict --split scaffold --seed 202 --smiles 'CCO'
```

Verification covers **40 passing tests**, **130 metric rows**, **29,770 prediction rows**, and **20 new model reloads**. Only project-specific integration code was added; RDKit and scikit-learn supply fingerprint/estimator implementations and Chemprop remains attributed. No author implementation or pretrained weights were copied into the source tree. See [code provenance](docs/CODE_PROVENANCE.md).

## Reading the code

The [code guide](docs/CODE_GUIDE.md) maps the modules and follows a molecule from standardization to ensemble prediction. Source files use consistent formatting and explanatory comments around the model mathematics, chemical grouping, calibration and experiment checks. Black settings in `pyproject.toml` keep future edits consistent and exclude frozen experiment sources.

## Earlier improvement: descriptor-augmented compact model

The [improvement report](reports/improvement-v1/REPORT.md) documents an exploratory follow-up on the same, already inspected benchmark. The new head combines our compact graph embedding with 200 molecular descriptors; a second candidate trains longer. Mean average precision across five seeds:

| Model | Random | Scaffold |
| --- | ---: | ---: |
| Original compact graph model | 0.310 | 0.232 |
| Graph + descriptors, same 30-epoch budget | 0.555 | 0.580 |
| Graph + descriptors, 60 epochs / revised optimizer settings | 0.698 | 0.661 |
| Validation-selected combination with the forest | 0.699 | 0.731 |
| Existing fixed forest | 0.716 | 0.746 |
| Existing Chemprop reference | 0.723 | 0.717 |

The descriptor-augmented 30-epoch model improves over the original compact model in all ten comparisons. The longer standalone model improves further on average, while remaining below Chemprop in mean AP. The selected combination does not beat the fixed forest on average, and it selects the forest alone in 3/10 partitions. All outcomes are retained. These are internal development gains, not independent validation or evidence of superiority to the paper.

Training and evaluation are separate commands. Training locks every candidate and blending-weight decision using validation AP before evaluation can access new test predictions. The [protocol](docs/IMPROVEMENT_PROTOCOL.md) fixes the candidates and selection rules in advance. Existing run directories are preserved.

```bash
# Requires the completed reference run and the existing pinned reference environment.
PYTHONPATH=src .reference-venv/bin/python -m molstudy.improvement train --reference runs/my-reference --output runs/my-improvement
PYTHONPATH=src .reference-venv/bin/python -m molstudy.improvement evaluate runs/my-improvement
PYTHONPATH=src .reference-venv/bin/python -m molstudy.improvement_report runs/my-improvement --output reports/my-improvement

# Inspect a standalone augmented graph model.
PYTHONPATH=src .reference-venv/bin/python -m molstudy.improved_predict --model runs/my-improvement/scaffold_202/fusion_60.pt --smiles 'CCO'

# Inspect the validation-selected combination, including its component scores and weight.
PYTHONPATH=src .reference-venv/bin/python -m molstudy.improvement predict runs/my-improvement --split scaffold --seed 202 --smiles 'CCO'
```

The architecture/training wrappers were generated here with AI assistance and use the existing attributed dependencies. No new library code or pretrained model was copied into the project. The stronger scores do not change data redistribution rights or the experimental nature of the predictions.

## Initial results

The current pilot is in `runs/pilot-v2/`. A shareable snapshot is in [reports/pilot-v2/REPORT.md](reports/pilot-v2/REPORT.md), with its metrics and figure. The snapshot should always be read with the limitations and actual split counts. The earlier `runs/pilot-v1/` is preserved as a diagnostic run: its original fold-based scaffold allocation produced uninformatively small test sets. See the protocol change log.

## Shared-split reference comparison

A separate, pinned Chemprop 1.6.1 environment adds a reference-method comparison on five new fixed seeds. It uses one configuration per family, identical partitions, a common neural training schedule, validation-only checkpoint selection, paired test comparisons and checkpoint reload verification. Read [the predeclared reference protocol](docs/REFERENCE_PROTOCOL.md). This compares a documented reference configuration, not the paper's optimized model under its original setup.

The [completed reference report](reports/reference-v1-1/REPORT.md) covers 40 evaluations and 30 saved/reloaded fitted models. Mean average precision across five seeds:

| Model | Random | Scaffold |
| --- | ---: | ---: |
| Project compact D-MPNN | 0.310 | 0.232 |
| Chemprop + descriptors | 0.723 | 0.717 |
| Fixed random forest | 0.716 | 0.746 |

The compact model trails Chemprop in all ten comparisons. The forest's scaffold AP advantage is +0.029 on average in this internal benchmark; it is not a demonstrated improvement over the published paper. The report includes every seed, paired differences, sample SDs and the limits of the changed training recipe. External validation remains outstanding.

Prepare the data in the main environment first. Run the reference experiment from the project directory in its separate Python 3.10 environment:

```bash
python3.10 -m venv .reference-venv
.reference-venv/bin/python -m pip install -r requirements-reference-lock.txt
PYTHONPATH=src .reference-venv/bin/python -m pytest -q
PYTHONPATH=src .reference-venv/bin/python -m molstudy.reference --output runs/my-reference
.venv/bin/python -m molstudy.reference_report runs/my-reference --output reports/my-reference
PYTHONPATH=src .reference-venv/bin/python -m molstudy.reference_predict --model runs/my-reference/random_101/chemprop.pt --smiles 'CCO'
```

The reference environment deliberately pins an older NumPy/SciPy/Torch combination for Chemprop 1.x compatibility and uses the project modules through `PYTHONPATH=src`. It does not replace the main environment or install the project's main dependency specification. The lock is an exact package-version list for this Python/macOS run, not a portable wheel-hash lock.

Reference checkpoints use `molstudy.reference_predict` in that environment; the original `molstudy predict` command below is for pilot artifacts. Keep the reference checkpoint alongside its saved `train.csv` configuration input. All returned values are uncalibrated assay scores.

The reference preflight initially rejected NaN descriptors before fitting; v1.1 uses Chemprop's native NaN-to-zero policy, logs replacements and still rejects infinities. That correction, including the failed run, is documented without changing seeds or model settings.

## Authorship and licenses

The project-specific pipeline and compact graph module were generated here with AI assistance. The D-MPNN method is established research. The new reference model intentionally uses third-party **MIT-licensed Chemprop**, and the rest of the stack also relies on external libraries. See [code provenance](docs/CODE_PROVENANCE.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [the dependency notice audit](docs/license-audit/README.md).

No general data redistribution license or outbound project-code license is asserted. A [source-only bundle builder](docs/PUBLISHING.md) excludes data, predictions, models and installed libraries. This project is not guaranteed globally unique or free of every legal restriction.

## Data

`molstudy download` fetches [the publisher's Table S1 workbook](https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx), verifies a pinned SHA-256 checksum, and reads **sheet S1B**. Labels come directly from the published `Activity` column. The ambiguous `Mean_Inhibition` heading is not reinterpreted or used to recreate labels.

| Stage | Molecules | Active |
| --- | ---: | ---: |
| Publisher Table S1B | 2,335 | 120 |
| After documented curation | 2,292 | 119 |

The curation audit records 40 merged same-label duplicate rows, 2 excluded rows belonging to one standardized structure with conflicting labels, and 1 invalid structure. The counts depend on the recorded RDKit version. Inspect `data/processed/curation_audit.csv` for source row numbers and reasons.

Data files are downloaded locally and ignored by Git. The code does not grant a new license to third-party research data. See [data provenance](docs/DATA_PROVENANCE.md).

## Models

| Model | Representation | Validation selection |
| --- | --- | --- |
| Class prior | Training positive rate | None; ranking reference |
| Logistic regression | 2,048-bit Morgan fingerprints, radius 2 | C = 0.1, 1, 10 |
| Random forest | Same fingerprints | Leaf size = 1, 3; 300 trees |
| Compact D-MPNN | Atom/bond features, directed bonds | Best validation AP epoch; max 30, patience 7 |

The graph network uses hidden size 64, depth 3, mean pooling, and an independently written PyTorch implementation. Compared with the original Chemprop study, this pilot omits descriptor augmentation, Bayesian architecture optimization and ensembling, and uses a smaller model. It is not a drop-in implementation of the original model. Classical and graph tuning budgets are different; the pilot does not establish architecture superiority.

## Evaluation contract

- Fix seeds before training. Every model receives identical partitions for each split/seed.
- Allocate whole groups once using a deterministic greedy objective balancing row and positive counts against 80/10/10 targets. Seeded group-priority perturbation supplies repeated partitions; ratios remain approximate for indivisible groups.
- Random partitioning still groups connectivity-equivalent molecules, keeping stereoisomers together.
- Scaffold partitioning groups Bemis–Murcko frameworks without chirality. All acyclic compounds share one group. Large groups can cause substantial size and class-prevalence differences.
- Assert no compound/connectivity overlap; additionally assert no scaffold overlap for scaffold runs.
- Select hyperparameters and graph checkpoints using **validation average precision only**. Never reroll a split based on performance. A one-class partition fails explicitly.
- Preflight all partitions before training: validation/test must each contain at least 150 molecules and 5 positives. The v0.2 pilot obtains 229 molecules / 12 positives in every validation and test partition.
- Report average precision (AP), ROC-AUC, precision@20, and Brier score. Constant-score ties are handled without exploiting dataset order.
- Record every held-out prediction and its maximum Tanimoto similarity to training molecules.
- Report mean and sample standard deviation across seeds. Overlapping test sets across seeds are not independent samples and the standard deviation is not a confidence interval.

Read [the research protocol](docs/RESEARCH_PROTOCOL.md) before extending the experiment.

## Prediction demo

```bash
molstudy predict --model runs/pilot-v2/random_11/forest.joblib --smiles 'CCO'
molstudy predict --model runs/pilot-v2/random_11/dmpnn.pt --smiles 'CCO'
```

These example commands test the interface. The output is an uncalibrated score for the dataset's assay, not an estimate of therapeutic efficacy or safety. Novel input molecules may lie outside the training distribution. Load only trusted, locally generated `.joblib` models.

## Structure

```text
src/molstudy/
  data.py          Publisher download, identity normalization, curation audit
  splits.py        Fixed group partitions and leakage assertions
  features.py      Morgan fingerprints and training similarity
  graph.py         Directed-bond neural network and validation checkpointing
  experiment.py    Shared evaluation runner and saved artifacts
  metrics.py       Ranking and probability-score metrics
  report.py        Static figures, tables, and interpretation limits
  cli.py           Download / prepare / run / report / predict
tests/             Chemical identity, leakage, metrics, and graph integrity tests
docs/              Protocol and provenance
reports/pilot-v2/  Shareable results snapshot
```

## Next milestones

1. Evaluate the shared-split Chemprop comparison, retaining its fixed-budget and reference-configuration limitations.
2. Test sensitivity to standardization and the treatment of acyclic compounds using a predeclared protocol.
3. Add probability calibration fitted on validation data, with calibration evaluation kept separate.
4. Reserve a compatible external assay dataset for a final evaluation; do not treat curated top-ranked compounds as an unbiased external test set.
5. Add a small molecular viewer after the prediction/evaluation contract is stable.

## References

- [Stokes et al. (2020), original study](https://doi.org/10.1016/j.cell.2020.01.021)
- [Stokes et al. (2020), published correction](https://doi.org/10.1016/j.cell.2020.04.001)
- [Yang et al. (2019), directed molecular representations](https://doi.org/10.1021/acs.jcim.9b00237)
- [Wu et al. (2018), MoleculeNet and molecular splitting](https://doi.org/10.1039/C7SC02664A)
- [Chemprop implementation](https://github.com/chemprop/chemprop)
