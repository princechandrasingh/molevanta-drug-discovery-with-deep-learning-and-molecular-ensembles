# Reading and maintaining Molevanta

Start with the feature and model modules, then follow the experiment runner. The code keeps data preparation, fitting, selection, evaluation and reporting separate so each step can be inspected.

## Where to start

| File | Responsibility |
| --- | --- |
| `src/molstudy/data.py` | Verify the publisher workbook, standardize molecules, resolve duplicates and record provenance. |
| `src/molstudy/features.py` | Convert molecules to binary fingerprints and calculate similarity to training molecules. |
| `src/molstudy/splits.py` | Keep related chemical structures together while assigning train, validation and test partitions. |
| `src/molstudy/graph.py` | Construct directed molecular graphs, batch them safely and define the compact neural network. |
| `src/molstudy/improved.py` | Combine graph embeddings with molecular descriptors and select neural checkpoints. |
| `src/molstudy/kernel.py` | Compute Tanimoto kernels, tune SVMs with grouped training folds and combine model scores. |
| `src/molstudy/kernel_study.py` | Run the binary-fingerprint experiment, save selection evidence and verify artifacts before evaluation. |
| `src/molstudy/count_study.py` | Compare count fingerprints with the frozen binary study using matched folds, locked selection, and verified aggregate reports. |
| `src/molstudy/metrics.py` | Define the shared ranking and calibration metrics, including tie handling. |
| `src/molstudy/kernel_report.py` | Recalculate saved results, check ensemble arithmetic and generate aggregate reports. |
| `src/molstudy/reference.py` | Adapt the attributed Chemprop implementation to the comparison protocol. |
| `scripts/run_frozen_kernel.py` | Run a historical kernel experiment from its verified source snapshot. |
| `scripts/build_source_bundle.py` | Package source, documentation and aggregate reports using an explicit allowlist. |

The earlier `experiment.py`, `improvement.py` and corresponding report modules preserve earlier stages of the study. The `*_predict.py` modules demonstrate inference from saved checkpoints.

## Follow one molecule through the code

1. `standardize` produces a canonical structure, compound identity and grouping keys.
2. `fingerprints` creates binary molecular features; `count_fingerprints` preserves environment multiplicities; `molecular_graph` creates atoms and paired directed bonds.
3. A saved graph model or SVM maps those features to an assay score.
4. `ensemble_scores` averages the declared component scores with fixed weights.
5. `evaluate` measures predictions only after the experiment's selection decisions have been locked.

The inline comments explain details that can silently change scientific results: stereoisomer grouping, reverse-edge exclusion, offsets in graph batches, integer overflow in fingerprint multiplication, out-of-fold calibration, dtype restoration, and validation-only selection.

## Read the array names

- `x` or `features`: a two-dimensional feature matrix, one molecule per row.
- `y` or `labels`: one activity label per molecule.
- `train_ix`, `val_ix` and `test`: row indices into the fixed dataset.
- `fold`: one inner-fold number per outer-training molecule.
- `decisions`: SVM margins, with one column per candidate setting.
- `score`: a model's assay scores; calibration outside this dataset is not established.

In graph batches, atom/edge indices refer to the concatenated batch. Reverse-edge indices refer to the opposite direction of the same chemical bond. Graph membership identifies which molecule owns each atom.

## Formatting and comments

Python files use Black formatting with a 100-character target line length and Python 3.10 syntax. The cleanup used the already available Black 24.8.0 executable; formatting is a development tool, not a model runtime dependency. The configuration is in `pyproject.toml`.

```bash
black --check src tests scripts
black src tests scripts
MPLCONFIGDIR=/tmp/molstudy-mpl PYTHONPATH=src .reference-venv/bin/python -m pytest -q
```

Use one statement per line and expand long calls, dictionaries and comprehensions. Keep comments close to the code they explain. Explain the scientific assumption, array relationship or reason for a check; avoid repeating an obvious assignment in prose. Some URLs and report text remain longer than the formatting target.

Historical source copies in `runs/` are excluded from formatting. Their exact bytes are part of the experiment evidence. Data, environments and distribution archives are excluded too.

## Running a historical checkpoint after editing source

The original kernel runner requires the executing model source to match the recorded source hashes exactly. Even adding a comment changes a hash. For a saved run created before this cleanup, use its verified source snapshot:

```bash
.reference-venv/bin/python scripts/run_frozen_kernel.py runs/kernel-v1 predict --split scaffold --seed 202 --smiles 'CCO'
```

The launcher verifies every recorded Python file, stages those bytes in a temporary package and starts the same interpreter with that package. The historical runner then performs its original model, split and selection-lock checks. The launcher does not change the saved manifest or relax the source check. Use it only with your own trusted local experiments, as with all existing checkpoint commands.

For new experiments trained with the current source, the normal `molstudy.kernel_study train`, `evaluate` and `predict` commands continue to apply. A historical, still-locked experiment can also be evaluated through the launcher using `evaluate`; a completed run remains protected against a second evaluation by its original status check.

## Count-fingerprint experiment

The count study extends the shared kernel implementation with an explicit `kernel_kind`. Existing checkpoints default to binary behavior; new count checkpoints store their representation. The count formula uses squared norms and float64 products. It is the dot-product Tanimoto kernel, not min/max overlap on integer vectors.

`count_study.train` reuses exact training-fold assignments and verifies old binary validation predictions before fitting count models. `evaluate_locked` scores new test molecules only after all model selections have been frozen. The report reader checks saved metrics, test membership and fixed ensemble arithmetic, and includes unfavorable outcomes. Run with the existing pinned reference environment:

```bash
PYTHONPATH=src .reference-venv/bin/python -m molstudy.count_study train --data-root . --base runs/my-kernel --output runs/my-count
PYTHONPATH=src .reference-venv/bin/python -m molstudy.count_study evaluate runs/my-count
MPLCONFIGDIR=/tmp/molstudy-mpl PYTHONPATH=src .reference-venv/bin/python -m molstudy.count_study report runs/my-count --output reports/my-count
```

`--data-root` can point to an existing local data workspace and `--base` to its completed kernel run; their absolute paths and hashes are recorded locally. Experiment and report output directories must be new. The trained count checkpoints can be loaded with `joblib` and scored using `feature_matrix` from `count_study`; only load trusted local artifacts. Counts do not establish a superior model unless supported by the reported comparisons.
