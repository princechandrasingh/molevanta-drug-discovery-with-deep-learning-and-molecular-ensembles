# Verification

Verified on 2026-09-23, macOS arm64 / Python 3.10.13.

- 14 unit/integrity tests passed (curation, salt/identity normalization, conflicting labels, group leakage, large-group allocation, tied ranking scores, directed edge indexing, graph batching and gradients).
- Publisher downloader succeeded from a clean temporary directory and verified the pinned source checksum.
- Completed all 24 model/split/seed evaluations, including the class-prior reference.
- Reloaded all 18 fitted models (Logistic Regression, Random Forest, D-MPNN across six partitions); all reproduced stored test scores within absolute tolerance 1e-7 and relative tolerance 1e-6.
- Rechecked chemical-group separation in all six saved partitions.
- Installed the package and ran the prediction CLI successfully.
- Dependency consistency passed pip check.
- Inspected the rendered comparison figure.

These checks establish software/data-pipeline behavior, not clinical validity, external generalization, or an exact reproduction of the paper.
# Reference integration checks, 2026-09-23

- Main environment: 18 tests passed, 4 optional Chemprop tests skipped because that library is isolated in the reference environment.
- Reference environment: all 22 tests passed. This includes graph gradient/prediction behavior, the Chemprop train-logit/evaluation-probability contract, batch permutation consistency, checkpoint reload, the learning-rate schedule, validation-only checkpoint choice, native missing-descriptor handling, paired-result alignment and source-bundle exclusions.
- Three SciPy CDF overflow warnings occurred while descriptor transforms were evaluated in the tests. The adapter explicitly rejects infinities and uses the reference package's NaN-to-zero rule. On the full dataset, 44 NaN cells in 11 molecules affected four partial-charge descriptors; the run retains an exact replacement audit. No compounds or seeds were removed.
- The initial reference preflight failed on those descriptor NaNs before fitting or scoring. Its source and protocol snapshot are preserved in `runs/reference-v1/`. The documented adapter correction is recorded in protocol v1.1.
- Both dependency environments passed `pip check`. Notice hashes were verified for every collected file. Five wheel notice omissions were supplemented from exact-version source archives after SHA-256 verification against PyPI metadata.
- `reference_report.py` requires a complete result matrix, rechecks every split, aligns each model's test IDs and labels, recomputes metrics, and checks source/protocol/environment/split hashes before producing a report. Each fitted model is saved/reloaded and its predictions compared during the run.

The reference experiment's completed counts and hashes are reported in its generated `verification.json`, rather than inferred from passing unit tests.

Completed reference run `reference-v1-1`: all **40 evaluations**, **9,160 prediction rows**, **10 partitions** and **30 fitted-model reload checks** passed. All metrics were recomputed at `rtol=1e-10, atol=1e-10`. The report reader restores neural predictions to their original float32 dtype; reading them as float64 changed Brier scores by at most 6.8e-10 while leaving ranking metrics unchanged. This serialization issue was corrected in the report reader, without changing training, predictions or stored metrics. The standalone chart was inspected visually, and the reference prediction CLI was exercised successfully with a locally trained Chemprop checkpoint.
# Improvement experiment checks

The descriptor-augmentation follow-up passed **30 tests** in the pinned reference environment, including unchanged graph-backbone initialization, descriptor/graph gradients, batch ordering, checkpoint reloads, validation tie rules, checkpoint selection, rejection of premature test evaluation and selection-lock tampering.

Run `improvement-v1` trained and reloaded **30 new candidate checkpoints**. All ten validation selections were locked before test evaluation began; the lock is hashed and revalidated against stored validation predictions. The completed report rechecked **80 evaluations and 18,320 prediction rows**, including the frozen reference models. Every metric matched at `rtol=1e-10, atol=1e-10`. This is verification of an exploratory development experiment, not independent external validation.

The standalone augmented-model prediction command and selected-combination prediction command were both exercised with a saved `scaffold_202` checkpoint. The latter reproduced the declared 0.75 neural / 0.25 forest combination. Source and prior-benchmark hashes remained unchanged, and the aggregate comparison chart was inspected visually.

# Fingerprint-kernel experiment checks

The pinned reference environment passed **37 tests**. New checks compare exact Tanimoto values with RDKit, catch uint8 dot-product overflow, check kernel symmetry/positive semidefiniteness on representative molecules, preserve chemical groups within inner folds, reject single-class/leaking folds, verify selection tie behavior, exercise training-only calibration, reproduce serialized predictions and batch-independent inference, check fixed float64 ensemble arithmetic, and reject premature/tampered evaluation locks.

Run `kernel-v1` completed **20 new kernel models**, each with eight candidate settings evaluated across three training-only folds (480 tuning fits, plus 20 final refits). No new graph models were fitted. All 20 saved models reproduced validation scores exactly after reload. The selection lock was written at `2026-09-23T20:28:22.852565+00:00`; evaluation began at `2026-09-23T20:28:55.079229+00:00`. Test evaluation verified current executing source and baseline hashes and reproduced the frozen forest, custom graph and Chemprop test predictions.

The generated report checked **130 metric rows**, **29,770 prediction rows**, **30 ensemble outputs**, **20 kernel selections**, and **30 inner folds**. All metrics match saved predictions at `rtol=1e-10, atol=1e-10`; ensemble arithmetic matches at `rtol=1e-14, atol=1e-14`. The report includes the unsupported primary standalone hypothesis and all five new candidates. The comparison figure was visually inspected.

The prediction CLI was exercised on `CCO` using saved `scaffold_202` models and returned all component and ensemble scores. For example, the custom ensemble equals the mean of kernel 0.045884298275301334, graph 0.0005293132271617651, and forest 0.0: 0.015471203834154366. These are experimental assay scores, not validated probabilities or efficacy estimates. The known SciPy descriptor CDF warning remains unchanged.

# Molevanta naming update

Applied the approved title, **Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles**, to the README, package description, CLI help and source-bundle metadata. The bundle now uses a `molevanta/` archive root. Existing module/package/CLI identifiers and historical experiment records retain their original names for compatibility.

The CLI help displays the approved title and the existing source-allowlist test passes. The completed kernel run's selection lock, model execution source, all recorded baseline artifact hashes, and metrics/predictions checksums were reverified after the naming update. No training or evaluation rerun was needed for this presentation change.

# Readability and comments cleanup

- Reformatted 27 existing Python files across `src/`, `scripts/` and `tests/` using Black 24.8.0, with Python 3.10 syntax and a 100-character target line length. All 28 pre-existing Python files retain identical syntax trees compared with their pre-cleanup copies. The original executable logic is unchanged.
- Added explanatory comments around chemical identity/grouping, graph edge directions and batch offsets, fingerprint overflow, out-of-fold calibration, validation-only selection, score dtypes and immutable experiment records. Added `docs/CODE_GUIDE.md` to explain the modules and array conventions.
- Added a standard-library launcher for historical kernel experiments. It verifies recorded source hashes and executes those exact bytes in an isolated temporary package, preserving the existing strict source checks. Three new tests cover source isolation/argument forwarding, rejection of modified source before execution, and rejection of manifest path traversal.
- **40 tests passed** in the pinned reference environment, with the same three known SciPy descriptor CDF warnings. The formatting check passes for all 30 current Python files. No formatting was applied to frozen `runs/` sources.
- Exercised the historical launcher with saved `scaffold_202` models on `CCO`; all eight scores matched direct inference through the cleaned model modules exactly. The custom ensemble remains `0.015471203834154366` for that example. This is a software regression check, not a new evaluation experiment.
- Reverified all **130 metric rows**, **29,770 saved prediction rows**, selection/model/source snapshots and recorded baseline hashes using the cleaned report code. Historical metrics, predictions, training source snapshots and protocols remain unchanged.

Current source files naturally have different byte hashes after formatting/comments. Use the frozen-source prediction command in the code guide for old kernel experiments; new experiments snapshot the current source as usual.
