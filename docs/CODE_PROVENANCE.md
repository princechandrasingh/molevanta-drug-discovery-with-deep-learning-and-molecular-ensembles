# Code provenance and attribution

Recorded 2026-09-23. This is a record of this project's development actions, not proof of universal uniqueness or a legal noninfringement opinion.

The project is named **Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles**. Earlier records use “Molecular Generalization.” This branding update retains the existing Python distribution/module/CLI identifiers and frozen experiment snapshots; it does not change the models, data, results, authorship record or third-party attributions.

The subsequent readability cleanup expands dense statements, applies consistent formatting and adds explanatory `#` comments. All 28 pre-existing Python files retained identical syntax trees after that cleanup. The additional `scripts/run_frozen_kernel.py` launcher was written here with AI assistance and uses Python's standard library to execute verified historical source snapshots; its tests check isolation and tamper rejection. The original source-hash requirements remain in place. See the [code guide](CODE_GUIDE.md) for the corresponding prediction command.

## Project-specific work

The data acquisition/curation pipeline, group partition allocator, compact PyTorch graph module, training orchestration, metrics, reports, CLI, tests, reference adapter and audit scripts were generated in this workspace with AI assistance. No Chemprop repository or third-party antibiotic reproduction repository was cloned or used as a project template. The compact `src/molstudy/graph.py` was written here in the initial pilot; it is not a copied Chemprop module. There has been no exhaustive similarity search against all existing code.

The D-MPNN concept, molecular fingerprints, descriptor augmentation, scaffold evaluation and other methods are established research ideas. This project does not claim to invent them. Cite [Yang et al. (2019)](https://doi.org/10.1021/acs.jcim.9b00237) and [Stokes et al. (2020)](https://doi.org/10.1016/j.cell.2020.01.021), as well as the software actually used.

## Explicit third-party use

- Both environments use external scientific libraries. For example, sklearn supplies the random forest and logistic-regression estimators, RDKit supplies chemistry operations, and PyTorch supplies tensor/autodiff operations.
- The new reference experiment intentionally imports **Chemprop 1.6.1**, an MIT-licensed third-party implementation. `src/molstudy/reference.py` calls its `MoleculeModel`, `TrainArgs`, `MolGraph` and `BatchMolGraph` APIs. Those model and featurization implementations belong to the Chemprop authors, not this project.
- The reference experiment also calls **descriptastorus 2.8.0** for 200 normalized descriptors, including its externally supplied CDF parameters. Its shipped notice is BSD-3-Clause text attributed to Novartis Institutes for BioMedical Research Inc.
- The adapter was written after consulting Chemprop's API documentation and implementation to establish correct behavior. The reference library is installed separately and is not vendored into the project source tree. Its complete shipped MIT notice is retained under `docs/license-audit/`.
- No pretrained author checkpoints are downloaded, reused or represented as newly trained work. Experiment checkpoints are trained locally on the cited published data.
- The publisher workbook is third-party research data. It is not newly created data, and access to it does not establish unrestricted redistribution rights.

An accurate portfolio description is: **“An AI-assisted, independently assembled molecular generalization study with custom curation, evaluation and a compact graph model, benchmarked against an attributed Chemprop reference.”** Do not describe the entire stack as original code, claim a novel D-MPNN architecture, or present the reference implementation as your own.

## What the audit establishes

The subsequent `improved.py` module adds a project-specific prediction head to the existing compact graph backbone and includes a descriptor-only ablation. `improvement.py` supplies new two-phase training/selection/evaluation orchestration and a selected-pipeline prediction CLI; `improvement_report.py` verifies and reports its results. These modules were generated here with AI assistance. They reuse this project's existing graph code and the same licensed descriptor, chemistry, forest and tensor dependencies. No new third-party implementation was vendored. Descriptor augmentation and weighted score averaging are established methods and are not claimed as new inventions. The experiment reuses previously inspected benchmark partitions, which is recorded as exploratory development.

The next extension, `kernel.py`, implements the standard binary Tanimoto formula and project-specific wrappers for training-only grouped cross-validation, regularized sigmoid calibration and fixed ensembles. `kernel_study.py` and `kernel_report.py` provide locked evaluation, saved-model inference and audits. RDKit implements Morgan fingerprints; scikit-learn implements SVC, LogisticRegression and StratifiedGroupKFold. These library implementations are not newly authored here. [Tripp et al. (2023)](https://arxiv.org/abs/2306.14809) informed the kernel experiment, but its proposed random-feature approximation and source implementation were not copied or reproduced. [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) informed the model-selection safeguards. The source changes were generated here with AI assistance, use already inventoried dependencies, and add no vendored third-party implementation or pretrained weights. There is no claim that an exact Tanimoto kernel, SVM, calibration or ensemble method is a novel algorithm.

`scripts/audit_licenses.py` records installed package versions, metadata and hashes of shipped license/notice files. Missing wheel notices can be supplemented from checksum-verified, exact-version PyPI source archives through `scripts/fetch_missing_notices.py`. This is dependency provenance and notice collection, not a comprehensive review of every compiled/bundled component, patent, trademark, contract, training-data right or possible code similarity.

The [U.S. Copyright Office's computer-program guidance](https://www.copyright.gov/circs/circ61.pdf) distinguishes copyrightable program expression from methods and algorithms. That distinction does not establish that any particular generated implementation is globally unique or free of all restrictions.

No outbound open-source license has yet been selected for project-specific material. `THIRD_PARTY_NOTICES.md` documents external licenses; it does not relicense external code or data. Choose an appropriate code license and confirm ownership before publishing an explicitly licensed release.

## Count-fingerprint extension, 2026-09-24

The `count_fingerprints` function calls RDKit's existing Morgan count API. `count_tanimoto` implements the established dot-product Tanimoto equation, and the shared classifier records whether it uses binary or count features. `count_study.py` and its tests were written here with AI assistance to compare those representations with matched training folds and tuning budgets. The existing scikit-learn SVM and sigmoid estimators are reused. No paper implementation, external model weights, or additional package was downloaded. The [count protocol](COUNT_PROTOCOL.md) distinguishes this experiment from Tripp et al.'s random-feature work and states the limits of reusing the benchmark.
