# Third-party notices

This project contains newly assembled study code and intentionally uses third-party software and research data. These components retain their own rights and license conditions. This notice is not an outbound license for the project.

## Reference implementation and descriptors

**Chemprop 1.6.1 — MIT.** Copyright (c) 2020 Wengong Jin, Kyle Swanson, Kevin Yang, Regina Barzilay, Tommi Jaakkola. The reference experiment imports this package directly. Its model code is not claimed as project-authored work. Preserve the copyright and permission notice when redistributing copies or substantial portions of the software. [Complete shipped notice](docs/license-audit/licenses/chemprop-1.6.1/chemprop-1.6.1.dist-info/LICENSE.txt). [Upstream project](https://github.com/chemprop/chemprop).

**Descriptastorus 2.8.0 — BSD-3-Clause notice.** Copyright (c) 2018–2023 Novartis Institutes for BioMedical Research Inc. The reference experiment uses its normalized descriptor implementation and supplied normalization parameters. [Complete shipped notice](docs/license-audit/licenses/descriptastorus-2.8.0/descriptastorus-2.8.0.dist-info/LICENSE). [Upstream project](https://github.com/bp-kelley/descriptastorus).

## Other scientific libraries

NumPy, pandas, SciPy, scikit-learn, RDKit, PyTorch and joblib ship BSD-family notices; openpyxl ships an MIT notice. Matplotlib and dependencies include their own notices and bundled-component terms. Transitive tools are not uniformly MIT/BSD: for example, certifi identifies MPL-2.0 and tqdm identifies MPL-2.0 AND MIT. Do not infer all redistribution obligations from this short list or from a package's top-level license label.

The complete environment inventories, including build/development/transitive packages, are in [the license audit](docs/license-audit/README.md). Preserved notices govern their respective components. A source-only project bundle contains dependency declarations and notices, not the installed library implementations or virtual environments. Packaging a Docker image, executable, environment or dependency source requires checking that distribution's additional obligations.

## Data and papers

Dataset: Stokes et al., “A Deep Learning Approach to Antibiotic Discovery,” Cell (2020), DOI [10.1016/j.cell.2020.01.021](https://doi.org/10.1016/j.cell.2020.01.021), Table S1B. Correction: [10.1016/j.cell.2020.04.001](https://doi.org/10.1016/j.cell.2020.04.001). Algorithmic background: Yang et al. (2019), DOI [10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237).

The fingerprint-kernel follow-up also cites Tripp et al., [Tanimoto Random Features for Scalable Molecular Machine Learning](https://arxiv.org/abs/2306.14809) (NeurIPS 2023), and Cawley and Talbot, [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html) (JMLR 2010). These are research references, not imported source implementations. The new SVM/calibration estimators are supplied by the existing attributed scikit-learn dependency.

No general dataset redistribution license was established in this review. Source spreadsheets, curated compound tables, molecular predictions and trained checkpoints stay outside the source bundle. Paper PDFs, figures and third-party model weights are not bundled. [Data provenance](docs/DATA_PROVENANCE.md) records the publisher URL, checksum and processing choices. The project's own license, if later selected, will not grant rights to those external materials.
