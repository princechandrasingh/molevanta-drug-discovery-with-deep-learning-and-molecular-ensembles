# Source-only publication preparation

Project title: **Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles**.

Use the complete title in the README and project description. The matching GitHub repository name is `molevanta-drug-discovery-with-deep-learning-and-molecular-ensembles`.

The source distribution contains project code, documentation, aggregate reports and dependency notices. The publisher's data, molecular predictions, trained models, virtual environments and installed reference software are excluded.

```bash
.venv/bin/python scripts/build_source_bundle.py
```

The archive builder uses an explicit allowlist, rejects symlinks/outside paths and molecule-level CSV columns, records a SHA-256 manifest, and verifies every zipped entry. It refuses to overwrite an earlier bundle. A bundle contains license notices as attribution, but no dependency implementation is vendored.

The default output is `dist/molevanta-source.zip`, with files under `molevanta/` and the full project title in `BUNDLE_MANIFEST.json`. Existing archives and experiment snapshots retain their historical names. The source's Python distribution identifier and CLI remain `molecular-generalization` and `molstudy` for compatibility.

The readability edition is packaged separately as `dist/molevanta-readable-source.zip` and includes the commented source, code guide and historical-run launcher. To build a distinct review archive without overwriting a prior one, pass an unused output path:

```bash
.venv/bin/python scripts/build_source_bundle.py --output dist/molevanta-readable-source.zip
```

## Remaining decisions before a licensed public release

- Choose an outbound license for project-specific material and confirm the rights to license that material. MIT is one common permissive option, but it has not been applied automatically. The existing third-party notices are not a project license.
- Keep the publisher workbook, curated structures, compound predictions and checkpoints outside the release until their applicable reuse terms are established. No unrestricted dataset redistribution license was found in this review. The [publisher's permissions guidance](https://www.elsevier.com/about/policies-and-standards/copyright/permissions) requires assessing the material and applicable license; public access alone is not a license grant.
- Describe the project as an AI-assisted computational study and credit Chemprop explicitly for the reference model. Do not describe the entire software stack or the D-MPNN algorithm as your invention.
- Report the shared-split results and their limitations. Do not use a comparison with the paper's 0.896 ROC-AUC as proof of superiority. No independent experimental discovery or clinical efficacy has been established.

The dependency review collected notices for all inventoried distributions, including five exact-version source notices missing from wheels. It does not certify absence of every copyright, patent, data, contractual or distribution issue. Bundling dependencies or deploying a commercial service would need a review of that specific distribution/use.

No GitHub repository, release, message or public upload is created by these scripts.
