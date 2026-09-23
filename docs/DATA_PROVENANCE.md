# Data provenance

## Primary source

- Study: Stokes et al., Cell (2020), DOI `10.1016/j.cell.2020.01.021`.
- File: `https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx`.
- Retrieved: 2026-09-23.
- SHA-256: `a75e9546c5af35c6eee29bbaa0a9e36fe97cb346b3f2c614e590d31962410a31`.
- Sheet: `S1B`, second row is the column header.
- Expected rows: 2,335; 120 `Active`, 2,215 `Inactive`.
- Label mapping: published `Active` -> 1, `Inactive` -> 0. Any other label is a hard error.
- Original source row numbers are retained, starting at Excel row 3.

No mirror data is used in this project. A third-party copy was investigated during acquisition, then superseded by the publisher's workbook. No original author model checkpoint is used or loaded.

The paper's correction (DOI `10.1016/j.cell.2020.04.001`) includes changes to Table S2B, figures and text. This project uses S1B directly and does not use S2B predictions or manually reconstruct corrected labels. We verified source location, file checksum, schema and published counts; this does not independently validate the underlying laboratory measurements.

## Curation choices

1. Parse SMILES with RDKit; reject invalid or empty structures.
2. Apply RDKit Cleanup, FragmentParent and Uncharger; remove atom map numbers.
3. Generate canonical isomeric SMILES as molecular identity. Preserve stereochemistry; do not canonicalize tautomers.
4. Merge repeated standardized identities if all their labels agree. Retain all original row numbers.
5. Exclude all source rows for a standardized identity whose labels disagree.
6. Use non-isomeric canonical connectivity as a grouping key so stereoisomers cannot span random partitions.
7. Use a non-chiral Bemis–Murcko scaffold for scaffold partitioning; all acyclic compounds share `__ACYCLIC__`.

These choices are experimental assumptions. Removing salts/neutralizing structures can collapse chemically distinct assay entries. The conflicting-label quarantine prevents us from choosing a convenient label; it does not resolve the biological ambiguity. Different tautomers or close analogues may still occur in separate random partitions. Curation changes the sample from the paper, so results are not directly comparable to its reported ROC-AUC.

Preparation writes a full audit plus a second checksum of the curated CSV, used to reject altered input before training. Each run embeds the data manifest and installed RDKit version.

## Distribution

Third-party data rights remain with their original owners. Raw/curated molecules and model checkpoints are ignored by Git by default; acquisition is reproducible through the pinned public source. The aggregate report can be inspected without downloading raw data.

The 2026-09-23 rights review did not establish an explicit unrestricted redistribution license for this workbook. Public download availability is not treated as such a grant. The source-only bundle excludes raw/curated data, molecular predictions, descriptors and checkpoints. This is a distribution choice while rights remain unresolved, not a legal conclusion that every form of dataset use is prohibited. See [publication preparation](PUBLISHING.md) and [publisher permissions guidance](https://www.elsevier.com/about/policies-and-standards/copyright/permissions).

The inspected workbook XML contained no matches for copyright, license/licence, Creative Commons or redistribution text. The [PMC author-manuscript page](https://pmc.ncbi.nlm.nih.gov/articles/PMC8349178/) links to a generic PMC copyright notice rather than supplying an explicit workbook license in the inspected text. These checks support recording the status as unresolved; absence of a matching text string does not prove either permission or prohibition.
