"""Publisher-data acquisition, structure curation, and auditable grouping."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx"
SOURCE_SHA256 = "a75e9546c5af35c6eee29bbaa0a9e36fe97cb346b3f2c614e590d31962410a31"
PAPER_DOI = "10.1016/j.cell.2020.01.021"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reject NaN/Infinity so a manifest cannot silently contain invalid JSON numbers.
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download_data(root: Path) -> Path:
    path = root / "data/raw/stokes_table_s1.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(
            Request(SOURCE_URL, headers={"User-Agent": "molecular-generalization/0.1"}), timeout=60
        ) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
            raise ValueError(
                "Publisher file checksum changed. Inspect the new version before updating the pinned checksum."
            )
        path.write_bytes(payload)
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("Local source checksum mismatch; refusing to use unverified data.")
    return path


def standardize(smiles: str) -> dict[str, str]:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Empty SMILES")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Invalid or empty molecular structure")
    # Keep this chemistry order fixed: changing it can change identities and splits.
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Identity preserves stereochemistry; connectivity groups stereoisomers together.
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return {
        "smiles": canonical,
        "compound_id": hashlib.sha256(canonical.encode()).hexdigest()[:20],
        "connectivity": connectivity,
        # Keep all acyclic compounds together rather than allowing empty-scaffold leakage.
        "scaffold": scaffold or "__ACYCLIC__",
    }


def curate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, rejected = [], []
    for i, row in frame.reset_index(drop=True).iterrows():
        # Excel data starts on row 3 after the title and column-header rows.
        base = {"source_row": i + 3, "name": str(row["Name"]).strip(), "raw_smiles": row["SMILES"]}
        activity = row["Activity"]
        if activity not in ("Active", "Inactive"):
            raise ValueError(f"Unexpected activity label in source row {i + 3}: {activity!r}")
        try:
            fields = standardize(row["SMILES"])
        except (ValueError, RuntimeError) as exc:
            rejected.append({**base, "reason": f"structure_error: {exc}"})
            continue
        rows.append({**base, **fields, "label": int(activity == "Active")})
    valid = pd.DataFrame(rows)
    if valid.empty:
        raise ValueError("No valid molecules remain")
    # Quarantine every conflicting measurement rather than choose a convenient label.
    conflicts = set(valid.groupby("smiles").label.nunique().loc[lambda s: s > 1].index)
    for record in valid[valid.smiles.isin(conflicts)].to_dict("records"):
        rejected.append({**record, "reason": "conflicting_labels_after_standardization"})
    valid = valid[~valid.smiles.isin(conflicts)].copy()
    curated = []
    for _, group in valid.groupby("smiles", sort=True):
        # Merge agreeing duplicates while retaining every source row for the audit.
        first = group.iloc[0].to_dict()
        first["source_rows"] = ";".join(map(str, group.source_row.tolist()))
        first["source_count"] = len(group)
        curated.append(first)
        for record in group.iloc[1:].to_dict("records"):
            rejected.append({**record, "reason": "duplicate_same_label_merged"})
    clean = pd.DataFrame(curated).sort_values("compound_id").reset_index(drop=True)
    audit = pd.DataFrame(rejected, columns=list(dict.fromkeys([*valid.columns, "reason"])))
    summary = {
        "raw_rows": len(frame),
        "raw_actives": int((frame.Activity == "Active").sum()),
        "curated_rows": len(clean),
        "curated_actives": int(clean.label.sum()),
        "unique_scaffolds": int(clean.scaffold.nunique()),
        "acyclic_compounds": int((clean.scaffold == "__ACYCLIC__").sum()),
        "conflicting_structures": len(conflicts),
        "excluded_or_merged": audit.reason.value_counts().to_dict(),
        "standardization": "RDKit Cleanup, FragmentParent, Uncharger, clear atom maps, canonical isomeric SMILES; no tautomer canonicalization",
        "stereochemistry": "Preserved in identity; connectivity grouping keeps stereoisomers in one partition. Fingerprints ignore chirality.",
    }
    return clean, audit, summary


def prepare(root: Path) -> dict:
    source = root / "data/raw/stokes_table_s1.xlsx"
    if not source.exists():
        raise FileNotFoundError("Run molstudy download first")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source file checksum mismatch")
    frame = pd.read_excel(source, sheet_name="S1B", header=1)
    if len(frame) != 2335 or int((frame.Activity == "Active").sum()) != 120:
        raise ValueError("The source does not match the expected Table S1B counts")
    clean, audit, summary = curate(frame)
    dest = root / "data/processed"
    dest.mkdir(parents=True, exist_ok=True)
    clean.to_csv(dest / "molecules.csv", index=False)
    audit.to_csv(dest / "curation_audit.csv", index=False)
    summary.update(
        {
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "paper_doi": PAPER_DOI,
            "sheet": "S1B",
            "label_source": "Published Activity column (not re-thresholded)",
            "processed_sha256": sha256(dest / "molecules.csv"),
            "rdkit_version": rdBase.rdkitVersion,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(dest / "data_manifest.json", summary)
    return summary
