import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
N_BITS = 2048


def fingerprints(smiles, radius=RADIUS):
    # Chirality is deliberately excluded to match the declared fingerprint baseline.
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=N_BITS, includeChirality=False
    )
    bits, arrays = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Cannot fingerprint invalid SMILES: {smi}")
        fp = generator.GetFingerprint(mol)
        # Keep both forms: NumPy for estimators and RDKit bits for similarity queries.
        array = np.zeros(N_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, array)
        bits.append(fp)
        arrays.append(array)
    return np.stack(arrays), bits


def nearest_train_similarity(query_bits, train_bits):
    if not train_bits:
        raise ValueError("Empty training reference set")
    # Each query is compared only with the corresponding training partition.
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, train_bits)) for fp in query_bits]
    )
