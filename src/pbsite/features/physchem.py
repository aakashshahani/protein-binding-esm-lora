"""Lightweight per-residue physicochemical features (optional concat to ESM).

Seven interpretable scalars per residue: Kyte-Doolittle hydrophobicity, formal
charge at pH 7, polarity flag, aromaticity flag, molecular volume, and two
flags (is_glycine, is_proline) that often matter at binding-site loops.
Values are normalized to roughly [-1, 1] so they play nicely with embeddings.
"""
from __future__ import annotations

import numpy as np

PHYSCHEM_DIM = 7

# Kyte-Doolittle hydrophobicity scale (normalized by /4.5).
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
_POLAR = set("STNQCYRKHDE")
_AROMATIC = set("FWYH")
# residue side-chain volumes (A^3), normalized by /230.
_VOL = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}


def physchem_features(seq: str) -> np.ndarray:
    """Return an (L, 7) float32 array for a sequence."""
    feats = np.zeros((len(seq), PHYSCHEM_DIM), dtype=np.float32)
    for i, aa in enumerate(seq):
        feats[i, 0] = _KD.get(aa, 0.0) / 4.5
        feats[i, 1] = _CHARGE.get(aa, 0.0)
        feats[i, 2] = 1.0 if aa in _POLAR else 0.0
        feats[i, 3] = 1.0 if aa in _AROMATIC else 0.0
        feats[i, 4] = _VOL.get(aa, 120.0) / 230.0
        feats[i, 5] = 1.0 if aa == "G" else 0.0
        feats[i, 6] = 1.0 if aa == "P" else 0.0
    return feats
