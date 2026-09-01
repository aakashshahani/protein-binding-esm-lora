"""Structure-aware per-residue features from AlphaFold DB (stretch goal #8).

For a UniProt accession we fetch the predicted structure from AlphaFold DB and
compute, per residue:
  - relative solvent accessibility (RSA) = SASA / max-ASA(residue), Tien 2013 scale
  - 3-state secondary structure (helix / sheet / coil) as a one-hot triple

Returns an (L, 4) float array aligned to the sequence, or None if the structure
is unavailable or does not match the sequence length. Structures are cached.

This is optional and gated behind `biotite`; import errors are surfaced clearly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

STRUCT_DIM = 4  # [RSA, is_helix, is_sheet, is_coil]

# Resolve the CIF URL via the API so we track AlphaFold DB's current model version
# (v6 as of 2026) instead of hard-coding a filename that 404s after a version bump.
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

# Tien et al. 2013 theoretical maximum accessible surface area (A^2), 3-letter.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def fetch_alphafold_cif(acc: str, cache_dir: str | Path, timeout: int = 60) -> Path | None:
    """Download the AlphaFold DB v4 mmCIF for a UniProt accession (cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"AF-{acc}.cif"
    if dest.exists():
        return dest
    try:
        api = requests.get(AF_API.format(acc=acc), timeout=timeout)
        if api.status_code != 200 or not api.json():
            return None
        cif_url = api.json()[0].get("cifUrl")
        if not cif_url:
            return None
        r = requests.get(cif_url, timeout=timeout)
    except (requests.RequestException, ValueError):
        return None
    if r.status_code != 200 or not r.text.startswith("data_"):
        return None
    dest.write_text(r.text, encoding="utf-8")
    return dest


def structure_features(acc: str, seq: str, cache_dir: str | Path) -> np.ndarray | None:
    """Return (len(seq), 4) [RSA, helix, sheet, coil] or None if unavailable/mismatched."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    cif = fetch_alphafold_cif(acc, cache_dir)
    if cif is None:
        return None

    f = pdbx.CIFFile.read(str(cif))
    arr = pdbx.get_structure(f, model=1)
    arr = arr[struc.filter_amino_acids(arr)]
    if arr.array_length() == 0:
        return None

    _, res_names = struc.get_residues(arr)
    n_res = len(res_names)
    if n_res != len(seq):
        # AF model should span the canonical sequence; skip on any mismatch
        return None

    atom_sasa = struc.sasa(arr, vdw_radii="Single")
    res_sasa = struc.apply_residue_wise(arr, atom_sasa, np.nansum)
    max_asa = np.array([MAX_ASA.get(rn, 197.0) for rn in res_names], dtype=np.float32)
    rsa = np.clip(res_sasa / max_asa, 0.0, 1.5).astype(np.float32)

    sse = struc.annotate_sse(arr)  # per-residue 'a' (helix) / 'b' (sheet) / 'c' (coil)
    feats = np.zeros((n_res, STRUCT_DIM), dtype=np.float32)
    feats[:, 0] = rsa
    feats[:, 1] = (sse == "a").astype(np.float32)
    feats[:, 2] = (sse == "b").astype(np.float32)
    feats[:, 3] = (sse == "c").astype(np.float32)
    return feats
