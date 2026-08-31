"""Minimal BioLiP2 parsing for the cross-dataset generalization check (stretch #9).

BioLiP2 ships tab-separated annotation dumps. The binding-site residue field
encodes residues like "A12 A15 A47" (chain + residue number). We map those onto
the UniProt canonical sequence (resolved via uniprot.fetch_fasta) to produce the
same per-residue 0/1 label vector used elsewhere.

This is intentionally light: it is NOT part of the comparable benchmark, only an
external test set. Full column spec: https://zhanggroup.org/BioLiP/download
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_RES = re.compile(r"[A-Za-z]?(\d+)")


@dataclass
class BioLiPRow:
    pdb_id: str
    receptor_chain: str
    ligand_id: str
    binding_residue_pdb: str      # e.g. "A12 A15 A47" (PDB numbering)
    uniprot_acc: str | None


def parse_binding_positions(field: str) -> list[int]:
    """Extract integer residue positions from a BioLiP binding-site field."""
    positions: list[int] = []
    for tok in field.split():
        m = _RES.search(tok)
        if m:
            positions.append(int(m.group(1)))
    return positions


# Ligands to EXCLUDE for a small-molecule (SMB) definition, matching CLAPE-SMB:
# metal ions, water, and common non-drug-like crystallization additives.
NON_SMB_LIGANDS = {
    "HOH", "MG", "ZN", "NA", "CA", "K", "MN", "FE", "FE2", "CU", "CU1", "NI",
    "CO", "CD", "HG", "CL", "SO4", "PO4", "GOL", "EDO", "PEG", "DMS", "ACT",
    "NAG", "BMA", "MAN", "FUC",
}


def is_small_molecule(ligand_id: str) -> bool:
    return ligand_id.upper() not in NON_SMB_LIGANDS
