"""Fetch canonical sequences from the UniProt REST API.

Used for (a) verifying CLAPE-SMB sequences against current UniProt canonical
forms and (b) resolving sequences for BioLiP2-derived cross-dataset records.
Every fetched FASTA is cached under the data dir so a repeat run is offline.
"""
from __future__ import annotations

from pathlib import Path

import requests

FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def fetch_fasta(acc: str, cache_dir: str | Path, timeout: int = 30) -> str | None:
    """Return the sequence (no header) for a UniProt accession, or None.

    Cached to ``cache_dir/{acc}.fasta``. Network is only touched on a miss.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{acc}.fasta"
    if cached.exists():
        text = cached.read_text(encoding="utf-8")
    else:
        resp = requests.get(FASTA_URL.format(acc=acc), timeout=timeout)
        if resp.status_code != 200 or not resp.text.startswith(">"):
            return None
        text = resp.text
        cached.write_text(text, encoding="utf-8")
    return "".join(line for line in text.splitlines() if not line.startswith(">"))
