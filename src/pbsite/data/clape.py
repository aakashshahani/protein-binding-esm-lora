"""Parse the CLAPE-SMB / UniProtSMB 3-line record format.

Each record is exactly three lines::

    >P12345
    MSTFARLF...            # amino-acid sequence
    000000010...           # per-residue label, '1' = ligand-binding residue

The label string is the same length as the sequence. This is the format used
by https://github.com/JueWangTHU/CLAPE-SMB (files under Raw_data/), which we
adopt verbatim so our test-set numbers are directly comparable to the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Record:
    id: str
    seq: str
    labels: tuple[int, ...]  # 0/1 per residue, len == len(seq)

    def __post_init__(self) -> None:
        if len(self.seq) != len(self.labels):
            raise ValueError(
                f"{self.id}: seq len {len(self.seq)} != labels len {len(self.labels)}"
            )

    @property
    def n_pos(self) -> int:
        return sum(self.labels)


def parse_clape_file(path: str | Path) -> list[Record]:
    """Read a 3-line-per-record file into a list of Records.

    Validates the label alphabet ({0,1}) and seq/label length agreement.
    Blank trailing lines are ignored.
    """
    lines = [ln.rstrip("\n\r") for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: line count {len(lines)} is not a multiple of 3")

    records: list[Record] = []
    for i in range(0, len(lines), 3):
        header, seq, lab = lines[i], lines[i + 1], lines[i + 2]
        if not header.startswith(">"):
            raise ValueError(f"{path}: record {i//3} header missing '>': {header!r}")
        if set(lab) - {"0", "1"}:
            raise ValueError(f"{path}: record {header} has non-binary label chars")
        records.append(
            Record(id=header[1:].strip(), seq=seq.strip(), labels=tuple(int(c) for c in lab))
        )
    return records


def write_clape_file(records: list[Record], path: str | Path) -> None:
    """Write records back out in the canonical 3-line format."""
    out = []
    for r in records:
        out.append(f">{r.id}")
        out.append(r.seq)
        out.append("".join(str(x) for x in r.labels))
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def dataset_stats(records: list[Record]) -> dict[str, float]:
    """Summary stats used for the honest data card / README."""
    n_res = sum(len(r.seq) for r in records)
    n_pos = sum(r.n_pos for r in records)
    lens = [len(r.seq) for r in records]
    return {
        "n_proteins": len(records),
        "n_residues": n_res,
        "n_binding_residues": n_pos,
        "pos_fraction": (n_pos / n_res) if n_res else 0.0,
        "min_len": min(lens) if lens else 0,
        "max_len": max(lens) if lens else 0,
        "mean_len": (sum(lens) / len(lens)) if lens else 0.0,
    }
