"""Assemble the honest benchmark table.

Our own runs contribute measured metrics. External methods contribute EITHER
metrics we reproduced by running their released tool on our split, OR numbers
copied from their paper. The latter are flagged ``source="published"`` and are
rendered with an explicit "(published, not reproduced)" note in the README so
the distinction is never blurred.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BenchmarkRow:
    method: str
    auprc: float | None = None
    auroc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    mcc: float | None = None
    source: str = "ours"          # {"ours", "reproduced", "published"}
    note: str = ""
    citation: str = ""


# Published reference numbers on the UniProtSMB test set. These are transcribed
# from the CLAPE-SMB paper (J. Cheminformatics 2024) and clearly marked as such.
# NOTE: only fields the paper actually reports are filled; unknown = None.
PUBLISHED: list[BenchmarkRow] = [
    BenchmarkRow(
        method="CLAPE-SMB",
        mcc=0.699,
        source="published",
        note="UniProtSMB test set, paper value",
        citation="Wang et al., J. Cheminformatics 2024, doi:10.1186/s13321-024-00920-2",
    ),
]


@dataclass
class BenchmarkTable:
    dataset: str = "UniProtSMB test"
    rows: list[BenchmarkRow] = field(default_factory=list)

    def add(self, row: BenchmarkRow) -> None:
        self.rows.append(row)

    def to_markdown(self) -> str:
        cols = ["method", "source", "auprc", "auroc", "f1", "precision", "recall", "mcc", "note"]
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [f"**Dataset: {self.dataset}**", "", head, sep]
        for r in self.rows:
            d = asdict(r)
            cells = []
            for c in cols:
                v = d[c]
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"dataset": self.dataset, "rows": [asdict(r) for r in self.rows]}, indent=2),
            encoding="utf-8",
        )
