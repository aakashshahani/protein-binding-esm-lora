"""Validate the downloaded splits and emit an honest data card.

Parses each UniProtSMB split, checks seq/label integrity, computes class balance
and length stats, and writes data/DATA_CARD.md + data/data_card.json. Run after
download_data.py. No network, no training.

Usage:
    python scripts/build_dataset.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import dataset_stats, parse_clape_file  # noqa: E402
from pbsite.utils import load_config  # noqa: E402

SPLITS = {"train": "train_UniProtSMB.txt", "valid": "valid_UniProtSMB.txt",
          "test": "test_UniProtSMB.txt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.data_dir) / "clape_smb"
    card: dict = {"source": "CLAPE-SMB / UniProtSMB",
                  "snapshot_date": cfg["clape_smb"]["snapshot_date"], "splits": {}}

    for name, fn in SPLITS.items():
        recs = parse_clape_file(raw / fn)
        card["splits"][name] = dataset_stats(recs)

    (Path(args.data_dir) / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    lines = ["# Data card — UniProtSMB (via CLAPE-SMB)", "",
             f"- Snapshot date: **{card['snapshot_date']}**",
             "- Source: https://github.com/JueWangTHU/CLAPE-SMB (Raw_data/UniProtSMB)",
             "- Task: per-residue small-molecule binding-site classification",
             "- Label: `1` = binding residue, `0` = non-binding", "",
             "| split | proteins | residues | binding residues | pos. fraction | mean len |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in SPLITS:
        s = card["splits"][name]
        lines.append(
            f"| {name} | {s['n_proteins']} | {s['n_residues']} | {s['n_binding_residues']} "
            f"| {s['pos_fraction']:.4f} | {s['mean_len']:.0f} |")
    lines += ["", "Binding residues are rare (see pos. fraction) — training uses focal / "
              "class-weighted loss to handle the imbalance."]
    (Path(args.data_dir) / "DATA_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
