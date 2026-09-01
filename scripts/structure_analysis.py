"""Structure-aware analysis (stretch goal #8): are binding residues structurally distinct?

Fetches AlphaFold DB structures for test proteins, computes per-residue RSA + 3-state
secondary structure, caches them (so a later structure-augmented retrain can reuse the
cache), and reports how binding vs non-binding residues differ in burial (RSA) and SS.

Usage:
    python scripts/structure_analysis.py --limit 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.features.structure import structure_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="data/clape_smb/test_UniProtSMB.txt")
    ap.add_argument("--struct-dir", default="data/structures")
    ap.add_argument("--out", default="data/structure_analysis.json")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    records = parse_clape_file(args.test_file)[: args.limit]
    rsa_bind, rsa_non = [], []
    ss_bind = {"a": 0, "b": 0, "c": 0}
    ss_non = {"a": 0, "b": 0, "c": 0}
    n_ok = 0
    for i, r in enumerate(records, 1):
        feats = structure_features(r.id, r.seq, args.struct_dir)
        if feats is None:
            continue
        n_ok += 1
        labels = np.array(r.labels)
        rsa = feats[:, 0]
        ss = feats[:, 1:4].argmax(axis=1)  # 0=helix,1=sheet,2=coil
        rsa_bind.extend(rsa[labels == 1].tolist())
        rsa_non.extend(rsa[labels == 0].tolist())
        for cls, key in zip([labels == 1, labels == 0], [ss_bind, ss_non], strict=True):
            sc = ss[cls]
            key["a"] += int((sc == 0).sum())
            key["b"] += int((sc == 1).sum())
            key["c"] += int((sc == 2).sum())
        if i % 20 == 0:
            print(f"  processed {i}/{len(records)} ({n_ok} with structures)")

    def frac(d):
        t = sum(d.values()) or 1
        return {k: round(v / t, 3) for k, v in d.items()}

    summary = {
        "n_proteins_requested": len(records),
        "n_with_structure": n_ok,
        "mean_RSA_binding": round(float(np.mean(rsa_bind)), 3) if rsa_bind else None,
        "mean_RSA_nonbinding": round(float(np.mean(rsa_non)), 3) if rsa_non else None,
        "ss_fraction_binding": frac(ss_bind),      # helix/sheet/coil among binding res
        "ss_fraction_nonbinding": frac(ss_non),
        "n_binding_residues": len(rsa_bind),
        "n_nonbinding_residues": len(rsa_non),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
