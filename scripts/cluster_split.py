"""Cluster sequences at 30% identity (MMseqs2) and build leakage-safe CV folds.

Audits how many test proteins share a 30%-identity cluster with any train
protein (an honest benchmark should be ~0) and writes GroupKFold folds over the
training set keyed by cluster id, so homologs never span folds.

Usage:
    python scripts/cluster_split.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.data.splits import (  # noqa: E402
    audit_leakage,
    group_kfold_indices,
    run_mmseqs_cluster,
    write_fasta,
)
from pbsite.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = cfg["mmseqs"]
    data_dir = Path(args.data_dir)
    raw = data_dir / "clape_smb"
    out_dir = data_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = parse_clape_file(raw / "train_UniProtSMB.txt")
    test = parse_clape_file(raw / "test_UniProtSMB.txt")
    all_records = train + test

    fasta = out_dir / "all.fasta"
    write_fasta(all_records, fasta)

    print("Running MMseqs2 clustering (this uses a native binary or Docker)...")
    clusters = run_mmseqs_cluster(
        fasta, out_dir / "mmseqs",
        min_seq_id=m["min_seq_id"], coverage=m["coverage"],
        docker_image=m["docker_image"],
    )
    (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2), encoding="utf-8")

    train_ids = [r.id for r in train]
    test_ids = [r.id for r in test]
    leak = audit_leakage(train_ids, test_ids, clusters)
    print(f"Leakage audit: {leak['n_leaked']}/{leak['n_test']} test proteins share a "
          f"30%-id cluster with train.")

    folds = group_kfold_indices(train_ids, clusters, n_folds=cfg["cv"]["n_folds"],
                                seed=cfg["seed"])
    fold_ids = [[train_ids[i] for i in fold] for fold in folds]
    (out_dir / "cv_folds.json").write_text(json.dumps(fold_ids, indent=2), encoding="utf-8")

    summary = {
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_clusters": len(set(clusters.values())),
        "leakage": leak,
        "n_folds": cfg["cv"]["n_folds"],
    }
    (out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote splits to {out_dir}  ({summary['n_clusters']} clusters)")


if __name__ == "__main__":
    main()
