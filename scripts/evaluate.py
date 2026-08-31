"""Evaluate a trained embedding-based model on the held-out test set.

Loads outputs/<run>/{model.pt, meta.json}, computes the full metric suite on the
UniProtSMB test split at the validation-selected threshold, appends the row to a
benchmark table alongside published reference numbers, and writes both JSON and a
Markdown fragment for the README. Every "ours" number here is measured now.

Usage:
    python scripts/evaluate.py --run outputs/bilstm_esm2_t33_650M_UR50D
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.data.dataset import ResidueEmbeddingDataset, pad_collate  # noqa: E402
from pbsite.eval.benchmark import PUBLISHED, BenchmarkRow, BenchmarkTable  # noqa: E402
from pbsite.eval.metrics import residue_metrics  # noqa: E402
from pbsite.utils import get_device  # noqa: E402


def load_model(meta: dict, input_dim: int):
    from pbsite.models.bilstm import BiLSTMTagger
    from pbsite.models.frozen_head import ResidueMLPHead

    if meta["model_type"] == "bilstm":
        return BiLSTMTagger(input_dim=input_dim, hidden_size=meta.get("hidden_size", 256),
                            num_layers=meta.get("num_layers", 2))
    return ResidueMLPHead(input_dim=input_dim, hidden=meta.get("hidden", 512),
                          num_layers=meta.get("num_layers", 2))


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/<run> dir with model.pt + meta.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--out", default=None, help="where to write benchmark markdown/json")
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    device = get_device()
    emb_dir = Path(args.emb_dir) / meta["model_id"].split("/")[-1]

    test_recs = parse_clape_file(Path(args.data_dir) / "clape_smb" / "test_UniProtSMB.txt")
    ds = ResidueEmbeddingDataset(test_recs, emb_dir=emb_dir,
                                 add_physchem=meta.get("add_physchem", False))
    from torch.utils.data import DataLoader

    ld = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pad_collate)

    model = load_model(meta, meta["input_dim"]).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.inference_mode():
        for X, Y, M in ld:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            prob = torch.sigmoid(model(X, M))
            ys.append(Y[M].cpu().numpy())
            ps.append(prob[M].cpu().numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)

    metrics = residue_metrics(y_true, y_prob, threshold=meta["threshold"])
    print("Test metrics:", json.dumps(metrics, indent=2))

    table = BenchmarkTable(dataset="UniProtSMB test")
    table.add(BenchmarkRow(
        method=f"{meta['model_type']} ({meta['model_id'].split('/')[-1]})",
        auprc=metrics["auprc"], auroc=metrics["auroc"], f1=metrics["f1"],
        precision=metrics["precision"], recall=metrics["recall"], mcc=metrics["mcc"],
        source="ours",
    ))
    for row in PUBLISHED:
        table.add(row)

    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table.save_json(out / "benchmark.json")
    (out / "benchmark.md").write_text(table.to_markdown() + "\n", encoding="utf-8")
    print(f"\nWrote benchmark to {out}/benchmark.md")
    print(table.to_markdown())


if __name__ == "__main__":
    main()
