"""Cluster-grouped k-fold cross-validation (grouped by MMseqs2 cluster).

Uses the folds in data/splits/cv_folds.json (GroupKFold over the training set,
keyed by 30%-identity cluster, so homologs never span folds). Trains a fresh
model on k-1 folds and evaluates on the held-out fold, then reports per-fold and
mean +/- std of the primary metric (AUPRC) and MCC. Runs on cached embeddings.

Usage:
    python scripts/cross_validate.py --config configs/bilstm.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.data.dataset import (  # noqa: E402
    ResidueEmbeddingDataset,
    pad_collate,
    pos_weight_from_records,
)
from pbsite.eval.metrics import best_f1_threshold, residue_metrics  # noqa: E402
from pbsite.utils import get_device, load_config, set_seed  # noqa: E402


def build_model(cfg, input_dim):
    from pbsite.models.bilstm import BiLSTMTagger
    from pbsite.models.frozen_head import ResidueMLPHead

    if cfg["model"] == "bilstm":
        return BiLSTMTagger(input_dim=input_dim, hidden_size=cfg["bilstm"]["hidden_size"],
                            num_layers=cfg["bilstm"]["num_layers"],
                            dropout=cfg["bilstm"]["dropout"])
    return ResidueMLPHead(input_dim=input_dim, hidden=cfg["head"]["hidden"],
                          num_layers=cfg["head"]["num_layers"], dropout=cfg["head"]["dropout"])


def gather(model, loader, device, torch):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for X, Y, M in loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            p = torch.sigmoid(model(X, M))
            ys.append(Y[M].cpu().numpy())
            ps.append(p[M].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from pbsite.models.losses import masked_focal

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bilstm.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--epochs", type=int, default=15, help="cap per fold (early-stopped)")
    ap.add_argument("--out", default="outputs/cv_bilstm.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    model_id = cfg["embeddings"]["model_id"]
    emb_dir = Path(args.emb_dir) / model_id.split("/")[-1]
    add_physchem = cfg["embeddings"].get("add_physchem", False)
    input_dim = cfg["embeddings"]["dim"] + (7 if add_physchem else 0)

    records = parse_clape_file(Path(args.data_dir) / "clape_smb" / "train_UniProtSMB.txt")
    by_id = {r.id: r for r in records}
    folds = json.loads((Path(args.data_dir) / "splits" / "cv_folds.json").read_text())
    print(f"{len(folds)} cluster-grouped folds over {len(records)} train proteins "
          f"({cfg['model']}, {model_id.split('/')[-1]})")

    fold_metrics = []
    for k, val_ids in enumerate(folds):
        val_set = set(val_ids)
        val_recs = [by_id[i] for i in val_ids if i in by_id]
        train_recs = [r for r in records if r.id not in val_set]
        tr = DataLoader(ResidueEmbeddingDataset(train_recs, emb_dir, add_physchem),
                        batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
        va = DataLoader(ResidueEmbeddingDataset(val_recs, emb_dir, add_physchem),
                        batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=pad_collate)

        model = build_model(cfg, input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                               weight_decay=cfg["train"]["weight_decay"])
        pw = pos_weight_from_records(train_recs)  # noqa: F841 (available if BCE chosen)
        best, best_state, patience = -1.0, None, 0
        for _ in range(args.epochs):
            model.train()
            for X, Y, M in tr:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                opt.zero_grad()
                loss = masked_focal(model(X, M), Y, M, cfg["loss"]["focal_gamma"],
                                    cfg["loss"]["focal_alpha"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
            yt, yp = gather(model, va, device, torch)
            ap_ = residue_metrics(yt, yp)["auprc"]
            if ap_ > best:
                best, best_state = ap_, {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break
        model.load_state_dict(best_state)
        yt, yp = gather(model, va, device, torch)
        thr = best_f1_threshold(yt, yp)
        m = residue_metrics(yt, yp, threshold=thr)
        fold_metrics.append(m)
        print(f"  fold {k+1}/{len(folds)}: n_val={len(val_recs)}  "
              f"AUPRC={m['auprc']:.4f}  MCC={m['mcc']:.4f}  F1={m['f1']:.4f}")

    def agg(key):
        vals = [m[key] for m in fold_metrics]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "model": cfg["model"], "model_id": model_id, "n_folds": len(folds),
        "grouped_by": "mmseqs_cluster_30pct",
        "auprc": agg("auprc"), "auroc": agg("auroc"), "f1": agg("f1"), "mcc": agg("mcc"),
        "per_fold": fold_metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCV {cfg['model']}: AUPRC {summary['auprc']['mean']:.4f} +/- "
          f"{summary['auprc']['std']:.4f} | MCC {summary['mcc']['mean']:.4f} +/- "
          f"{summary['mcc']['std']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
