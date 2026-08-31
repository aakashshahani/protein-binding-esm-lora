"""Train a model from a config and log to Weights & Biases (offline by default).

Dispatches on the config's ``model`` field:
  - bilstm / frozen_head : train over cached ESM-2 embeddings.
  - lora                 : LoRA fine-tune ESM-2 end-to-end (--track local|colab).

Saves outputs/<run_name>/{model.pt, meta.json, val_metrics.json}. The best
checkpoint (by validation AUPRC) is kept. All metrics written here are measured,
never hand-entered.

Usage:
    python scripts/train.py --config configs/bilstm.yaml
    python scripts/train.py --config configs/lora.yaml --track local
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

FILES = {"train": "train_UniProtSMB.txt", "valid": "valid_UniProtSMB.txt",
         "test": "test_UniProtSMB.txt"}


def _emb_tag(model_id: str) -> str:
    return model_id.split("/")[-1]


def _init_wandb(cfg: dict, run_name: str):
    try:
        import os

        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "protein-binding-esm-lora"),
            name=run_name,
            config=cfg,
            mode=os.environ.get("WANDB_MODE", "offline"),
        )
        return wandb
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")
        return None


def _compute_loss(cfg, logits, y, mask, pos_weight):
    from pbsite.models.losses import masked_bce, masked_focal

    lt = cfg["loss"]["type"]
    if lt == "focal":
        return masked_focal(logits, y, mask, cfg["loss"]["focal_gamma"], cfg["loss"]["focal_alpha"])
    if lt == "weighted_bce":
        return masked_bce(logits, y, mask, pos_weight=pos_weight)
    return masked_bce(logits, y, mask)


def _gather(model, loader, device):
    """Run model over a loader, return flat (y_true, y_prob) for masked positions."""
    import torch

    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for X, Y, M in loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits = model(X, M)
            prob = torch.sigmoid(logits)
            ys.append(Y[M].cpu().numpy())
            ps.append(prob[M].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_embedding_model(cfg: dict, args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from pbsite.models.bilstm import BiLSTMTagger
    from pbsite.models.frozen_head import ResidueMLPHead

    set_seed(cfg["seed"])
    device = get_device()
    model_id = cfg["embeddings"]["model_id"]
    emb_dir = Path(args.emb_dir) / _emb_tag(model_id)
    raw = Path(args.data_dir) / "clape_smb"
    add_physchem = cfg["embeddings"].get("add_physchem", False)

    train_recs = parse_clape_file(raw / FILES["train"])
    val_recs = parse_clape_file(raw / FILES["valid"])

    input_dim = cfg["embeddings"]["dim"] + (7 if add_physchem else 0)
    ds_kw = {"emb_dir": emb_dir, "add_physchem": add_physchem}
    train_ds = ResidueEmbeddingDataset(train_recs, **ds_kw)
    val_ds = ResidueEmbeddingDataset(val_recs, **ds_kw)
    bs = cfg["train"]["batch_size"]
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=pad_collate)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=pad_collate)

    if cfg["model"] == "bilstm":
        model = BiLSTMTagger(
            input_dim=input_dim,
            hidden_size=cfg["bilstm"]["hidden_size"],
            num_layers=cfg["bilstm"]["num_layers"],
            dropout=cfg["bilstm"]["dropout"],
            bidirectional=cfg["bilstm"]["bidirectional"],
        )
    else:
        model = ResidueMLPHead(
            input_dim=input_dim, hidden=cfg["head"]["hidden"],
            num_layers=cfg["head"]["num_layers"], dropout=cfg["head"]["dropout"],
        )
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=cfg["train"]["scheduler_patience"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", False) and device == "cuda")
    pos_weight = pos_weight_from_records(train_recs)

    run_name = args.run_name or f"{cfg['model']}_{_emb_tag(model_id)}"
    wandb = _init_wandb(cfg, run_name)
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auprc, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        for X, Y, M in train_ld:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(X, M)
                loss = _compute_loss(cfg, logits, Y, M, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        y_true, y_prob = _gather(model, val_ld, device)
        thr = best_f1_threshold(y_true, y_prob)
        val = residue_metrics(y_true, y_prob, threshold=thr)
        sched.step(val["auprc"])
        print(f"epoch {epoch:02d}  loss={running/len(train_ld):.4f}  "
              f"val_auprc={val['auprc']:.4f}  val_mcc={val['mcc']:.4f}  thr={thr:.2f}")
        if wandb:
            wandb.log({"epoch": epoch, "train_loss": running / len(train_ld), **{f"val_{k}": v for k, v in val.items()}})

        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
            patience = 0
            (out_dir / "val_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"early stop at epoch {epoch}")
                break

    torch.save(best_state, out_dir / "model.pt")
    meta = {
        "model_type": cfg["model"],
        "model_id": model_id,
        "input_dim": input_dim,
        "add_physchem": add_physchem,
        "threshold": float(best_thr),
        "best_val_auprc": float(best_auprc),
    }
    if cfg["model"] == "bilstm":
        meta.update(hidden_size=cfg["bilstm"]["hidden_size"], num_layers=cfg["bilstm"]["num_layers"])
    else:
        meta.update(hidden=cfg["head"]["hidden"], num_layers=cfg["head"]["num_layers"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved best model (val AUPRC={best_auprc:.4f}) -> {out_dir}")
    if wandb:
        wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--track", default="local", choices=["local", "colab"],
                    help="LoRA hardware track (only used when model=lora)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["model"] in ("bilstm", "frozen_head"):
        train_embedding_model(cfg, args)
    elif cfg["model"] == "lora":
        from pbsite.train_lora import train_lora  # separate, tokenization-based loop

        train_lora(cfg, args)
    else:
        raise SystemExit(f"unknown model type: {cfg['model']}")


if __name__ == "__main__":
    main()
