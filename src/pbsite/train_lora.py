"""LoRA fine-tuning loop for per-residue binding classification.

Tokenizes sequences with the ESM-2 tokenizer, aligns per-residue labels to
tokens (special tokens get label -100 = ignore), and trains only the LoRA
adapter + classification head. Two hardware tracks are configured in
configs/lora.yaml: ``local`` (150M on a 4 GB GPU) and ``colab`` (650M on T4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.clape import parse_clape_file
from .eval.metrics import best_f1_threshold, residue_metrics
from .models.losses import masked_focal
from .utils import get_device, set_seed

FILES = {"train": "train_UniProtSMB.txt", "valid": "valid_UniProtSMB.txt"}
IGNORE = -100


class _TokenDataset:
    def __init__(self, records, tokenizer, max_len):
        self.records = records
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        seq = r.seq[: self.max_len]
        labels = list(r.labels[: self.max_len])
        enc = self.tok(seq, add_special_tokens=True)
        ids = enc["input_ids"]
        # ids = [CLS] + residues + [EOS]; label the specials as IGNORE
        lab = [IGNORE] + labels + [IGNORE]
        assert len(ids) == len(lab), (len(ids), len(lab))
        return ids, lab


def _collate(batch, pad_id):
    import torch

    maxlen = max(len(ids) for ids, _ in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), IGNORE, dtype=torch.float32)
    for i, (ids, lab) in enumerate(batch):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids)
        attn[i, :L] = 1
        labels[i, :L] = torch.tensor(lab, dtype=torch.float32)
    return input_ids, attn, labels


def _eval(model, loader, device):
    import torch

    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for input_ids, attn, labels in loader:
            input_ids, attn = input_ids.to(device), attn.to(device)
            logits = model(input_ids, attn)
            prob = torch.sigmoid(logits).cpu()
            keep = labels != IGNORE
            ys.append(labels[keep].numpy())
            ps.append(prob[keep].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_lora(cfg: dict, args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from .models.lora_esm import LoraESMTokenClassifier

    set_seed(cfg["seed"])
    device = get_device()
    track = cfg["tracks"][args.track]
    model_id = track["model_id"]
    max_len = track["max_len"]
    raw = Path(args.data_dir) / "clape_smb"

    tok = AutoTokenizer.from_pretrained(model_id)
    train_recs = parse_clape_file(raw / FILES["train"])
    val_recs = parse_clape_file(raw / FILES["valid"])

    def collate(b):
        return _collate(b, tok.pad_token_id)

    train_ld = DataLoader(_TokenDataset(train_recs, tok, max_len),
                          batch_size=track["batch_size"], shuffle=True, collate_fn=collate)
    val_ld = DataLoader(_TokenDataset(val_recs, tok, max_len),
                        batch_size=track["batch_size"], shuffle=False, collate_fn=collate)

    model = LoraESMTokenClassifier(
        model_id=model_id, lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
        head_dropout=cfg["head"]["dropout"],
        grad_checkpointing=track.get("grad_checkpointing", True),
    ).to(device)
    print("param summary:", model.trainable_parameter_summary())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    steps = max(len(train_ld) * cfg["train"]["epochs"], 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.cuda.amp.GradScaler(enabled=track.get("amp", False) and device == "cuda")
    accum = track.get("grad_accum", 1)

    run_name = args.run_name or f"lora_{model_id.split('/')[-1]}_{args.track}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional W&B
    wandb = None
    try:
        import os

        import wandb as _wb
        _wb.init(project=os.environ.get("WANDB_PROJECT", "protein-binding-esm-lora"),
                 name=run_name, config={**cfg, "track": args.track},
                 mode=os.environ.get("WANDB_MODE", "offline"))
        wandb = _wb
    except Exception as exc:
        print(f"[wandb disabled: {exc}]")

    g, a = cfg["loss"]["focal_gamma"], cfg["loss"]["focal_alpha"]
    best_auprc, best_thr = -1.0, 0.5
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        opt.zero_grad()
        for step, (input_ids, attn, labels) in enumerate(train_ld, 1):
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            mask = (labels != IGNORE).float()
            targets = labels.clamp_min(0.0)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(input_ids, attn)
                loss = masked_focal(logits, targets, mask, gamma=g, alpha=a) / accum
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            running += loss.item() * accum

        y_true, y_prob = _eval(model, val_ld, device)
        thr = best_f1_threshold(y_true, y_prob)
        val = residue_metrics(y_true, y_prob, threshold=thr)
        print(f"epoch {epoch}  loss={running/len(train_ld):.4f}  "
              f"val_auprc={val['auprc']:.4f}  val_mcc={val['mcc']:.4f}")
        if wandb:
            wandb.log({"epoch": epoch, "train_loss": running / len(train_ld),
                       **{f"val_{k}": v for k, v in val.items()}})
        if val["auprc"] > best_auprc:
            best_auprc, best_thr = val["auprc"], thr
            model.backbone.save_pretrained(out_dir / "lora_adapter")
            torch.save(model.classifier.state_dict(), out_dir / "head.pt")
            (out_dir / "val_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")

    meta = {"model_type": "lora", "model_id": model_id, "track": args.track,
            "max_len": max_len, "threshold": float(best_thr), "best_val_auprc": float(best_auprc)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter + head (val AUPRC={best_auprc:.4f}) -> {out_dir}")
    if wandb:
        wandb.finish()
