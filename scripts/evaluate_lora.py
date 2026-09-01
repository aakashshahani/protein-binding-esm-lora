"""Evaluate a trained LoRA ESM-2 checkpoint on the held-out test set.

Loads the saved PEFT adapter + classification head, scores every residue by
tiling sequences into non-overlapping windows of <= `win` residues (matching the
training context length so no residue is dropped), and reports the full metric
suite at the validation-selected threshold. Also writes meta.json for serving.

Usage:
    python scripts/evaluate_lora.py --run outputs/lora_esm2_t30_150M_UR50D_local \
        --model-id facebook/esm2_t30_150M_UR50D --win 510
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.eval.benchmark import PUBLISHED, BenchmarkRow, BenchmarkTable  # noqa: E402
from pbsite.eval.metrics import residue_metrics  # noqa: E402
from pbsite.utils import get_device  # noqa: E402


def score_sequence(model, tok, seq, device, win, torch):
    """Return per-residue probabilities for the full sequence via window tiling."""
    probs = np.zeros(len(seq), dtype=np.float32)
    for start in range(0, len(seq), win):
        chunk = seq[start:start + win]
        enc = tok(chunk, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            logits = model(enc["input_ids"], enc["attention_mask"])
        p = torch.sigmoid(logits)[0].float().cpu().numpy()
        # drop CLS (0) and EOS (last); the middle aligns 1:1 with chunk residues
        probs[start:start + len(chunk)] = p[1:1 + len(chunk)]
    return probs


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-id", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--track", default="local")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--win", type=int, default=510)
    args = ap.parse_args()

    run = Path(args.run)
    device = get_device()
    val = json.loads((run / "val_metrics.json").read_text(encoding="utf-8"))
    threshold = float(val["threshold"])

    # rebuild: base encoder + trained LoRA adapter + trained head
    base = AutoModel.from_pretrained(args.model_id)
    backbone = PeftModel.from_pretrained(base, str(run / "lora_adapter")).to(device).eval()
    hidden = base.config.hidden_size
    head = torch.nn.Linear(hidden, 1).to(device)
    head.load_state_dict(torch.load(run / "head.pt", map_location=device))
    head.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)

    class Wrapped(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            h = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return head(h).squeeze(-1)

    model = Wrapped()

    test = parse_clape_file(Path(args.data_dir) / "clape_smb" / "test_UniProtSMB.txt")
    ys, ps = [], []
    for i, r in enumerate(test, 1):
        prob = score_sequence(model, tok, r.seq, device, args.win, torch)
        ys.append(np.array(r.labels, dtype=np.int64))
        ps.append(prob)
        if i % 100 == 0 or i == len(test):
            print(f"  scored {i}/{len(test)}")
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    metrics = residue_metrics(y_true, y_prob, threshold=threshold)
    print("LoRA test metrics:", json.dumps(metrics, indent=2))

    meta = {
        "model_type": "lora", "model_id": args.model_id, "track": args.track,
        "threshold": threshold, "best_val_auprc": float(val["auprc"]), "win": args.win,
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    table = BenchmarkTable(dataset="UniProtSMB test")
    table.add(BenchmarkRow(
        method=f"LoRA ({args.model_id.split('/')[-1]}, {args.track})",
        auprc=metrics["auprc"], auroc=metrics["auroc"], f1=metrics["f1"],
        precision=metrics["precision"], recall=metrics["recall"], mcc=metrics["mcc"],
        source="ours", note=f"stopped at best val epoch; {args.track} track",
    ))
    for row in PUBLISHED:
        table.add(row)
    table.save_json(run / "benchmark.json")
    (run / "benchmark.md").write_text(table.to_markdown() + "\n", encoding="utf-8")
    print(f"\nWrote {run}/benchmark.md and meta.json")


if __name__ == "__main__":
    main()
