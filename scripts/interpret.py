"""Per-residue interpretability: plot predicted binding probability along a protein.

For a chosen test protein, loads its cached ESM-2 embedding + a trained embedding
model (bilstm/frozen_head), computes the per-residue binding probability track,
and renders a figure highlighting predicted binding pockets (prob >= threshold)
with the ground-truth binding residues marked for comparison.

Usage:
    python scripts/interpret.py --run outputs/bilstm_esm2_t33_650M_UR50D \
        --protein-id P0A6F5 --out outputs/interpret
    # or pick the test protein with the most binding residues:
    python scripts/interpret.py --run outputs/bilstm_esm2_t33_650M_UR50D --auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.utils import get_device  # noqa: E402


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from pbsite.features.physchem import physchem_features
    from pbsite.models.bilstm import BiLSTMTagger
    from pbsite.models.frozen_head import ResidueMLPHead

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--protein-id", default=None)
    ap.add_argument("--auto", action="store_true", help="pick test protein with most binding sites")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--out", default="outputs/interpret")
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    device = get_device()
    emb_dir = Path(args.emb_dir) / meta["model_id"].split("/")[-1]

    records = parse_clape_file(Path(args.data_dir) / "clape_smb" / "test_UniProtSMB.txt")
    by_id = {r.id: r for r in records}
    if args.auto or not args.protein_id:
        rec = max(records, key=lambda r: r.n_pos)
    else:
        rec = by_id[args.protein_id]

    emb = np.load(emb_dir / f"{rec.id}.npy").astype(np.float32)
    if meta.get("add_physchem"):
        emb = np.concatenate([emb, physchem_features(rec.seq)], axis=1)

    if meta["model_type"] == "bilstm":
        model = BiLSTMTagger(input_dim=meta["input_dim"], hidden_size=meta.get("hidden_size", 256),
                             num_layers=meta.get("num_layers", 2))
    else:
        model = ResidueMLPHead(input_dim=meta["input_dim"], hidden=meta.get("hidden", 512),
                               num_layers=meta.get("num_layers", 2))
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    model.to(device).eval()

    x = torch.from_numpy(emb).unsqueeze(0).to(device)
    mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=device)
    with torch.inference_mode():
        probs = torch.sigmoid(model(x, mask))[0].float().cpu().numpy()

    thr = float(meta["threshold"])
    true = np.array(rec.labels)
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (true == 1)).sum())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(min(16, 4 + len(rec.seq) / 60), 3.2))
    xs = np.arange(len(rec.seq))
    ax.fill_between(xs, 0, probs, where=probs >= thr, color="crimson", alpha=0.35,
                    label=f"predicted pocket (p>={thr:.2f})", step="mid")
    ax.plot(xs, probs, color="black", lw=0.8, label="P(binding)")
    ax.axhline(thr, color="grey", ls="--", lw=0.7)
    true_idx = np.where(true == 1)[0]
    ax.scatter(true_idx, np.full_like(true_idx, -0.04, dtype=float), marker="|",
               color="royalblue", s=80, label="true binding residue")
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlabel("residue index")
    ax.set_ylabel("P(binding)")
    ax.set_title(f"{rec.id}  |  {meta['model_type']}  |  "
                 f"{tp}/{int(true.sum())} true sites recovered")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    png = out / f"{rec.id}_{meta['model_type']}.png"
    fig.savefig(png, dpi=130)
    (out / f"{rec.id}_probs.json").write_text(
        json.dumps({"id": rec.id, "sequence": rec.seq, "threshold": thr,
                    "probabilities": [round(float(p), 4) for p in probs],
                    "true_binding": true_idx.tolist(),
                    "predicted_binding": np.where(pred == 1)[0].tolist()}, indent=2),
        encoding="utf-8")
    print(f"protein {rec.id}: {len(rec.seq)} residues, {int(true.sum())} true sites, "
          f"{tp} recovered at thr={thr:.2f}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
