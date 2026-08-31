"""Extract and cache ESM-2 per-residue embeddings for every split.

Writes embeddings/{model_tag}/{id}.npy (float16). Idempotent: existing files are
skipped, so extraction can be resumed after an interruption (important on a 4 GB
GPU where this is the slow step).

Usage:
    python scripts/extract_embeddings.py --config configs/bilstm.yaml \
        --splits train valid test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import parse_clape_file  # noqa: E402
from pbsite.features.esm import ESMEmbedder  # noqa: E402
from pbsite.utils import get_device, load_config  # noqa: E402

FILES = {
    "train": "train_UniProtSMB.txt",
    "valid": "valid_UniProtSMB.txt",
    "test": "test_UniProtSMB.txt",
}


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bilstm.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--model-id", default=None, help="override embeddings.model_id")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = args.model_id or cfg["embeddings"]["model_id"]
    raw = Path(args.data_dir) / "clape_smb"
    emb_dir = Path(args.emb_dir) / model_tag(model_id)

    print(f"Device: {get_device()}  |  model: {model_id}")
    embedder = ESMEmbedder(model_id=model_id)
    print(f"Embedding dim: {embedder.dim}")

    for split in args.splits:
        records = parse_clape_file(raw / FILES[split])
        print(f"[{split}] {len(records)} proteins -> {emb_dir}")
        t0 = time.time()
        for i, r in enumerate(records, 1):
            embedder.embed_to_cache(r.id, r.seq, emb_dir)
            if i % 25 == 0 or i == len(records):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(records)}  ({rate:.1f} prot/s)")
    print("Done.")


if __name__ == "__main__":
    main()
