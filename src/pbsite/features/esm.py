"""ESM-2 per-residue embedding extraction with disk caching.

Design for a 4 GB GPU:
  - fp16 weights, ``torch.no_grad`` / ``inference_mode``,
  - batch size 1,
  - sequences longer than ``max_len`` are split into overlapping windows and
    the per-residue embeddings are stitched back (overlap regions averaged),
  - each protein cached as ``emb_dir/{id}.npy`` in float16.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ESMEmbedder:
    def __init__(
        self,
        model_id: str = "facebook/esm2_t33_650M_UR50D",
        device: str | None = None,
        fp16: bool = True,
        max_len: int = 1022,
        overlap: int = 64,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.max_len = max_len
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.fp16 else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def _embed_window(self, seq: str) -> np.ndarray:
        """Embed a single window (<= max_len residues) -> (L, D) float32."""
        torch = self.torch
        enc = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc).last_hidden_state[0]  # (L+2, D) incl. BOS/EOS
        # drop BOS (index 0) and EOS (last) to align 1:1 with residues
        residues = out[1:-1]
        return residues.float().cpu().numpy()

    def embed(self, seq: str) -> np.ndarray:
        """Return (len(seq), D) float32 embeddings, windowing long sequences."""
        if len(seq) <= self.max_len:
            return self._embed_window(seq)

        step = self.max_len - self.overlap
        acc = np.zeros((len(seq), self.dim), dtype=np.float32)
        cnt = np.zeros((len(seq), 1), dtype=np.float32)
        for start in range(0, len(seq), step):
            end = min(start + self.max_len, len(seq))
            win = self._embed_window(seq[start:end])
            acc[start:end] += win
            cnt[start:end] += 1.0
            if end == len(seq):
                break
        return acc / np.clip(cnt, 1.0, None)

    def embed_to_cache(self, seq_id: str, seq: str, emb_dir: str | Path) -> Path:
        emb_dir = Path(emb_dir)
        emb_dir.mkdir(parents=True, exist_ok=True)
        out = emb_dir / f"{seq_id}.npy"
        if out.exists():
            return out
        emb = self.embed(seq).astype(np.float16)  # halve disk footprint
        np.save(out, emb)
        return out
