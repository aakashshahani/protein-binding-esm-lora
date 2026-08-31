"""Load a trained embedding-based model and predict per-residue binding probs.

A saved model directory contains:
  - ``model.pt``     : state_dict
  - ``meta.json``    : {model_type, model_id, input_dim, add_physchem, threshold, ...}

ESM-2 embeddings are computed on demand and cached (so repeated queries for the
same sequence are cheap). Supports the ``frozen_head`` and ``bilstm`` models;
the LoRA model has its own serving path and is out of scope for this endpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, model_dir: str | Path, emb_cache: str | Path | None = None):
        import torch

        from ..features.esm import ESMEmbedder
        from ..features.physchem import physchem_features
        from ..models.bilstm import BiLSTMTagger
        from ..models.frozen_head import ResidueMLPHead

        self.torch = torch
        self._physchem = physchem_features
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.model_type = self.meta["model_type"]
        self.add_physchem = self.meta.get("add_physchem", False)
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedder = ESMEmbedder(model_id=self.meta["model_id"], device=self.device)
        input_dim = self.meta["input_dim"]
        if self.model_type == "bilstm":
            self.model = BiLSTMTagger(
                input_dim=input_dim,
                hidden_size=self.meta.get("hidden_size", 256),
                num_layers=self.meta.get("num_layers", 2),
            )
        elif self.model_type == "frozen_head":
            self.model = ResidueMLPHead(
                input_dim=input_dim,
                hidden=self.meta.get("hidden", 512),
                num_layers=self.meta.get("num_layers", 2),
            )
        else:
            raise ValueError(f"unsupported model_type for serving: {self.model_type}")

        state = torch.load(model_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.emb_cache = Path(emb_cache) if emb_cache else None
        if self.emb_cache:
            self.emb_cache.mkdir(parents=True, exist_ok=True)

    def _features(self, seq: str) -> np.ndarray:
        emb = None
        key = None
        if self.emb_cache:
            key = hashlib.sha1(f"{self.meta['model_id']}:{seq}".encode()).hexdigest()
            cached = self.emb_cache / f"{key}.npy"
            if cached.exists():
                emb = np.load(cached).astype(np.float32)
        if emb is None:
            emb = self.embedder.embed(seq).astype(np.float32)
            if self.emb_cache and key:
                np.save(self.emb_cache / f"{key}.npy", emb.astype(np.float16))
        if self.add_physchem:
            emb = np.concatenate([emb, self._physchem(seq)], axis=1)
        return emb

    def predict(self, seq: str, threshold: float | None = None) -> dict:
        torch = self.torch
        thr = self.threshold if threshold is None else float(threshold)
        feats = self._features(seq)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        mask = torch.ones(1, feats.shape[0], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logits = self.model(x, mask)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
        binding = [int(i) for i, p in enumerate(probs) if p >= thr]
        return {
            "probabilities": [round(float(p), 4) for p in probs],
            "binding_residues": binding,
            "threshold": thr,
            "model": f"{self.model_type}:{self.meta['model_id']}",
        }
