"""Torch Dataset / collation over cached per-residue embeddings + labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .clape import Record


class ResidueEmbeddingDataset(Dataset):
    """Serves (embedding[L, D], label[L], mask[L]) per protein.

    Embeddings are loaded lazily from ``emb_dir/{id}.npy`` (float16 on disk,
    upcast to float32 here) so we never hold the whole corpus in RAM.
    """

    def __init__(self, records: list[Record], emb_dir: str | Path, add_physchem: bool = False):
        self.records = records
        self.emb_dir = Path(emb_dir)
        self.add_physchem = add_physchem
        if add_physchem:
            from ..features.physchem import physchem_features

            self._physchem = physchem_features

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        emb = np.load(self.emb_dir / f"{r.id}.npy").astype(np.float32)
        if emb.shape[0] != len(r.labels):
            raise ValueError(
                f"{r.id}: cached emb len {emb.shape[0]} != labels {len(r.labels)}"
            )
        if self.add_physchem:
            emb = np.concatenate([emb, self._physchem(r.seq)], axis=1)
        x = torch.from_numpy(emb)
        y = torch.tensor(r.labels, dtype=torch.float32)
        mask = torch.ones(len(r.labels), dtype=torch.bool)
        return x, y, mask


def pad_collate(batch):
    """Pad a batch of variable-length proteins; return (X, Y, mask)."""
    lengths = [x.shape[0] for x, _, _ in batch]
    max_len = max(lengths)
    dim = batch[0][0].shape[1]
    bsz = len(batch)

    X = torch.zeros(bsz, max_len, dim, dtype=torch.float32)
    Y = torch.zeros(bsz, max_len, dtype=torch.float32)
    M = torch.zeros(bsz, max_len, dtype=torch.bool)
    for i, (x, y, _) in enumerate(batch):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = True
    return X, Y, M


def pos_weight_from_records(records: list[Record]) -> float:
    """neg/pos ratio used for weighted BCE on the heavily imbalanced labels."""
    pos = sum(r.n_pos for r in records)
    neg = sum(len(r.seq) for r in records) - pos
    return float(neg) / float(max(pos, 1))
