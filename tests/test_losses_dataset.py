import numpy as np
import torch
from tests.conftest import FIXTURES

from pbsite.data.clape import parse_clape_file
from pbsite.data.dataset import ResidueEmbeddingDataset, pad_collate, pos_weight_from_records
from pbsite.models.losses import masked_bce, masked_focal


def test_masked_losses_ignore_padding():
    logits = torch.tensor([[2.0, -2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    mask_full = torch.tensor([[1.0, 1.0, 1.0]])
    mask_part = torch.tensor([[1.0, 1.0, 0.0]])
    # masking off the 3rd position must change the loss
    assert masked_bce(logits, targets, mask_full) != masked_bce(logits, targets, mask_part)
    assert masked_focal(logits, targets, mask_full) >= 0.0


def test_pos_weight():
    recs = parse_clape_file(FIXTURES / "mini.txt")
    pw = pos_weight_from_records(recs)
    # 6 positives / 24 negatives = 4.0
    assert abs(pw - 4.0) < 1e-6


def test_dataset_and_collate(tmp_path):
    recs = parse_clape_file(FIXTURES / "mini.txt")
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    for r in recs:
        np.save(emb_dir / f"{r.id}.npy", np.random.rand(len(r.seq), 8).astype(np.float16))
    ds = ResidueEmbeddingDataset(recs, emb_dir=emb_dir, add_physchem=True)
    x, y, m = ds[0]
    assert x.shape == (10, 8 + 7)  # emb dim 8 + 7 physchem
    X, Y, M = pad_collate([ds[0], ds[1], ds[2]])
    assert X.shape[0] == 3 and X.shape[2] == 15
    assert M.dtype == torch.bool
