import torch

from pbsite.models.bilstm import BiLSTMTagger
from pbsite.models.frozen_head import ResidueMLPHead
from pbsite.models.losses import masked_focal


def _batch(dim=16):
    x = torch.randn(2, 7, dim)
    y = (torch.rand(2, 7) > 0.7).float()
    m = torch.ones(2, 7)
    m[1, 5:] = 0  # second sample is length 5
    return x, y, m


def test_bilstm_forward_backward():
    model = BiLSTMTagger(input_dim=16, hidden_size=8, num_layers=1)
    x, y, m = _batch()
    logits = model(x, m.bool())
    assert logits.shape == (2, 7)
    loss = masked_focal(logits, y, m)
    loss.backward()
    assert loss.item() >= 0.0


def test_mlp_head_forward_backward():
    model = ResidueMLPHead(input_dim=16, hidden=8, num_layers=2)
    x, y, m = _batch()
    logits = model(x, m.bool())
    assert logits.shape == (2, 7)
    loss = masked_focal(logits, y, m)
    loss.backward()
    assert loss.item() >= 0.0
