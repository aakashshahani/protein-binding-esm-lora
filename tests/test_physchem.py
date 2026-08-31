import numpy as np

from pbsite.features.physchem import PHYSCHEM_DIM, physchem_features


def test_shape_and_dim():
    feats = physchem_features("ACDEFG")
    assert feats.shape == (6, PHYSCHEM_DIM)
    assert feats.dtype == np.float32


def test_charge_and_flags():
    feats = physchem_features("DKGP")
    # D negative, K positive charge
    assert feats[0, 1] == -1.0
    assert feats[1, 1] == 1.0
    # glycine / proline flags
    assert feats[2, 5] == 1.0
    assert feats[3, 6] == 1.0
