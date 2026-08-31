import pytest
from tests.conftest import FIXTURES

from pbsite.data.clape import dataset_stats, parse_clape_file, write_clape_file


def test_parse_mini():
    recs = parse_clape_file(FIXTURES / "mini.txt")
    assert len(recs) == 3
    assert recs[0].id == "P00001"
    assert recs[0].seq == "MKTAYIAKQR"
    assert recs[0].labels == (0, 0, 0, 1, 1, 0, 0, 0, 0, 0)
    assert recs[0].n_pos == 2


def test_roundtrip(tmp_path):
    recs = parse_clape_file(FIXTURES / "mini.txt")
    out = tmp_path / "rt.txt"
    write_clape_file(recs, out)
    assert parse_clape_file(out) == recs


def test_length_mismatch_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(">X\nMKT\n0000\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_clape_file(bad)


def test_non_binary_label_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(">X\nMKT\n0A0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_clape_file(bad)


def test_stats():
    recs = parse_clape_file(FIXTURES / "mini.txt")
    s = dataset_stats(recs)
    assert s["n_proteins"] == 3
    assert s["n_residues"] == 30
    assert s["n_binding_residues"] == 6
    assert abs(s["pos_fraction"] - 6 / 30) < 1e-9
