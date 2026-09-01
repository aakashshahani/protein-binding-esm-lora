from pbsite.data.splits import audit_leakage, group_kfold_indices, parse_cluster_tsv


def test_parse_cluster_tsv(tmp_path):
    tsv = tmp_path / "c.tsv"
    tsv.write_text("repA\trepA\nrepA\tmem1\nrepB\trepB\n", encoding="utf-8")
    mapping = parse_cluster_tsv(tsv)
    assert mapping["mem1"] == "repA"
    assert mapping["repB"] == "repB"


def test_parse_cluster_tsv_crlf_robust(tmp_path):
    # MMseqs2 fed a CRLF FASTA embeds \r into IDs; parser must strip it and not
    # mis-split on the carriage return (regression test).
    tsv = tmp_path / "c.tsv"
    tsv.write_bytes(b"repA\r\trepA\r\nrepA\r\tmem1\r\nrepB\r\trepB\r\n")
    mapping = parse_cluster_tsv(tsv)
    assert mapping == {"repA": "repA", "mem1": "repA", "repB": "repB"}


def test_write_fasta_uses_lf(tmp_path):
    from pbsite.data.clape import Record
    from pbsite.data.splits import write_fasta

    write_fasta([Record("X", "MKT", (0, 1, 0))], tmp_path / "a.fasta")
    assert b"\r" not in (tmp_path / "a.fasta").read_bytes()


def test_audit_leakage():
    clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c3"}
    # test id 'b' shares cluster c1 with train id 'a' -> leaked
    res = audit_leakage(["a"], ["b", "c"], clusters)
    assert res["n_test"] == 2
    assert res["n_leaked"] == 1


def test_group_kfold_disjoint_clusters():
    ids = [f"p{i}" for i in range(10)]
    clusters = {p: f"c{i//2}" for i, p in enumerate(ids)}  # 5 clusters of 2
    folds = group_kfold_indices(ids, clusters, n_folds=5)
    seen = set()
    for fold in folds:
        fold_clusters = {clusters[ids[i]] for i in fold}
        assert fold_clusters.isdisjoint(seen)
        seen |= fold_clusters
