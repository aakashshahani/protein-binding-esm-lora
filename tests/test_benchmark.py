from pbsite.eval.benchmark import PUBLISHED, BenchmarkRow, BenchmarkTable


def test_published_row_labeled():
    assert any(r.source == "published" for r in PUBLISHED)
    clape = next(r for r in PUBLISHED if r.method == "CLAPE-SMB")
    assert clape.mcc == 0.699
    assert "not reproduced" in clape.note or "paper" in clape.note


def test_table_markdown_contains_rows():
    t = BenchmarkTable(dataset="UniProtSMB test")
    t.add(BenchmarkRow(method="bilstm (esm2)", auprc=0.5, source="ours"))
    for r in PUBLISHED:
        t.add(r)
    md = t.to_markdown()
    assert "bilstm (esm2)" in md
    assert "CLAPE-SMB" in md
    assert "published" in md
