"""Redundancy reduction and leakage-safe splitting.

We run MMseqs2 `easy-cluster` (via Docker) at 30% sequence identity, then:
  1. audit how many test proteins share a cluster with any train protein
     (should be ~0 for an honest benchmark), and
  2. build GroupKFold CV folds keyed by cluster id so homologs never span folds.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .clape import Record


def write_fasta(records: list[Record], path: str | Path) -> None:
    lines = []
    for r in records:
        lines.append(f">{r.id}")
        lines.append(r.seq)
    # newline="\n": avoid Windows CRLF, which MMseqs2 would fold into sequence IDs
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def run_mmseqs_cluster(
    fasta_path: str | Path,
    out_dir: str | Path,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    docker_image: str = "ghcr.io/soedinglab/mmseqs2:latest",
    use_docker: bool | None = None,
) -> dict[str, str]:
    """Cluster sequences with MMseqs2; return {seq_id: cluster_rep_id}.

    Prefers a native `mmseqs` binary; falls back to Docker (mounts out_dir).
    Raises if neither is available.
    """
    fasta_path = Path(fasta_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "clust"

    native = shutil.which("mmseqs")
    if use_docker is None:
        use_docker = native is None

    if not use_docker and native:
        cmd = [
            native, "easy-cluster", str(fasta_path), str(prefix), str(out_dir / "tmp"),
            "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
        ]
    else:
        if not shutil.which("docker"):
            raise RuntimeError("Neither native mmseqs nor docker is available for clustering.")
        mount = out_dir
        # copy fasta into the mounted dir so it is visible in-container
        local_fasta = out_dir / "input.fasta"
        shutil.copy(fasta_path, local_fasta)
        cmd = [
            "docker", "run", "--rm", "-v", f"{mount}:/work", docker_image,
            "mmseqs", "easy-cluster", "/work/input.fasta", "/work/clust", "/work/tmp",
            "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
        ]

    subprocess.run(cmd, check=True)
    return parse_cluster_tsv(out_dir / "clust_cluster.tsv")


def parse_cluster_tsv(tsv_path: str | Path) -> dict[str, str]:
    """MMseqs2 writes <rep>\\t<member> lines; map member -> rep (cluster id)."""
    mapping: dict[str, str] = {}
    # Read raw bytes and drop all carriage returns before splitting, so IDs that
    # picked up a stray \r from a CRLF FASTA or a Windows/OneDrive copy still parse
    # (reading as text would let universal-newline translation mangle embedded \r).
    text = Path(tsv_path).read_bytes().decode("utf-8").replace("\r", "")
    for line in text.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        mapping[parts[1]] = parts[0]
    return mapping


def audit_leakage(
    train_ids: list[str], test_ids: list[str], clusters: dict[str, str]
) -> dict[str, int]:
    """Count test proteins whose cluster also contains a train protein."""
    train_clusters = {clusters.get(i, i) for i in train_ids}
    leaked = [i for i in test_ids if clusters.get(i, i) in train_clusters]
    return {"n_test": len(test_ids), "n_leaked": len(leaked)}


def group_kfold_indices(
    ids: list[str], clusters: dict[str, str], n_folds: int = 5, seed: int = 42
) -> list[list[int]]:
    """Return a list of n_folds index-lists; each fold's clusters are disjoint."""
    from sklearn.model_selection import GroupKFold

    groups = [clusters.get(i, i) for i in ids]
    gkf = GroupKFold(n_splits=n_folds)
    # GroupKFold is deterministic given group order; seed kept for API symmetry
    return [list(val_idx) for _, val_idx in gkf.split(ids, groups=groups)]
