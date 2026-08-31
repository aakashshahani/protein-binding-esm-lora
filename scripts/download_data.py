"""Download the CLAPE-SMB / UniProtSMB splits and record the exact release.

Pins the CLAPE-SMB repo to a concrete commit SHA (so the split is reproducible),
fetches the train/valid/test files, and writes data/DATA_RELEASE.json with the
SHA, retrieval timestamp, and a SHA-256 checksum of every downloaded file.

No large BioLiP2 bulk download happens here (that is opt-in, --biolip).

Usage:
    python scripts/download_data.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbsite.data.clape import dataset_stats, parse_clape_file  # noqa: E402
from pbsite.utils import load_config  # noqa: E402

API = "https://api.github.com/repos/{repo}/commits/{ref}"
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"


def resolve_sha(repo: str, ref: str) -> str:
    r = requests.get(API.format(repo=repo, ref=ref), timeout=30)
    r.raise_for_status()
    return r.json()["sha"]


def download(repo: str, sha: str, path: str, dest: Path) -> str:
    url = RAW.format(repo=repo, sha=sha, path=path.replace(" ", "%20"))
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return hashlib.sha256(r.content).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["clape_smb"]
    repo, ref = c["repo"], c["ref"]
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "clape_smb"

    print(f"Resolving {repo}@{ref} ...")
    sha = resolve_sha(repo, ref)
    print(f"  commit {sha}")

    release = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "clape_smb": {"repo": repo, "commit": sha, "snapshot_date": c["snapshot_date"]},
        "files": {},
        "stats": {},
    }

    all_files = {**c["files"], **c.get("extra", {})}
    for name, path in all_files.items():
        dest = raw_dir / Path(path).name
        print(f"Downloading {name}: {path}")
        checksum = download(repo, sha, path, dest)
        release["files"][name] = {"path": str(dest), "sha256": checksum}
        # per-split stats for splits we can parse in the 3-line format
        try:
            recs = parse_clape_file(dest)
            release["stats"][name] = dataset_stats(recs)
        except Exception as exc:  # standard datasets may differ in format
            release["stats"][name] = {"parse_error": str(exc)}

    out = data_dir / "DATA_RELEASE.json"
    out.write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    for name in c["files"]:
        s = release["stats"].get(name, {})
        if "n_proteins" in s:
            print(
                f"  {name:6s}: {s['n_proteins']} proteins, "
                f"{s['n_residues']} residues, pos_frac={s['pos_fraction']:.4f}"
            )


if __name__ == "__main__":
    main()
