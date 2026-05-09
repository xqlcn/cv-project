#!/usr/bin/env python3
"""
Download ModelNet40 into ``data/modelnet40/`` (train / test folder layout).

Default URL (Princeton): http://modelnet.cs.princeton.edu/ModelNet40.zip

The official archive may redirect or require manual download if the server blocks
automated requests; this script prints clear instructions on failure.

Usage::

    python scripts/download_modelnet40.py
    python scripts/download_modelnet40.py --output-dir data/modelnet40 --no-unzip
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from normalize_modelnet40_layout import normalize_modelnet40_root


DEFAULT_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download(url: str, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n-> {dest_zip}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(dest_zip, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            n = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                if total:
                    print(f"\r  {n / total * 100:.1f}%", end="", flush=True)
            print()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("\nAutomatic download failed:", exc, file=sys.stderr)
        print(
            "\nManual steps:\n"
            "  1) Open https://modelnet.cs.princeton.edu/ (or the download page linked there)\n"
            "  2) Download **ModelNet40** (ModelNet40.zip)\n"
            f"  3) Unzip so you have: <root>/train/<category>/*.off and <root>/test/<category>/*.off\n"
            f"  4) Move or symlink that root to: {_project_root() / 'data' / 'modelnet40'}\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ModelNet40 to data/modelnet40/")
    parser.add_argument("--url", default=DEFAULT_URL, help="ZIP URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "data" / "modelnet40",
        help="Final dataset root (train/ and test/ live here)",
    )
    parser.add_argument("--zip-path", type=Path, default=None, help="Save ZIP here (default: output-dir/../ModelNet40.zip)")
    parser.add_argument("--no-unzip", action="store_true", help="Only download the archive")
    parser.add_argument("--keep-zip", action="store_true", help="Do not delete ZIP after unzip")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    zip_path = args.zip_path or (out.parent / "ModelNet40.zip")

    if (out / "train").is_dir() and any((out / "train").iterdir()):
        print(f"Already looks populated: {out / 'train'}. Skipping download.")
        return

    _download(args.url, zip_path)

    if args.no_unzip:
        print("Skipping unzip (--no-unzip).")
        return

    print(f"Extracting to {out} ...")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    normalize_modelnet40_root(out)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)

    print("Done. Expected layout:", out / "train" / "<category>" / "*.off")
    if not (out / "train").is_dir():
        print(
            "Warning: `train` folder not found. Your ZIP layout may differ; "
            "manually arrange into data/modelnet40/train/ and data/modelnet40/test/.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
