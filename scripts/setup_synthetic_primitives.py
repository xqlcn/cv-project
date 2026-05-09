#!/usr/bin/env python3
"""Generate cached synthetic primitive meshes (default controlled dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.synthetic_primitives import build_synthetic_primitive_records, default_synthetic_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None, help="Output root (default: data/synthetic_primitives)")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = args.root or default_synthetic_root()
    recs = build_synthetic_primitive_records(
        root=root,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )
    print(f"Wrote {len(recs)} meshes under {root.resolve()}")


if __name__ == "__main__":
    main()
