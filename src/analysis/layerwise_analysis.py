"""
Train one linear probe per intermediate layer listed in `features.layers`.

Launches `python -m src.training.train_probe` as a subprocess so each run gets a
fresh Hydra working directory and isolated `output_dir`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="clip_probe")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.paths.project_root).resolve()
    layers: List[int] = list(cfg.features.layers) if cfg.features.layers is not None else []
    base_out = Path(cfg.paths.output_dir)
    if not base_out.is_absolute():
        base_out = project_root / base_out

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["CV_PROJECT_ROOT"] = str(project_root)

    for layer in layers:
        layer_out = base_out.parent / f"{base_out.name}_layer{layer}"
        cmd = [
            sys.executable,
            "-m",
            "src.training.train_probe",
            f"probe.target_layer={layer}",
            f"paths.output_dir={layer_out}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(project_root), env=env, check=True)


if __name__ == "__main__":
    main()
