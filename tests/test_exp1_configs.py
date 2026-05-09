from pathlib import Path

import pytest
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_NAMES = ["exp1_smoke", "exp1_mvp", "exp1_full"]
TEXTURE_CONDITIONS = ["photorealistic", "flat", "random_noise"]
REQUIRED_TOP_LEVEL_KEYS = (
    "paths",
    "experiment",
    "assets",
    "render",
    "textures",
    "camera_grid",
    "lighting_grid",
    "scale_grid",
    "splits",
    "models",
    "tasks",
    "probe",
)


def _compose_with_hydra(name: str):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except SystemExit as exc:
        pytest.skip(f"Hydra import failed in this environment: {exc}")
    except Exception as exc:
        pytest.skip(f"Hydra is not importable in this environment: {exc}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    return cfg


def _compose_with_omegaconf(name: str):
    OmegaConf.register_new_resolver("now", lambda fmt: "now", replace=True)
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "exp1" / "paths.yaml"),
        OmegaConf.load(CONFIG_DIR / "exp1" / "render.yaml"),
        OmegaConf.load(CONFIG_DIR / "exp1" / "tasks.yaml"),
        OmegaConf.load(CONFIG_DIR / f"{name}.yaml"),
    )
    OmegaConf.resolve(cfg)
    return cfg


def _assert_exp1_config_contract(cfg) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)

    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in resolved

    assert list(cfg.textures.conditions) == TEXTURE_CONDITIONS
    assert cfg.splits.object_disjoint is True
    assert abs(sum(float(v) for v in cfg.splits.fractions.values()) - 1.0) < 1e-8
    assert cfg.features.use_random_augmentations is False
    assert cfg.probe.head == "linear"


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_exp1_configs_load_with_hydra(config_name: str) -> None:
    _assert_exp1_config_contract(_compose_with_hydra(config_name))


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_exp1_configs_resolve_with_omegaconf(config_name: str) -> None:
    _assert_exp1_config_contract(_compose_with_omegaconf(config_name))


def test_exp1_smoke_config_stays_tiny() -> None:
    cfg = _compose_with_omegaconf("exp1_smoke")

    assert cfg.assets.max_objects == 3
    assert list(cfg.models.enabled) == ["clip_vit_b16"]
    assert list(cfg.models.layers) == ["final"]
    assert list(cfg.tasks.enabled) == ["surface_normal_aggregate"]
    assert max(int(x) for x in cfg.render.resolution) <= 128


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_exp1_configs_do_not_require_local_absolute_paths(config_name: str) -> None:
    cfg = _compose_with_omegaconf(config_name)
    text = OmegaConf.to_yaml(cfg, resolve=False)

    assert "/Users/" not in text
    assert "\\Users\\" not in text
