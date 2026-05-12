from pathlib import Path

import pytest
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_NAMES = ["exp1_smoke", "exp1_mvp", "exp1_bounded", "exp1_full"]
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


def test_exp1_bounded_config_focuses_depth_and_normals() -> None:
    cfg = _compose_with_omegaconf("exp1_bounded")

    assert list(cfg.tasks.enabled) == [
        "surface_normal_aggregate",
        "relative_depth_regions",
    ]
    assert cfg.assets.max_objects == 50
    assert cfg.datasets.objaverse.max_download_gb <= 10.0
    assert str(cfg.paths.hf_cache_root).endswith("data/hf_cache")


def test_exp1_shapenet_source_defaults_to_accessible_core_zip_repo() -> None:
    cfg = _compose_with_omegaconf("exp1_smoke")
    sources = {str(source.name): source for source in cfg.assets.sources}

    assert sources["shapenet_hf"].repo_id == "ShapeNet/ShapeNetCore"
    assert str(cfg.paths.shapenet_hf_root).endswith("data/shapenet_hf/ShapeNetCore")
    assert sources["shapenet_hf"].download is False
    assert sources["shapenet_hf"].enabled is True
    assert list(sources["shapenet_hf"].categories)
    assert sources["shapenet_hf"].max_objects > 0


def test_exp1_default_sources_are_shapenetcore_and_objaverse() -> None:
    cfg = _compose_with_omegaconf("exp1_smoke")
    sources = {str(source.name): source for source in cfg.assets.sources}

    assert sources["modelnet40"].enabled is False
    assert sources["synthetic_primitives"].enabled is False
    assert sources["objaverse_manifest"].enabled is True
    assert sources["objaverse"].enabled is True
    assert str(cfg.paths.objaverse_root).endswith("data/objaverse")
    assert str(cfg.paths.objaverse_manifest).endswith("manifests/objaverse_assets.jsonl")
    assert cfg.datasets.objaverse.max_objects > 0
    assert cfg.datasets.objaverse.max_download_gb > 0


def test_exp1_full_probe_tasks_have_label_paths() -> None:
    cfg = _compose_with_omegaconf("exp1_full")

    for task in cfg.tasks.enabled:
        assert cfg.tasks.definitions[task].get("label_path") is not None


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_exp1_configs_do_not_require_local_absolute_paths(config_name: str) -> None:
    cfg = _compose_with_omegaconf(config_name)
    text = OmegaConf.to_yaml(cfg, resolve=False)

    assert "/Users/" not in text
    assert "\\Users\\" not in text
