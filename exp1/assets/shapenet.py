"""ShapeNet asset ingestion helpers for Experiment 1.

The Hugging Face ShapeNet repositories are file repositories rather than
row-oriented datasets, so preprocessing treats them as snapshots and then scans
for mesh files. This targets the original ShapeNetCore ZIP repository
(`ShapeNet/ShapeNetCore`) by default, while still supporting local GLB mirrors
when explicitly provided.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from exp1.assets.discover import ALLOWED_MESH_EXTENSIONS


SHAPENETCORE_ZIP_HF_REPO_ID = "ShapeNet/ShapeNetCore"
SHAPENETCORE_GLB_HF_REPO_ID = "ShapeNet/shapenetcore-glb"
DEFAULT_SHAPENET_HF_REPO_ID = SHAPENETCORE_ZIP_HF_REPO_ID

SHAPENETCORE_SYNSET_TO_CATEGORY = {
    "02691156": "airplane",
    "02747177": "ashcan",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}

SHAPENETCORE_CATEGORY_TO_SYNSET = {
    category: synset for synset, category in SHAPENETCORE_SYNSET_TO_CATEGORY.items()
}

SPLIT_ALIASES = {"valid": "val", "validation": "val"}
VALID_SPLIT_PARTS = {"train", "val", "valid", "validation", "test"}


def _clean_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _safe_component(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "unknown"


def _as_optional_list(
    values: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",")]
    else:
        raw = [str(value).strip() for value in values]
    out = [value for value in raw if value]
    return out or None


def _category_filter_values(
    categories: Optional[Union[str, Sequence[str]]],
) -> Optional[set[str]]:
    values = _as_optional_list(categories)
    if values is None:
        return None

    filters: set[str] = set()
    for value in values:
        clean = _clean_token(value)
        if not clean:
            continue
        filters.add(clean)
        if clean in SHAPENETCORE_SYNSET_TO_CATEGORY:
            filters.add(_clean_token(SHAPENETCORE_SYNSET_TO_CATEGORY[clean]))
        if clean in SHAPENETCORE_CATEGORY_TO_SYNSET:
            filters.add(SHAPENETCORE_CATEGORY_TO_SYNSET[clean])
    return filters or None


def _synset_for_category(value: str) -> Optional[str]:
    clean = _clean_token(value)
    if clean in SHAPENETCORE_SYNSET_TO_CATEGORY:
        return clean
    return SHAPENETCORE_CATEGORY_TO_SYNSET.get(clean)


def shapenet_allow_patterns(
    categories: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    """Build Hugging Face allow patterns for selected ShapeNet categories."""
    values = _as_optional_list(categories)
    if values is None:
        return None

    patterns: list[str] = []
    for value in values:
        clean = _clean_token(value)
        if not clean:
            continue
        synset = _synset_for_category(clean)
        names = {clean}
        if synset is not None:
            names.add(synset)
            names.add(_clean_token(SHAPENETCORE_SYNSET_TO_CATEGORY[synset]))

        for name in sorted(names):
            patterns.extend([f"{name}/**", f"{name}.zip"])
    return sorted(set(patterns))


def _normalize_hf_token(token: Any) -> Any:
    if token is None:
        return None
    if isinstance(token, bool):
        return token
    text = str(token).strip()
    lower = text.lower()
    if lower in {"true", "1", "yes", "y"}:
        return True
    if lower in {"", "auto", "none", "null"}:
        return None
    if lower in {"false", "0", "no", "n"}:
        return False
    if lower == "env":
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None
        )
    return text


def _download_snapshot(
    *,
    repo_id: str,
    local_dir: Optional[Union[str, Path]],
    revision: Optional[str],
    token: Any,
    allow_patterns: Optional[Sequence[str]],
    ignore_patterns: Optional[Sequence[str]],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ShapeNet Hugging Face ingestion requires huggingface_hub. "
            "Install requirements.txt or run `pip install huggingface_hub`."
        ) from exc

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "token": _normalize_hf_token(token),
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
        "ignore_patterns": list(ignore_patterns) if ignore_patterns else None,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(Path(local_dir).expanduser())

    return Path(snapshot_download(**kwargs)).expanduser().resolve()


def _glb_category_name(value: str) -> str:
    clean = _clean_token(value)
    if clean in SHAPENETCORE_SYNSET_TO_CATEGORY:
        return SHAPENETCORE_SYNSET_TO_CATEGORY[clean]
    return clean


def _download_selected_glbs(
    *,
    repo_id: str,
    local_dir: Union[str, Path],
    revision: Optional[str],
    token: Any,
    categories: Optional[Union[str, Sequence[str]]],
    max_objects_per_category: int,
) -> Path:
    """Download a bounded per-category GLB subset from the ShapeNet GLB mirror."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "ShapeNet Hugging Face ingestion requires huggingface_hub. "
            "Install requirements.txt or run `pip install huggingface_hub`."
        ) from exc

    category_values = _as_optional_list(categories)
    if not category_values:
        raise ValueError(
            "categories are required for bounded ShapeNet GLB downloads"
        )

    token_value = _normalize_hf_token(token)
    local_root = Path(local_dir).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    selected_paths: list[str] = []

    for value in category_values:
        category = _glb_category_name(value)
        items = api.list_repo_tree(
            repo_id,
            repo_type="dataset",
            path_in_repo=category,
            recursive=False,
            revision=revision,
            token=token_value,
        )
        paths = sorted(
            str(getattr(item, "path", ""))
            for item in items
            if str(getattr(item, "path", "")).lower().endswith(".glb")
        )
        if len(paths) < int(max_objects_per_category):
            raise ValueError(
                f"ShapeNet GLB category {category!r} has {len(paths)} files, "
                f"required {int(max_objects_per_category)}"
            )
        selected_paths.extend(paths[: int(max_objects_per_category)])

    for path in selected_paths:
        hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
            revision=revision,
            token=token_value,
            local_dir=str(local_root),
        )

    return local_root


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            target = (output_dir / member.filename).resolve()
            if output_root not in target.parents and target != output_root:
                raise ValueError(
                    f"Refusing to extract path outside output root: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())


def extract_shapenet_archives(
    snapshot_root: Union[str, Path],
    output_root: Union[str, Path],
    *,
    categories: Optional[Union[str, Sequence[str]]] = None,
    overwrite: bool = False,
    max_archives: Optional[int] = None,
) -> Path:
    """Extract ShapeNetCore category ZIPs into a reusable local directory."""
    snapshot_root = Path(snapshot_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    filters = _category_filter_values(categories)

    archives = sorted(snapshot_root.rglob("*.zip"))
    selected: list[Path] = []
    for archive_path in archives:
        synset = archive_path.stem
        category = SHAPENETCORE_SYNSET_TO_CATEGORY.get(synset, synset)
        tokens = {_clean_token(synset), _clean_token(category)}
        if filters is not None and not tokens.intersection(filters):
            continue
        selected.append(archive_path)

    if max_archives is not None:
        selected = selected[: int(max_archives)]

    for archive_path in selected:
        destination = output_root / archive_path.stem
        has_contents = destination.exists() and any(destination.rglob("*"))
        if overwrite or not has_contents:
            _safe_extract_zip(archive_path, destination)

    return output_root


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return path.parts


def _split_from_parts(parts: Sequence[str], *, default_split: str) -> str:
    for part in parts:
        clean = _clean_token(part)
        if clean in VALID_SPLIT_PARTS:
            return SPLIT_ALIASES.get(clean, clean)
    return default_split


def _infer_shapenet_identity(
    mesh_path: Path,
    root: Path,
    *,
    default_split: str,
) -> Dict[str, str]:
    parts = _relative_parts(mesh_path, root)
    split = _split_from_parts(parts, default_split=default_split)
    synset_id = ""
    category = ""
    model_id = mesh_path.stem

    for idx, part in enumerate(parts):
        clean = _clean_token(part)
        if clean in SHAPENETCORE_SYNSET_TO_CATEGORY:
            synset_id = clean
            category = SHAPENETCORE_SYNSET_TO_CATEGORY[clean]
            for candidate in parts[idx + 1 :]:
                candidate_clean = _clean_token(candidate)
                if candidate_clean and candidate_clean not in {
                    synset_id,
                    "models",
                    "images",
                    "screenshots",
                    "textures",
                }:
                    model_id = Path(candidate).stem
                    break
            break

    if not category:
        non_split_parts = [
            part for part in parts[:-1] if _clean_token(part) not in VALID_SPLIT_PARTS
        ]
        category = non_split_parts[0] if non_split_parts else mesh_path.parent.name
        synset_id = _synset_for_category(category) or ""

    return {
        "split": split,
        "category": _clean_token(category) or "unknown",
        "shapenet_synset_id": synset_id,
        "shapenet_model_id": _safe_component(model_id),
    }


def _has_obj_material(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip().lower()
                if stripped.startswith("mtllib "):
                    return True
    except OSError:
        return False
    return any(path.with_suffix(ext).is_file() for ext in (".mtl", ".MTL"))


def infer_photorealistic_material_available(path: Union[str, Path]) -> bool:
    """Conservatively infer whether a mesh may carry original material data."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf", ".fbx"}:
        return True
    if suffix == ".obj":
        return _has_obj_material(path)
    return False


def _object_id(category: str, model_id: str, *, used: set[str]) -> str:
    base = f"{_safe_component(category)}_{_safe_component(model_id)}"
    object_id = base
    counter = 2
    while object_id in used:
        object_id = f"{base}_{counter}"
        counter += 1
    used.add(object_id)
    return object_id


def _iter_mesh_paths(
    roots: Iterable[Path],
    *,
    allowed_extensions: Sequence[str],
) -> Iterable[tuple[Path, Path]]:
    allowed = {ext.lower() for ext in allowed_extensions}
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for mesh_path in sorted(root.rglob("*")):
            if not mesh_path.is_file() or mesh_path.name.startswith("."):
                continue
            if mesh_path.suffix.lower() not in allowed:
                continue
            resolved = mesh_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield root, resolved


def discover_huggingface_shapenet_assets(
    *,
    repo_id: str = DEFAULT_SHAPENET_HF_REPO_ID,
    local_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
    revision: Optional[str] = None,
    token: Any = True,
    categories: Optional[Union[str, Sequence[str]]] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    extract_archives: bool = True,
    extracted_dir: Optional[Union[str, Path]] = None,
    overwrite_extract: bool = False,
    source_dataset: str = "shapenet",
    allowed_extensions: Sequence[str] = ALLOWED_MESH_EXTENSIONS,
    default_split: str = "train",
    max_objects: Optional[int] = None,
    max_objects_per_category: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Discover ShapeNet meshes from a Hugging Face snapshot.

    If ``download`` is true, this uses ``huggingface_hub.snapshot_download`` and
    expects the user to have accepted the gated dataset license and logged in
    with the Hugging Face CLI or ``HF_TOKEN``. Tests and offline workflows can
    pass ``download=False`` with ``local_dir`` pointing at an existing snapshot.
    """
    if allow_patterns is None:
        allow_patterns = shapenet_allow_patterns(categories)

    snapshot_root: Path
    if download:
        if (
            repo_id == SHAPENETCORE_GLB_HF_REPO_ID
            and max_objects_per_category is not None
        ):
            if local_dir is None:
                raise ValueError(
                    "local_dir is required for bounded ShapeNet GLB downloads"
                )
            snapshot_root = _download_selected_glbs(
                repo_id=repo_id,
                local_dir=local_dir,
                revision=revision,
                token=token,
                categories=categories,
                max_objects_per_category=int(max_objects_per_category),
            )
        else:
            snapshot_root = _download_snapshot(
                repo_id=repo_id,
                local_dir=local_dir,
                revision=revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
    else:
        if local_dir is None:
            raise ValueError("local_dir is required when download=False")
        snapshot_root = Path(local_dir).expanduser().resolve()
        if not snapshot_root.is_dir():
            raise FileNotFoundError(
                f"ShapeNet snapshot directory not found: {snapshot_root}"
            )

    scan_roots = [snapshot_root]
    if extract_archives and any(snapshot_root.rglob("*.zip")):
        archive_output = (
            Path(extracted_dir).expanduser()
            if extracted_dir is not None
            else snapshot_root / "_extracted"
        )
        scan_roots.append(
            extract_shapenet_archives(
                snapshot_root,
                archive_output,
                categories=categories,
                overwrite=overwrite_extract,
            )
        )

    filters = _category_filter_values(categories)
    rows: list[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for root, mesh_path in _iter_mesh_paths(
        scan_roots,
        allowed_extensions=allowed_extensions,
    ):
        identity = _infer_shapenet_identity(
            mesh_path,
            root,
            default_split=default_split,
        )
        tokens = {
            _clean_token(identity["category"]),
            _clean_token(identity["shapenet_synset_id"]),
        }
        if filters is not None and not tokens.intersection(filters):
            continue

        rows.append(
            {
                "object_id": _object_id(
                    identity["category"],
                    identity["shapenet_model_id"],
                    used=used_ids,
                ),
                "source_dataset": source_dataset,
                "category": identity["category"],
                "split": identity["split"],
                "raw_mesh_path": str(mesh_path),
                "normalized_mesh_path": "",
                "asset_status": "discovered",
                "asset_error_message": "",
                "has_photorealistic_material": infer_photorealistic_material_available(
                    mesh_path
                ),
                "hf_repo_id": repo_id,
                "hf_revision": str(revision or ""),
                "shapenet_synset_id": identity["shapenet_synset_id"],
                "shapenet_model_id": identity["shapenet_model_id"],
            }
        )

        if max_objects is not None and len(rows) >= int(max_objects):
            break

    return rows
