#!/usr/bin/env python3
"""Strict configuration and lineage checks for the OT2 Model V2 pipeline.

The model consumes canonical preprocessing artifacts; it never rebuilds labels
or fits normalization statistics.  This module deliberately contains no torch
dependency so a run can fail fast before allocating a GPU.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

MODEL_SCHEMA_VERSION = "ot2-model-v2-1"
PRIMARY_MODALITIES: Tuple[str, ...] = (
    "clinical_data",
    "cardiac_ecg",
    "environment",
    "retinal_flio",
    "retinal_oct",
    "retinal_octa",
    "retinal_photography",
    "wearable_activity_monitor",
    "wearable_blood_glucose",
)


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    """Match preprocessing_v2 without importing its numpy/pandas runtime."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _probability(value: Any, name: str, *, allow_zero: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not lower_ok or result > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bracket}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _path(value: Any, name: str, base: Path) -> Path:
    text = _nonempty_string(value, name)
    candidate = Path(text).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _optional_path(value: Any, name: str, base: Path) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return _path(value, name, base)


def _load_serialized(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML configuration requires PyYAML. Install "
                "requirements-model-v2.txt or use a JSON config."
            ) from exc
        value = yaml.safe_load(text)
    return _mapping(value, str(path))


@dataclass(frozen=True)
class DatasetV2Config:
    task: str
    master_csv: Path
    master_metadata_json: Path
    scaler_json: Path
    clinical_schema_json: Path
    environment_schema_json: Path
    wearable_schema_json: Path
    ecg_schema_json: Path
    artifact_root: Optional[Path]
    strict_lineage: bool
    allow_missing_modalities: bool
    retinal_max_assets: Mapping[str, int]
    deterministic_retinal_selection: bool


@dataclass(frozen=True)
class ModelV2Config:
    source_order: Tuple[str, ...]
    d_model: int
    dropout: float
    task_heads: Mapping[str, int]
    longitudinal_patch_size_slots: int
    ecg_patch_size_samples: int
    fusion_layers: int
    fusion_heads: int
    retinal_backbone: str
    retinal_backbone_dim: int
    retinal_weights: Optional[str]
    retinal_weights_path: Optional[Path]
    retinal_backbone_frozen: bool
    retinal_input_size: int
    retinal_input_normalization: str
    retinal_backbone_forward_batch_size: int


@dataclass(frozen=True)
class TrainingV2Config:
    batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    selection_metric: str
    early_stopping_patience: int
    reload_best_before_test: bool


@dataclass(frozen=True)
class ImportanceV2Config:
    methods: Tuple[str, ...]
    permutation_repeats: int
    bootstrap_samples: int
    allow_test_set_model_selection: bool


@dataclass(frozen=True)
class ModelV2RunConfig:
    schema_version: str
    seed: int
    output_dir: Path
    dataset: DatasetV2Config
    model: ModelV2Config
    training: TrainingV2Config
    importance: ImportanceV2Config
    raw: Mapping[str, Any]
    config_hash: str


def _parse_dataset(value: Any, base: Path) -> DatasetV2Config:
    cfg = _mapping(value, "dataset")
    retinal = _mapping(cfg.get("retinal", {}), "dataset.retinal")
    caps_raw = _mapping(retinal.get("max_assets", {}), "dataset.retinal.max_assets")
    retinal_names = ("photography", "oct", "octa", "flio")
    missing_caps = sorted(set(retinal_names) - set(caps_raw))
    extra_caps = sorted(set(caps_raw) - set(retinal_names))
    if missing_caps or extra_caps:
        raise ValueError(
            "dataset.retinal.max_assets must contain exactly photography, oct, "
            f"octa and flio; missing={missing_caps}, extra={extra_caps}"
        )
    caps = {
        name: _positive_int(caps_raw[name], f"dataset.retinal.max_assets.{name}")
        for name in retinal_names
    }
    return DatasetV2Config(
        task=_nonempty_string(cfg.get("task"), "dataset.task"),
        master_csv=_path(cfg.get("master_csv"), "dataset.master_csv", base),
        master_metadata_json=_path(
            cfg.get("master_metadata_json"), "dataset.master_metadata_json", base
        ),
        scaler_json=_path(cfg.get("scaler_json"), "dataset.scaler_json", base),
        clinical_schema_json=_path(
            cfg.get("clinical_schema_json"), "dataset.clinical_schema_json", base
        ),
        environment_schema_json=_path(
            cfg.get("environment_schema_json"),
            "dataset.environment_schema_json",
            base,
        ),
        wearable_schema_json=_path(
            cfg.get("wearable_schema_json"), "dataset.wearable_schema_json", base
        ),
        ecg_schema_json=_path(
            cfg.get("ecg_schema_json"), "dataset.ecg_schema_json", base
        ),
        artifact_root=_optional_path(
            cfg.get("artifact_root"), "dataset.artifact_root", base
        ),
        strict_lineage=bool(cfg.get("strict_lineage", True)),
        allow_missing_modalities=bool(cfg.get("allow_missing_modalities", True)),
        retinal_max_assets=caps,
        deterministic_retinal_selection=bool(
            retinal.get("deterministic_selection", True)
        ),
    )


def _parse_model(value: Any, task: str, base: Path) -> ModelV2Config:
    cfg = _mapping(value, "model")
    order = tuple(str(item).strip() for item in cfg.get("source_order", ()))
    if order != PRIMARY_MODALITIES:
        raise ValueError(
            "model.source_order must exactly match the nine canonical modality "
            f"groups: {list(PRIMARY_MODALITIES)}"
        )
    heads_raw = _mapping(cfg.get("task_heads", {}), "model.task_heads")
    heads = {
        _nonempty_string(name, "model.task_heads key"): _positive_int(
            output_dim, f"model.task_heads.{name}"
        )
        for name, output_dim in heads_raw.items()
    }
    if task not in heads:
        raise ValueError(f"dataset.task={task!r} is absent from model.task_heads")
    longitudinal = _mapping(cfg.get("longitudinal", {}), "model.longitudinal")
    ecg = _mapping(cfg.get("ecg", {}), "model.ecg")
    retinal = _mapping(cfg.get("retinal", {}), "model.retinal")
    fusion = _mapping(cfg.get("fusion", {}), "model.fusion")

    retinal_backbone = _nonempty_string(
        retinal.get("backbone", "small_cnn"), "model.retinal.backbone"
    ).lower()
    if retinal_backbone not in {"small_cnn", "resnet18"}:
        raise ValueError("model.retinal.backbone must be small_cnn or resnet18")
    weights_value = retinal.get("weights")
    retinal_weights = (
        None
        if weights_value is None
        else _nonempty_string(weights_value, "model.retinal.weights").upper()
    )
    if retinal_weights not in {None, "IMAGENET1K_V1"}:
        raise ValueError(
            "model.retinal.weights must be null or IMAGENET1K_V1; DEFAULT is "
            "not reproducible across torchvision versions"
        )
    retinal_weights_path = _optional_path(
        retinal.get("weights_path"), "model.retinal.weights_path", base
    )
    if retinal_weights is not None and retinal_weights_path is not None:
        raise ValueError(
            "set at most one of model.retinal.weights and weights_path"
        )
    if retinal_weights_path is not None and not retinal_weights_path.is_file():
        raise FileNotFoundError(
            f"model.retinal.weights_path does not exist: {retinal_weights_path}"
        )
    retinal_dim = _positive_int(
        retinal.get("backbone_dim", 512 if retinal_backbone == "resnet18" else 128),
        "model.retinal.backbone_dim",
    )
    retinal_normalization = _nonempty_string(
        retinal.get(
            "normalization",
            "imagenet1k_v1" if retinal_backbone == "resnet18" else "none",
        ),
        "model.retinal.normalization",
    ).lower()
    retinal_input_size = _positive_int(
        retinal.get("input_size", 224), "model.retinal.input_size"
    )
    retinal_frozen = _boolean(
        retinal.get("frozen", False), "model.retinal.frozen"
    )
    if retinal_backbone == "small_cnn":
        if retinal_weights is not None or retinal_weights_path is not None:
            raise ValueError("small_cnn does not accept pretrained weights")
        if retinal_normalization != "none":
            raise ValueError("small_cnn requires model.retinal.normalization=none")
    else:
        if retinal_dim != 512:
            raise ValueError("resnet18 requires model.retinal.backbone_dim=512")
        if retinal_input_size != 224:
            raise ValueError("resnet18 requires model.retinal.input_size=224")
        if retinal_normalization != "imagenet1k_v1":
            raise ValueError(
                "resnet18 requires model.retinal.normalization=imagenet1k_v1"
            )
    return ModelV2Config(
        source_order=order,
        d_model=_positive_int(cfg.get("d_model"), "model.d_model"),
        dropout=_probability(cfg.get("dropout", 0.1), "model.dropout"),
        task_heads=heads,
        longitudinal_patch_size_slots=_positive_int(
            longitudinal.get("patch_size_slots"),
            "model.longitudinal.patch_size_slots",
        ),
        ecg_patch_size_samples=_positive_int(
            ecg.get("patch_size_samples"), "model.ecg.patch_size_samples"
        ),
        fusion_layers=_positive_int(fusion.get("layers"), "model.fusion.layers"),
        fusion_heads=_positive_int(fusion.get("heads"), "model.fusion.heads"),
        retinal_backbone=retinal_backbone,
        retinal_backbone_dim=retinal_dim,
        retinal_weights=retinal_weights,
        retinal_weights_path=retinal_weights_path,
        retinal_backbone_frozen=retinal_frozen,
        retinal_input_size=retinal_input_size,
        retinal_input_normalization=retinal_normalization,
        retinal_backbone_forward_batch_size=_positive_int(
            retinal.get("backbone_forward_batch_size", 64),
            "model.retinal.backbone_forward_batch_size",
        ),
    )


def _parse_training(value: Any) -> TrainingV2Config:
    cfg = _mapping(value, "training")
    selection_metric = _nonempty_string(
        cfg.get("selection_metric", "auroc"), "training.selection_metric"
    ).lower()
    if selection_metric not in {"auroc", "auprc", "log_loss"}:
        raise ValueError("training.selection_metric must be auroc, auprc or log_loss")
    learning_rate = float(cfg.get("learning_rate", 5e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    result = TrainingV2Config(
        batch_size=_positive_int(cfg.get("batch_size", 8), "training.batch_size"),
        gradient_accumulation_steps=_positive_int(
            cfg.get("gradient_accumulation_steps", 1),
            "training.gradient_accumulation_steps",
        ),
        max_epochs=_positive_int(cfg.get("max_epochs", 30), "training.max_epochs"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        selection_metric=selection_metric,
        early_stopping_patience=_positive_int(
            cfg.get("early_stopping_patience", 7),
            "training.early_stopping_patience",
        ),
        reload_best_before_test=bool(cfg.get("reload_best_before_test", True)),
    )
    if not result.reload_best_before_test:
        raise ValueError("training.reload_best_before_test must be true")
    return result


def _parse_importance(value: Any) -> ImportanceV2Config:
    cfg = _mapping(value or {}, "importance")
    methods = tuple(str(item).strip() for item in cfg.get("methods", ()))
    allowed = {"ablation", "permutation", "matched_permutation"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"importance.methods contains unknown methods: {unknown}")
    result = ImportanceV2Config(
        methods=methods,
        permutation_repeats=_positive_int(
            cfg.get("permutation_repeats", 200),
            "importance.permutation_repeats",
        ),
        bootstrap_samples=_positive_int(
            cfg.get("bootstrap_samples", 2000), "importance.bootstrap_samples"
        ),
        allow_test_set_model_selection=bool(
            cfg.get("allow_test_set_model_selection", False)
        ),
    )
    if result.allow_test_set_model_selection:
        raise ValueError("test-set model selection is forbidden")
    return result


def load_model_v2_config(
    path: str | Path, *, validate_files: bool = False
) -> ModelV2RunConfig:
    """Read JSON/YAML, validate the contract, and optionally verify lineage."""
    config_path = Path(path).expanduser().resolve()
    raw = _load_serialized(config_path)
    schema_version = _nonempty_string(raw.get("schema_version"), "schema_version")
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version!r}; expected {MODEL_SCHEMA_VERSION!r}"
        )
    base = config_path.parent
    dataset = _parse_dataset(raw.get("dataset"), base)
    model = _parse_model(raw.get("model"), dataset.task, base)
    result = ModelV2RunConfig(
        schema_version=schema_version,
        seed=int(raw.get("seed", 42)),
        output_dir=_path(raw.get("output_dir"), "output_dir", base),
        dataset=dataset,
        model=model,
        training=_parse_training(raw.get("training", {})),
        importance=_parse_importance(raw.get("importance", {})),
        raw=raw,
        config_hash=canonical_json_hash(raw),
    )
    if result.model.d_model % result.model.fusion_heads:
        raise ValueError("model.d_model must be divisible by model.fusion.heads")
    if validate_files or result.dataset.strict_lineage:
        validate_model_v2_lineage(result)
    return result


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _declared_schema_hash(payload: Mapping[str, Any]) -> Optional[str]:
    for value in (
        payload.get("config_hash"),
        _mapping(payload.get("artifact_config", {}), "artifact_config").get(
            "config_hash"
        ),
        _mapping(payload.get("preprocessing_config", {}), "preprocessing_config").get(
            "config_hash"
        ),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _source_hash(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key in ("config_hash", "artifact_config_hash", "task_config_hash"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return None
    text = str(value or "").strip()
    return text or None


def validate_model_v2_lineage(config: ModelV2RunConfig) -> Dict[str, str]:
    """Fail on task, hash, feature-order, or train-fit lineage mismatches."""
    dataset = config.dataset
    required_paths: Sequence[Tuple[str, Path]] = (
        ("master CSV", dataset.master_csv),
        ("master metadata", dataset.master_metadata_json),
        ("scaler", dataset.scaler_json),
        ("clinical schema", dataset.clinical_schema_json),
        ("environment schema", dataset.environment_schema_json),
        ("wearable schema", dataset.wearable_schema_json),
        ("ECG schema", dataset.ecg_schema_json),
    )
    for name, path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    master = _read_json(dataset.master_metadata_json, "master metadata")
    scaler = _read_json(dataset.scaler_json, "train-only scaler")
    clinical = _read_json(dataset.clinical_schema_json, "clinical schema")
    environment = _read_json(dataset.environment_schema_json, "environment schema")
    wearable = _read_json(dataset.wearable_schema_json, "wearable schema")
    ecg = _read_json(dataset.ecg_schema_json, "ECG schema")

    for name, payload in (("master metadata", master), ("clinical schema", clinical)):
        declared_task = str(payload.get("task") or "").strip()
        if declared_task and declared_task != dataset.task:
            raise ValueError(
                f"{name} task mismatch: {declared_task!r} != {dataset.task!r}"
            )
    if str(clinical.get("fit_split") or "").strip().lower() != "train":
        raise ValueError("clinical schema was not fit on the official train split")
    if scaler.get("fit_split") != "recommended_split=train":
        raise ValueError("scaler was not fit on recommended_split=train")

    scaler_without_hash = dict(scaler)
    declared_scaler_hash = str(
        scaler_without_hash.pop("scaler_config_hash", "") or ""
    ).strip()
    if not declared_scaler_hash:
        raise ValueError("scaler is missing scaler_config_hash")
    computed_scaler_hash = canonical_json_hash(scaler_without_hash)
    if declared_scaler_hash != computed_scaler_hash:
        raise ValueError(
            "scaler_config_hash mismatch; the scaler file was modified or truncated"
        )

    master_preprocessing = _mapping(
        master.get("preprocessing_config", {}), "master.preprocessing_config"
    )
    if str(master_preprocessing.get("task") or "").strip() != dataset.task:
        raise ValueError("master preprocessing_config task does not match dataset.task")
    declared_master_hash = str(master.get("config_hash") or "").strip()
    if not declared_master_hash or canonical_json_hash(master_preprocessing) != declared_master_hash:
        raise ValueError("master metadata config_hash does not match preprocessing_config")

    declared_clinical_hash = str(master.get("clinical_schema_hash") or "").strip()
    computed_clinical_hash = canonical_json_hash(clinical)
    if declared_clinical_hash and declared_clinical_hash != computed_clinical_hash:
        raise ValueError("master metadata is bound to a different clinical schema")

    master_task_config_hash = str(
        master_preprocessing.get("task_config_hash") or ""
    ).strip()
    clinical_task_config_hash = str(clinical.get("task_config_hash") or "").strip()
    if (
        master_task_config_hash
        and clinical_task_config_hash
        and master_task_config_hash != clinical_task_config_hash
    ):
        raise ValueError("label/clinical task_config_hash mismatch")

    clinical_fit_ids_hash = str(clinical.get("fit_person_ids_hash") or "").strip()
    scaler_fit_population = _mapping(
        scaler.get("fit_population", {}), "scaler.fit_population"
    )
    scaler_fit_ids_hash = str(
        scaler_fit_population.get("fit_person_ids_hash") or ""
    ).strip()
    if (
        clinical_fit_ids_hash
        and scaler_fit_ids_hash
        and clinical_fit_ids_hash != scaler_fit_ids_hash
    ):
        raise ValueError("clinical/scaler official train participant hash mismatch")

    schema_feature_names = [str(name) for name in clinical.get("feature_names", [])]
    scalers = _mapping(scaler.get("scalers", {}), "scaler.scalers")
    clinical_scaler = _mapping(scalers.get("clinical", {}), "scaler.clinical")
    scaler_feature_names = [
        str(name) for name in clinical_scaler.get("feature_order", [])
    ]
    if not schema_feature_names or scaler_feature_names != schema_feature_names:
        raise ValueError(
            "clinical feature order differs between schema and train-only scaler"
        )

    master_hashes = _mapping(
        master_preprocessing.get("source_index_config_hashes", {}),
        "master source_index_config_hashes",
    )
    scaler_hashes = _mapping(
        scaler.get("source_config_hashes", {}), "scaler.source_config_hashes"
    )
    clinical_config_hash = _declared_schema_hash(clinical)
    clinical_scaler_hash = _source_hash(scaler_hashes.get("clinical"))
    if (
        clinical_config_hash
        and clinical_scaler_hash
        and clinical_config_hash != clinical_scaler_hash
    ):
        raise ValueError("clinical config hash differs between schema and scaler")
    schema_hashes = {
        "environment": _declared_schema_hash(environment),
        "wearable": _declared_schema_hash(wearable),
        "ecg": _declared_schema_hash(ecg),
    }
    for source, schema_hash in schema_hashes.items():
        master_hash = _source_hash(master_hashes.get(source))
        scaler_hash = _source_hash(scaler_hashes.get(source))
        known = [value for value in (schema_hash, master_hash, scaler_hash) if value]
        if len(set(known)) > 1:
            raise ValueError(
                f"{source} config hash mismatch across master/schema/scaler: {known}"
            )
        if dataset.strict_lineage and len(known) < 3:
            raise ValueError(
                f"strict lineage requires {source} hashes in master, schema and scaler"
            )

    return {
        "task": dataset.task,
        "master_config_hash": declared_master_hash,
        "clinical_schema_hash": computed_clinical_hash,
        "scaler_config_hash": declared_scaler_hash,
        **{
            f"{name}_config_hash": str(value)
            for name, value in schema_hashes.items()
            if value
        },
    }


__all__ = [
    "MODEL_SCHEMA_VERSION",
    "PRIMARY_MODALITIES",
    "canonical_json_hash",
    "DatasetV2Config",
    "ImportanceV2Config",
    "ModelV2Config",
    "ModelV2RunConfig",
    "TrainingV2Config",
    "load_model_v2_config",
    "validate_model_v2_lineage",
]
