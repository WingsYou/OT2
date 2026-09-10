#!/usr/bin/env python3
"""Supervised training CLI for the OT2 Model V2 binary tasks.

This is the composition boundary for the strict V2 configuration, lazy V2
dataset, nine-source model, and validation-only training engine.  It does not
rebuild preprocessing artifacts, refit scalers, or inspect test outcomes while
selecting a model or decision threshold.

Example::

    python ot2/train_task2_v2.py \
        --config ot2/configs/task2_v2_moca.yaml \
        --device cuda --workers 4 --bf16

Use ``--dry-run`` to validate lineage, construct all three split-specific
datasets/loaders, build the dimension-bound model, and execute one train-batch
forward pass without optimizing or iterating validation/test loaders.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ot2.task2_v2_config import (
        MODEL_SCHEMA_VERSION,
        PRIMARY_MODALITIES,
        ModelV2RunConfig,
        load_model_v2_config,
        validate_model_v2_lineage,
    )
    from ot2.task2_v2_dataset import (
        OT2V2Dataset,
        PRIMARY_SOURCE_ORDER,
        SOURCE_FLAG_BY_GROUP,
        task2_v2_collate,
    )
    from ot2.task2_v2_engine import (
        ENGINE_FORMAT_VERSION,
        BinaryTrainingConfig,
        CheckpointProvenance,
        FitResult,
        fit_evaluate_binary,
        recursive_to_device,
        set_deterministic_seed,
    )
    from ot2.task2_v2_model import (
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )
except ModuleNotFoundError:  # Support direct execution from the ot2 directory.
    from task2_v2_config import (  # type: ignore
        MODEL_SCHEMA_VERSION,
        PRIMARY_MODALITIES,
        ModelV2RunConfig,
        load_model_v2_config,
        validate_model_v2_lineage,
    )
    from task2_v2_dataset import (  # type: ignore
        OT2V2Dataset,
        PRIMARY_SOURCE_ORDER,
        SOURCE_FLAG_BY_GROUP,
        task2_v2_collate,
    )
    from task2_v2_engine import (  # type: ignore
        ENGINE_FORMAT_VERSION,
        BinaryTrainingConfig,
        CheckpointProvenance,
        FitResult,
        fit_evaluate_binary,
        recursive_to_device,
        set_deterministic_seed,
    )
    from task2_v2_model import (  # type: ignore
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )


SUPPORTED_TASKS = frozenset(
    {"moca_binary", "cvd_broad_binary", "cvd_strict_binary"}
)
SCHEDULER_NAME = "cosine_with_linear_warmup"
PRETRAIN_CHECKPOINT_FORMAT_VERSION = "ot2-task2-v2-pretrain-cli-1"

_MODEL_KEYS = frozenset(
    {
        "source_order",
        "d_model",
        "dropout",
        "task_heads",
        "clinical",
        "longitudinal",
        "ecg",
        "retinal",
        "fusion",
    }
)
_MODEL_SECTION_KEYS = {
    "clinical": frozenset({"hidden_dims"}),
    "longitudinal": frozenset(
        {"patch_size_slots", "layers", "heads", "ff_dim"}
    ),
    "ecg": frozenset(
        {
            "patch_size_samples",
            "layers",
            "heads",
            "ff_dim",
            "combine_numeric_features",
        }
    ),
    "retinal": frozenset(
        {
            "backbone",
            "backbone_dim",
            "weights",
            "weights_path",
            "frozen",
            "input_size",
            "normalization",
            "backbone_forward_batch_size",
            "shared_backbone",
            "view_pool",
            "asset_pool",
        }
    ),
    "fusion": frozenset(
        {"layers", "heads", "ff_dim", "use_learned_missing_tokens"}
    ),
}
_DATASET_KEYS = frozenset(
    {
        "task",
        "master_csv",
        "master_metadata_json",
        "scaler_json",
        "clinical_schema_json",
        "environment_schema_json",
        "wearable_schema_json",
        "ecg_schema_json",
        "artifact_root",
        "strict_lineage",
        "allow_missing_modalities",
        "retinal",
    }
)
_TRAINING_KEYS = frozenset(
    {
        "batch_size",
        "gradient_accumulation_steps",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "selection_metric",
        "early_stopping_patience",
        "reload_best_before_test",
        "num_workers",
        "workers",
        "device",
        "bf16",
        "use_bf16_amp",
        "grad_clip_norm",
        "use_pos_weight",
        "selection_min_delta",
        "threshold_method",
        "fixed_threshold",
        "deterministic",
        "optimizer",
        "scheduler",
        "warmup_ratio",
        "warmup_steps",
        "initialization_checkpoint",
        "disabled_groups",
        "evaluation_phase",
    }
)


@dataclass(frozen=True)
class DatasetBundle:
    train: OT2V2Dataset
    validation: OT2V2Dataset
    test: OT2V2Dataset
    feature_dimensions: Mapping[str, int]


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    workers: int
    pin_memory: bool


@dataclass(frozen=True)
class CosineWarmupSpec:
    name: str
    interval: str
    warmup_steps: int
    warmup_ratio: float
    total_optimizer_steps: int
    optimizer_steps_per_epoch: int


@dataclass
class PreparedRun:
    config_path: Path
    config: ModelV2RunConfig
    device: torch.device
    use_bf16_amp: bool
    datasets: DatasetBundle
    loaders: LoaderBundle
    model_config: Task2V2ModelConfig
    model: Task2V2Model
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.LambdaLR
    scheduler_spec: CosineWarmupSpec
    scheduler_hook: Any
    engine_config: BinaryTrainingConfig
    provenance: CheckpointProvenance
    output_dir: Path
    initialization_audit: Mapping[str, Any]

    def remove_scheduler_hook(self) -> None:
        if self.scheduler_hook is not None:
            self.scheduler_hook.remove()
            self.scheduler_hook = None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a YAML boolean")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: frozenset[str], name: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{name} contains unsupported keys: {unknown}")


def _json_safe(value: Any) -> Any:
    """Convert provenance to values accepted by both json and torch.save."""

    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    return payload


def _resolve_optional_path(value: Any, *, base: Path, label: str) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolve_disabled_groups(
    config: ModelV2RunConfig,
    override: Optional[Sequence[str]],
) -> tuple[str, ...]:
    training = _mapping(config.raw["training"], "training")
    raw = training.get("disabled_groups", ()) if override is None else override
    if raw is None:
        raw = ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("training.disabled_groups/--disabled-groups must be a sequence")
    requested = [str(value).strip() for value in raw]
    if any(not value for value in requested):
        raise ValueError("disabled_groups cannot contain empty names")
    if len(requested) != len(set(requested)):
        raise ValueError("disabled_groups cannot contain duplicates")
    unknown = sorted(set(requested) - set(PRIMARY_MODALITY_GROUPS))
    if unknown:
        raise ValueError(f"disabled_groups contains unknown modalities: {unknown}")
    disabled = tuple(
        group for group in PRIMARY_MODALITY_GROUPS if group in set(requested)
    )
    if len(disabled) == len(PRIMARY_MODALITY_GROUPS):
        raise ValueError("at least one primary modality must remain enabled")
    return disabled


def load_pretrained_student_checkpoint(
    model: Task2V2Model,
    checkpoint_path: str | Path,
    *,
    expected_task: str,
    expected_lineage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Strictly initialize the downstream backbone from label-free pretraining.

    Task heads are deliberately not loaded: they receive no objective during
    pretraining.  Every non-head tensor must otherwise exist with exactly the
    same name and shape, which prevents partial shape-mismatch loading from
    silently leaving random encoders frozen or unevaluated.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pretraining checkpoint does not exist: {path}")
    payload = _safe_torch_load(path)
    if payload.get("format_version") != PRETRAIN_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported pretraining checkpoint format_version")
    if payload.get("checkpoint_kind") != "fixed_budget_pretrained_student":
        raise ValueError("checkpoint is not a fixed-budget pretrained student")
    if payload.get("outcomes_read") is not False:
        raise ValueError("pretraining checkpoint does not prove outcomes_read=false")
    if payload.get("task_agnostic_objectives") is not True:
        raise ValueError("pretraining checkpoint lacks task-agnostic objectives")
    if str(payload.get("clinical_view_task") or "") != str(expected_task):
        raise ValueError("pretraining clinical view does not match downstream task")

    expected_base_hash = str(expected_lineage.get("base_config_hash") or "").strip()
    if not expected_base_hash or payload.get("base_config_hash") != expected_base_hash:
        raise ValueError(
            "pretraining checkpoint base config hash differs from the current run"
        )
    expected_sources = expected_lineage.get("source_files_sha256")
    observed_sources = payload.get("source_files")
    if not isinstance(expected_sources, Mapping) or not isinstance(
        observed_sources, Mapping
    ):
        raise ValueError("pretraining checkpoint lacks source-file lineage")
    for name, expected_hash in expected_sources.items():
        observed = observed_sources.get(name)
        if (
            not isinstance(observed, Mapping)
            or str(observed.get("sha256") or "") != str(expected_hash)
        ):
            raise ValueError(
                f"pretraining/current source file hash mismatch: {name}"
            )
    cohort_audit = payload.get("cohort_audit")
    if not isinstance(cohort_audit, Mapping):
        raise ValueError("pretraining checkpoint lacks cohort lineage")
    expected_train_hash = str(
        expected_lineage.get("train_person_ids_hash") or ""
    ).strip()
    if (
        not expected_train_hash
        or str(cohort_audit.get("train_person_ids_hash") or "")
        != expected_train_hash
    ):
        raise ValueError(
            "pretraining train cohort differs from the scaler-fit population"
        )
    clinical_audit = cohort_audit.get("clinical_features_csv")
    expected_clinical_hash = str(
        expected_lineage.get("clinical_features_sha256") or ""
    ).strip()
    if (
        not expected_clinical_hash
        or not isinstance(clinical_audit, Mapping)
        or str(clinical_audit.get("sha256") or "") != expected_clinical_hash
    ):
        raise ValueError(
            "pretraining/current clinical wide CSV hash mismatch"
        )

    observed_config = _json_safe(payload.get("resolved_model_config"))
    expected_config = _json_safe(model.config)
    if observed_config != expected_config:
        raise ValueError(
            "pretraining/downstream resolved model configs differ; refusing partial load"
        )
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("pretraining checkpoint is missing model_state")
    current = model.state_dict()
    expected_backbone = {key for key in current if not key.startswith("heads.")}
    observed_backbone = {str(key) for key in raw_state if not str(key).startswith("heads.")}
    missing = sorted(expected_backbone - observed_backbone)
    unexpected = sorted(observed_backbone - expected_backbone)
    if missing or unexpected:
        raise ValueError(
            "pretraining backbone key mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    selected: Dict[str, torch.Tensor] = {}
    for key in sorted(expected_backbone):
        value = raw_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"pretraining model_state[{key!r}] is not a tensor")
        if tuple(value.shape) != tuple(current[key].shape):
            raise ValueError(
                f"pretraining tensor shape mismatch for {key}: "
                f"{tuple(value.shape)} != {tuple(current[key].shape)}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"pretraining tensor {key} contains non-finite values")
        selected[key] = value
    incompatibility = model.load_state_dict(selected, strict=False)
    expected_missing_heads = sorted(key for key in current if key.startswith("heads."))
    if sorted(incompatibility.missing_keys) != expected_missing_heads:
        raise RuntimeError("unexpected missing keys during pretraining initialization")
    if incompatibility.unexpected_keys:
        raise RuntimeError("unexpected keys during pretraining initialization")
    return {
        "kind": "corrected_label_free_pretrained_student_backbone",
        "path": str(path),
        "sha256": _sha256_file(path),
        "format_version": payload.get("format_version"),
        "pretraining_epoch": int(payload.get("epoch", -1)),
        "clinical_view_task": expected_task,
        "outcomes_read": False,
        "selection_policy": payload.get("selection_policy"),
        "lineage_verified": True,
        "base_config_hash": expected_base_hash,
        "train_person_ids_hash": expected_train_hash,
        "clinical_features_sha256": expected_clinical_hash,
        "loaded_backbone_tensor_count": len(selected),
        "excluded_random_task_head_keys": expected_missing_heads,
    }


def _person_ids_hash(dataset: OT2V2Dataset) -> str:
    digest = hashlib.sha256()
    for person_id in sorted(str(value) for value in dataset.index["person_id"]):
        digest.update(person_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def pretraining_lineage_expectations(
    config_path: Path,
    config: ModelV2RunConfig,
    dataset: OT2V2Dataset,
) -> Dict[str, Any]:
    """Resolve the exact data semantics a pretraining checkpoint must match."""

    fit_population = dataset.scaler_payload.get("fit_population")
    if not isinstance(fit_population, Mapping):
        raise ValueError("scaler lacks fit_population pretraining lineage")
    train_hash = str(fit_population.get("fit_person_ids_hash") or "").strip()
    if not train_hash:
        raise ValueError("scaler lacks fit_population.fit_person_ids_hash")
    source_paths = {
        "base_supervised_config": config_path,
        "scaler": config.dataset.scaler_json,
        "clinical_schema": config.dataset.clinical_schema_json,
        "environment_schema": config.dataset.environment_schema_json,
        "wearable_schema": config.dataset.wearable_schema_json,
        "ecg_schema": config.dataset.ecg_schema_json,
    }
    return {
        "base_config_hash": config.config_hash,
        "source_files_sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
        "clinical_features_sha256": _sha256_file(
            dataset.clinical_features_path
        ),
        "train_person_ids_hash": train_hash,
    }


def load_supervised_config(path: str | Path) -> tuple[Path, ModelV2RunConfig]:
    """Strictly load a YAML run config and validate every lineage file."""

    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("the supervised V2 CLI requires a .yaml or .yml config")
    config = load_model_v2_config(config_path, validate_files=True)
    # The three study tasks above are shipped presets, not a closed registry.
    # Any future audited binary label/master/clinical-schema bundle can add a
    # one-logit head without changing the backbone or this CLI.

    raw_dataset = _mapping(config.raw.get("dataset"), "dataset")
    raw_model = _mapping(config.raw.get("model"), "model")
    raw_training = _mapping(config.raw.get("training"), "training")
    _reject_unknown_keys(raw_dataset, _DATASET_KEYS, "dataset")
    _reject_unknown_keys(raw_model, _MODEL_KEYS, "model")
    _reject_unknown_keys(raw_training, _TRAINING_KEYS, "training")
    retinal_dataset = _mapping(raw_dataset.get("retinal"), "dataset.retinal")
    _reject_unknown_keys(
        retinal_dataset,
        frozenset({"deterministic_selection", "max_assets"}),
        "dataset.retinal",
    )
    for key, default in (
        ("strict_lineage", True),
        ("allow_missing_modalities", True),
    ):
        _strict_bool(raw_dataset.get(key, default), f"dataset.{key}")
    _strict_bool(
        retinal_dataset.get("deterministic_selection", True),
        "dataset.retinal.deterministic_selection",
    )
    reload_best = _strict_bool(
        raw_training.get("reload_best_before_test", True),
        "training.reload_best_before_test",
    )
    if not reload_best:
        raise ValueError("training.reload_best_before_test must be true")
    if "num_workers" in raw_training and "workers" in raw_training:
        raise ValueError("set only one of training.num_workers and training.workers")
    for section, allowed in _MODEL_SECTION_KEYS.items():
        nested = _mapping(raw_model.get(section, {}), f"model.{section}")
        _reject_unknown_keys(nested, allowed, f"model.{section}")
    return config_path, config


def resolve_device(requested: str) -> torch.device:
    """Resolve only CPU/CUDA devices, never silently downgrade explicit CUDA."""

    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if not value.startswith("cuda"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested, but torch.cuda.is_available() is false"
        )
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise ValueError(f"invalid CUDA device {requested!r}") from exc
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device.index} is not visible; "
            f"device_count={torch.cuda.device_count()}"
        )
    return device


def _retinal_caps(config: ModelV2RunConfig) -> Dict[str, int]:
    # Config uses protocol names; Dataset uses the nine scientific group names.
    return {
        "retinal_photography": int(config.dataset.retinal_max_assets["photography"]),
        "retinal_oct": int(config.dataset.retinal_max_assets["oct"]),
        "retinal_octa": int(config.dataset.retinal_max_assets["octa"]),
        "retinal_flio": int(config.dataset.retinal_max_assets["flio"]),
    }


def build_datasets(
    config: ModelV2RunConfig,
    *,
    loaded_source_groups: Optional[Sequence[str]] = None,
) -> DatasetBundle:
    """Construct isolated official train/val/test datasets with one contract."""

    if not config.dataset.deterministic_retinal_selection:
        raise ValueError(
            "dataset.retinal.deterministic_selection=false is unsupported; "
            "the V2 Dataset implements deterministic audited selection only"
        )
    if tuple(PRIMARY_MODALITIES) != tuple(PRIMARY_SOURCE_ORDER) or tuple(
        PRIMARY_MODALITIES
    ) != tuple(PRIMARY_MODALITY_GROUPS):
        raise RuntimeError("config, Dataset, and model disagree on nine-source order")

    common = {
        "master_csv": config.dataset.master_csv,
        "scaler_json": config.dataset.scaler_json,
        "task": config.dataset.task,
        "artifact_root": config.dataset.artifact_root,
        "master_metadata_json": config.dataset.master_metadata_json,
        "environment_schema_json": config.dataset.environment_schema_json,
        "max_retinal_assets_per_group": _retinal_caps(config),
        "require_y": True,
        "validate_scaler_hash": True,
        "loaded_source_groups": loaded_source_groups,
    }
    train = OT2V2Dataset(split="train", **common)
    validation = OT2V2Dataset(split="val", **common)
    test = OT2V2Dataset(split="test", **common)

    dimensions = dict(train.model_feature_dimensions)
    for split_name, dataset in (("val", validation), ("test", test)):
        if dict(dataset.model_feature_dimensions) != dimensions:
            raise ValueError(
                f"{split_name} feature dimensions differ from the training Dataset"
            )
        if dataset.environment_feature_names != train.environment_feature_names:
            raise ValueError(
                f"{split_name} environment model-view columns differ from train"
            )
        if dataset.clinical_features_path != train.clinical_features_path:
            raise ValueError(
                f"{split_name} resolves a different clinical wide CSV from train"
            )

    split_sets = {
        "train": set(train.index["person_id"].astype(str)),
        "val": set(validation.index["person_id"].astype(str)),
        "test": set(test.index["person_id"].astype(str)),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left].intersection(split_sets[right])
        if overlap:
            raise ValueError(
                f"official {left}/{right} splits overlap: {sorted(overlap)[:10]}"
            )

    if not config.dataset.allow_missing_modalities:
        flags = sorted(set(SOURCE_FLAG_BY_GROUP.values()))
        for split_name, dataset in (
            ("train", train),
            ("val", validation),
            ("test", test),
        ):
            missing = (~dataset.index[flags].astype(bool)).any(axis=1)
            if bool(missing.any()):
                raise ValueError(
                    "dataset.allow_missing_modalities=false, but "
                    f"split={split_name} contains {int(missing.sum())} incomplete rows"
                )
    return DatasetBundle(train, validation, test, dimensions)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    datasets: DatasetBundle,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> LoaderBundle:
    """Build reproducible split-specific loaders without test shuffling."""

    batch_size = _positive_int(batch_size, "training.batch_size")
    workers = _nonnegative_int(workers, "workers")
    pin_memory = device.type == "cuda"

    def make(dataset: OT2V2Dataset, *, shuffle: bool, offset: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + offset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=task2_v2_collate,
            pin_memory=pin_memory,
            persistent_workers=workers > 0,
            worker_init_fn=_seed_worker if workers > 0 else None,
            generator=generator,
            drop_last=False,
        )

    return LoaderBundle(
        train=make(datasets.train, shuffle=True, offset=0),
        validation=make(datasets.validation, shuffle=False, offset=1),
        test=make(datasets.test, shuffle=False, offset=2),
        workers=workers,
        pin_memory=pin_memory,
    )


def _optional_positive_int(
    mapping: Mapping[str, Any], key: str, default: int, qualified_name: str
) -> int:
    return _positive_int(mapping.get(key, default), qualified_name)


def build_model_config(
    config: ModelV2RunConfig, feature_dimensions: Mapping[str, int]
) -> Task2V2ModelConfig:
    """Bind YAML architecture fields to dimensions derived by the Dataset."""

    expected_dimensions = {
        "clinical_dim",
        "environment_dim",
        "activity_dim",
        "cgm_dim",
        "ecg_waveform_dim",
        "ecg_numeric_dim",
    }
    if set(feature_dimensions) != expected_dimensions:
        raise ValueError(
            "Dataset model_feature_dimensions must contain exactly "
            f"{sorted(expected_dimensions)}; got {sorted(feature_dimensions)}"
        )
    dimensions = {
        name: _positive_int(feature_dimensions[name], f"Dataset {name}")
        for name in sorted(expected_dimensions)
    }
    if int(config.model.task_heads[config.dataset.task]) != 1:
        raise ValueError(
            f"binary task {config.dataset.task!r} must configure exactly one logit"
        )
    non_binary_heads = {
        name: width
        for name, width in config.model.task_heads.items()
        if int(width) != 1
    }
    if non_binary_heads:
        raise ValueError(
            "Task2V2Model currently exposes binary heads only; unsupported "
            f"task_heads={non_binary_heads}"
        )
    raw = _mapping(config.raw["model"], "model")
    clinical = _mapping(raw.get("clinical", {}), "model.clinical")
    longitudinal = _mapping(raw.get("longitudinal", {}), "model.longitudinal")
    ecg = _mapping(raw.get("ecg", {}), "model.ecg")
    retinal = _mapping(raw.get("retinal", {}), "model.retinal")
    fusion = _mapping(raw.get("fusion", {}), "model.fusion")

    temporal_layers = _optional_positive_int(
        longitudinal, "layers", 2, "model.longitudinal.layers"
    )
    ecg_layers = _optional_positive_int(
        ecg, "layers", temporal_layers, "model.ecg.layers"
    )
    if ecg_layers != temporal_layers:
        raise ValueError(
            "Task2V2Model shares temporal layer count across longitudinal and ECG; "
            f"got {temporal_layers} and {ecg_layers}"
        )

    fusion_heads = int(config.model.fusion_heads)
    longitudinal_heads = _optional_positive_int(
        longitudinal, "heads", fusion_heads, "model.longitudinal.heads"
    )
    ecg_heads = _optional_positive_int(ecg, "heads", fusion_heads, "model.ecg.heads")
    if len({fusion_heads, longitudinal_heads, ecg_heads}) != 1:
        raise ValueError(
            "Task2V2Model uses one num_heads value for longitudinal, ECG, and "
            f"fusion; got {longitudinal_heads}, {ecg_heads}, {fusion_heads}"
        )

    d_model = int(config.model.d_model)
    expected_ff = d_model * 2
    ff_dims = {
        "longitudinal": _optional_positive_int(
            longitudinal, "ff_dim", expected_ff, "model.longitudinal.ff_dim"
        ),
        "ecg": _optional_positive_int(ecg, "ff_dim", expected_ff, "model.ecg.ff_dim"),
        "fusion": _optional_positive_int(
            fusion, "ff_dim", expected_ff, "model.fusion.ff_dim"
        ),
    }
    if len(set(ff_dims.values())) != 1:
        raise ValueError(
            "Task2V2Model uses one FFN multiplier across encoders/fusion; "
            f"got {ff_dims}"
        )
    common_ff = next(iter(ff_dims.values()))
    if common_ff % d_model:
        raise ValueError("model ff_dim must be an integer multiple of model.d_model")

    implied_clinical_hidden = max(
        d_model * 2, min(512, dimensions["clinical_dim"])
    )
    if "hidden_dims" in clinical:
        hidden = clinical["hidden_dims"]
        if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
            raise ValueError("model.clinical.hidden_dims must be a sequence")
        observed_hidden = [_positive_int(value, "model.clinical.hidden_dims") for value in hidden]
        expected_hidden = [implied_clinical_hidden, d_model]
        if observed_hidden != expected_hidden:
            raise ValueError(
                "model.clinical.hidden_dims does not match the implemented "
                f"MaskAwareVectorEncoder: expected {expected_hidden}, got {observed_hidden}"
            )

    for key in ("shared_backbone",):
        if key in retinal and not _strict_bool(retinal[key], f"model.retinal.{key}"):
            raise ValueError("Task2V2Model requires a shared retinal backbone")
    for key in ("view_pool", "asset_pool"):
        if str(retinal.get(key, "masked_attention")).strip().lower() != "masked_attention":
            raise ValueError(f"Task2V2Model currently requires retinal {key}=masked_attention")
    if "combine_numeric_features" in ecg and not _strict_bool(
        ecg["combine_numeric_features"], "model.ecg.combine_numeric_features"
    ):
        raise ValueError("Task2V2Model always combines ECG waveform and numeric features")
    if "use_learned_missing_tokens" in fusion and not _strict_bool(
        fusion["use_learned_missing_tokens"],
        "model.fusion.use_learned_missing_tokens",
    ):
        raise ValueError("Task2V2Model requires learned missing-source tokens")

    return Task2V2ModelConfig(
        **dimensions,
        d_model=d_model,
        temporal_layers=temporal_layers,
        fusion_layers=int(config.model.fusion_layers),
        num_heads=fusion_heads,
        ffn_multiplier=common_ff // d_model,
        dropout=float(config.model.dropout),
        environment_patch_size=int(config.model.longitudinal_patch_size_slots),
        activity_patch_size=int(config.model.longitudinal_patch_size_slots),
        cgm_patch_size=int(config.model.longitudinal_patch_size_slots),
        ecg_patch_size=int(config.model.ecg_patch_size_samples),
        retinal_backbone=config.model.retinal_backbone,
        retinal_backbone_dim=int(config.model.retinal_backbone_dim),
        retinal_weights=config.model.retinal_weights,
        retinal_weights_path=(
            None
            if config.model.retinal_weights_path is None
            else str(config.model.retinal_weights_path)
        ),
        retinal_backbone_frozen=bool(config.model.retinal_backbone_frozen),
        retinal_input_size=int(config.model.retinal_input_size),
        retinal_input_normalization=config.model.retinal_input_normalization,
        retinal_backbone_forward_batch_size=int(
            config.model.retinal_backbone_forward_batch_size
        ),
        task_names=tuple(config.model.task_heads),
    )


def _scheduler_parameters(config: ModelV2RunConfig) -> tuple[float, Optional[int]]:
    training = _mapping(config.raw["training"], "training")
    raw_scheduler = training.get("scheduler")
    scheduler: Mapping[str, Any]
    if raw_scheduler is None:
        scheduler = {}
    else:
        scheduler = _mapping(raw_scheduler, "training.scheduler")
        _reject_unknown_keys(
            scheduler,
            frozenset({"name", "warmup_ratio", "warmup_steps", "interval"}),
            "training.scheduler",
        )
        name = str(scheduler.get("name", SCHEDULER_NAME)).strip().lower()
        if name not in {SCHEDULER_NAME, "cosine", "cosine_warmup"}:
            raise ValueError(
                "training.scheduler.name must be cosine_with_linear_warmup"
            )
        interval = str(scheduler.get("interval", "optimizer_step")).strip().lower()
        if interval not in {"optimizer_step", "step"}:
            raise ValueError("cosine warmup scheduler interval must be optimizer_step")
    if "warmup_ratio" in training and "warmup_ratio" in scheduler:
        raise ValueError("warmup_ratio cannot be set both flat and in scheduler")
    if "warmup_steps" in training and "warmup_steps" in scheduler:
        raise ValueError("warmup_steps cannot be set both flat and in scheduler")
    ratio = _finite_float(
        scheduler.get("warmup_ratio", training.get("warmup_ratio", 0.1)),
        "training.scheduler.warmup_ratio",
    )
    if not 0.0 <= ratio < 1.0:
        raise ValueError("training.scheduler.warmup_ratio must be in [0, 1)")
    raw_steps = scheduler.get("warmup_steps", training.get("warmup_steps"))
    steps = (
        None
        if raw_steps is None
        else _nonnegative_int(raw_steps, "training.scheduler.warmup_steps")
    )
    return ratio, steps


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: ModelV2RunConfig,
    *,
    train_loader_batches: int,
) -> tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
    CosineWarmupSpec,
    Any,
]:
    """Build AdamW and attach a scheduler step to every optimizer update."""

    training = _mapping(config.raw["training"], "training")
    raw_optimizer = training.get("optimizer", "adamw")
    if isinstance(raw_optimizer, Mapping):
        _reject_unknown_keys(
            raw_optimizer,
            frozenset({"name", "betas", "eps", "amsgrad"}),
            "training.optimizer",
        )
        optimizer_settings = raw_optimizer
        optimizer_name = str(raw_optimizer.get("name", "adamw")).strip().lower()
    else:
        optimizer_settings = {}
        optimizer_name = str(raw_optimizer).strip().lower()
    if optimizer_name != "adamw":
        raise ValueError("OT2 V2 supervised training requires optimizer=AdamW")

    betas_raw = optimizer_settings.get("betas", (0.9, 0.999))
    if not isinstance(betas_raw, Sequence) or isinstance(betas_raw, (str, bytes)) or len(betas_raw) != 2:
        raise ValueError("training.optimizer.betas must contain two numbers")
    betas = tuple(_finite_float(value, "training.optimizer.betas") for value in betas_raw)
    if not all(0.0 <= value < 1.0 for value in betas):
        raise ValueError("training.optimizer.betas values must lie in [0, 1)")
    eps = _finite_float(optimizer_settings.get("eps", 1e-8), "training.optimizer.eps")
    if eps <= 0:
        raise ValueError("training.optimizer.eps must be positive")
    amsgrad = optimizer_settings.get("amsgrad", False)
    amsgrad = _strict_bool(amsgrad, "training.optimizer.amsgrad")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
        amsgrad=amsgrad,
    )
    batches = _positive_int(train_loader_batches, "number of train loader batches")
    updates_per_epoch = math.ceil(
        batches / int(config.training.gradient_accumulation_steps)
    )
    total_updates = updates_per_epoch * int(config.training.max_epochs)
    warmup_ratio, explicit_warmup_steps = _scheduler_parameters(config)
    warmup_steps = (
        int(total_updates * warmup_ratio)
        if explicit_warmup_steps is None
        else explicit_warmup_steps
    )
    if warmup_steps >= total_updates:
        raise ValueError(
            "warmup_steps must be smaller than total optimizer steps so the "
            "cosine phase is non-empty"
        )

    def multiplier(current_step: int) -> float:
        step = int(current_step)
        if warmup_steps and step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_updates - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if not hasattr(optimizer, "register_step_post_hook"):
        raise RuntimeError(
            "this PyTorch version lacks optimizer.register_step_post_hook; "
            "PyTorch >=2 is required for exact accumulation-aware scheduling"
        )
    hook = optimizer.register_step_post_hook(
        lambda _optimizer, _args, _kwargs: scheduler.step()
    )
    spec = CosineWarmupSpec(
        name=SCHEDULER_NAME,
        interval="optimizer_step",
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        total_optimizer_steps=total_updates,
        optimizer_steps_per_epoch=updates_per_epoch,
    )
    return optimizer, scheduler, spec, hook


def _effective_runtime(
    config: ModelV2RunConfig,
    *,
    requested_device: Optional[str],
    requested_workers: Optional[int],
    requested_bf16: Optional[bool],
) -> tuple[torch.device, int, bool]:
    training = _mapping(config.raw["training"], "training")
    device = resolve_device(requested_device or str(training.get("device", "auto")))
    workers_value = (
        requested_workers
        if requested_workers is not None
        else training.get("num_workers", training.get("workers", 4))
    )
    workers = _nonnegative_int(workers_value, "workers")
    if requested_bf16 is None:
        if "bf16" in training and "use_bf16_amp" in training:
            raise ValueError("set only one of training.bf16 and training.use_bf16_amp")
        raw_bf16 = training.get("bf16", training.get("use_bf16_amp", False))
        use_bf16 = _strict_bool(raw_bf16, "training.bf16")
    else:
        use_bf16 = bool(requested_bf16)
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested, but the selected CUDA device lacks support")
    return device, workers, use_bf16


def _engine_config(
    config: ModelV2RunConfig,
    *,
    use_bf16_amp: bool,
    disabled_groups: Sequence[str] = (),
    seed: Optional[int] = None,
    evaluation_phase: Optional[str] = None,
) -> BinaryTrainingConfig:
    raw = _mapping(config.raw["training"], "training")
    phase = str(
        raw.get("evaluation_phase", "validation_screen")
        if evaluation_phase is None
        else evaluation_phase
    ).strip().lower()
    if phase != "validation_screen":
        raise ValueError(
            "training is validation-only; evaluate a locked best checkpoint on "
            "test with evaluate_frozen_checkpoint_v2.py"
        )
    return BinaryTrainingConfig(
        task=config.dataset.task,
        max_epochs=int(config.training.max_epochs),
        gradient_accumulation_steps=int(
            config.training.gradient_accumulation_steps
        ),
        grad_clip_norm=_finite_float(
            raw.get("grad_clip_norm", 1.0), "training.grad_clip_norm"
        ),
        use_bf16_amp=use_bf16_amp,
        use_pos_weight=_strict_bool(
            raw.get("use_pos_weight", True), "training.use_pos_weight"
        ),
        selection_metric=config.training.selection_metric,
        selection_min_delta=_finite_float(
            raw.get("selection_min_delta", 0.0),
            "training.selection_min_delta",
        ),
        early_stopping_patience=int(config.training.early_stopping_patience),
        threshold_method=str(raw.get("threshold_method", "youden")),
        fixed_threshold=_finite_float(
            raw.get("fixed_threshold", 0.5), "training.fixed_threshold"
        ),
        seed=int(config.seed if seed is None else seed),
        deterministic=_strict_bool(
            raw.get("deterministic", True), "training.deterministic"
        ),
        disabled_groups=tuple(disabled_groups),
        evaluation_phase=phase,
    )


def build_provenance(
    prepared_config_path: Path,
    config: ModelV2RunConfig,
    datasets: DatasetBundle,
    loaders: LoaderBundle,
    device: torch.device,
    use_bf16_amp: bool,
    model_config: Task2V2ModelConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler_spec: CosineWarmupSpec,
    engine_config: BinaryTrainingConfig,
    initialization_audit: Mapping[str, Any],
    output_dir: Path,
) -> CheckpointProvenance:
    """Create JSON-safe, reconstruction-oriented checkpoint lineage."""

    lineage = validate_model_v2_lineage(config)
    paths = {
        "master_csv": config.dataset.master_csv,
        "master_metadata_json": config.dataset.master_metadata_json,
        "scaler_json": config.dataset.scaler_json,
        "clinical_schema_json": config.dataset.clinical_schema_json,
        "environment_schema_json": config.dataset.environment_schema_json,
        "wearable_schema_json": config.dataset.wearable_schema_json,
        "ecg_schema_json": config.dataset.ecg_schema_json,
    }
    scaler_payload = datasets.train.scaler_payload
    optimizer_group = optimizer.param_groups[0]
    enabled_groups = [
        group
        for group in PRIMARY_MODALITY_GROUPS
        if group not in set(engine_config.disabled_groups)
    ]
    for split_name, dataset in (
        ("train", datasets.train),
        ("val", datasets.validation),
        ("test", datasets.test),
    ):
        if list(dataset.loaded_source_groups) != enabled_groups:
            raise ValueError(
                f"{split_name} Dataset load contract does not match the model "
                f"subset: loaded={list(dataset.loaded_source_groups)}, "
                f"enabled={enabled_groups}"
            )

    clinical_features_path = datasets.train.clinical_features_path
    clinical_features_provenance: dict[str, Any] = {
        "path": str(clinical_features_path),
        "read": "clinical_data" in enabled_groups,
    }
    if "clinical_data" in enabled_groups:
        clinical_features_provenance["sha256"] = _sha256_file(
            clinical_features_path
        )
    else:
        # A true acquisition ablation must remain runnable when the omitted
        # participant-level artifact is not deployed.  The feature order and
        # preprocessing contract remain bound through the clinical schema and
        # scaler hashes above; deliberately do not open/stat the wide CSV.
        clinical_features_provenance["sha256"] = None
        clinical_features_provenance["binding"] = (
            "omitted_source_bound_by_schema_scaler_and_master_contract"
        )

    config_provenance = {
        "config_path": str(prepared_config_path),
        "config_hash": config.config_hash,
        "schema_version": MODEL_SCHEMA_VERSION,
        "engine_format_version": ENGINE_FORMAT_VERSION,
        "raw_yaml": _json_safe(config.raw),
        "resolved_model_config": _json_safe(model_config),
        "runtime": {
            "device": str(device),
            "bf16_amp": use_bf16_amp,
            "workers": loaders.workers,
            "pin_memory": loaders.pin_memory,
            "batch_size": int(config.training.batch_size),
            "gradient_accumulation_steps": int(
                config.training.gradient_accumulation_steps
            ),
            "seed": int(engine_config.seed),
            "output_dir": str(output_dir),
        },
        "modality_subset": {
            "enabled_groups": enabled_groups,
            "disabled_groups": list(engine_config.disabled_groups),
            "dataset_loaded_groups": enabled_groups,
            "disabled_participant_artifacts_opened": False,
            "semantics": (
                "disabled participant artifacts are not opened, disabled encoders "
                "are not evaluated, and fusion receives the corresponding learned "
                "missing token"
            ),
        },
        "initialization": _json_safe(initialization_audit),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(config.training.learning_rate),
            "weight_decay": float(config.training.weight_decay),
            "betas": list(optimizer_group["betas"]),
            "eps": float(optimizer_group["eps"]),
            "amsgrad": bool(optimizer_group["amsgrad"]),
        },
        "scheduler": _json_safe(scheduler_spec),
        "retinal_backbone": _json_safe(
            model.retinal_backbone_provenance()
            if isinstance(model, Task2V2Model)
            else None
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "resume_supported": False,
    }
    data_provenance = {
        "task": config.dataset.task,
        "artifact_root": str(config.dataset.artifact_root)
        if config.dataset.artifact_root is not None
        else None,
        "lineage": lineage,
        "files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "clinical_features_csv": clinical_features_provenance,
        "split_counts": {
            "train": len(datasets.train),
            "val": len(datasets.validation),
            "test": len(datasets.test),
        },
        "split_person_ids_sha256": {
            "train": _person_ids_hash(datasets.train),
            "val": _person_ids_hash(datasets.validation),
            "test": _person_ids_hash(datasets.test),
        },
        "model_feature_dimensions": dict(datasets.feature_dimensions),
        "data_contract_audit": _json_safe(datasets.train.data_contract_audit),
        "test_role": (
            "not_iterated_during_validation_screen"
            if engine_config.evaluation_phase == "validation_screen"
            else "final_evaluation_only_after_best_validation_checkpoint_reload"
        ),
    }
    scaler_provenance = {
        "path": str(config.dataset.scaler_json),
        "sha256": _sha256_file(config.dataset.scaler_json),
        "scaler_format_version": scaler_payload.get("scaler_format_version"),
        "preprocessing_version": scaler_payload.get("preprocessing_version"),
        "scaler_config_hash": scaler_payload.get("scaler_config_hash"),
        "fit_split": scaler_payload.get("fit_split"),
        "fit_population": _json_safe(scaler_payload.get("fit_population", {})),
        "source_config_hashes": _json_safe(
            scaler_payload.get("source_config_hashes", {})
        ),
    }
    return CheckpointProvenance(
        config=_json_safe(config_provenance),
        data=_json_safe(data_provenance),
        scaler=_json_safe(scaler_provenance),
    )


def prepare_run(
    config_path: str | Path,
    *,
    device: Optional[str] = None,
    workers: Optional[int] = None,
    bf16: Optional[bool] = None,
    pretrained_checkpoint: Optional[str | Path] = None,
    disabled_groups: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
    seed: Optional[int] = None,
    evaluation_phase: Optional[str] = None,
) -> PreparedRun:
    """Validate and construct every component without beginning optimization."""

    resolved_path, config = load_supervised_config(config_path)
    resolved_device, resolved_workers, use_bf16 = _effective_runtime(
        config,
        requested_device=device,
        requested_workers=workers,
        requested_bf16=bf16,
    )
    effective_disabled = _resolve_disabled_groups(config, disabled_groups)
    effective_seed = int(config.seed if seed is None else seed)
    if effective_seed < 0:
        raise ValueError("seed must be non-negative")
    effective_output_dir = (
        config.output_dir
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    engine_config = _engine_config(
        config,
        use_bf16_amp=use_bf16,
        disabled_groups=effective_disabled,
        seed=effective_seed,
        evaluation_phase=evaluation_phase,
    )
    # The engine seeds at fit time too, but model initialization happens here.
    set_deterministic_seed(effective_seed, deterministic=engine_config.deterministic)
    datasets = build_datasets(
        config,
        loaded_source_groups=[
            group
            for group in PRIMARY_MODALITY_GROUPS
            if group not in set(effective_disabled)
        ],
    )
    loaders = build_dataloaders(
        datasets,
        batch_size=config.training.batch_size,
        workers=resolved_workers,
        seed=effective_seed,
        device=resolved_device,
    )
    model_config = build_model_config(config, datasets.feature_dimensions)
    model = Task2V2Model(model_config)
    raw_training = _mapping(config.raw["training"], "training")
    requested_checkpoint = (
        None
        if pretrained_checkpoint is None
        else _resolve_optional_path(
            pretrained_checkpoint,
            base=Path.cwd(),
            label="--pretrained-checkpoint",
        )
    )
    configured_checkpoint = (
        None
        if requested_checkpoint is not None
        else _resolve_optional_path(
            raw_training.get("initialization_checkpoint"),
            base=resolved_path.parent,
            label="training.initialization_checkpoint",
        )
    )
    checkpoint_path = requested_checkpoint or configured_checkpoint
    initialization_audit: Mapping[str, Any]
    if checkpoint_path is None:
        initialization_audit = {
            "kind": "no_multimodal_pretraining_checkpoint",
            "task_heads": "random_initialization",
            "note": (
                "retinal backbone initialization is recorded separately; all "
                "other trainable modules use deterministic random initialization"
            ),
        }
    else:
        initialization_audit = load_pretrained_student_checkpoint(
            model,
            checkpoint_path,
            expected_task=config.dataset.task,
            expected_lineage=pretraining_lineage_expectations(
                resolved_path, config, datasets.train
            ),
        )
    optimizer, scheduler, scheduler_spec, scheduler_hook = (
        build_optimizer_and_scheduler(
            model,
            config,
            train_loader_batches=len(loaders.train),
        )
    )
    try:
        provenance = build_provenance(
            resolved_path,
            config,
            datasets,
            loaders,
            resolved_device,
            use_bf16,
            model_config,
            model,
            optimizer,
            scheduler_spec,
            engine_config,
            initialization_audit,
            effective_output_dir,
        )
    except BaseException:
        scheduler_hook.remove()
        raise
    return PreparedRun(
        config_path=resolved_path,
        config=config,
        device=resolved_device,
        use_bf16_amp=use_bf16,
        datasets=datasets,
        loaders=loaders,
        model_config=model_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_spec=scheduler_spec,
        scheduler_hook=scheduler_hook,
        engine_config=engine_config,
        provenance=provenance,
        output_dir=effective_output_dir,
        initialization_audit=initialization_audit,
    )


def dry_run_forward(prepared: PreparedRun) -> Dict[str, Any]:
    """Run one train batch only; validation/test loaders are never iterated."""

    prepared.model.to(prepared.device)
    prepared.model.eval()
    raw_batch = next(iter(prepared.loaders.train))
    if set(str(value) for value in raw_batch["split"]) != {"train"}:
        raise ValueError("dry-run train loader yielded a non-train sample")
    batch = recursive_to_device(
        raw_batch, prepared.device, non_blocking=prepared.device.type == "cuda"
    )
    autocast = (
        torch.autocast(device_type=prepared.device.type, dtype=torch.bfloat16)
        if prepared.use_bf16_amp
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        output = prepared.model(
            batch,
            task=prepared.config.dataset.task,
            disabled_groups=prepared.engine_config.disabled_groups,
        )
    logits = output["logits"]
    labels = batch["y"].reshape(-1)
    if logits.shape != labels.shape or not torch.isfinite(logits.float()).all():
        raise ValueError(
            f"dry-run logits {tuple(logits.shape)} are invalid for labels "
            f"{tuple(labels.shape)}"
        )
    return {
        "status": "dry_run_ok",
        "task": prepared.config.dataset.task,
        "device": str(prepared.device),
        "bf16_amp": prepared.use_bf16_amp,
        "workers": prepared.loaders.workers,
        "split_counts": {
            "train": len(prepared.datasets.train),
            "val": len(prepared.datasets.validation),
            "test": len(prepared.datasets.test),
        },
        "feature_dimensions": dict(prepared.datasets.feature_dimensions),
        "train_batch_size": int(labels.numel()),
        "logits_shape": list(logits.shape),
        "scheduler": _json_safe(prepared.scheduler_spec),
        "enabled_groups": [
            group
            for group in PRIMARY_MODALITY_GROUPS
            if group not in set(prepared.engine_config.disabled_groups)
        ],
        "disabled_groups": list(prepared.engine_config.disabled_groups),
        "initialization": _json_safe(prepared.initialization_audit),
        "evaluation_phase": prepared.engine_config.evaluation_phase,
        "test_loader_iterated": False,
        "validation_or_test_batches_iterated": False,
    }


def run_supervised_training(
    config_path: str | Path,
    *,
    device: Optional[str] = None,
    workers: Optional[int] = None,
    bf16: Optional[bool] = None,
    dry_run: bool = False,
    pretrained_checkpoint: Optional[str | Path] = None,
    disabled_groups: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
    seed: Optional[int] = None,
    evaluation_phase: Optional[str] = None,
) -> FitResult | Dict[str, Any]:
    prepared = prepare_run(
        config_path,
        device=device,
        workers=workers,
        bf16=bf16,
        pretrained_checkpoint=pretrained_checkpoint,
        disabled_groups=disabled_groups,
        output_dir=output_dir,
        seed=seed,
        evaluation_phase=evaluation_phase,
    )
    try:
        if dry_run:
            return dry_run_forward(prepared)
        # Avoid a costly preliminary traversal of lazy retinal artifacts and
        # prevent the class-weight pass from advancing the shuffled loader RNG.
        train_labels = torch.as_tensor(
            prepared.datasets.train.index["y"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return fit_evaluate_binary(
            prepared.model,
            prepared.loaders.train,
            prepared.loaders.validation,
            prepared.loaders.test,
            prepared.optimizer,
            output_dir=prepared.output_dir,
            config=prepared.engine_config,
            provenance=prepared.provenance,
            device=prepared.device,
            train_labels=train_labels,
        )
    finally:
        # Hooks are process-local execution state and must not leak if this
        # function is called repeatedly in a notebook or test worker.
        prepared.remove_scheduler_hook()


def _fit_result_summary(result: FitResult) -> Dict[str, Any]:
    return {
        "status": "complete",
        "best_epoch": result.best_epoch,
        "best_score": result.best_score,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "pos_weight": result.pos_weight,
        "selected_threshold": dataclasses.asdict(result.threshold_selection),
        "validation_metrics": result.validation.metrics,
        "evaluation_phase": (
            "final_test" if result.test is not None else "validation_screen"
        ),
        "test_metrics": None if result.test is None else result.test.metrics,
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
        "validation_predictions": str(result.validation_predictions_csv),
        "test_predictions": (
            None if result.predictions_csv is None else str(result.predictions_csv)
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an OT2 Model V2 supervised binary task from strict YAML."
    )
    parser.add_argument("--config", required=True, type=Path, help="V2 YAML config")
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime override: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--workers",
        "--num-workers",
        "--num_workers",
        dest="workers",
        type=int,
        default=None,
        help="DataLoader worker override (non-negative)",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override YAML bf16 autocast setting",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Strict corrected-pretraining student checkpoint; task heads are "
            "always freshly initialized"
        ),
    )
    parser.add_argument(
        "--disabled-groups",
        nargs="*",
        default=None,
        choices=PRIMARY_MODALITY_GROUPS,
        help=(
            "Fixed modality groups omitted throughout train/validation/test; "
            "use for single-modality and leave-one-modality-out retraining"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run-specific output directory override (recorded in provenance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run-specific non-negative seed override (recorded in provenance)",
    )
    parser.add_argument(
        "--evaluation-phase",
        choices=("validation_screen",),
        default=None,
        help=(
            "training never iterates test; use evaluate_frozen_checkpoint_v2.py "
            "with a validation selection lock for final confirmation"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/build and forward one train batch; do not iterate val/test loaders",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_supervised_training(
        args.config,
        device=args.device,
        workers=args.workers,
        bf16=args.bf16,
        dry_run=args.dry_run,
        pretrained_checkpoint=args.pretrained_checkpoint,
        disabled_groups=args.disabled_groups,
        output_dir=args.output_dir,
        seed=args.seed,
        evaluation_phase=args.evaluation_phase,
    )
    payload = result if isinstance(result, Mapping) else _fit_result_summary(result)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CosineWarmupSpec",
    "DatasetBundle",
    "LoaderBundle",
    "PreparedRun",
    "SUPPORTED_TASKS",
    "build_arg_parser",
    "build_dataloaders",
    "build_datasets",
    "build_model_config",
    "build_optimizer_and_scheduler",
    "build_provenance",
    "dry_run_forward",
    "load_supervised_config",
    "main",
    "prepare_run",
    "resolve_device",
    "run_supervised_training",
]
