#!/usr/bin/env python3
"""Cache full-teacher tokens and train an OT2 universal subset student.

This is a validation-only pilot runner.  It deliberately never iterates the
test loader.  ``cache`` performs one full native-encoder pass over train and
validation; ``train`` then learns whole-modality masking and latent
reconstruction entirely in the compact ``[participant, 9, d_model]`` token
space.  The resulting checkpoint retains every native encoder so a later
deployment evaluator can accept different observed modality subsets.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - requirements-model-v2 includes tqdm.
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore
        return iterable

try:
    from ot2.task2_v2_engine import (
        atomic_torch_save,
        binary_classification_metrics,
        recursive_to_device,
        select_validation_threshold,
        set_deterministic_seed,
    )
    from ot2.task2_v2_dataset import (
        SOURCE_FLAG_BY_GROUP,
        OT2V2Dataset,
        task2_v2_collate,
    )
    from ot2.task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model
    from ot2.task2_v2_universal_reconstruction import (
        ModalityMaskSampler,
        TargetQueryCrossModalReconstructor,
        Task2V2UniversalDistiller,
        Task2V2UniversalReconstructionStudent,
        UNIVERSAL_RECONSTRUCTION_FORMAT_VERSION,
        UniversalMaskConfig,
        UniversalReconstructionConfig,
    )
    from ot2.train_cheap_student_v2 import load_full_teacher_checkpoint
    from ot2.train_task2_v2 import (
        build_model_config,
        load_supervised_config,
        resolve_device,
    )
except ModuleNotFoundError:  # Direct execution from ``ot2/``.
    from task2_v2_engine import (  # type: ignore
        atomic_torch_save,
        binary_classification_metrics,
        recursive_to_device,
        select_validation_threshold,
        set_deterministic_seed,
    )
    from task2_v2_dataset import (  # type: ignore
        SOURCE_FLAG_BY_GROUP,
        OT2V2Dataset,
        task2_v2_collate,
    )
    from task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model  # type: ignore
    from task2_v2_universal_reconstruction import (  # type: ignore
        ModalityMaskSampler,
        TargetQueryCrossModalReconstructor,
        Task2V2UniversalDistiller,
        Task2V2UniversalReconstructionStudent,
        UNIVERSAL_RECONSTRUCTION_FORMAT_VERSION,
        UniversalMaskConfig,
        UniversalReconstructionConfig,
    )
    from train_cheap_student_v2 import load_full_teacher_checkpoint  # type: ignore
    from train_task2_v2 import (  # type: ignore
        build_model_config,
        load_supervised_config,
        resolve_device,
    )


TOKEN_CACHE_FORMAT_VERSION = "ot2-full-teacher-token-cache-v1"
DEPLOYMENT_FORMAT_VERSION = "ot2-universal-reconstruction-deployment-v1"
TRAINING_FORMAT_VERSION = "ot2-universal-reconstruction-training-v1"


def _runtime_code_sha256() -> Dict[str, str]:
    """Hash every local Python module that defines this cache/model contract."""

    root = Path(__file__).resolve().parent
    names = (
        "train_universal_reconstruction_v2.py",
        "task2_v2_universal_reconstruction.py",
        "task2_v2_model.py",
        "task2_v2_distill.py",
        "task2_v2_engine.py",
        "task2_v2_config.py",
        "task2_v2_dataset.py",
        "train_task2_v2.py",
        "train_cheap_student_v2.py",
        "fit_train_scalers_v2.py",
        "preprocessing_v2.py",
    )
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"universal runtime code files are missing: {missing}")
    return {name: _sha256_file(root / name) for name in names}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _well_formed_sha256(value: Any, label: str) -> str:
    """Return a normalized SHA-256 digest or fail closed."""

    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _cohort_hashes(
    person_ids: Sequence[str], labels: torch.Tensor
) -> Dict[str, str]:
    normalized_ids = [str(value) for value in person_ids]
    normalized_labels = [int(value) for value in labels.reshape(-1).tolist()]
    if len(normalized_ids) != len(normalized_labels):
        raise ValueError("person IDs and labels differ in length")
    pairs = sorted(zip(normalized_ids, normalized_labels))
    return {
        "ordered_person_ids_sha256": _canonical_hash(normalized_ids),
        "person_id_set_sha256": _canonical_hash(sorted(normalized_ids)),
        "person_label_map_sha256": _canonical_hash(pairs),
    }


def _dataset_person_labels(dataset: Any) -> Dict[str, int]:
    frame = dataset.index
    identifiers = [str(value) for value in frame["person_id"]]
    labels = [int(value) for value in frame["y"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("current official dataset contains duplicate person IDs")
    return dict(zip(identifiers, labels))


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _safe_torch_load(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - cluster uses modern torch.
        raise RuntimeError("safe loading requires torch.load(weights_only=True)") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"torch artifact is not a mapping: {resolved}")
    return dict(payload)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--bf16 requested but the CUDA device lacks bf16 support")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bf16 is supported only on CPU/CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _validate_teacher_validation_screen_payload(
    payload: Mapping[str, Any], *, label: str
) -> Dict[str, Any]:
    """Require a validation-selected teacher that carries no test results.

    The full teacher is allowed to provide targets only when its checkpoint
    proves that it was selected in the validation-screen phase.  Optional
    explicit test fields are fail-closed when present.
    """

    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError(f"{label} lacks training_config")
    training_phase = str(
        training_config.get("evaluation_phase") or ""
    ).strip().lower()
    if training_phase != "validation_screen":
        raise ValueError(
            f"{label} must have training_config.evaluation_phase="
            "validation_screen"
        )
    data_provenance = payload.get("data_provenance")
    if not isinstance(data_provenance, Mapping):
        raise ValueError(f"{label} lacks data_provenance")
    test_role = str(data_provenance.get("test_role") or "").strip()
    if test_role != "not_iterated_during_validation_screen":
        raise ValueError(
            f"{label} does not prove that test was excluded during selection"
        )
    top_level_phase = payload.get("evaluation_phase")
    if top_level_phase is not None and str(top_level_phase).strip().lower() != (
        "validation_screen"
    ):
        raise ValueError(f"{label} has a conflicting top-level evaluation_phase")
    if "test_evaluated" in payload and payload.get("test_evaluated") is not False:
        raise ValueError(f"{label} records test_evaluated=true")
    if "test_metrics" in payload:
        test_metrics = payload.get("test_metrics")
        empty_metrics = test_metrics is None or (
            isinstance(test_metrics, (Mapping, list, tuple))
            and len(test_metrics) == 0
        )
        if not empty_metrics:
            raise ValueError(f"{label} contains populated test metrics")
    return {
        "training_evaluation_phase": training_phase,
        "data_provenance_test_role": test_role,
        "top_level_evaluation_phase": (
            None
            if top_level_phase is None
            else str(top_level_phase).strip().lower()
        ),
        "test_evaluated_field": payload.get("test_evaluated"),
        "test_metrics_present": "test_metrics" in payload,
        "contract_passed": True,
    }


def _validate_teacher_validation_screen_contract(
    checkpoint: str | Path,
) -> Dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    payload = _safe_torch_load(path)
    return {
        "checkpoint": str(path),
        **_validate_teacher_validation_screen_payload(
            payload, label=f"teacher checkpoint {path}"
        ),
    }


@dataclasses.dataclass(frozen=True)
class _TrainValidationDatasets:
    train: OT2V2Dataset
    validation: OT2V2Dataset
    feature_dimensions: Mapping[str, int]
    master_view_record: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class _TrainValidationLoaders:
    train: DataLoader
    validation: DataLoader
    workers: int
    pin_memory: bool


def _write_train_val_master_view(
    source: Path,
    destination: Path,
    source_metadata: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Materialize a deterministic master view containing no test rows.

    ``OT2V2Dataset`` validates labels before applying its split filter.  Feeding
    it the original three-way master would therefore inspect test outcomes even
    though this pilot never iterates a test loader.  This streaming projection
    selects rows by the ``split`` field and writes only train/validation rows;
    excluded test labels are never parsed or validated by model code.
    """

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"supervised master CSV does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    split_only = pd.read_csv(
        source,
        usecols=["person_id", "split"],
        dtype={"person_id": "string", "split": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    normalized = split_only["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized).difference({"train", "val", "test"}))
    if unknown:
        raise ValueError(f"supervised master contains unsupported splits: {unknown}")
    included_positions = np.flatnonzero(normalized.isin(["train", "val"]).to_numpy())
    keep_file_rows = {int(position) + 1 for position in included_positions}
    projected = pd.read_csv(
        source,
        skiprows=lambda row_number: (
            row_number != 0 and row_number not in keep_file_rows
        ),
        dtype={"person_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    projected_splits = projected["split"].astype(str).str.strip().str.lower()
    counts = {
        "train": int(projected_splits.eq("train").sum()),
        "val": int(projected_splits.eq("val").sum()),
        "test_excluded": int(normalized.eq("test").sum()),
    }
    if len(projected) != counts["train"] + counts["val"]:
        raise RuntimeError("train/validation master projection is inconsistent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output_stream:
            projected.to_csv(output_stream, index=False, lineterminator="\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if counts["train"] <= 0 or counts["val"] <= 0:
            raise ValueError(
                "supervised master must contain both train and validation rows"
            )
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    metadata_record: Optional[Dict[str, Any]] = None
    if source_metadata is not None:
        source_metadata = Path(source_metadata).expanduser().resolve()
        if not source_metadata.is_file():
            raise FileNotFoundError(
                f"supervised master metadata does not exist: {source_metadata}"
            )
        try:
            raw_metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"could not parse supervised master metadata: {source_metadata}"
            ) from exc
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("supervised master metadata must contain a JSON object")
        if raw_metadata.get("n_rows") is not None and int(
            raw_metadata["n_rows"]
        ) != len(split_only):
            raise ValueError("supervised master metadata n_rows differs from CSV")
        source_indexes = raw_metadata.get("source_indexes")
        if not isinstance(source_indexes, Mapping):
            raise ValueError("supervised master metadata lacks source_indexes")
        sanitized_source_indexes: Dict[str, Dict[str, str]] = {}
        for source_name in ("environment", "wearable", "ecg", "retinal"):
            raw_record = source_indexes.get(source_name)
            if not isinstance(raw_record, Mapping):
                raise ValueError(
                    "supervised master metadata lacks source lineage for "
                    f"{source_name}"
                )
            sanitized_record = {
                "config_hash": _well_formed_sha256(
                    raw_record.get("config_hash"),
                    f"source_indexes.{source_name}.config_hash",
                )
            }
            mapping_hash = str(raw_record.get("mapping_hash") or "").strip()
            if mapping_hash:
                sanitized_record["mapping_hash"] = _well_formed_sha256(
                    mapping_hash,
                    f"source_indexes.{source_name}.mapping_hash",
                )
            if source_name == "retinal":
                sanitized_record["model_view_config_hash"] = _well_formed_sha256(
                    raw_record.get("model_view_config_hash"),
                    "source_indexes.retinal.model_view_config_hash",
                )
            sanitized_source_indexes[source_name] = sanitized_record
        retinal_hash = sanitized_source_indexes["retinal"]["config_hash"]
        model_view_hash = sanitized_source_indexes["retinal"][
            "model_view_config_hash"
        ]
        task_name = str(raw_metadata.get("task", "") or "").strip()
        clinical_schema_hash = _well_formed_sha256(
            raw_metadata.get("clinical_schema_hash"),
            "supervised master clinical_schema_hash",
        )
        if not task_name:
            raise ValueError(
                "supervised master metadata must bind task and clinical schema"
            )
        derived_metadata_path = destination.with_suffix(".metadata.json")
        # Keep precisely the fields consumed by OT2V2Dataset's artifact
        # contract checks.  Outcome summaries and split-level aggregates from
        # the three-way parent metadata are intentionally not copied.
        derived_metadata = {
            "format_version": "ot2-train-validation-master-metadata-v1",
            "task": task_name,
            "n_rows": int(len(projected)),
            "preprocessing_version": raw_metadata.get("preprocessing_version"),
            "clinical_schema_hash": clinical_schema_hash,
            # Only lineage fields consumed by OT2V2Dataset are retained.  The
            # parent metadata contains full-cohort/test aggregates, including
            # nested fields, and must never be copied wholesale here.
            "source_indexes": sanitized_source_indexes,
            "derived_view": {
                "included_splits": ["train", "val"],
                "source_master_csv_sha256": _sha256_file(source),
                "source_master_metadata_sha256": _sha256_file(source_metadata),
                "filter_first_pass_columns": ["person_id", "split"],
                "non_train_validation_y_materialized": False,
                "test_rows_in_derived_view": 0,
                "metadata_policy": "lineage_fields_allowlist",
            },
        }
        _atomic_json(derived_metadata, derived_metadata_path)
        metadata_record = {
            "source_master_metadata_json": str(source_metadata),
            "source_master_metadata_sha256": _sha256_file(source_metadata),
            "view_master_metadata_json": str(derived_metadata_path),
            "view_master_metadata_sha256": _sha256_file(derived_metadata_path),
            "retinal_artifact_config_hash": retinal_hash,
            "retinal_model_view_config_hash": model_view_hash,
            "clinical_schema_hash": clinical_schema_hash,
            "outcome_or_test_aggregate_fields_copied": False,
        }
    record = {
        "format_version": "ot2-train-validation-master-view-v1",
        "source_master_csv": str(source),
        "source_master_sha256": _sha256_file(source),
        "view_master_csv": str(destination),
        "view_master_sha256": _sha256_file(destination),
        "row_counts": counts,
        "included_splits": ["train", "val"],
        "first_pass_columns": ["person_id", "split"],
        "non_train_validation_y_materialized": False,
        "test_rows_present_in_view": False,
        "test_labels_validated_by_dataset": False,
        "metadata_contract": metadata_record,
    }
    _atomic_json(record, destination.with_suffix(".lineage.json"))
    return record


def _build_train_val_datasets(
    run_config: Any,
    *,
    split_view_dir: Path,
) -> _TrainValidationDatasets:
    if not run_config.dataset.deterministic_retinal_selection:
        raise ValueError(
            "dataset.retinal.deterministic_selection=false is unsupported"
        )
    view_path = (
        Path(split_view_dir).expanduser().resolve()
        / f"{run_config.dataset.task}.train_val_only.csv"
    )
    if run_config.dataset.master_metadata_json is None:
        raise ValueError(
            "universal cache requires dataset.master_metadata_json so artifact "
            "lineage remains enforceable after the train/validation projection"
        )
    view_record = _write_train_val_master_view(
        Path(run_config.dataset.master_csv),
        view_path,
        Path(run_config.dataset.master_metadata_json),
    )
    retinal_caps = {
        "retinal_photography": int(
            run_config.dataset.retinal_max_assets["photography"]
        ),
        "retinal_oct": int(run_config.dataset.retinal_max_assets["oct"]),
        "retinal_octa": int(run_config.dataset.retinal_max_assets["octa"]),
        "retinal_flio": int(run_config.dataset.retinal_max_assets["flio"]),
    }
    common = {
        "master_csv": view_path,
        "scaler_json": run_config.dataset.scaler_json,
        "task": run_config.dataset.task,
        "artifact_root": run_config.dataset.artifact_root,
        "master_metadata_json": view_path.with_suffix(".metadata.json"),
        "environment_schema_json": run_config.dataset.environment_schema_json,
        "max_retinal_assets_per_group": retinal_caps,
        "require_y": True,
        "validate_scaler_hash": True,
    }
    train = OT2V2Dataset(split="train", **common)
    validation = OT2V2Dataset(split="val", **common)
    metadata_contract = view_record.get("metadata_contract")
    if not isinstance(metadata_contract, Mapping):
        raise RuntimeError("train/validation master view lacks metadata contract")
    expected_retinal_hash = str(
        metadata_contract["retinal_artifact_config_hash"]
    )
    expected_model_view_hash = str(
        metadata_contract["retinal_model_view_config_hash"]
    )
    for split, dataset in (("train", train), ("val", validation)):
        if dataset.retinal_artifact_config_hash != expected_retinal_hash:
            raise ValueError(
                f"{split} Dataset lost the retinal artifact lineage contract"
            )
        if dataset.retinal_model_view_config_hash != expected_model_view_hash:
            raise ValueError(
                f"{split} Dataset lost the retinal model-view lineage contract"
            )
    dimensions = dict(train.model_feature_dimensions)
    if dict(validation.model_feature_dimensions) != dimensions:
        raise ValueError("validation feature dimensions differ from training")
    if validation.environment_feature_names != train.environment_feature_names:
        raise ValueError("validation environment columns differ from training")
    if validation.clinical_features_path != train.clinical_features_path:
        raise ValueError("validation resolves a different clinical feature CSV")
    overlap = set(train.index["person_id"].astype(str)).intersection(
        validation.index["person_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"official train/validation splits overlap: {sorted(overlap)[:10]}"
        )
    if not run_config.dataset.allow_missing_modalities:
        flags = sorted(set(SOURCE_FLAG_BY_GROUP.values()))
        for split, dataset in (("train", train), ("val", validation)):
            missing = (~dataset.index[flags].astype(bool)).any(axis=1)
            if bool(missing.any()):
                raise ValueError(
                    "dataset.allow_missing_modalities=false, but "
                    f"split={split} contains {int(missing.sum())} incomplete rows"
                )
    return _TrainValidationDatasets(
        train=train,
        validation=validation,
        feature_dimensions=dimensions,
        master_view_record=view_record,
    )


def _seed_loader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_train_val_loaders(
    datasets: _TrainValidationDatasets,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> _TrainValidationLoaders:
    batch_size = int(batch_size)
    workers = int(workers)
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
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
            worker_init_fn=_seed_loader_worker if workers > 0 else None,
            generator=generator,
            drop_last=False,
        )

    return _TrainValidationLoaders(
        train=make(datasets.train, shuffle=True, offset=0),
        validation=make(datasets.validation, shuffle=False, offset=1),
        workers=workers,
        pin_memory=pin_memory,
    )


def _teacher_context(
    *,
    config_path: Path,
    teacher_checkpoint: Path,
    device: torch.device,
    workers: int,
    batch_size: Optional[int],
    split_view_dir: Path,
) -> tuple[Any, Any, Any, Task2V2Model, Any, Mapping[str, Any]]:
    teacher_screen_audit = _validate_teacher_validation_screen_contract(
        teacher_checkpoint
    )
    resolved_config, run_config = load_supervised_config(config_path)
    datasets = _build_train_val_datasets(
        run_config, split_view_dir=split_view_dir
    )
    effective_batch = int(batch_size or run_config.training.batch_size)
    loaders = _build_train_val_loaders(
        datasets,
        batch_size=effective_batch,
        workers=int(workers),
        seed=int(run_config.seed),
        device=device,
    )
    model_config = build_model_config(run_config, datasets.feature_dimensions)
    teacher = Task2V2Model(model_config)
    data_paths = {
        "master_csv": run_config.dataset.master_csv,
        "master_metadata_json": run_config.dataset.master_metadata_json,
        "clinical_schema_json": run_config.dataset.clinical_schema_json,
        "environment_schema_json": run_config.dataset.environment_schema_json,
        "wearable_schema_json": run_config.dataset.wearable_schema_json,
        "ecg_schema_json": run_config.dataset.ecg_schema_json,
    }
    teacher_audit = load_full_teacher_checkpoint(
        teacher,
        teacher_checkpoint,
        expected_task=run_config.dataset.task,
        expected_config_hash=run_config.config_hash,
        expected_model_config=model_config,
        expected_master_sha256=_sha256_file(run_config.dataset.master_csv),
        expected_scaler_sha256=_sha256_file(run_config.dataset.scaler_json),
        expected_clinical_features_sha256=_sha256_file(
            datasets.train.clinical_features_path
        ),
        expected_data_file_sha256={
            name: _sha256_file(path) for name, path in data_paths.items()
        },
    )
    teacher.to(device)
    teacher.eval()
    return (
        resolved_config,
        run_config,
        datasets,
        teacher,
        loaders,
        {
            "teacher_audit": teacher_audit,
            "teacher_validation_screen_contract": teacher_screen_audit,
            "model_config": model_config,
            "master_view_record": datasets.master_view_record,
        },
    )


@torch.no_grad()
def _cache_one_split(
    *,
    split: str,
    loader: Any,
    teacher: Task2V2Model,
    task: str,
    device: torch.device,
    use_bf16_amp: bool,
    common_metadata: Mapping[str, Any],
    destination: Path,
) -> Mapping[str, Any]:
    token_chunks: list[torch.Tensor] = []
    presence_chunks: list[torch.Tensor] = []
    embedding_chunks: list[torch.Tensor] = []
    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    person_ids: list[str] = []
    progress = tqdm(loader, desc=f"Teacher token cache [{task}/{split}]", unit="batch")
    for raw_batch in progress:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("loader must yield mapping batches")
        observed_splits = {str(value).strip().lower() for value in raw_batch["split"]}
        if observed_splits != {split}:
            raise ValueError(f"expected split={split}, observed={sorted(observed_splits)}")
        batch = recursive_to_device(
            raw_batch, device, non_blocking=device.type == "cuda"
        )
        with _autocast(device, use_bf16_amp):
            encoded = teacher.encode_sources(batch)
            output = teacher.predict_from_source_tokens(
                encoded["source_tokens"], encoded["source_present"], task=task
            )
        labels = batch.get("y")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("cache input is missing y")
        token_chunks.append(encoded["source_tokens"].float().cpu())
        presence_chunks.append(encoded["source_present"].bool().cpu())
        embedding_chunks.append(output["embedding"].float().cpu())
        logit_chunks.append(output["logits"].float().cpu().reshape(-1))
        label_chunks.append(labels.float().cpu().reshape(-1))
        identifiers = [str(value) for value in raw_batch["person_id"]]
        if len(identifiers) != labels.numel():
            raise ValueError("person_id and label counts differ")
        person_ids.extend(identifiers)

    if not person_ids or len(person_ids) != len(set(person_ids)):
        raise ValueError("token cache has no people or duplicate person IDs")
    payload = {
        "format_version": TOKEN_CACHE_FORMAT_VERSION,
        "split": split,
        "task": task,
        "person_ids": person_ids,
        "source_order": list(PRIMARY_MODALITY_GROUPS),
        "source_tokens": torch.cat(token_chunks, dim=0).contiguous(),
        "source_present": torch.cat(presence_chunks, dim=0).contiguous(),
        "teacher_embedding": torch.cat(embedding_chunks, dim=0).contiguous(),
        "teacher_logits": torch.cat(logit_chunks, dim=0).contiguous(),
        "labels": torch.cat(label_chunks, dim=0).contiguous(),
        "metadata": dict(common_metadata),
    }
    payload["cohort_hashes"] = _cohort_hashes(
        payload["person_ids"], payload["labels"]
    )
    n = len(person_ids)
    expected_shapes = {
        "source_tokens": (n, len(PRIMARY_MODALITY_GROUPS), teacher.config.d_model),
        "source_present": (n, len(PRIMARY_MODALITY_GROUPS)),
        "teacher_embedding": (n, teacher.config.d_model),
        "teacher_logits": (n,),
        "labels": (n,),
    }
    for name, shape in expected_shapes.items():
        if tuple(payload[name].shape) != tuple(shape):
            raise RuntimeError(f"cache {name} shape mismatch: {payload[name].shape} != {shape}")
    if not torch.isfinite(payload["source_tokens"]).all():
        raise ValueError("cached source tokens contain non-finite values")
    if not torch.isfinite(payload["teacher_embedding"]).all():
        raise ValueError("cached teacher embeddings contain non-finite values")
    if not torch.isfinite(payload["teacher_logits"]).all():
        raise ValueError("cached teacher logits contain non-finite values")
    if not torch.all((payload["labels"] == 0) | (payload["labels"] == 1)):
        raise ValueError("cached labels are not binary")
    atomic_torch_save(payload, destination)
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "split": split,
        "n": n,
        "shape": list(payload["source_tokens"].shape),
        "source_presence_counts": {
            group: int(payload["source_present"][:, index].sum().item())
            for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
        },
    }


def build_token_cache(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.reuse_existing and args.overwrite:
        raise ValueError("choose only one of --reuse-existing and --overwrite")
    device = resolve_device(args.device)
    set_deterministic_seed(int(args.seed), deterministic=True)
    output = Path(args.cache_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (
        config_path,
        run_config,
        datasets,
        teacher,
        loaders,
        context,
    ) = _teacher_context(
        config_path=args.config,
        teacher_checkpoint=args.teacher_checkpoint,
        device=device,
        workers=args.workers,
        batch_size=args.batch_size,
        split_view_dir=output / "_master_views",
    )
    if int(args.seed) != int(run_config.seed):
        raise ValueError(
            "cache seed must match the full teacher config; use the teacher's YAML seed"
        )
    teacher_record = {
        "path": str(Path(args.teacher_checkpoint).expanduser().resolve()),
        "sha256": _sha256_file(args.teacher_checkpoint),
    }
    common = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "config_hash": run_config.config_hash,
        "teacher_checkpoint": teacher_record,
        "teacher_audit": context["teacher_audit"],
        "teacher_validation_screen_contract": context[
            "teacher_validation_screen_contract"
        ],
        "train_validation_master_view": context["master_view_record"],
        "resolved_model_config": _json_safe(context["model_config"]),
        "seed": int(args.seed),
        "test_dataset_constructed": False,
        "test_iterated": False,
        "cache_role": "train_and_validation_full_teacher_tokens_only",
        "encoding_contract": {
            "autocast_bf16": bool(args.bf16),
            "stored_tensor_dtype": "float32",
            "device_type": str(device.type),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "runtime_code_sha256": _runtime_code_sha256(),
        },
    }
    common["cache_config_hash"] = _canonical_hash(common)
    records: Dict[str, Any] = {}
    for split, loader in (("train", loaders.train), ("val", loaders.validation)):
        destination = output / f"{run_config.dataset.task}_{split}_teacher_tokens.pt"
        if destination.exists() and args.reuse_existing:
            cached = _safe_torch_load(destination)
            if (
                cached.get("format_version") != TOKEN_CACHE_FORMAT_VERSION
                or cached.get("split") != split
                or cached.get("task") != run_config.dataset.task
                or not isinstance(cached.get("metadata"), Mapping)
                or cached["metadata"].get("cache_config_hash")
                != common["cache_config_hash"]
            ):
                raise ValueError(f"existing cache is stale or incompatible: {destination}")
            cached = _validate_cache(
                destination,
                split=split,
                task=run_config.dataset.task,
                teacher_sha256=teacher_record["sha256"],
                config_hash=run_config.config_hash,
                expected_d_model=int(teacher.config.d_model),
                expected_person_labels=_dataset_person_labels(
                    getattr(datasets, "validation" if split == "val" else split)
                ),
            )
            records[split] = {
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "split": split,
                "n": len(cached.get("person_ids", [])),
                "shape": list(cached["source_tokens"].shape),
                "status": "reused",
            }
            continue
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"cache already exists; pass --reuse-existing or --overwrite: {destination}"
            )
        records[split] = dict(
            _cache_one_split(
                split=split,
                loader=loader,
                teacher=teacher,
                task=run_config.dataset.task,
                device=device,
                use_bf16_amp=args.bf16,
                common_metadata=common,
                destination=destination,
            )
        )
        records[split]["status"] = "processed"
    summary = {
        "format_version": TOKEN_CACHE_FORMAT_VERSION,
        "task": run_config.dataset.task,
        "seed": int(args.seed),
        "cache_config_hash": common["cache_config_hash"],
        "teacher_checkpoint": teacher_record,
        "splits": records,
        "train_validation_master_view": context["master_view_record"],
        "test_dataset_constructed": False,
        "test_iterated": False,
    }
    _atomic_json(summary, output / f"{run_config.dataset.task}_token_cache_summary.json")
    return summary


class CachedTokenDataset(Dataset[Dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.person_ids = [str(value) for value in payload["person_ids"]]
        self.source_tokens = payload["source_tokens"].float()
        self.source_present = payload["source_present"].bool()
        self.teacher_embedding = payload["teacher_embedding"].float()
        self.teacher_logits = payload["teacher_logits"].float()
        self.labels = payload["labels"].float()

    def __len__(self) -> int:
        return len(self.person_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "person_id": self.person_ids[index],
            "source_tokens": self.source_tokens[index],
            "source_present": self.source_present[index],
            "teacher_embedding": self.teacher_embedding[index],
            "teacher_logits": self.teacher_logits[index],
            "y": self.labels[index],
        }


def _validate_cache(
    path: Path,
    *,
    split: str,
    task: str,
    teacher_sha256: str,
    config_hash: str,
    expected_d_model: int,
    expected_person_labels: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    payload = _safe_torch_load(path)
    if payload.get("format_version") != TOKEN_CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported token cache: {path}")
    if payload.get("split") != split or payload.get("task") != task:
        raise ValueError(f"token cache task/split mismatch: {path}")
    if list(payload.get("source_order", ())) != list(PRIMARY_MODALITY_GROUPS):
        raise ValueError(f"token cache modality order mismatch: {path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"token cache lacks metadata: {path}")
    teacher_record = metadata.get("teacher_checkpoint")
    if not isinstance(teacher_record, Mapping) or teacher_record.get(
        "sha256"
    ) != teacher_sha256:
        raise ValueError(f"token cache was built from a different teacher: {path}")
    if metadata.get("config_hash") != config_hash:
        raise ValueError(f"token cache config hash mismatch: {path}")
    raw_presence = payload.get("source_present")
    if not isinstance(raw_presence, torch.Tensor):
        raise ValueError(f"token cache source_present is not a tensor: {path}")
    if raw_presence.dtype != torch.bool:
        raise ValueError(
            f"token cache source_present must use torch.bool without coercion: {path}"
        )
    dataset = CachedTokenDataset(payload)
    n = len(dataset)
    if n == 0 or len(set(dataset.person_ids)) != n:
        raise ValueError(f"token cache contains no people or duplicate IDs: {path}")
    expected_source_shape = (n, len(PRIMARY_MODALITY_GROUPS), int(expected_d_model))
    if tuple(dataset.source_tokens.shape) != expected_source_shape:
        raise ValueError(f"token cache source shape is invalid: {path}")
    if tuple(dataset.source_present.shape) != expected_source_shape[:2]:
        raise ValueError(f"token cache presence shape is invalid: {path}")
    if tuple(dataset.teacher_embedding.shape) != (n, int(expected_d_model)):
        raise ValueError(f"token cache teacher embedding shape is invalid: {path}")
    if tuple(dataset.teacher_logits.shape) != (n,):
        raise ValueError(f"token cache teacher logit shape is invalid: {path}")
    if tuple(dataset.labels.shape) != (n,):
        raise ValueError(f"token cache label shape is invalid: {path}")
    for name, tensor in (
        ("source_tokens", dataset.source_tokens),
        ("teacher_embedding", dataset.teacher_embedding),
        ("teacher_logits", dataset.teacher_logits),
        ("labels", dataset.labels),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"token cache {name} contains non-finite values: {path}")
    if not torch.all((dataset.labels == 0) | (dataset.labels == 1)):
        raise ValueError(f"token cache labels are not binary: {path}")
    observed_hashes = _cohort_hashes(dataset.person_ids, dataset.labels)
    if payload.get("cohort_hashes") != observed_hashes:
        raise ValueError(f"token cache cohort hashes are missing or invalid: {path}")
    if expected_person_labels is not None:
        cached_person_labels = {
            person_id: int(label)
            for person_id, label in zip(
                dataset.person_ids, dataset.labels.to(torch.int64).tolist()
            )
        }
        expected = {
            str(person_id): int(label)
            for person_id, label in expected_person_labels.items()
        }
        if cached_person_labels != expected:
            raise ValueError(
                f"token cache cohort/labels differ from the official {split} split: {path}"
            )
    return payload


def _load_universal_config(path: Path, *, task: str, d_model: int) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("universal YAML requires PyYAML") from exc
    resolved = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("universal config must be a mapping")
    allowed = {
        "format_version",
        "mask",
        "reconstructor",
        "loss",
        "training",
        "selection",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unknown universal config keys: {unknown}")
    if raw.get("format_version") != UNIVERSAL_RECONSTRUCTION_FORMAT_VERSION:
        raise ValueError("universal config format_version mismatch")
    mask_raw = dict(raw.get("mask") or {})
    family_mapping = mask_raw.pop("families", None)
    validation_families: Dict[str, tuple[str, ...]] = {}
    if family_mapping is not None:
        if not isinstance(family_mapping, Mapping):
            raise ValueError("mask.families must be a mapping")
        reserved_family_names = {
            *PRIMARY_MODALITY_GROUPS,
            "teacher_full",
            "student_full",
            "deployment_mix",
        }
        seen_family_names: set[str] = set()
        for family_name, groups in family_mapping.items():
            normalized_name = str(family_name).strip().lower()
            if (
                not normalized_name
                or not normalized_name.replace("_", "").isalnum()
            ):
                raise ValueError(
                    f"mask family name must be alphanumeric/underscore: {family_name!r}"
                )
            if normalized_name.startswith("without_"):
                raise ValueError("mask family names cannot start with 'without_'")
            if normalized_name in reserved_family_names:
                raise ValueError(
                    f"mask family name is reserved or collides with a modality: "
                    f"{normalized_name!r}"
                )
            if normalized_name in seen_family_names:
                raise ValueError(
                    f"mask family names collide after normalization: {normalized_name!r}"
                )
            seen_family_names.add(normalized_name)
            if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
                raise ValueError(
                    f"mask.families.{family_name} must be a sequence of groups"
                )
            validation_families[normalized_name] = tuple(
                str(group) for group in groups
            )
        mask_raw["families"] = tuple(
            tuple(str(group) for group in groups)
            for groups in family_mapping.values()
        )
    mask_config = UniversalMaskConfig(**mask_raw)
    if not validation_families:
        validation_families = {
            f"family_{index}": tuple(groups)
            for index, groups in enumerate(mask_config.families)
        }
    reconstructor_raw = dict(raw.get("reconstructor") or {})
    loss_raw = dict(raw.get("loss") or {})
    unknown_reconstructor = sorted(
        set(reconstructor_raw).difference(
            {"num_heads", "decoder_layers", "ffn_multiplier", "dropout"}
        )
    )
    if unknown_reconstructor:
        raise ValueError(
            f"unknown universal reconstructor keys: {unknown_reconstructor}"
        )
    unknown_loss = sorted(
        set(loss_raw).difference(
            {
                "temperature",
                "supervised_weight",
                "observed_only_supervised_weight",
                "logit_kd_weight",
                "patient_embedding_weight",
                "token_reconstruction_weight",
            }
        )
    )
    if unknown_loss:
        raise ValueError(f"unknown universal loss keys: {unknown_loss}")
    objective = UniversalReconstructionConfig(
        task_name=task,
        d_model=int(d_model),
        num_heads=int(reconstructor_raw.get("num_heads", 4)),
        decoder_layers=int(reconstructor_raw.get("decoder_layers", 2)),
        ffn_multiplier=int(reconstructor_raw.get("ffn_multiplier", 2)),
        dropout=float(reconstructor_raw.get("dropout", 0.10)),
        temperature=float(loss_raw.get("temperature", 2.0)),
        supervised_weight=float(loss_raw.get("supervised_weight", 1.0)),
        observed_only_supervised_weight=float(
            loss_raw.get("observed_only_supervised_weight", 0.5)
        ),
        logit_kd_weight=float(loss_raw.get("logit_kd_weight", 1.0)),
        patient_embedding_weight=float(
            loss_raw.get("patient_embedding_weight", 1.0)
        ),
        token_reconstruction_weight=float(
            loss_raw.get("token_reconstruction_weight", 1.0)
        ),
    )
    training = dict(raw.get("training") or {})
    allowed_training = {
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "grad_clip_norm",
        "early_stopping_patience",
    }
    unknown_training = sorted(set(training).difference(allowed_training))
    if unknown_training:
        raise ValueError(f"unknown universal training keys: {unknown_training}")
    values = {
        "batch_size": int(training.get("batch_size", 128)),
        "max_epochs": int(training.get("max_epochs", 50)),
        "learning_rate": float(training.get("learning_rate", 1e-4)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.10)),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 1)
        ),
        "grad_clip_norm": float(training.get("grad_clip_norm", 1.0)),
        "early_stopping_patience": int(
            training.get("early_stopping_patience", 8)
        ),
    }
    if values["batch_size"] <= 0 or values["max_epochs"] <= 0:
        raise ValueError("universal batch_size/max_epochs must be positive")
    finite_training_values = (
        values["learning_rate"],
        values["weight_decay"],
        values["warmup_ratio"],
        values["grad_clip_norm"],
    )
    if any(not math.isfinite(value) for value in finite_training_values):
        raise ValueError("universal floating-point training values must be finite")
    if values["learning_rate"] <= 0 or values["weight_decay"] < 0:
        raise ValueError("universal optimizer values are invalid")
    if not 0.0 <= values["warmup_ratio"] <= 1.0:
        raise ValueError("universal warmup_ratio must lie in [0, 1]")
    if values["gradient_accumulation_steps"] <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if values["grad_clip_norm"] < 0 or values["early_stopping_patience"] < 0:
        raise ValueError("gradient/patience values cannot be negative")
    selection_raw = dict(raw.get("selection") or {})
    allowed_selection = {
        "full_teacher_auroc_tolerance",
        "deployment_mask_seed",
    }
    unknown_selection = sorted(set(selection_raw).difference(allowed_selection))
    if unknown_selection:
        raise ValueError(f"unknown universal selection keys: {unknown_selection}")
    selection = {
        "full_teacher_auroc_tolerance": float(
            selection_raw.get("full_teacher_auroc_tolerance", 0.02)
        ),
        "deployment_mask_seed": int(
            selection_raw.get("deployment_mask_seed", 20_260_906)
        ),
    }
    if any(
        not math.isfinite(float(value))
        for key, value in selection.items()
        if key != "deployment_mask_seed"
    ):
        raise ValueError("universal selection values must be finite")
    if not 0.0 <= selection["full_teacher_auroc_tolerance"] <= 1.0:
        raise ValueError("full_teacher_auroc_tolerance must lie in [0, 1]")
    if selection["deployment_mask_seed"] < 0:
        raise ValueError("selection.deployment_mask_seed cannot be negative")
    return {
        "path": resolved,
        "sha256": _sha256_file(resolved),
        "config_hash": _canonical_hash(raw),
        "raw": dict(raw),
        "mask": mask_config,
        "validation_families": validation_families,
        "objective": objective,
        "training": values,
        "selection": selection,
    }


def _scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(math.floor(total_steps * warmup_ratio))
    if warmup_ratio > 0 and total_steps:
        warmup = max(1, warmup)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        denominator = max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup + 1) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> Dict[str, float]:
    probabilities = _sigmoid(logits.astype(np.float64))
    metrics = binary_classification_metrics(
        labels.astype(np.int64),
        probabilities=probabilities,
        threshold=float(threshold),
    )
    truth = labels.astype(np.int64)
    predicted = (probabilities >= float(threshold)).astype(np.int64)
    true_positive = int(np.sum((truth == 1) & (predicted == 1)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    false_negative = int(np.sum((truth == 1) & (predicted == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else math.nan
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else math.nan
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0
        else math.nan
    )
    return {
        **metrics,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "true_negative": float(np.sum((truth == 0) & (predicted == 0))),
        "false_negative": float(false_negative),
    }


def _person_mask_seed(global_seed: int, person_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(global_seed)}\0{str(person_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _deterministic_deployment_masks(
    native_present: torch.Tensor,
    person_ids: Sequence[str],
    *,
    sampler: ModalityMaskSampler,
    global_seed: int,
) -> tuple[Dict[str, torch.Tensor], list[Dict[str, Any]]]:
    """Sample deployment masks independently by participant identity.

    This keeps assignments unchanged when validation batch size or row order
    changes and yields a fully auditable per-person masking record.
    """

    if native_present.ndim != 2 or native_present.shape[0] != len(person_ids):
        raise ValueError("native presence and person_ids differ in batch size")
    chunks: Dict[str, list[torch.Tensor]] = {
        "native_present": [],
        "observed_present": [],
        "artificially_dropped": [],
        "requested_reconstruction": [],
        "mask_mode": [],
    }
    audits: list[Dict[str, Any]] = []
    for row_index, person_id in enumerate(person_ids):
        participant_seed = _person_mask_seed(global_seed, str(person_id))
        sampled = sampler.sample(
            native_present[row_index : row_index + 1],
            generator=torch.Generator(device="cpu").manual_seed(participant_seed),
        )
        for key in chunks:
            chunks[key].append(sampled[key])
        native = sampled["native_present"][0].detach().cpu().bool()
        observed = sampled["observed_present"][0].detach().cpu().bool()
        dropped = sampled["artificially_dropped"][0].detach().cpu().bool()
        mode_index = int(sampled["mask_mode"][0].detach().cpu().item())
        audits.append(
            {
                "person_id": str(person_id),
                "global_mask_seed": int(global_seed),
                "participant_mask_seed": participant_seed,
                "mask_mode": ModalityMaskSampler.MODE_NAMES[mode_index],
                "native_groups": "|".join(
                    group
                    for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
                    if bool(native[index])
                ),
                "observed_groups": "|".join(
                    group
                    for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
                    if bool(observed[index])
                ),
                "dropped_groups": "|".join(
                    group
                    for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
                    if bool(dropped[index])
                ),
                "condition_eligible": bool(dropped.any() and observed.any()),
            }
        )
    return (
        {key: torch.cat(values, dim=0) for key, values in chunks.items()},
        audits,
    )


def _validation_selection_record(
    *,
    teacher_full_auroc: float,
    student_full_auroc: float,
    reconstructed_aurocs: Sequence[float],
    reconstruction_gains: Sequence[float],
    fidelity_cosines: Sequence[float],
    selection_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a hard-gated, validation-only checkpoint ranking contract.

    A checkpoint is eligible only if its full-input AUROC remains within the
    predeclared tolerance of the frozen teacher.  Eligible checkpoints are
    ranked lexicographically: absolute reconstructed AUROC is primary;
    fidelity and reconstruction gain are report-level tie breakers only.  In
    particular, lowering the no-reconstruction arm can never compensate for a
    lower absolute reconstructed AUROC.
    """

    config = {
        "full_teacher_auroc_tolerance": 0.02,
        **dict(selection_config or {}),
    }
    unknown = sorted(
        set(config).difference({"full_teacher_auroc_tolerance", "deployment_mask_seed"})
    )
    if unknown:
        raise ValueError(f"unknown validation selection keys: {unknown}")
    tolerance = float(config["full_teacher_auroc_tolerance"])
    scalar_values = [float(teacher_full_auroc), float(student_full_auroc), tolerance]
    reconstructed = [float(value) for value in reconstructed_aurocs]
    gains = [float(value) for value in reconstruction_gains]
    cosines = [float(value) for value in fidelity_cosines]
    if not reconstructed or not gains or not cosines:
        raise ValueError("validation selection requires non-empty protocol values")
    if len(reconstructed) != len(gains):
        raise ValueError("reconstructed AUROC and gain vectors must align")
    if any(
        not math.isfinite(value)
        for value in (*scalar_values, *reconstructed, *gains, *cosines)
    ):
        raise ValueError("validation selection values must be finite")
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("full_teacher_auroc_tolerance must lie in [0, 1]")
    mean_reconstructed_auroc = float(np.mean(reconstructed))
    mean_reconstruction_gain = float(np.mean(gains))
    mean_fidelity_cosine = float(np.mean(cosines))
    full_floor = float(teacher_full_auroc) - tolerance
    eligible = float(student_full_auroc) + 1e-12 >= full_floor
    rank = [
        mean_reconstructed_auroc,
        mean_fidelity_cosine,
        mean_reconstruction_gain,
        float(student_full_auroc),
    ]
    return {
        "config": {
            "full_teacher_auroc_tolerance": tolerance,
            **(
                {"deployment_mask_seed": int(config["deployment_mask_seed"])}
                if "deployment_mask_seed" in config
                else {}
            ),
        },
        "teacher_full_auroc": float(teacher_full_auroc),
        "student_full_auroc": float(student_full_auroc),
        "full_auroc_floor": full_floor,
        "full_auroc_floor_deficit": max(
            0.0, full_floor - float(student_full_auroc)
        ),
        "hard_full_auroc_floor_passed": bool(eligible),
        "checkpoint_eligible": bool(eligible),
        "mean_masked_reconstructed_auroc": mean_reconstructed_auroc,
        "mean_masked_reconstruction_gain_auroc": mean_reconstruction_gain,
        "mean_modality_fidelity_cosine": mean_fidelity_cosine,
        "rank_order": [
            "mean_masked_reconstructed_auroc",
            "mean_modality_fidelity_cosine",
            "mean_masked_reconstruction_gain_auroc",
            "student_full_auroc",
        ],
        "rank": rank,
        "score": mean_reconstructed_auroc,
    }


@torch.no_grad()
def evaluate_validation_panel(
    student: Task2V2UniversalReconstructionStudent,
    dataset: CachedTokenDataset,
    *,
    device: torch.device,
    batch_size: int,
    use_bf16_amp: bool,
    threshold: float = 0.5,
    validation_families: Optional[Mapping[str, Sequence[str]]] = None,
    deployment_mask_sampler: Optional[ModalityMaskSampler] = None,
    deployment_mask_seed: int = 20_260_906,
    selection_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    student.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    protocol_names = ["teacher_full", "student_full"]
    for group in PRIMARY_MODALITY_GROUPS:
        protocol_names.extend((f"without_{group}_no_reconstruction", f"without_{group}_reconstructed"))
    normalized_families = {
        str(name): tuple(str(group) for group in groups)
        for name, groups in dict(validation_families or {}).items()
    }
    for family_name in normalized_families:
        protocol_names.extend(
            (
                f"without_{family_name}_no_reconstruction",
                f"without_{family_name}_reconstructed",
            )
        )
    if deployment_mask_sampler is not None:
        protocol_names.extend(
            ("deployment_mix_no_reconstruction", "deployment_mix_reconstructed")
        )
    logits: Dict[str, list[torch.Tensor]] = {name: [] for name in protocol_names}
    labels: list[torch.Tensor] = []
    native_presence: list[torch.Tensor] = []
    person_ids: list[str] = []
    confidence: Dict[str, list[torch.Tensor]] = {
        group: [] for group in PRIMARY_MODALITY_GROUPS
    }
    fidelity_squared: Dict[str, list[torch.Tensor]] = {
        group: [] for group in PRIMARY_MODALITY_GROUPS
    }
    fidelity_cosine: Dict[str, list[torch.Tensor]] = {
        group: [] for group in PRIMARY_MODALITY_GROUPS
    }
    fidelity_eligible: Dict[str, list[torch.Tensor]] = {
        group: [] for group in PRIMARY_MODALITY_GROUPS
    }
    eligibility: Dict[str, list[torch.Tensor]] = {
        name: [] for name in protocol_names
    }
    deployment_mask_audit: list[Dict[str, Any]] = []

    source_index = {
        group: index for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
    }
    for family_name, groups in normalized_families.items():
        unknown = sorted(set(groups).difference(source_index))
        if unknown:
            raise ValueError(
                f"validation family {family_name!r} has unknown groups: {unknown}"
            )

    for raw in loader:
        batch = recursive_to_device(raw, device, non_blocking=device.type == "cuda")
        tokens = batch["source_tokens"]
        present = batch["source_present"].bool()
        labels.append(batch["y"].float().cpu())
        native_presence.append(present.cpu())
        person_ids.extend(str(value) for value in raw["person_id"])
        logits["teacher_full"].append(batch["teacher_logits"].float().cpu())
        eligibility["teacher_full"].append(torch.ones_like(present[:, 0]).cpu())
        eligibility["student_full"].append(torch.ones_like(present[:, 0]).cpu())
        with _autocast(device, use_bf16_amp):
            full = student.forward_from_source_tokens(
                tokens, present, observed_source_mask=present,
                requested_reconstruction=torch.zeros_like(present),
                inject_reconstruction=False,
            )
        logits["student_full"].append(full["logits"].float().cpu())
        for group_index, group in enumerate(PRIMARY_MODALITY_GROUPS):
            observed = present.clone()
            observed[:, group_index] = False
            requested = torch.zeros_like(present)
            requested[:, group_index] = present[:, group_index]
            with _autocast(device, use_bf16_amp):
                no_reconstruction = student.forward_from_source_tokens(
                    tokens, present, observed_source_mask=observed,
                    requested_reconstruction=requested,
                    inject_reconstruction=False,
                )
                reconstructed = student.forward_from_source_tokens(
                    tokens, present, observed_source_mask=observed,
                    requested_reconstruction=requested,
                    inject_reconstruction=True,
                )
            logits[f"without_{group}_no_reconstruction"].append(
                no_reconstruction["logits"].float().cpu()
            )
            logits[f"without_{group}_reconstructed"].append(
                reconstructed["logits"].float().cpu()
            )
            confidence[group].append(
                reconstructed["reconstruction_confidence"][:, group_index]
                .float().cpu()
            )
            raw_prediction = reconstructed[
                "raw_reconstructed_source_tokens"
            ][:, group_index].float()
            target = tokens[:, group_index].float()
            squared = (raw_prediction - target).square().mean(dim=-1)
            cosine = torch.nn.functional.cosine_similarity(
                raw_prediction, target, dim=-1, eps=1e-8
            )
            eligible = present[:, group_index] & observed.any(dim=1)
            fidelity_squared[group].append(squared.cpu())
            fidelity_cosine[group].append(cosine.cpu())
            fidelity_eligible[group].append(eligible.cpu())
            eligibility[f"without_{group}_no_reconstruction"].append(eligible.cpu())
            eligibility[f"without_{group}_reconstructed"].append(eligible.cpu())

        for family_name, groups in normalized_families.items():
            indices = [source_index[group] for group in groups]
            observed = present.clone()
            observed[:, indices] = False
            requested = torch.zeros_like(present)
            requested[:, indices] = present[:, indices]
            with _autocast(device, use_bf16_amp):
                no_reconstruction = student.forward_from_source_tokens(
                    tokens,
                    present,
                    observed_source_mask=observed,
                    requested_reconstruction=requested,
                    inject_reconstruction=False,
                )
                reconstructed = student.forward_from_source_tokens(
                    tokens,
                    present,
                    observed_source_mask=observed,
                    requested_reconstruction=requested,
                    inject_reconstruction=True,
                )
            no_name = f"without_{family_name}_no_reconstruction"
            rec_name = f"without_{family_name}_reconstructed"
            logits[no_name].append(no_reconstruction["logits"].float().cpu())
            logits[rec_name].append(reconstructed["logits"].float().cpu())
            eligible = requested.any(dim=1) & observed.any(dim=1)
            eligibility[no_name].append(eligible.cpu())
            eligibility[rec_name].append(eligible.cpu())

        if deployment_mask_sampler is not None:
            deployment_masks, batch_audit = _deterministic_deployment_masks(
                present,
                [str(value) for value in raw["person_id"]],
                sampler=deployment_mask_sampler,
                global_seed=int(deployment_mask_seed),
            )
            deployment_mask_audit.extend(batch_audit)
            with _autocast(device, use_bf16_amp):
                no_reconstruction = student.forward_from_source_tokens(
                    tokens,
                    present,
                    observed_source_mask=deployment_masks["observed_present"],
                    requested_reconstruction=deployment_masks[
                        "requested_reconstruction"
                    ],
                    inject_reconstruction=False,
                )
                reconstructed = student.forward_from_source_tokens(
                    tokens,
                    present,
                    observed_source_mask=deployment_masks["observed_present"],
                    requested_reconstruction=deployment_masks[
                        "requested_reconstruction"
                    ],
                    inject_reconstruction=True,
                )
            logits["deployment_mix_no_reconstruction"].append(
                no_reconstruction["logits"].float().cpu()
            )
            logits["deployment_mix_reconstructed"].append(
                reconstructed["logits"].float().cpu()
            )
            eligible = (
                deployment_masks["requested_reconstruction"].any(dim=1)
                & deployment_masks["observed_present"].any(dim=1)
            )
            eligibility["deployment_mix_no_reconstruction"].append(eligible.cpu())
            eligibility["deployment_mix_reconstructed"].append(eligible.cpu())

    y = torch.cat(labels).numpy().astype(np.int64)
    native = torch.cat(native_presence).numpy().astype(bool)
    logit_arrays = {
        name: torch.cat(chunks).numpy().astype(np.float64)
        for name, chunks in logits.items()
    }
    metric_map = {
        name: _metrics(y, values, threshold) for name, values in logit_arrays.items()
    }
    eligibility_arrays = {
        name: torch.cat(chunks).numpy().astype(bool)
        for name, chunks in eligibility.items()
    }
    fidelity: Dict[str, Any] = {}
    confidence_arrays: Dict[str, np.ndarray] = {}
    for group in PRIMARY_MODALITY_GROUPS:
        squared = torch.cat(fidelity_squared[group]).numpy()
        cosine = torch.cat(fidelity_cosine[group]).numpy()
        eligible = torch.cat(fidelity_eligible[group]).numpy().astype(bool)
        confidence_values = torch.cat(confidence[group]).numpy().astype(np.float64)
        confidence_arrays[group] = confidence_values
        fidelity[group] = {
            "n": int(eligible.sum()),
            "mse": float(squared[eligible].mean()) if eligible.any() else math.nan,
            "cosine_similarity": (
                float(cosine[eligible].mean()) if eligible.any() else math.nan
            ),
            "mean_confidence": (
                float(confidence_values[eligible].mean())
                if eligible.any()
                else math.nan
            ),
        }
    masked_pairs = [
        (
            f"without_{group}_no_reconstruction",
            f"without_{group}_reconstructed",
        )
        for group in PRIMARY_MODALITY_GROUPS
    ]
    masked_pairs.extend(
        (
            f"without_{family_name}_no_reconstruction",
            f"without_{family_name}_reconstructed",
        )
        for family_name in normalized_families
    )
    if deployment_mask_sampler is not None:
        masked_pairs.append(
            ("deployment_mix_no_reconstruction", "deployment_mix_reconstructed")
        )
    teacher_full_auroc = float(metric_map["teacher_full"]["auroc"])
    student_full_auroc = float(metric_map["student_full"]["auroc"])
    if not math.isfinite(teacher_full_auroc) or not math.isfinite(
        student_full_auroc
    ):
        raise ValueError("full validation cohort must contain both label classes")
    protocol_details: Dict[str, Any] = {
        "teacher_full": metric_map["teacher_full"],
        "student_full": metric_map["student_full"],
    }
    reconstructed_aurocs: list[float] = []
    reconstruction_gains: list[float] = []
    for no_name, reconstructed_name in masked_pairs:
        selected = eligibility_arrays[reconstructed_name]
        if not selected.any() or np.unique(y[selected]).size != 2:
            raise ValueError(
                "checkpoint-selection protocol lacks both validation classes: "
                f"{reconstructed_name} (n={int(selected.sum())})"
            )
        no_metrics = _metrics(
            y[selected], logit_arrays[no_name][selected], threshold
        )
        reconstructed_metrics = _metrics(
            y[selected], logit_arrays[reconstructed_name][selected], threshold
        )
        no_auroc = float(no_metrics["auroc"])
        reconstructed_auroc = float(reconstructed_metrics["auroc"])
        if not math.isfinite(no_auroc) or not math.isfinite(reconstructed_auroc):
            raise ValueError(
                f"non-finite AUROC in fixed selection pair {no_name!r}/"
                f"{reconstructed_name!r}"
            )
        protocol_details[no_name] = no_metrics
        protocol_details[reconstructed_name] = reconstructed_metrics
        reconstructed_aurocs.append(reconstructed_auroc)
        reconstruction_gains.append(reconstructed_auroc - no_auroc)
    fidelity_cosines = [
        float(fidelity[group]["cosine_similarity"])
        for group in PRIMARY_MODALITY_GROUPS
    ]
    if not reconstructed_aurocs or any(
        not math.isfinite(value) for value in fidelity_cosines
    ):
        raise ValueError(
            "fixed selection panel requires finite masked AUROCs and all nine "
            "modality fidelity cosines"
        )
    selection_details = {
        **_validation_selection_record(
            teacher_full_auroc=teacher_full_auroc,
            student_full_auroc=student_full_auroc,
            reconstructed_aurocs=reconstructed_aurocs,
            reconstruction_gains=reconstruction_gains,
            fidelity_cosines=fidelity_cosines,
            selection_config=selection_config,
        ),
        "protocol_metrics": protocol_details,
        "n_masked_protocol_pairs": len(masked_pairs),
    }
    sorted_mask_audit = sorted(
        deployment_mask_audit, key=lambda row: str(row["person_id"])
    )
    return {
        "person_ids": person_ids,
        "labels": y,
        "native_present": native,
        "logits": logit_arrays,
        "probabilities": {name: _sigmoid(value) for name, value in logit_arrays.items()},
        "metrics": metric_map,
        "fidelity": fidelity,
        "confidence": confidence_arrays,
        "eligibility": eligibility_arrays,
        "validation_families": normalized_families,
        "selection_details": selection_details,
        "selection_score": float(selection_details["score"]),
        "selection_rank": list(selection_details["rank"]),
        "selection_eligible": bool(selection_details["checkpoint_eligible"]),
        "selection_metric": (
            "hard_teacher_full_auroc_floor_then_lexicographic_absolute_"
            "masked_reconstructed_auroc_fidelity_gain"
        ),
        "deployment_mask_seed": int(deployment_mask_seed),
        "deployment_mask_audit": sorted_mask_audit,
        "deployment_mask_audit_hash": (
            _canonical_hash(sorted_mask_audit) if sorted_mask_audit else None
        ),
        "threshold": float(threshold),
    }


def _train_epoch(
    distiller: Task2V2UniversalDistiller,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    device: torch.device,
    use_bf16_amp: bool,
    pos_weight: Optional[torch.Tensor],
    accumulation_steps: int,
    grad_clip_norm: float,
    mask_seed: int,
) -> Dict[str, Any]:
    distiller.train()
    optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(mask_seed))
    component_names = tuple(distiller.COMPONENTS)
    epoch_numerators = {name: 0.0 for name in component_names}
    epoch_denominators = {name: 0 for name in component_names}
    window_numerators: Dict[str, torch.Tensor] = {}
    window_denominators = {name: 0 for name in component_names}
    mask_counts = {name: 0 for name in ModalityMaskSampler.MODE_NAMES.values()}
    target_counts = {group: 0 for group in PRIMARY_MODALITY_GROUPS}
    optimizer_steps = 0
    n_batches = len(loader)
    progress = tqdm(loader, desc="Universal masked training", unit="batch", leave=False)
    for batch_index, raw in enumerate(progress, start=1):
        batch = recursive_to_device(raw, device, non_blocking=device.type == "cuda")
        with _autocast(device, use_bf16_amp):
            output = distiller.forward_cached(
                source_tokens=batch["source_tokens"],
                native_present=batch["source_present"],
                teacher_embedding=batch["teacher_embedding"],
                teacher_logits=batch["teacher_logits"],
                labels=batch["y"],
                generator=generator,
                pos_weight=pos_weight,
            )
        for name in component_names:
            numerator = output["component_numerators"][name]
            denominator = int(output["component_denominators"][name])
            window_numerators[name] = (
                numerator
                if name not in window_numerators
                else window_numerators[name] + numerator
            )
            window_denominators[name] += denominator
            epoch_numerators[name] += float(numerator.detach().float().cpu())
            epoch_denominators[name] += denominator
        for name, count in output["mask_mode_counts"].items():
            mask_counts[name] += int(count)
        for group, record in output["reconstruction_by_modality"].items():
            target_counts[group] += int(record["vectors"])
        boundary = batch_index % accumulation_steps == 0 or batch_index == n_batches
        if boundary:
            terms = [
                window_numerators[name]
                / float(window_denominators[name])
                * weight
                for name, weight in distiller.component_weights.items()
                if weight > 0 and window_denominators[name] > 0
            ]
            if not terms:
                raise ValueError("gradient accumulation window has no eligible targets")
            loss = torch.stack(terms).sum()
            if not bool(torch.isfinite(loss.float())):
                raise FloatingPointError("universal reconstruction loss is non-finite")
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in distiller.student.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            window_numerators = {}
            window_denominators = {name: 0 for name in component_names}

    means = {
        name: (
            epoch_numerators[name] / float(epoch_denominators[name])
            if epoch_denominators[name]
            else 0.0
        )
        for name in component_names
    }
    return {
        "loss": float(
            sum(distiller.component_weights[name] * means[name] for name in component_names)
        ),
        "component_losses": means,
        "component_denominators": epoch_denominators,
        "mask_mode_counts": mask_counts,
        "reconstruction_target_counts": target_counts,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def compute_relation_matrix(
    student: Task2V2UniversalReconstructionStudent,
    dataset: CachedTokenDataset,
    *,
    device: torch.device,
    batch_size: int,
    use_bf16_amp: bool,
) -> list[Dict[str, Any]]:
    """Measure directed source->target reconstruction dependence on validation."""

    student.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    accumulators: Dict[tuple[int, int], Dict[str, float]] = {
        (source, target): {
            "n": 0.0,
            "base_squared": 0.0,
            "without_squared": 0.0,
            "base_cosine": 0.0,
            "without_cosine": 0.0,
            "probability_change": 0.0,
            "base_confidence": 0.0,
            "without_confidence": 0.0,
        }
        for source in range(len(PRIMARY_MODALITY_GROUPS))
        for target in range(len(PRIMARY_MODALITY_GROUPS))
        if source != target
    }
    for raw in tqdm(loader, desc="Validation relation matrix", unit="batch"):
        batch = recursive_to_device(raw, device, non_blocking=device.type == "cuda")
        tokens = batch["source_tokens"]
        present = batch["source_present"].bool()
        for target in range(len(PRIMARY_MODALITY_GROUPS)):
            base_observed = present.clone()
            base_observed[:, target] = False
            requested = torch.zeros_like(present)
            requested[:, target] = present[:, target]
            with _autocast(device, use_bf16_amp):
                base = student.forward_from_source_tokens(
                    tokens,
                    present,
                    observed_source_mask=base_observed,
                    requested_reconstruction=requested,
                    inject_reconstruction=True,
                )
            base_raw = base["raw_reconstructed_source_tokens"][:, target].float()
            target_token = tokens[:, target].float()
            base_squared = (base_raw - target_token).square().mean(dim=-1)
            base_cosine = torch.nn.functional.cosine_similarity(
                base_raw, target_token, dim=-1, eps=1e-8
            )
            base_probability = torch.sigmoid(base["logits"].float())
            base_confidence = base["reconstruction_confidence"][:, target].float()
            for source in range(len(PRIMARY_MODALITY_GROUPS)):
                if source == target:
                    continue
                without_observed = base_observed.clone()
                without_observed[:, source] = False
                eligible = (
                    present[:, target]
                    & present[:, source]
                    & without_observed.any(dim=1)
                )
                count = int(eligible.sum().item())
                if not count:
                    continue
                with _autocast(device, use_bf16_amp):
                    without = student.forward_from_source_tokens(
                        tokens,
                        present,
                        observed_source_mask=without_observed,
                        requested_reconstruction=requested,
                        inject_reconstruction=True,
                    )
                without_raw = without[
                    "raw_reconstructed_source_tokens"
                ][:, target].float()
                without_squared = (without_raw - target_token).square().mean(dim=-1)
                without_cosine = torch.nn.functional.cosine_similarity(
                    without_raw, target_token, dim=-1, eps=1e-8
                )
                without_probability = torch.sigmoid(without["logits"].float())
                without_confidence = without[
                    "reconstruction_confidence"
                ][:, target].float()
                record = accumulators[(source, target)]
                record["n"] += count
                record["base_squared"] += float(base_squared[eligible].sum().cpu())
                record["without_squared"] += float(
                    without_squared[eligible].sum().cpu()
                )
                record["base_cosine"] += float(base_cosine[eligible].sum().cpu())
                record["without_cosine"] += float(
                    without_cosine[eligible].sum().cpu()
                )
                record["probability_change"] += float(
                    (without_probability[eligible] - base_probability[eligible])
                    .abs().sum().cpu()
                )
                record["base_confidence"] += float(
                    base_confidence[eligible].sum().cpu()
                )
                record["without_confidence"] += float(
                    without_confidence[eligible].sum().cpu()
                )
    rows: list[Dict[str, Any]] = []
    for (source, target), record in accumulators.items():
        n = int(record["n"])
        base_mse = record["base_squared"] / n if n else math.nan
        without_mse = record["without_squared"] / n if n else math.nan
        base_cosine = record["base_cosine"] / n if n else math.nan
        without_cosine = record["without_cosine"] / n if n else math.nan
        rows.append(
            {
                "source_group": PRIMARY_MODALITY_GROUPS[source],
                "target_group": PRIMARY_MODALITY_GROUPS[target],
                "n_source_and_target_observed": n,
                "target_mse_with_source": base_mse,
                "target_mse_without_source": without_mse,
                "delta_mse_without_minus_with": (
                    without_mse - base_mse if n else math.nan
                ),
                "target_cosine_with_source": base_cosine,
                "target_cosine_without_source": without_cosine,
                "delta_cosine_with_minus_without": (
                    base_cosine - without_cosine if n else math.nan
                ),
                "mean_abs_probability_change_when_source_removed_from_both_"
                "reconstruction_memory_and_prediction_fusion": (
                    record["probability_change"] / n if n else math.nan
                ),
                "mean_confidence_with_source": (
                    record["base_confidence"] / n if n else math.nan
                ),
                "mean_confidence_without_source": (
                    record["without_confidence"] / n if n else math.nan
                ),
                "interpretation": (
                    "positive delta_mse isolates help reconstructing target; the "
                    "probability-change column also includes source direct prediction"
                ),
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _panel_metric_rows(panel: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for protocol, all_metrics in panel["metrics"].items():
        eligible = panel["eligibility"][protocol].astype(bool)
        for scope, selected, metrics in (
            (
                "all_validation",
                np.ones(len(panel["labels"]), dtype=bool),
                all_metrics,
            ),
            (
                "condition_eligible",
                eligible,
                (
                    _metrics(
                        panel["labels"][eligible],
                        panel["logits"][protocol][eligible],
                        panel["threshold"],
                    )
                    if eligible.any()
                    else None
                ),
            ),
        ):
            if metrics is None:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "cohort_scope": scope,
                    "n_evaluated": int(selected.sum()),
                    "threshold_origin": "universal_student_validation_youden",
                    **{key: float(value) for key, value in metrics.items()},
                }
            )
    return rows


def _importance_rows(panel: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    metric_names = (
        "auroc",
        "auprc",
        "log_loss",
        "brier",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    )
    for group_index, group in enumerate(PRIMARY_MODALITY_GROUPS):
        no_name = f"without_{group}_no_reconstruction"
        rec_name = f"without_{group}_reconstructed"
        present = panel["native_present"][:, group_index].astype(bool)
        scopes = (
            ("all_validation", np.ones(len(panel["labels"]), dtype=bool)),
            ("native_present_only", present),
        )
        for scope_name, selected in scopes:
            if not selected.any():
                continue
            scoped_metrics = {
                protocol: _metrics(
                    panel["labels"][selected],
                    panel["logits"][protocol][selected],
                    panel["threshold"],
                )
                for protocol in (
                    "teacher_full",
                    "student_full",
                    no_name,
                    rec_name,
                )
            }
            full = scoped_metrics["student_full"]
            teacher = scoped_metrics["teacher_full"]
            no_metrics = scoped_metrics[no_name]
            rec_metrics = scoped_metrics[rec_name]
            base: Dict[str, Any] = {
                "modality": group,
                "cohort_scope": scope_name,
                "n_evaluated": int(selected.sum()),
                "n_validation": len(panel["labels"]),
                "n_native_present": int(present.sum()),
                "native_present_fraction": float(present.mean()),
                "latent_reconstruction_mse": panel["fidelity"][group]["mse"],
                "latent_reconstruction_cosine": panel["fidelity"][group][
                    "cosine_similarity"
                ],
                "mean_reconstruction_confidence": panel["fidelity"][group][
                    "mean_confidence"
                ],
            }
            for metric in metric_names:
                higher_is_better = metric not in {"log_loss", "brier"}
                full_value = float(full[metric])
                no_value = float(no_metrics[metric])
                rec_value = float(rec_metrics[metric])
                teacher_value = float(teacher[metric])
                if higher_is_better:
                    removal_harm = full_value - no_value
                    gain = rec_value - no_value
                    remaining = full_value - rec_value
                    teacher_gap = teacher_value - rec_value
                else:
                    removal_harm = no_value - full_value
                    gain = no_value - rec_value
                    remaining = rec_value - full_value
                    teacher_gap = rec_value - teacher_value
                base[f"student_full_{metric}"] = full_value
                base[f"removed_no_reconstruction_{metric}"] = no_value
                base[f"removed_reconstructed_{metric}"] = rec_value
                base[f"teacher_full_{metric}_at_student_threshold"] = teacher_value
                base[f"removal_harm_{metric}"] = removal_harm
                base[f"reconstruction_gain_{metric}"] = gain
                base[f"remaining_student_gap_{metric}"] = remaining
                base[
                    f"remaining_teacher_gap_{metric}_at_student_threshold"
                ] = teacher_gap
                base[f"recovery_fraction_{metric}"] = (
                    gain / removal_harm if removal_harm > 1e-8 else math.nan
                )
            rows.append(base)
    return rows


def _local_importance_rows(panel: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    full_probability = panel["probabilities"]["student_full"]
    full_logits = panel["logits"]["student_full"]
    for group_index, group in enumerate(PRIMARY_MODALITY_GROUPS):
        no_probability = panel["probabilities"][
            f"without_{group}_no_reconstruction"
        ]
        no_logits = panel["logits"][f"without_{group}_no_reconstruction"]
        rec_probability = panel["probabilities"][f"without_{group}_reconstructed"]
        rec_logits = panel["logits"][f"without_{group}_reconstructed"]
        for index, person_id in enumerate(panel["person_ids"]):
            native = bool(panel["native_present"][index, group_index])
            rows.append(
                {
                    "person_id": person_id,
                    "modality": group,
                    "native_present": native,
                    "y_true": int(panel["labels"][index]),
                    "student_full_probability": float(full_probability[index]),
                    "removed_no_reconstruction_probability": float(
                        no_probability[index]
                    ),
                    "removed_reconstructed_probability": float(rec_probability[index]),
                    "signed_probability_change_on_removal": (
                        float(full_probability[index] - no_probability[index])
                        if native
                        else math.nan
                    ),
                    "absolute_probability_change_on_removal": (
                        float(abs(full_probability[index] - no_probability[index]))
                        if native
                        else math.nan
                    ),
                    "signed_logit_change_on_removal": (
                        float(full_logits[index] - no_logits[index])
                        if native
                        else math.nan
                    ),
                    "probability_recovered_by_reconstruction": (
                        float(rec_probability[index] - no_probability[index])
                        if native
                        else math.nan
                    ),
                    "logit_recovered_by_reconstruction": (
                        float(rec_logits[index] - no_logits[index])
                        if native
                        else math.nan
                    ),
                    "reconstruction_confidence": (
                        float(panel["confidence"][group][index])
                        if native
                        else math.nan
                    ),
                }
            )
    return rows


def _prediction_rows(panel: Mapping[str, Any], *, task: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for protocol, values in panel["probabilities"].items():
        for index, person_id in enumerate(panel["person_ids"]):
            rows.append(
                {
                    "person_id": person_id,
                    "task": task,
                    "split": "val",
                    "protocol": protocol,
                    "y_true": int(panel["labels"][index]),
                    "y_logit": float(panel["logits"][protocol][index]),
                    "y_probability": float(values[index]),
                    "threshold": float(panel["threshold"]),
                    "threshold_origin": "universal_student_validation_youden",
                }
            )
    return rows


def _state_payload(
    *,
    kind: str,
    epoch: int,
    student: Task2V2UniversalReconstructionStudent,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: Sequence[Mapping[str, Any]],
    training_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": TRAINING_FORMAT_VERSION,
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "task": student.config.task_name,
        "base_model_state": student.base_model.state_dict(),
        "reconstructor_state": student.reconstructor.state_dict(),
        "universal_reconstruction_config": _json_safe(student.config),
        "training_config": _json_safe(training_config),
        "provenance": _json_safe(provenance),
        "validation_selection_score": float(validation["selection_score"]),
        "validation_selection_rank": _json_safe(validation["selection_rank"]),
        "validation_selection_eligible": bool(
            validation["selection_eligible"]
        ),
        "validation_selection_metric": validation["selection_metric"],
        "validation_selection_details": _json_safe(
            validation["selection_details"]
        ),
        "validation_deployment_mask_seed": validation["deployment_mask_seed"],
        "validation_deployment_mask_audit_hash": validation[
            "deployment_mask_audit_hash"
        ],
        "validation_metrics": _json_safe(validation["metrics"]),
        "validation_fidelity": _json_safe(validation["fidelity"]),
        "history": _json_safe(history),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "test_evaluated": False,
    }


def train_universal(args: argparse.Namespace) -> Mapping[str, Any]:
    device = resolve_device(args.device)
    set_deterministic_seed(int(args.seed), deterministic=True)
    config_path, run_config = load_supervised_config(args.config)
    if int(args.seed) != int(run_config.seed):
        raise ValueError(
            "pilot seed must match the full teacher YAML; create a seed-specific teacher first"
        )
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    # Strict teacher/data/config validation is intentionally repeated even
    # though optimization uses cached tokens.
    (
        _resolved,
        _run,
        datasets,
        teacher,
        _loaders,
        context,
    ) = _teacher_context(
        config_path=config_path,
        teacher_checkpoint=args.teacher_checkpoint,
        device=torch.device("cpu"),
        workers=0,
        batch_size=1,
        split_view_dir=cache_dir / "_master_views",
    )
    task = run_config.dataset.task
    teacher_sha = _sha256_file(args.teacher_checkpoint)
    train_path = cache_dir / f"{task}_train_teacher_tokens.pt"
    val_path = cache_dir / f"{task}_val_teacher_tokens.pt"
    train_payload = _validate_cache(
        train_path,
        split="train",
        task=task,
        teacher_sha256=teacher_sha,
        config_hash=run_config.config_hash,
        expected_d_model=int(context["model_config"].d_model),
        expected_person_labels=_dataset_person_labels(datasets.train),
    )
    val_payload = _validate_cache(
        val_path,
        split="val",
        task=task,
        teacher_sha256=teacher_sha,
        config_hash=run_config.config_hash,
        expected_d_model=int(context["model_config"].d_model),
        expected_person_labels=_dataset_person_labels(datasets.validation),
    )
    train_metadata = train_payload.get("metadata")
    val_metadata = val_payload.get("metadata")
    if not isinstance(train_metadata, Mapping) or not isinstance(
        val_metadata, Mapping
    ):
        raise ValueError("token caches are missing metadata")
    if train_metadata.get("cache_config_hash") != val_metadata.get(
        "cache_config_hash"
    ):
        raise ValueError("train and validation token caches use different contracts")
    if train_metadata.get("encoding_contract") != val_metadata.get(
        "encoding_contract"
    ):
        raise ValueError("train and validation token encoding contracts differ")
    encoding_contract = train_metadata.get("encoding_contract")
    if not isinstance(encoding_contract, Mapping) or encoding_contract.get(
        "runtime_code_sha256"
    ) != _runtime_code_sha256():
        raise ValueError(
            "token caches were generated by a different runtime code bundle; "
            "rebuild both cache splits"
        )
    train_dataset = CachedTokenDataset(train_payload)
    val_dataset = CachedTokenDataset(val_payload)
    if set(train_dataset.person_ids).intersection(val_dataset.person_ids):
        raise ValueError("train and validation token caches overlap")
    universal = _load_universal_config(
        args.universal_config,
        task=task,
        d_model=int(context["model_config"].d_model),
    )
    training = dict(universal["training"])
    if args.max_epochs is not None:
        training["max_epochs"] = int(args.max_epochs)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None:
        training["learning_rate"] = float(args.learning_rate)
    if args.early_stopping_patience is not None:
        training["early_stopping_patience"] = int(args.early_stopping_patience)
    if training["batch_size"] <= 0 or training["max_epochs"] <= 0:
        raise ValueError("overridden batch_size/max_epochs must be positive")
    if not math.isfinite(float(training["learning_rate"])):
        raise ValueError("overridden learning_rate must be finite")
    if training["learning_rate"] <= 0:
        raise ValueError("overridden learning_rate must be positive")
    if training["early_stopping_patience"] < 0:
        raise ValueError("overridden early_stopping_patience cannot be negative")

    # The universal pilot deliberately keeps the teacher token space fixed.
    # Native encoders are cloned for later raw-input deployment, while only
    # fusion, the selected task head, and the new reconstructor are optimized.
    student_base = copy.deepcopy(teacher)
    student_base.requires_grad_(False)
    student_base.fusion.requires_grad_(True)
    student_base.heads[task].requires_grad_(True)
    reconstructor = TargetQueryCrossModalReconstructor(universal["objective"])
    student = Task2V2UniversalReconstructionStudent(
        student_base, reconstructor, universal["objective"]
    )
    distiller = Task2V2UniversalDistiller(
        teacher,
        student,
        ModalityMaskSampler(universal["mask"]),
    )
    validation_mask_sampler = ModalityMaskSampler(universal["mask"])
    student.to(device)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("universal student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 10_000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / int(training["gradient_accumulation_steps"])
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=steps_per_epoch * int(training["max_epochs"]),
        warmup_ratio=float(training["warmup_ratio"]),
    )
    labels = train_dataset.labels
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if not positives or not negatives:
        raise ValueError("training cache must contain both classes")
    pos_weight = torch.tensor(
        negatives / positives, device=device, dtype=torch.float32
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        output / "best_universal_training.pt",
        output / "last_universal_training.pt",
        output / "deployment_universal_student.pt",
        output / "run_summary.json",
    )
    if not args.dry_run and not args.overwrite_output:
        existing = [str(path) for path in protected_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "training output already exists; use a new --output-dir or pass "
                f"--overwrite-output explicitly: {existing}"
            )
    provenance = {
        "task": task,
        "seed": int(args.seed),
        "teacher_checkpoint": {
            "path": str(Path(args.teacher_checkpoint).expanduser().resolve()),
            "sha256": teacher_sha,
        },
        "supervised_config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "config_hash": run_config.config_hash,
        },
        "universal_config": {
            "path": str(universal["path"]),
            "sha256": universal["sha256"],
            "config_hash": universal["config_hash"],
        },
        "token_caches": {
            "train": {"path": str(train_path), "sha256": _sha256_file(train_path)},
            "val": {"path": str(val_path), "sha256": _sha256_file(val_path)},
        },
        "resolved_model_config": _json_safe(context["model_config"]),
        "teacher_audit": context["teacher_audit"],
        "teacher_validation_screen_contract": context[
            "teacher_validation_screen_contract"
        ],
        "train_validation_master_view": context["master_view_record"],
        "runtime_code_sha256": _runtime_code_sha256(),
        "optimization_splits": ["train"],
        "selection_splits": ["val"],
        "test_dataset_constructed": False,
        "test_evaluated": False,
    }

    if args.dry_run:
        raw = next(iter(train_loader))
        batch = recursive_to_device(raw, device)
        with _autocast(device, args.bf16):
            result = distiller.forward_cached(
                source_tokens=batch["source_tokens"],
                native_present=batch["source_present"],
                teacher_embedding=batch["teacher_embedding"],
                teacher_logits=batch["teacher_logits"],
                labels=batch["y"],
                generator=torch.Generator(device="cpu").manual_seed(int(args.seed)),
                pos_weight=pos_weight,
            )
        result["loss"].backward()
        summary = {
            "status": "dry_run_ok",
            "task": task,
            "seed": int(args.seed),
            "loss": float(result["loss"].detach().float().cpu()),
            "component_losses": _json_safe(result["component_losses"]),
            "mask_mode_counts": result["mask_mode_counts"],
            "trainable_parameters": int(sum(p.numel() for p in parameters)),
            "train_cache_n": len(train_dataset),
            "val_cache_n": len(val_dataset),
            "test_evaluated": False,
        }
        _atomic_json(summary, output / "dry_run_summary.json")
        return summary

    history: list[Dict[str, Any]] = []
    initial_panel = evaluate_validation_panel(
        student,
        val_dataset,
        device=device,
        batch_size=int(training["batch_size"]),
        use_bf16_amp=args.bf16,
        threshold=0.5,
        validation_families=universal["validation_families"],
        deployment_mask_sampler=validation_mask_sampler,
        deployment_mask_seed=int(universal["selection"]["deployment_mask_seed"]),
        selection_config=universal["selection"],
    )
    if not bool(initial_panel["selection_eligible"]):
        raise RuntimeError(
            "the freshly cloned frozen teacher failed its own full-input AUROC "
            "floor; cached tokens or numerical/runtime lineage is inconsistent"
        )
    best_score: Optional[float] = float(initial_panel["selection_score"])
    best_rank: Optional[tuple[float, ...]] = tuple(
        float(value) for value in initial_panel["selection_rank"]
    )
    best_epoch = 0
    no_improvement = 0
    best_path = output / "best_universal_training.pt"
    last_path = output / "last_universal_training.pt"
    initial_payload = _state_payload(
        kind="best",
        epoch=0,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        training_config={
            **training,
            "seed": int(args.seed),
            "bf16": bool(args.bf16),
            "mask": _json_safe(universal["mask"]),
            "selection": _json_safe(universal["selection"]),
        },
        provenance=provenance,
        validation=initial_panel,
    )
    atomic_torch_save(initial_payload, best_path)
    print(
        json.dumps(
            {
                "epoch": 0,
                "role": "frozen_teacher_clone_initialization",
                "validation_score": best_score,
                "validation_rank": list(best_rank),
                "validation_checkpoint_eligible": True,
                "student_full_auroc": initial_panel["metrics"]["student_full"][
                    "auroc"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_metrics = _train_epoch(
            distiller,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            use_bf16_amp=args.bf16,
            pos_weight=pos_weight,
            accumulation_steps=int(training["gradient_accumulation_steps"]),
            grad_clip_norm=float(training["grad_clip_norm"]),
            mask_seed=int(args.seed) * 1_000_003 + epoch,
        )
        panel = evaluate_validation_panel(
            student,
            val_dataset,
            device=device,
            batch_size=int(training["batch_size"]),
            use_bf16_amp=args.bf16,
            threshold=0.5,
            validation_families=universal["validation_families"],
            deployment_mask_sampler=validation_mask_sampler,
            deployment_mask_seed=int(
                universal["selection"]["deployment_mask_seed"]
            ),
            selection_config=universal["selection"],
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_selection_score": panel["selection_score"],
            "validation_selection_rank": panel["selection_rank"],
            "validation_selection_eligible": panel["selection_eligible"],
            "validation_selection_metric": panel["selection_metric"],
            "validation_student_full": panel["metrics"]["student_full"],
            "validation_reconstruction_fidelity": panel["fidelity"],
        }
        history.append(epoch_record)
        payload = _state_payload(
            kind="last",
            epoch=epoch,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            training_config={
                **training,
                "seed": int(args.seed),
                "bf16": bool(args.bf16),
                "mask": _json_safe(universal["mask"]),
                "selection": _json_safe(universal["selection"]),
            },
            provenance=provenance,
            validation=panel,
        )
        atomic_torch_save(payload, last_path)
        score = float(panel["selection_score"])
        rank = tuple(float(value) for value in panel["selection_rank"])
        eligible = bool(panel["selection_eligible"])
        improved = eligible and (best_rank is None or rank > best_rank)
        if improved:
            best_score = score
            best_rank = rank
            best_epoch = epoch
            no_improvement = 0
            payload["checkpoint_kind"] = "best"
            atomic_torch_save(payload, best_path)
        elif best_rank is not None:
            no_improvement += 1
        else:
            # Do not early-stop before any checkpoint satisfies the hard
            # teacher-relative full-input validation floor.
            no_improvement = 0
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_score": score,
                    "validation_rank": list(rank),
                    "validation_checkpoint_eligible": eligible,
                    "student_full_auroc": panel["metrics"]["student_full"]["auroc"],
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "no_improvement": no_improvement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if no_improvement >= int(training["early_stopping_patience"]):
            break
    if not best_path.is_file() or best_score is None or best_rank is None:
        raise RuntimeError(
            "universal training produced no checkpoint satisfying the hard "
            "teacher-relative full-input validation AUROC floor; inspect "
            "last_universal_training.pt and revise on validation only"
        )

    best = _safe_torch_load(best_path)
    student.base_model.load_state_dict(best["base_model_state"], strict=True)
    student.reconstructor.load_state_dict(best["reconstructor_state"], strict=True)
    student.to(device)
    preliminary = evaluate_validation_panel(
        student,
        val_dataset,
        device=device,
        batch_size=int(training["batch_size"]),
        use_bf16_amp=args.bf16,
        threshold=0.5,
        validation_families=universal["validation_families"],
        deployment_mask_sampler=validation_mask_sampler,
        deployment_mask_seed=int(universal["selection"]["deployment_mask_seed"]),
        selection_config=universal["selection"],
    )
    threshold_selection = select_validation_threshold(
        preliminary["labels"],
        preliminary["probabilities"]["student_full"],
        method="youden",
        fixed_threshold=0.5,
        split="val",
    )
    panel = evaluate_validation_panel(
        student,
        val_dataset,
        device=device,
        batch_size=int(training["batch_size"]),
        use_bf16_amp=args.bf16,
        threshold=threshold_selection.threshold,
        validation_families=universal["validation_families"],
        deployment_mask_sampler=validation_mask_sampler,
        deployment_mask_seed=int(universal["selection"]["deployment_mask_seed"]),
        selection_config=universal["selection"],
    )
    importance = _importance_rows(panel)
    local_importance = _local_importance_rows(panel)
    relation = compute_relation_matrix(
        student,
        val_dataset,
        device=device,
        batch_size=int(training["batch_size"]),
        use_bf16_amp=args.bf16,
    )
    importance_path = output / "validation_modality_importance.csv"
    local_path = output / "validation_local_importance.csv"
    relation_path = output / "validation_source_target_relations.csv"
    predictions_path = output / "validation_panel_predictions.csv"
    panel_metrics_path = output / "validation_subset_panel_metrics.csv"
    deployment_mask_audit_path = output / "validation_deployment_mask_audit.csv"
    _write_csv(importance, importance_path)
    _write_csv(local_importance, local_path)
    _write_csv(relation, relation_path)
    _write_csv(_prediction_rows(panel, task=task), predictions_path)
    _write_csv(_panel_metric_rows(panel), panel_metrics_path)
    _write_csv(panel["deployment_mask_audit"], deployment_mask_audit_path)

    deployment = {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "checkpoint_kind": "universal_reconstruction_student",
        "model_role": "universal_dynamic_subset_reconstruction",
        "task": task,
        "seed": int(args.seed),
        "base_model_state": student.base_model.state_dict(),
        "reconstructor_state": student.reconstructor.state_dict(),
        "resolved_model_config": _json_safe(context["model_config"]),
        "universal_reconstruction_config": _json_safe(student.config),
        "mask_config": _json_safe(universal["mask"]),
        "selection_config": _json_safe(universal["selection"]),
        "native_encoder_groups": list(PRIMARY_MODALITY_GROUPS),
        "runtime_observed_groups_required": True,
        "supported_inference_modes": [
            "dynamic_subset_observed_only",
            "dynamic_subset_latent_reconstruction",
        ],
        "teacher_required": False,
        "raw_modality_reconstruction": False,
        "latent_target_space": "full_teacher_native_source_token_pre_fusion",
        "selected_threshold": _json_safe(threshold_selection),
        "validation_deployment_mask_seed": panel["deployment_mask_seed"],
        "validation_deployment_mask_audit_hash": panel[
            "deployment_mask_audit_hash"
        ],
        "validation_selection_details": _json_safe(panel["selection_details"]),
        "best_epoch": best_epoch,
        "provenance": provenance,
        "test_evaluated": False,
    }
    deployment_path = output / "deployment_universal_student.pt"
    atomic_torch_save(deployment, deployment_path)
    deployment_record = {
        "path": str(deployment_path),
        "sha256": _sha256_file(deployment_path),
        "size_bytes": int(deployment_path.stat().st_size),
    }
    summary = {
        "format_version": TRAINING_FORMAT_VERSION,
        "status": "complete",
        "task": task,
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_rank": list(best_rank),
        "selected_checkpoint_passed_hard_full_auroc_floor": True,
        "selection_metric": panel["selection_metric"],
        "epochs_completed": len(history),
        "initial_teacher_clone_validation_selection_details": initial_panel[
            "selection_details"
        ],
        "selected_threshold": _json_safe(threshold_selection),
        "teacher_full_validation_metrics_at_student_threshold": panel["metrics"][
            "teacher_full"
        ],
        "student_full_validation_metrics": panel["metrics"]["student_full"],
        "validation_reconstruction_fidelity": panel["fidelity"],
        "validation_selection_details": panel["selection_details"],
        "validation_deployment_mask_seed": panel["deployment_mask_seed"],
        "validation_deployment_mask_audit_hash": panel[
            "deployment_mask_audit_hash"
        ],
        "artifacts": {
            "best_training_checkpoint": str(best_path),
            "last_training_checkpoint": str(last_path),
            "deployment_checkpoint": deployment_record,
            "modality_importance": str(importance_path),
            "local_importance": str(local_path),
            "source_target_relations": str(relation_path),
            "validation_predictions": str(predictions_path),
            "subset_panel_metrics": str(panel_metrics_path),
            "deployment_mask_audit": str(deployment_mask_audit_path),
        },
        "history": history,
        "provenance": provenance,
        "test_evaluated": False,
        "notes": [
            "Single-seed validation-only exploratory pilot.",
            "Importance is task-specific model reliance, not causal clinical importance.",
            "Reconstruction predicts latent tokens, never raw medical measurements.",
            "Teacher thresholded metrics in this run use the universal student's "
            "validation-selected threshold for paired protocol comparison; the "
            "locked teacher report remains the authoritative teacher result.",
        ],
    }
    _atomic_json(summary, output / "run_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache", help="Cache full-teacher train/val tokens")
    cache.add_argument("--config", required=True, type=Path)
    cache.add_argument("--teacher-checkpoint", required=True, type=Path)
    cache.add_argument("--cache-dir", required=True, type=Path)
    cache.add_argument("--seed", type=int, default=42)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--workers", type=int, default=2)
    cache.add_argument("--batch-size", type=int)
    cache.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    cache.add_argument("--reuse-existing", action="store_true")
    cache.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Train universal token reconstructor")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--universal-config", required=True, type=Path)
    train.add_argument("--teacher-checkpoint", required=True, type=Path)
    train.add_argument("--cache-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--early-stopping-patience", type=int)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--overwrite-output", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_token_cache(args) if args.command == "cache" else train_universal(args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CachedTokenDataset",
    "DEPLOYMENT_FORMAT_VERSION",
    "TOKEN_CACHE_FORMAT_VERSION",
    "TRAINING_FORMAT_VERSION",
    "build_arg_parser",
    "build_token_cache",
    "compute_relation_matrix",
    "evaluate_validation_panel",
    "main",
    "train_universal",
    "_validation_selection_record",
    "_write_train_val_master_view",
]
