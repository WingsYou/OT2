#!/usr/bin/env python3
"""Evaluate one universal reconstruction student on validation raw inputs.

V1 is intentionally validation-only.  It accepts a static set of modalities
that would be available at deployment, constructs the OT2 Dataset with exactly
that acquisition set, and runs either the observed-only student or the same
student with latent reconstruction for the disabled modality groups.

The evaluator never selects a threshold and never opens the test split.  It
uses the validation-derived threshold embedded in
``deployment_universal_student.pt`` verbatim.
"""
from __future__ import annotations

import argparse
import contextlib
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
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements-model-v2.
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore
        return iterable

try:
    from ot2.task2_v2_dataset import (
        SOURCE_FLAG_BY_GROUP,
        OT2V2Dataset,
        task2_v2_collate,
    )
    from ot2.task2_v2_engine import (
        binary_classification_metrics,
        recursive_to_device,
        set_deterministic_seed,
    )
    from ot2.task2_v2_model import (
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )
    from ot2.task2_v2_universal_reconstruction import (
        TargetQueryCrossModalReconstructor,
        Task2V2UniversalReconstructionStudent,
        UniversalReconstructionConfig,
    )
    from ot2.train_task2_v2 import (
        build_model_config,
        load_supervised_config,
        resolve_device,
    )
    from ot2.train_universal_reconstruction_v2 import (
        DEPLOYMENT_FORMAT_VERSION,
        _runtime_code_sha256,
    )
except ModuleNotFoundError:  # Direct execution from ``ot2/``.
    from task2_v2_dataset import (  # type: ignore
        SOURCE_FLAG_BY_GROUP,
        OT2V2Dataset,
        task2_v2_collate,
    )
    from task2_v2_engine import (  # type: ignore
        binary_classification_metrics,
        recursive_to_device,
        set_deterministic_seed,
    )
    from task2_v2_model import (  # type: ignore
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )
    from task2_v2_universal_reconstruction import (  # type: ignore
        TargetQueryCrossModalReconstructor,
        Task2V2UniversalReconstructionStudent,
        UniversalReconstructionConfig,
    )
    from train_task2_v2 import (  # type: ignore
        build_model_config,
        load_supervised_config,
        resolve_device,
    )
    from train_universal_reconstruction_v2 import (  # type: ignore
        DEPLOYMENT_FORMAT_VERSION,
        _runtime_code_sha256,
    )


RAW_VALIDATION_EVALUATION_FORMAT_VERSION = (
    "ot2-universal-raw-validation-evaluation-v1"
)
RECONSTRUCTION_MODES = frozenset({"off", "on"})
RECONSTRUCTION_TARGET_POLICIES = frozenset(
    {"declared-disabled", "all-disabled-exploratory"}
)


@dataclasses.dataclass(frozen=True)
class UniversalDeploymentContract:
    """Strict, typed subset of the self-contained deployment checkpoint."""

    task: str
    seed: int
    threshold: float
    threshold_record: Mapping[str, Any]
    model_config: Task2V2ModelConfig
    reconstruction_config: UniversalReconstructionConfig
    provenance: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class RawValidationResult:
    person_ids: tuple[str, ...]
    labels: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    observed_native_counts: np.ndarray
    reconstructed_counts: np.ndarray
    observed_source_present: np.ndarray
    requested_reconstruction: np.ndarray
    reconstructed_source_present: np.ndarray
    mean_reconstruction_confidence: np.ndarray
    metrics: Mapping[str, float]
    metrics_by_reconstruction_eligibility: Mapping[str, Mapping[str, float]]
    threshold: float


def _sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _well_formed_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return normalized


def _safe_torch_load(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"deployment checkpoint does not exist: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - cluster uses modern torch.
        raise RuntimeError(
            "secure deployment loading requires torch.load(weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("deployment checkpoint must contain a mapping")
    return dict(payload)


def normalize_observed_groups(values: Sequence[str]) -> tuple[str, ...]:
    """Return a non-empty canonical-order acquisition set."""

    raw = tuple(str(value).strip() for value in values)
    if not raw or any(not value for value in raw):
        raise ValueError("--observed-groups must contain at least one modality")
    if len(raw) != len(set(raw)):
        raise ValueError("--observed-groups cannot contain duplicates")
    unknown = sorted(set(raw).difference(PRIMARY_MODALITY_GROUPS))
    if unknown:
        raise ValueError(f"unknown observed modality groups: {unknown}")
    selected = set(raw)
    return tuple(group for group in PRIMARY_MODALITY_GROUPS if group in selected)


def normalize_reconstruction_target_policy(value: str) -> str:
    """Validate the participant-level reconstruction-target policy."""

    policy = str(value).strip().lower()
    if policy not in RECONSTRUCTION_TARGET_POLICIES:
        raise ValueError(
            "reconstruction_target_policy must be one of "
            f"{sorted(RECONSTRUCTION_TARGET_POLICIES)}"
        )
    return policy


def _sanitized_validation_metadata(
    source_metadata: Mapping[str, Any],
    *,
    n_validation_rows: int,
    source_master_csv_sha256: str,
    source_master_metadata_sha256: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Create the minimum val-only metadata contract used by the Dataset.

    The source master metadata contains full-cohort label counts, split counts,
    coverage tables, and assignment hashes.  Copying it and merely changing
    ``n_rows`` would turn those full-cohort aggregates into false metadata for
    the projected file.  This function instead retains only the source hashes
    needed for clinical and per-modality artifact lineage checks.
    """

    task = str(source_metadata.get("task") or "").strip()
    if not task:
        raise ValueError("source master metadata task is missing")
    clinical_schema_hash = _well_formed_sha256(
        source_metadata.get("clinical_schema_hash"),
        "source master clinical_schema_hash",
    )
    raw_indexes = _mapping(
        source_metadata.get("source_indexes"), "source master source_indexes"
    )
    source_indexes: Dict[str, Dict[str, str]] = {}
    for source in ("environment", "wearable", "ecg", "retinal"):
        raw_record = _mapping(
            raw_indexes.get(source), f"source master source_indexes.{source}"
        )
        record = {
            "config_hash": _well_formed_sha256(
                raw_record.get("config_hash"),
                f"source master source_indexes.{source}.config_hash",
            )
        }
        mapping_hash = str(raw_record.get("mapping_hash") or "").strip()
        if mapping_hash:
            record["mapping_hash"] = _well_formed_sha256(
                mapping_hash,
                f"source master source_indexes.{source}.mapping_hash",
            )
        if source == "retinal":
            model_view_hash = str(
                raw_record.get("model_view_config_hash") or ""
            ).strip()
            if not model_view_hash:
                raise ValueError(
                    "source master retinal model_view_config_hash is required"
                )
            record["model_view_config_hash"] = _well_formed_sha256(
                model_view_hash,
                "source master retinal model_view_config_hash",
            )
        source_indexes[source] = record

    retained_optional: Dict[str, Any] = {}
    preprocessing_version = str(
        source_metadata.get("preprocessing_version") or ""
    ).strip()
    if preprocessing_version:
        retained_optional["preprocessing_version"] = preprocessing_version
    source_master_config_hash = str(
        source_metadata.get("config_hash") or ""
    ).strip()
    if source_master_config_hash:
        retained_optional["source_master_config_hash"] = _well_formed_sha256(
            source_master_config_hash, "source master config_hash"
        )

    derived_view = {
        "split": "val",
        "source_master_csv_sha256": source_master_csv_sha256,
        "source_master_metadata_sha256": source_master_metadata_sha256,
        "filter_first_pass_columns": ["person_id", "split"],
        "non_validation_y_materialized": False,
        "test_rows_in_derived_view": 0,
        "metadata_policy": "lineage_fields_allowlist",
    }
    derived_metadata: Dict[str, Any] = {
        "format_version": "ot2-validation-master-view-metadata-v1",
        **retained_optional,
        "task": task,
        "n_rows": int(n_validation_rows),
        "clinical_schema_hash": clinical_schema_hash,
        "source_indexes": source_indexes,
        "derived_view": derived_view,
    }
    retained_source_fields = {
        "preprocessing_version",
        "config_hash",
        "task",
        "clinical_schema_hash",
        "source_indexes",
    }
    audit = {
        **derived_view,
        "retained_source_metadata_fields": sorted(
            retained_source_fields.intersection(source_metadata)
        ),
        "omitted_source_metadata_fields": sorted(
            set(str(key) for key in source_metadata).difference(
                retained_source_fields.union({"n_rows"})
            )
        ),
        "outcome_aggregate_fields_copied": [],
    }
    return derived_metadata, audit


@contextlib.contextmanager
def isolated_validation_master_view(
    master_csv: str | Path,
    master_metadata_json: str | Path,
    *,
    temporary_parent: Path,
):
    """Yield a temporary val-only master without parsing non-val outcomes.

    The first pass requests only ``person_id`` and ``split``.  The second pass
    skips every non-validation record before pandas materializes the full row,
    including ``y``.  Consequently OT2V2Dataset can validate the derived file
    normally without ever seeing a train or test outcome.
    """

    source = Path(master_csv).expanduser().resolve()
    metadata_source = Path(master_metadata_json).expanduser().resolve()
    if not source.is_file() or not metadata_source.is_file():
        raise FileNotFoundError("master CSV and metadata are required")
    split_only = pd.read_csv(
        source,
        usecols=["person_id", "split"],
        dtype={"person_id": "string", "split": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    normalized_splits = split_only["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized_splits).difference({"train", "val", "test"}))
    if unknown:
        raise ValueError(f"master contains invalid split values: {unknown}")
    val_positions = np.flatnonzero(normalized_splits.eq("val").to_numpy())
    if val_positions.size == 0:
        raise ValueError("master contains no validation rows")
    keep_file_rows = {int(position) + 1 for position in val_positions}
    validation = pd.read_csv(
        source,
        skiprows=lambda row_number: (
            row_number != 0 and row_number not in keep_file_rows
        ),
        dtype={"person_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    observed_splits = set(validation["split"].astype(str).str.strip().str.lower())
    if observed_splits != {"val"} or len(validation) != len(val_positions):
        raise RuntimeError("derived validation master row selection is inconsistent")
    if validation["person_id"].astype(str).duplicated().any():
        raise ValueError("derived validation master contains duplicate person IDs")
    source_metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    if not isinstance(source_metadata, Mapping):
        raise ValueError("master metadata must contain a mapping")
    declared_rows = source_metadata.get("n_rows")
    if declared_rows is not None and int(declared_rows) != len(split_only):
        raise ValueError("source master metadata n_rows differs from the CSV")
    source_master_csv_sha256 = _sha256_file(source)
    source_master_metadata_sha256 = _sha256_file(metadata_source)
    derived_metadata, metadata_audit = _sanitized_validation_metadata(
        source_metadata,
        n_validation_rows=len(validation),
        source_master_csv_sha256=source_master_csv_sha256,
        source_master_metadata_sha256=source_master_metadata_sha256,
    )
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".universal-val-master.", dir=temporary_parent
    ) as directory:
        root = Path(directory)
        derived_csv = root / "master_validation_only.csv"
        derived_metadata_path = root / "master_validation_only.metadata.json"
        validation.to_csv(derived_csv, index=False)
        _atomic_json(derived_metadata, derived_metadata_path)
        audit = {
            "source_master_csv": str(source),
            "source_master_csv_sha256": source_master_csv_sha256,
            "source_master_metadata_json": str(metadata_source),
            "source_master_metadata_sha256": source_master_metadata_sha256,
            "derived_master_csv_sha256": _sha256_file(derived_csv),
            "derived_master_metadata_sha256": _sha256_file(
                derived_metadata_path
            ),
            "n_source_rows": int(len(split_only)),
            "n_validation_rows": int(len(validation)),
            "filter_first_pass_columns": ["person_id", "split"],
            "non_validation_y_materialized": False,
            "test_rows_passed_to_dataset": 0,
            "temporary_view_retained": False,
            "derived_metadata_policy": metadata_audit,
        }
        yield derived_csv, derived_metadata_path, audit


def _model_config(value: Any) -> Task2V2ModelConfig:
    raw = dict(_mapping(value, "resolved_model_config"))
    if "task_names" in raw:
        task_names = raw["task_names"]
        if not isinstance(task_names, Sequence) or isinstance(task_names, (str, bytes)):
            raise ValueError("resolved_model_config.task_names must be a sequence")
        raw["task_names"] = tuple(str(item) for item in task_names)
    try:
        return Task2V2ModelConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment resolved_model_config is invalid") from exc


def _reconstruction_config(value: Any) -> UniversalReconstructionConfig:
    raw = dict(_mapping(value, "universal_reconstruction_config"))
    try:
        return UniversalReconstructionConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment universal reconstruction config is invalid") from exc


def _validation_threshold(payload: Mapping[str, Any]) -> tuple[float, Mapping[str, Any]]:
    record = _mapping(payload.get("selected_threshold"), "selected_threshold")
    try:
        threshold = float(record.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment threshold is invalid") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("deployment threshold must be finite and in [0, 1]")
    if str(record.get("method") or "").strip().lower() not in {"fixed", "youden"}:
        raise ValueError("deployment threshold method is unsupported")
    objective = record.get("objective")
    if objective is not None:
        try:
            numeric_objective = float(objective)
        except (TypeError, ValueError) as exc:
            raise ValueError("deployment threshold objective is invalid") from exc
        if not math.isfinite(numeric_objective):
            raise ValueError("deployment threshold objective must be finite")
    return threshold, dict(record)


def inspect_deployment_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_task: str,
    expected_seed: int,
    expected_config_hash: str,
    expected_config_sha256: str,
    verify_runtime_code: bool = True,
) -> UniversalDeploymentContract:
    """Fail closed on checkpoint semantics before participant artifacts load."""

    if payload.get("format_version") != DEPLOYMENT_FORMAT_VERSION:
        raise ValueError("unsupported universal deployment checkpoint format")
    if payload.get("checkpoint_kind") != "universal_reconstruction_student":
        raise ValueError("checkpoint_kind is not a universal reconstruction student")
    if payload.get("model_role") != "universal_dynamic_subset_reconstruction":
        raise ValueError("checkpoint model_role is invalid")
    if str(payload.get("task") or "") != str(expected_task):
        raise ValueError("deployment task differs from the supervised config")
    if payload.get("test_evaluated") is not False:
        raise ValueError("V1 accepts only checkpoints that have not evaluated test")
    if payload.get("teacher_required") is not False:
        raise ValueError("deployment checkpoint unexpectedly requires a teacher")
    if payload.get("raw_modality_reconstruction") is not False:
        raise ValueError("V1 supports latent reconstruction only")
    if payload.get("runtime_observed_groups_required") is not True:
        raise ValueError("deployment checkpoint lacks an explicit runtime subset contract")
    if list(payload.get("native_encoder_groups", ())) != list(PRIMARY_MODALITY_GROUPS):
        raise ValueError("deployment native encoder order is incompatible")
    supported_modes = payload.get("supported_inference_modes")
    if not isinstance(supported_modes, Sequence) or isinstance(supported_modes, (str, bytes)):
        raise ValueError("deployment supported_inference_modes is invalid")
    required_modes = {
        "dynamic_subset_observed_only",
        "dynamic_subset_latent_reconstruction",
    }
    if set(str(item) for item in supported_modes) != required_modes:
        raise ValueError("deployment inference modes are incompatible")
    if payload.get("latent_target_space") != "full_teacher_native_source_token_pre_fusion":
        raise ValueError("deployment latent target space is incompatible")
    for forbidden in ("teacher_state", "optimizer_state", "scheduler_state"):
        if forbidden in payload:
            raise ValueError(f"deployment checkpoint contains forbidden {forbidden}")

    try:
        seed = int(payload.get("seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment seed is invalid") from exc
    if seed != int(expected_seed):
        raise ValueError("deployment seed differs from the supervised config")
    threshold, threshold_record = _validation_threshold(payload)
    model_config = _model_config(payload.get("resolved_model_config"))
    reconstruction_config = _reconstruction_config(
        payload.get("universal_reconstruction_config")
    )
    if reconstruction_config.task_name != expected_task:
        raise ValueError("reconstruction config task differs from deployment task")
    if int(reconstruction_config.d_model) != int(model_config.d_model):
        raise ValueError("base and reconstruction model widths differ")
    if expected_task not in model_config.task_names:
        raise ValueError("deployment base model lacks the requested task head")

    provenance = _mapping(payload.get("provenance"), "provenance")
    if provenance.get("test_evaluated") is not False:
        raise ValueError("deployment provenance does not prove test isolation")
    if str(provenance.get("task") or "") != expected_task:
        raise ValueError("deployment provenance task mismatch")
    if int(provenance.get("seed", -1)) != seed:
        raise ValueError("deployment/provenance seed mismatch")
    if list(provenance.get("optimization_splits", ())) != ["train"]:
        raise ValueError("deployment optimization split provenance is invalid")
    if list(provenance.get("selection_splits", ())) != ["val"]:
        raise ValueError("deployment threshold/model selection is not validation-only")
    supervised = _mapping(
        provenance.get("supervised_config"), "provenance.supervised_config"
    )
    if supervised.get("config_hash") != expected_config_hash:
        raise ValueError("deployment supervised config hash mismatch")
    if _well_formed_sha256(
        supervised.get("sha256"), "provenance supervised config sha256"
    ) != expected_config_sha256:
        raise ValueError("deployment supervised config bytes differ from current config")
    if _json_safe(provenance.get("resolved_model_config")) != _json_safe(model_config):
        raise ValueError("deployment has inconsistent resolved model configs")

    teacher_record = _mapping(
        provenance.get("teacher_checkpoint"), "provenance.teacher_checkpoint"
    )
    teacher_audit = _mapping(
        provenance.get("teacher_audit"), "provenance.teacher_audit"
    )
    teacher_sha = _well_formed_sha256(
        teacher_record.get("sha256"), "provenance teacher checkpoint sha256"
    )
    if _well_formed_sha256(
        teacher_audit.get("sha256"), "teacher audit sha256"
    ) != teacher_sha:
        raise ValueError("teacher checkpoint and teacher audit hashes differ")
    if teacher_audit.get("config_hash") != expected_config_hash:
        raise ValueError("teacher audit config hash mismatch")
    if teacher_audit.get("full_modality_teacher") is not True:
        raise ValueError("deployment was not derived from a full-modality teacher")
    for key in (
        "master_csv_sha256",
        "scaler_sha256",
        "clinical_features_csv_sha256",
    ):
        _well_formed_sha256(teacher_audit.get(key), f"teacher audit {key}")

    universal_provenance = _mapping(
        provenance.get("universal_config"), "provenance.universal_config"
    )
    _well_formed_sha256(
        universal_provenance.get("sha256"), "provenance universal config sha256"
    )
    if not str(universal_provenance.get("config_hash") or "").strip():
        raise ValueError("provenance universal config hash is missing")
    cache_provenance = _mapping(
        provenance.get("token_caches"), "provenance.token_caches"
    )
    if set(cache_provenance) != {"train", "val"}:
        raise ValueError("deployment token-cache provenance must contain train and val only")
    for split in ("train", "val"):
        record = _mapping(
            cache_provenance.get(split), f"provenance token cache {split}"
        )
        _well_formed_sha256(
            record.get("sha256"), f"provenance token cache {split} sha256"
        )

    runtime = _mapping(
        provenance.get("runtime_code_sha256"), "provenance.runtime_code_sha256"
    )
    for name, digest in runtime.items():
        _well_formed_sha256(digest, f"runtime code hash for {name}")
    if verify_runtime_code and dict(runtime) != _runtime_code_sha256():
        raise ValueError(
            "deployment checkpoint was built with a different universal runtime code bundle"
        )
    return UniversalDeploymentContract(
        task=expected_task,
        seed=seed,
        threshold=threshold,
        threshold_record=threshold_record,
        model_config=model_config,
        reconstruction_config=reconstruction_config,
        provenance=dict(provenance),
    )


def _audit_state_dict(
    value: Any,
    module: torch.nn.Module,
    *,
    label: str,
) -> Dict[str, torch.Tensor]:
    state = _mapping(value, label)
    expected = module.state_dict()
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"{label} key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    audited: Dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        candidate = state[key]
        if not isinstance(candidate, torch.Tensor):
            raise ValueError(f"{label}[{key!r}] is not a tensor")
        if tuple(candidate.shape) != tuple(target.shape):
            raise ValueError(
                f"{label}[{key!r}] shape mismatch: "
                f"{tuple(candidate.shape)} != {tuple(target.shape)}"
            )
        if candidate.dtype != target.dtype:
            raise ValueError(
                f"{label}[{key!r}] dtype mismatch: {candidate.dtype} != {target.dtype}"
            )
        if (candidate.is_floating_point() or candidate.is_complex()) and not bool(
            torch.isfinite(candidate).all()
        ):
            raise ValueError(f"{label}[{key!r}] contains non-finite values")
        audited[key] = candidate
    return audited


def build_deployment_model(
    payload: Mapping[str, Any],
    contract: UniversalDeploymentContract,
) -> Task2V2UniversalReconstructionStudent:
    """Construct and strictly load the self-contained universal student."""

    # The deployment state already contains the complete retinal backbone.
    # Suppress constructor-time ImageNet/custom initialization so evaluation
    # never downloads a weight file or depends on the training host path.
    construction_config = dataclasses.replace(
        contract.model_config,
        retinal_weights=None,
        retinal_weights_path=None,
    )
    base = Task2V2Model(construction_config)
    reconstructor = TargetQueryCrossModalReconstructor(
        contract.reconstruction_config
    )
    student = Task2V2UniversalReconstructionStudent(
        base, reconstructor, contract.reconstruction_config
    )
    base_state = _audit_state_dict(
        payload.get("base_model_state"), base, label="base_model_state"
    )
    reconstructor_state = _audit_state_dict(
        payload.get("reconstructor_state"),
        reconstructor,
        label="reconstructor_state",
    )
    base.load_state_dict(base_state, strict=True)
    reconstructor.load_state_dict(reconstructor_state, strict=True)
    student.requires_grad_(False)
    student.eval()
    return student


def _retinal_caps(run_config: Any) -> Dict[str, int]:
    caps = run_config.dataset.retinal_max_assets
    return {
        "retinal_photography": int(caps["photography"]),
        "retinal_oct": int(caps["oct"]),
        "retinal_octa": int(caps["octa"]),
        "retinal_flio": int(caps["flio"]),
    }


def build_validation_dataset(
    run_config: Any,
    observed_groups: Sequence[str],
    *,
    master_csv: Optional[str | Path] = None,
    master_metadata_json: Optional[str | Path] = None,
) -> OT2V2Dataset:
    """Build only validation and load only runtime-observed modalities."""

    observed = normalize_observed_groups(observed_groups)
    if not run_config.dataset.deterministic_retinal_selection:
        raise ValueError("deterministic retinal selection is required")
    return OT2V2Dataset(
        master_csv=(
            run_config.dataset.master_csv if master_csv is None else master_csv
        ),
        scaler_json=run_config.dataset.scaler_json,
        task=run_config.dataset.task,
        artifact_root=run_config.dataset.artifact_root,
        split="val",
        master_metadata_json=(
            run_config.dataset.master_metadata_json
            if master_metadata_json is None
            else master_metadata_json
        ),
        environment_schema_json=run_config.dataset.environment_schema_json,
        max_retinal_assets_per_group=_retinal_caps(run_config),
        require_y=True,
        validate_scaler_hash=True,
        loaded_source_groups=observed,
    )


def _loader(
    dataset: OT2V2Dataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(workers) < 0:
        raise ValueError("workers cannot be negative")

    def seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator().manual_seed(int(seed) + 91)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=task2_v2_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(workers) > 0,
        worker_init_fn=seed_worker if int(workers) > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--bf16 requested but CUDA bf16 is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bf16 evaluation is supported only on CPU/CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _strict_declared_presence_vector(value: Any, *, person_id: str) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (len(PRIMARY_MODALITY_GROUPS),)
    if array.shape != expected_shape:
        raise ValueError(
            f"declared availability for {person_id} must have shape {expected_shape}"
        )
    if array.dtype != np.bool_:
        try:
            numeric = array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"declared availability for {person_id} is not boolean"
            ) from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError(
                f"declared availability for {person_id} is not binary"
            )
    return array.astype(bool, copy=False)


def declared_availability_by_person(
    dataset: OT2V2Dataset,
) -> Dict[str, tuple[bool, ...]]:
    """Read original (pre-acquisition-ablation) availability from the master."""

    result: Dict[str, tuple[bool, ...]] = {}
    for _, row in dataset.index.iterrows():
        person_id = str(row["person_id"])
        if person_id in result:
            raise ValueError(f"duplicate person ID in validation Dataset: {person_id}")
        result[person_id] = tuple(
            bool(row[SOURCE_FLAG_BY_GROUP[group]])
            for group in PRIMARY_MODALITY_GROUPS
        )
    if not result:
        raise ValueError("validation Dataset contains no people")
    return result


def _metrics_with_classification_counts(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    metrics = binary_classification_metrics(
        labels, probabilities=probabilities, threshold=float(threshold)
    )
    truth = labels.astype(np.int64)
    predicted = (probabilities >= float(threshold)).astype(np.int64)
    true_positive = int(np.sum((truth == 1) & (predicted == 1)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    true_negative = int(np.sum((truth == 0) & (predicted == 0)))
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
        "true_negative": float(true_negative),
        "false_negative": float(false_negative),
    }


def _stratified_reconstruction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    requested_reconstruction: np.ndarray,
    observed_source_present: np.ndarray,
    threshold: float,
) -> Dict[str, Mapping[str, float]]:
    """Report fixed-threshold performance for reconstruction eligibility strata."""

    requested_any = requested_reconstruction.any(axis=1)
    context_any = observed_source_present.any(axis=1)
    masks = {
        "requested_and_context_available": requested_any & context_any,
        "requested_but_no_observed_context": requested_any & ~context_any,
        "no_requested_target": ~requested_any,
    }
    template = (
        "n",
        "prevalence",
        "auroc",
        "auprc",
        "log_loss",
        "brier",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "threshold",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    )
    result: Dict[str, Mapping[str, float]] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count:
            result[name] = _metrics_with_classification_counts(
                labels[mask], probabilities[mask], threshold=float(threshold)
            )
        else:
            result[name] = {
                key: (
                    0.0
                    if key
                    in {
                        "n",
                        "true_positive",
                        "false_positive",
                        "true_negative",
                        "false_negative",
                    }
                    else float(threshold)
                    if key == "threshold"
                    else math.nan
                )
                for key in template
            }
    if sum(int(mask.sum()) for mask in masks.values()) != int(labels.size):
        raise RuntimeError("reconstruction eligibility strata do not partition validation")
    return result


@torch.no_grad()
def predict_validation(
    model: Task2V2UniversalReconstructionStudent,
    loader: Any,
    *,
    task: str,
    disabled_groups: Sequence[str],
    declared_availability: Mapping[str, Sequence[bool]],
    reconstruction_mode: str,
    reconstruction_target_policy: str = "declared-disabled",
    threshold: float,
    device: torch.device,
    use_bf16_amp: bool,
) -> RawValidationResult:
    """Predict validation with one fixed deployment acquisition contract."""

    mode = str(reconstruction_mode).strip().lower()
    if mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"reconstruction_mode must be one of {sorted(RECONSTRUCTION_MODES)}")
    disabled = tuple(str(group) for group in disabled_groups)
    unknown = sorted(set(disabled).difference(PRIMARY_MODALITY_GROUPS))
    if unknown or len(disabled) != len(set(disabled)):
        raise ValueError("disabled_groups must be unique canonical modalities")
    canonical_disabled = tuple(
        group for group in PRIMARY_MODALITY_GROUPS if group in set(disabled)
    )
    if disabled != canonical_disabled:
        raise ValueError("disabled_groups must follow canonical modality order")
    policy = normalize_reconstruction_target_policy(reconstruction_target_policy)
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    declared_lookup = {
        str(person_id): _strict_declared_presence_vector(
            values, person_id=str(person_id)
        )
        for person_id, values in declared_availability.items()
    }
    if len(declared_lookup) != len(declared_availability):
        raise ValueError(
            "declared_availability contains duplicate person IDs after string normalization"
        )
    if not declared_lookup:
        raise ValueError("declared_availability cannot be empty")
    disabled_vector = torch.tensor(
        [group in set(disabled) for group in PRIMARY_MODALITY_GROUPS],
        dtype=torch.bool,
        device=device,
    )
    model.eval()
    identifiers: list[str] = []
    label_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    observed_count_parts: list[torch.Tensor] = []
    reconstructed_count_parts: list[torch.Tensor] = []
    observed_state_parts: list[torch.Tensor] = []
    requested_state_parts: list[torch.Tensor] = []
    reconstructed_state_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    iterator = tqdm(loader, desc=f"Universal raw validation [{mode}]", unit="batch")
    for raw_batch in iterator:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("validation loader must yield mapping batches")
        observed_splits = {
            str(value).strip().lower() for value in raw_batch.get("split", ())
        }
        if observed_splits != {"val"}:
            raise ValueError(
                f"V1 raw evaluator received non-validation data: {sorted(observed_splits)}"
            )
        batch_ids = [str(value) for value in raw_batch.get("person_id", ())]
        if not batch_ids or any(not person_id.strip() for person_id in batch_ids):
            raise ValueError("validation batch contains no person IDs or an empty person ID")
        missing_declared = [
            person_id for person_id in batch_ids if person_id not in declared_lookup
        ]
        if missing_declared:
            raise ValueError(
                "validation batch IDs are absent from declared_availability: "
                f"{missing_declared[:10]}"
            )
        declared_batch = torch.as_tensor(
            np.stack([declared_lookup[person_id] for person_id in batch_ids]),
            dtype=torch.bool,
            device=device,
        )
        requested = disabled_vector.unsqueeze(0).expand(len(batch_ids), -1)
        if policy == "declared-disabled":
            requested = requested & declared_batch
        labels = raw_batch.get("y")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("validation batch is missing y")
        batch = recursive_to_device(
            raw_batch, device, non_blocking=device.type == "cuda"
        )
        with _autocast(device, use_bf16_amp):
            output = model(
                batch,
                task=task,
                disabled_groups=disabled,
                requested_reconstruction=requested,
                reconstruct_disabled=False,
                inject_reconstruction=mode == "on",
            )
        logits = output.get("logits")
        if not isinstance(logits, torch.Tensor):
            raise ValueError("universal student did not return logits")
        logits = logits.detach().float().cpu().reshape(-1)
        batch_labels = labels.detach().float().cpu().reshape(-1)
        if len(batch_ids) != logits.numel() or logits.shape != batch_labels.shape:
            raise ValueError("person IDs, labels, and logits are misaligned")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("universal student returned non-finite logits")
        observed = output.get("observed_source_present")
        output_requested = output.get("requested_reconstruction")
        reconstructed = output.get("reconstructed_source_present")
        confidence = output.get("reconstruction_confidence")
        if not all(
            isinstance(item, torch.Tensor)
            for item in (
                observed,
                output_requested,
                reconstructed,
                confidence,
            )
        ):
            raise ValueError("universal student did not return source-state tensors")
        if any(
            item.dtype != torch.bool
            for item in (observed, output_requested, reconstructed)
        ):
            raise ValueError("universal student source-state tensors must be boolean")
        expected_shape = (logits.numel(), len(PRIMARY_MODALITY_GROUPS))
        if (
            tuple(observed.shape) != expected_shape
            or tuple(output_requested.shape) != expected_shape
            or tuple(reconstructed.shape) != expected_shape
        ):
            raise ValueError("universal student source-state shape mismatch")
        if tuple(confidence.shape) != expected_shape:
            raise ValueError("universal student confidence shape mismatch")
        if not torch.equal(output_requested.bool(), requested.bool()):
            raise ValueError("universal student changed the requested reconstruction mask")
        expected_reconstructed = (
            requested.bool() & observed.bool().any(dim=1, keepdim=True)
            if mode == "on"
            else torch.zeros_like(requested, dtype=torch.bool)
        )
        if not torch.equal(reconstructed.bool(), expected_reconstructed):
            raise ValueError(
                "universal student reconstructed sources outside the requested/eligible set"
            )
        invalid_confidence = (
            ~torch.isfinite(confidence)
            | confidence.lt(0.0)
            | confidence.gt(1.0)
        )
        if bool((reconstructed & invalid_confidence).any().item()):
            raise ValueError(
                "universal student returned invalid confidence for a reconstructed source"
            )
        observed_count_parts.append(observed.sum(dim=1).cpu())
        reconstructed_count_parts.append(reconstructed.sum(dim=1).cpu())
        observed_state_parts.append(observed.bool().cpu())
        requested_state_parts.append(output_requested.bool().cpu())
        reconstructed_state_parts.append(reconstructed.bool().cpu())
        if mode == "on" and disabled:
            indices = [PRIMARY_MODALITY_GROUPS.index(group) for group in disabled]
            selected_confidence = confidence[:, indices]
            eligible = reconstructed[:, indices]
            numerator = torch.where(
                eligible, selected_confidence, torch.zeros_like(selected_confidence)
            ).sum(dim=1)
            denominator = eligible.sum(dim=1)
            mean_confidence = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1),
                torch.full_like(numerator, float("nan")),
            )
        else:
            mean_confidence = torch.full(
                (logits.numel(),), float("nan"), device=confidence.device
            )
        confidence_parts.append(mean_confidence.float().cpu())
        identifiers.extend(batch_ids)
        label_parts.append(batch_labels)
        logit_parts.append(logits)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("validation predictions are empty or contain duplicate people")
    if set(identifiers) != set(declared_lookup):
        raise ValueError(
            "declared_availability and validation prediction cohorts differ"
        )
    labels_array = torch.cat(label_parts).numpy().astype(np.int64)
    logits_array = torch.cat(logit_parts).numpy().astype(np.float64)
    probabilities = np.empty_like(logits_array)
    nonnegative = logits_array >= 0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-logits_array[nonnegative])
    )
    exponent = np.exp(logits_array[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = _metrics_with_classification_counts(
        labels_array, probabilities, threshold=float(threshold)
    )
    observed_states = torch.cat(observed_state_parts).numpy().astype(bool)
    requested_states = torch.cat(requested_state_parts).numpy().astype(bool)
    eligibility_metrics = _stratified_reconstruction_metrics(
        labels_array,
        probabilities,
        requested_reconstruction=requested_states,
        observed_source_present=observed_states,
        threshold=float(threshold),
    )
    return RawValidationResult(
        person_ids=tuple(identifiers),
        labels=labels_array,
        logits=logits_array,
        probabilities=probabilities,
        predictions=predictions,
        observed_native_counts=torch.cat(observed_count_parts).numpy().astype(np.int64),
        reconstructed_counts=torch.cat(reconstructed_count_parts).numpy().astype(np.int64),
        observed_source_present=observed_states,
        requested_reconstruction=requested_states,
        reconstructed_source_present=torch.cat(reconstructed_state_parts)
        .numpy()
        .astype(bool),
        mean_reconstruction_confidence=torch.cat(confidence_parts).numpy().astype(np.float64),
        metrics=metrics,
        metrics_by_reconstruction_eligibility=eligibility_metrics,
        threshold=float(threshold),
    )


def audit_availability_states(
    dataset: OT2V2Dataset,
    result: RawValidationResult,
    *,
    observed_groups: Sequence[str],
    reconstruction_mode: str,
    reconstruction_target_policy: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Separate acquisition ablation from natural source missingness."""

    if tuple(str(value) for value in dataset.index["person_id"]) != result.person_ids:
        raise ValueError("availability audit person order differs from predictions")
    declared = np.asarray(
        [
            [bool(row[SOURCE_FLAG_BY_GROUP[group]]) for group in PRIMARY_MODALITY_GROUPS]
            for _, row in dataset.index.iterrows()
        ],
        dtype=bool,
    )
    if declared.shape != result.observed_source_present.shape:
        raise ValueError("declared/model source-state shapes differ")
    observed_contract = np.asarray(
        [group in set(observed_groups) for group in PRIMARY_MODALITY_GROUPS],
        dtype=bool,
    )
    disabled_contract = ~observed_contract
    mode = str(reconstruction_mode).strip().lower()
    if mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"unknown reconstruction mode in availability audit: {mode}")
    policy = normalize_reconstruction_target_policy(
        reconstruction_target_policy
    )
    if (result.observed_source_present & ~declared).any():
        raise ValueError("model encoded a source not declared available in the master")
    if (result.observed_source_present & disabled_contract[None, :]).any():
        raise ValueError("model encoded a source outside the observed acquisition set")
    if (
        result.observed_source_present & result.reconstructed_source_present
    ).any():
        raise ValueError("a source cannot be both observed and reconstructed")
    if (result.requested_reconstruction & observed_contract[None, :]).any():
        raise ValueError("an observed acquisition was requested for reconstruction")
    declared_and_disabled = declared & disabled_contract[None, :]
    natural_missing_and_disabled = ~declared & disabled_contract[None, :]
    expected_requested = (
        declared_and_disabled
        if policy == "declared-disabled"
        else np.broadcast_to(disabled_contract, declared.shape)
    )
    if not np.array_equal(result.requested_reconstruction, expected_requested):
        raise ValueError(
            "requested reconstruction states disagree with the declared policy"
        )
    reconstructed = result.reconstructed_source_present
    expected_reconstructed = (
        expected_requested & result.observed_source_present.any(axis=1, keepdims=True)
        if mode == "on"
        else np.zeros_like(expected_requested, dtype=bool)
    )
    if not np.array_equal(reconstructed, expected_reconstructed):
        raise ValueError(
            "reconstructed states disagree with mode/requested/eligibility states"
        )
    rows: Dict[str, Any] = {}
    for index, group in enumerate(PRIMARY_MODALITY_GROUPS):
        rows[group] = {
            "declared_available": int(declared[:, index].sum()),
            "declared_naturally_absent": int((~declared[:, index]).sum()),
            "in_observed_acquisition_set": bool(observed_contract[index]),
            "disabled_but_declared_available": int(
                declared_and_disabled[:, index].sum()
            ),
            "disabled_and_naturally_absent": int(
                natural_missing_and_disabled[:, index].sum()
            ),
            "successfully_encoded_observed": int(
                result.observed_source_present[:, index].sum()
            ),
            "requested_reconstruction_declared_available": int(
                (result.requested_reconstruction[:, index] & declared[:, index]).sum()
            ),
            "requested_reconstruction_naturally_absent": int(
                (result.requested_reconstruction[:, index] & ~declared[:, index]).sum()
            ),
            "reconstructed_declared_available": int(
                (reconstructed[:, index] & declared[:, index]).sum()
            ),
            "reconstructed_naturally_absent": int(
                (reconstructed[:, index] & ~declared[:, index]).sum()
            ),
        }
    audit = {
        "state_definitions": {
            "declared_available": "source availability flag in the original master row",
            "disabled_but_declared_available": (
                "source existed upstream but this deployment experiment withheld it"
            ),
            "disabled_and_naturally_absent": (
                "source was not available upstream and is also outside the observed set"
            ),
            "successfully_encoded_observed": (
                "source was in the observed set and survived artifact/QC masks"
            ),
            "reconstructed": "latent token injected; no raw modality was synthesized",
        },
        "reconstruction_mode": mode,
        "reconstruction_target_policy": policy,
        "reconstruction_target_policy_definition": (
            "request only sources declared available in the original master and "
            "withheld by the static acquisition contract"
            if policy == "declared-disabled"
            else "exploratory: request every statically disabled source, including "
            "sources naturally absent in the original master"
        ),
        "by_modality": rows,
        "n_people_with_any_disabled_declared_available": int(
            declared_and_disabled.any(axis=1).sum()
        ),
        "n_people_with_any_disabled_naturally_absent": int(
            natural_missing_and_disabled.any(axis=1).sum()
        ),
        "n_reconstructed_tokens_for_declared_available_sources": int(
            (reconstructed & declared).sum()
        ),
        "n_reconstructed_tokens_for_naturally_absent_sources": int(
            (reconstructed & ~declared).sum()
        ),
        "n_requested_tokens_for_declared_available_sources": int(
            (result.requested_reconstruction & declared).sum()
        ),
        "n_requested_tokens_for_naturally_absent_sources": int(
            (result.requested_reconstruction & ~declared).sum()
        ),
    }
    return declared, audit


def _atomic_predictions(
    result: RawValidationResult,
    destination: Path,
    *,
    task: str,
    observed_groups: Sequence[str],
    reconstruction_mode: str,
    reconstruction_target_policy: str,
    declared_available: np.ndarray,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    fields = (
        "person_id",
        "task",
        "split",
        "observed_groups",
        "reconstruction_mode",
        "reconstruction_target_policy",
        "y_true",
        "y_logit",
        "y_probability",
        "y_pred",
        "threshold",
        "n_observed_native_sources",
        "n_requested_reconstruction_sources",
        "n_reconstructed_sources",
        "n_disabled_declared_available_sources",
        "n_disabled_naturally_absent_sources",
        "n_reconstructed_declared_available_sources",
        "n_reconstructed_naturally_absent_sources",
        "declared_available_groups",
        "successfully_encoded_observed_groups",
        "requested_reconstruction_groups",
        "reconstructed_groups",
        "mean_reconstruction_confidence",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            observed_text = ";".join(observed_groups)
            disabled_mask = np.asarray(
                [
                    group not in set(observed_groups)
                    for group in PRIMARY_MODALITY_GROUPS
                ],
                dtype=bool,
            )
            for index, person_id in enumerate(result.person_ids):
                confidence = result.mean_reconstruction_confidence[index]
                declared = declared_available[index]
                observed_state = result.observed_source_present[index]
                requested = result.requested_reconstruction[index]
                reconstructed = result.reconstructed_source_present[index]
                groups_for = lambda mask: ";".join(  # noqa: E731
                    group
                    for group, present in zip(PRIMARY_MODALITY_GROUPS, mask)
                    if bool(present)
                )
                writer.writerow(
                    {
                        "person_id": person_id,
                        "task": task,
                        "split": "val",
                        "observed_groups": observed_text,
                        "reconstruction_mode": reconstruction_mode,
                        "reconstruction_target_policy": reconstruction_target_policy,
                        "y_true": int(result.labels[index]),
                        "y_logit": float(result.logits[index]),
                        "y_probability": float(result.probabilities[index]),
                        "y_pred": int(result.predictions[index]),
                        "threshold": float(result.threshold),
                        "n_observed_native_sources": int(
                            result.observed_native_counts[index]
                        ),
                        "n_requested_reconstruction_sources": int(
                            requested.sum()
                        ),
                        "n_reconstructed_sources": int(
                            result.reconstructed_counts[index]
                        ),
                        "n_disabled_declared_available_sources": int(
                            (declared & disabled_mask).sum()
                        ),
                        "n_disabled_naturally_absent_sources": int(
                            (~declared & disabled_mask).sum()
                        ),
                        "n_reconstructed_declared_available_sources": int(
                            (reconstructed & declared).sum()
                        ),
                        "n_reconstructed_naturally_absent_sources": int(
                            (reconstructed & ~declared).sum()
                        ),
                        "declared_available_groups": groups_for(declared),
                        "successfully_encoded_observed_groups": groups_for(
                            observed_state
                        ),
                        "requested_reconstruction_groups": groups_for(requested),
                        "reconstructed_groups": groups_for(reconstructed),
                        "mean_reconstruction_confidence": (
                            float(confidence) if math.isfinite(confidence) else ""
                        ),
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _person_label_hashes(result: RawValidationResult) -> Mapping[str, str]:
    pairs = sorted(
        (person_id, int(label))
        for person_id, label in zip(result.person_ids, result.labels)
    )
    return {
        "ordered_person_ids_sha256": _canonical_json_hash(result.person_ids),
        "person_id_set_sha256": _canonical_json_hash(sorted(result.person_ids)),
        "person_label_map_sha256": _canonical_json_hash(pairs),
    }


def _input_file_records(
    *,
    config_path: Path,
    checkpoint_path: Path,
    run_config: Any,
    dataset: OT2V2Dataset,
    observed_groups: Sequence[str],
) -> Mapping[str, Any]:
    paths = {
        "supervised_config": config_path,
        "deployment_checkpoint": checkpoint_path,
        "master_csv": run_config.dataset.master_csv,
        "master_metadata_json": run_config.dataset.master_metadata_json,
        "scaler_json": run_config.dataset.scaler_json,
        "clinical_schema_json": run_config.dataset.clinical_schema_json,
        "environment_schema_json": run_config.dataset.environment_schema_json,
        "wearable_schema_json": run_config.dataset.wearable_schema_json,
        "ecg_schema_json": run_config.dataset.ecg_schema_json,
    }
    records: Dict[str, Any] = {
        name: {"path": str(path), "sha256": _sha256_file(path), "read": True}
        for name, path in paths.items()
    }
    clinical_read = "clinical_data" in set(observed_groups)
    records["clinical_features_csv"] = {
        "path": str(dataset.clinical_features_path),
        "sha256": (
            _sha256_file(dataset.clinical_features_path) if clinical_read else None
        ),
        "read": clinical_read,
        "binding_when_not_read": (
            None
            if clinical_read
            else "disabled source bound by master, clinical schema, and scaler hashes"
        ),
    }
    return records


def run_raw_validation(args: argparse.Namespace) -> Mapping[str, Any]:
    """Run a static-subset validation evaluation; test is impossible in V1."""

    split = str(args.split).strip().lower()
    if split != "val":
        raise ValueError("V1 universal raw evaluation is validation-only; split must be val")
    mode = str(args.reconstruction_mode).strip().lower()
    if mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"reconstruction_mode must be one of {sorted(RECONSTRUCTION_MODES)}")
    target_policy = normalize_reconstruction_target_policy(
        getattr(args, "reconstruction_target_policy", "declared-disabled")
    )
    observed = normalize_observed_groups(args.observed_groups)
    disabled = tuple(
        group for group in PRIMARY_MODALITY_GROUPS if group not in set(observed)
    )
    output_path = Path(args.out).expanduser().resolve()
    metadata_path = (
        output_path.with_suffix(".metadata.json")
        if args.metadata_out is None
        else Path(args.metadata_out).expanduser().resolve()
    )
    if output_path == metadata_path:
        raise ValueError("prediction and metadata destinations must differ")
    if not args.overwrite:
        existing = [str(path) for path in (output_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"evaluation output exists; use --overwrite or a new path: {existing}"
            )

    config_path, run_config = load_supervised_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = _safe_torch_load(checkpoint_path)
    contract = inspect_deployment_checkpoint(
        payload,
        expected_task=run_config.dataset.task,
        expected_seed=int(run_config.seed),
        expected_config_hash=run_config.config_hash,
        expected_config_sha256=_sha256_file(config_path),
    )
    # State validation occurs before any participant-level artifact is opened.
    model = build_deployment_model(payload, contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with isolated_validation_master_view(
        run_config.dataset.master_csv,
        run_config.dataset.master_metadata_json,
        temporary_parent=output_path.parent,
    ) as (validation_master, validation_metadata, isolation_audit):
        dataset = build_validation_dataset(
            run_config,
            observed,
            master_csv=validation_master,
            master_metadata_json=validation_metadata,
        )
        expected_model_config = build_model_config(
            run_config, dataset.model_feature_dimensions
        )
        if _json_safe(expected_model_config) != _json_safe(contract.model_config):
            raise ValueError(
                "deployment model architecture differs from current "
                "Dataset/scaler dimensions"
            )
        teacher_audit = _mapping(
            contract.provenance.get("teacher_audit"), "provenance.teacher_audit"
        )
        if _sha256_file(run_config.dataset.master_csv) != teacher_audit.get(
            "master_csv_sha256"
        ):
            raise ValueError("current master CSV differs from deployment provenance")
        if _sha256_file(run_config.dataset.scaler_json) != teacher_audit.get(
            "scaler_sha256"
        ):
            raise ValueError("current scaler differs from deployment provenance")
        if "clinical_data" in set(observed) and _sha256_file(
            dataset.clinical_features_path
        ) != teacher_audit.get("clinical_features_csv_sha256"):
            raise ValueError(
                "current clinical feature table differs from deployment provenance"
            )

        device = resolve_device(args.device)
        set_deterministic_seed(contract.seed, deterministic=True)
        model.to(device)
        effective_batch_size = int(
            args.batch_size
            if args.batch_size is not None
            else run_config.training.batch_size
        )
        loader = _loader(
            dataset,
            batch_size=effective_batch_size,
            workers=int(args.workers),
            seed=contract.seed,
            device=device,
        )
        declared_lookup = declared_availability_by_person(dataset)
        result = predict_validation(
            model,
            loader,
            task=contract.task,
            disabled_groups=disabled,
            declared_availability=declared_lookup,
            reconstruction_mode=mode,
            reconstruction_target_policy=target_policy,
            threshold=contract.threshold,
            device=device,
            use_bf16_amp=bool(args.bf16),
        )
        expected_ids = tuple(str(value) for value in dataset.index["person_id"])
        if result.person_ids != expected_ids:
            raise ValueError(
                "validation prediction order differs from the Dataset index"
            )
        declared_available, availability_audit = audit_availability_states(
            dataset,
            result,
            observed_groups=observed,
            reconstruction_mode=mode,
            reconstruction_target_policy=target_policy,
        )
        _atomic_predictions(
            result,
            output_path,
            task=contract.task,
            observed_groups=observed,
            reconstruction_mode=mode,
            reconstruction_target_policy=target_policy,
            declared_available=declared_available,
        )
        inputs = _input_file_records(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            run_config=run_config,
            dataset=dataset,
            observed_groups=observed,
        )
        metadata: Dict[str, Any] = {
            "format_version": RAW_VALIDATION_EVALUATION_FORMAT_VERSION,
            "evaluation_role": "validation_only_dynamic_subset_feedback",
            "task": contract.task,
            "seed": contract.seed,
            "split": "val",
            "n": len(result.person_ids),
            "observed_groups": list(observed),
            "disabled_groups": list(disabled),
            "reconstruction_mode": mode,
            "reconstruction_target_policy": target_policy,
            "reconstruct_disabled": mode == "on",
            "raw_modality_reconstruction": False,
            "latent_reconstruction_target": (
                "full_teacher_native_source_token_pre_fusion"
                if mode == "on"
                else None
            ),
            "threshold": contract.threshold,
            "threshold_source": "checkpoint_validation_selected_student_full",
            "selected_threshold_record": contract.threshold_record,
            "metrics": result.metrics,
            "metrics_by_reconstruction_eligibility": (
                result.metrics_by_reconstruction_eligibility
            ),
            "cohort_hashes": _person_label_hashes(result),
            "validation_master_isolation": isolation_audit,
            "availability_state_audit": availability_audit,
            "inputs": inputs,
            "prediction_csv": {
                "path": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            "dataset_load_contract": {
                "loaded_source_groups": list(dataset.loaded_source_groups),
                "unloaded_source_groups": list(disabled),
                "disabled_participant_artifacts_opened": False,
                "data_contract_audit": dataset.data_contract_audit,
            },
            "model_forward_contract": {
                "disabled_groups": list(disabled),
                "inject_reconstruction": mode == "on",
                "reconstruct_disabled": False,
                "requested_reconstruction_policy": target_policy,
                "requested_reconstruction_passed_explicitly": True,
            },
            "deployment_checkpoint_contract_hash": _canonical_json_hash(
                {
                    "task": contract.task,
                    "seed": contract.seed,
                    "threshold_record": contract.threshold_record,
                    "model_config": contract.model_config,
                    "reconstruction_config": contract.reconstruction_config,
                    "observed_groups": observed,
                    "reconstruction_mode": mode,
                    "reconstruction_target_policy": target_policy,
                }
            ),
            "runtime": {
                "device": str(device),
                "batch_size": effective_batch_size,
                "workers": int(args.workers),
                "bf16": bool(args.bf16),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
            },
            "test_iterated": False,
            "test_evaluated": False,
            "notes": [
                "V1 is validation-only; this output cannot be a final test result.",
                "The Dataset received a derived val-only master; no test row or y was validated.",
                "Disabled participant artifacts were not loaded by the Dataset.",
                "Natural absence and experiment-disabled availability are audited separately.",
                "Reconstruction, when enabled, creates latent task tokens only.",
                (
                    "Default reconstruction targets match training/panel supervision: "
                    "only declared-available sources withheld by this experiment."
                    if target_policy == "declared-disabled"
                    else "Exploratory policy injects tokens for naturally absent sources; "
                    "this is outside direct token-reconstruction supervision."
                ),
            ],
        }
        _atomic_json(metadata, metadata_path)
        return {
            "status": "complete",
            "task": contract.task,
            "split": "val",
            "observed_groups": list(observed),
            "disabled_groups": list(disabled),
            "reconstruction_mode": mode,
            "reconstruction_target_policy": target_policy,
            "n": len(result.person_ids),
            "threshold": contract.threshold,
            "metrics": result.metrics,
            "metrics_by_reconstruction_eligibility": (
                result.metrics_by_reconstruction_eligibility
            ),
            "predictions": str(output_path),
            "metadata": str(metadata_path),
            "test_evaluated": False,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--observed-groups", required=True, nargs="+")
    parser.add_argument(
        "--reconstruction-mode", required=True, choices=sorted(RECONSTRUCTION_MODES)
    )
    parser.add_argument(
        "--reconstruction-target-policy",
        choices=sorted(RECONSTRUCTION_TARGET_POLICIES),
        default="declared-disabled",
        help=(
            "Which disabled sources receive latent reconstruction requests. "
            "The default matches train/panel supervision; all-disabled-exploratory "
            "also requests originally absent sources."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_raw_validation(args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RAW_VALIDATION_EVALUATION_FORMAT_VERSION",
    "RECONSTRUCTION_TARGET_POLICIES",
    "RawValidationResult",
    "UniversalDeploymentContract",
    "audit_availability_states",
    "build_arg_parser",
    "build_deployment_model",
    "build_validation_dataset",
    "declared_availability_by_person",
    "inspect_deployment_checkpoint",
    "isolated_validation_master_view",
    "main",
    "normalize_observed_groups",
    "normalize_reconstruction_target_policy",
    "predict_validation",
    "run_raw_validation",
]
