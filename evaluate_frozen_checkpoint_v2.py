#!/usr/bin/env python3
"""Evaluate one frozen OT2 V2 checkpoint without retraining or test selection.

Validation evaluation is useful for reconstructing missing prediction evidence.
Test evaluation is deliberately gated by either the immutable cost-selection
lock or the standalone supervised-final lock.  Both accept only an exact
locked model/seed/checkpoint/config/data tuple, and both reuse the
validation-derived threshold stored in the checkpoint verbatim.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

try:
    from ot2.analyze_cost_noninferiority_v2 import (
        SELECTION_LOCK_VERSION,
        _json_without_duplicate_keys,
    )
    from ot2.analyze_modality_importance_v2 import (
        _validate_checkpoint_input_file_hashes,
        _validate_clinical_feature_bytes,
        _validate_evaluation_cohort_provenance,
        _validate_resolved_model_config,
    )
    from ot2.task2_v2_config import validate_model_v2_lineage
    from ot2.task2_v2_dataset import OT2V2Dataset, task2_v2_collate
    from ot2.task2_v2_engine import (
        ENGINE_FORMAT_VERSION,
        evaluate_binary,
        save_prediction_csv,
        set_deterministic_seed,
    )
    from ot2.task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model
    from ot2.task2_v2_distill import (
        LATENT_RECONSTRUCTION_MASK_SEMANTICS,
        LATENT_RECONSTRUCTION_TARGET_SPACE,
        LEGACY_DISTILLED_INFERENCE_MODE,
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE,
    )
    from ot2.train_cheap_student_v2 import (
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        load_deployment_inference_model,
    )
    from ot2.train_task2_v2 import (
        build_model_config,
        load_supervised_config,
        resolve_device,
    )
    from ot2.freeze_supervised_final_lock_v2 import (
        SUPERVISED_FINAL_LOCK_VERSION,
        authorize_supervised_final_lock,
        validate_supervised_final_lock_preflight,
    )
except ModuleNotFoundError:  # Direct execution from the ot2 directory.
    from analyze_cost_noninferiority_v2 import (  # type: ignore
        SELECTION_LOCK_VERSION,
        _json_without_duplicate_keys,
    )
    from analyze_modality_importance_v2 import (  # type: ignore
        _validate_checkpoint_input_file_hashes,
        _validate_clinical_feature_bytes,
        _validate_evaluation_cohort_provenance,
        _validate_resolved_model_config,
    )
    from task2_v2_config import validate_model_v2_lineage  # type: ignore
    from task2_v2_dataset import OT2V2Dataset, task2_v2_collate  # type: ignore
    from task2_v2_engine import (  # type: ignore
        ENGINE_FORMAT_VERSION,
        evaluate_binary,
        save_prediction_csv,
        set_deterministic_seed,
    )
    from task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model  # type: ignore
    from task2_v2_distill import (  # type: ignore
        LATENT_RECONSTRUCTION_MASK_SEMANTICS,
        LATENT_RECONSTRUCTION_TARGET_SPACE,
        LEGACY_DISTILLED_INFERENCE_MODE,
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE,
    )
    from train_cheap_student_v2 import (  # type: ignore
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        load_deployment_inference_model,
    )
    from train_task2_v2 import (  # type: ignore
        build_model_config,
        load_supervised_config,
        resolve_device,
    )
    from freeze_supervised_final_lock_v2 import (  # type: ignore
        SUPERVISED_FINAL_LOCK_VERSION,
        authorize_supervised_final_lock,
        validate_supervised_final_lock_preflight,
    )


EVALUATION_FORMAT_VERSION = "ot2-frozen-evaluation-v2-1"
LATENT_RECONSTRUCTION_EVALUATION_CAPABILITY = (
    "ot2-frozen-evaluation-latent-reconstruction-v1"
)
MODEL_ROLES = frozenset({"full_teacher", "cheap_baseline", "distilled_student"})


@dataclasses.dataclass(frozen=True)
class FrozenCheckpointContract:
    checkpoint_kind: str
    model_role: str
    seed: int
    threshold: float
    enabled_groups: tuple[str, ...]
    disabled_groups: tuple[str, ...]
    state: Mapping[str, torch.Tensor]
    inference_mode: str
    checkpoint_model_id: str
    reconstructors_required: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "secure checkpoint loading requires a PyTorch version with "
            "torch.load(..., weights_only=True) support"
        ) from error
    return _mapping(value, "checkpoint")


def _normalized(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _threshold(payload: Mapping[str, Any]) -> float:
    selected = _mapping(payload.get("selected_threshold"), "selected threshold")
    try:
        threshold = float(selected.get("threshold"))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint threshold is invalid") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("checkpoint threshold must be finite and in [0, 1]")
    if str(selected.get("method") or "").lower() not in {"fixed", "youden"}:
        raise ValueError("checkpoint threshold is not validation-derived")
    for key in ("selection_split", "threshold_selection_split", "source_split"):
        split = str(selected.get(key) or "").strip().lower()
        if split and split not in {"val", "validation"}:
            raise ValueError("checkpoint threshold source is not validation")
    return threshold


def _ordered_partition(
    enabled: Sequence[str], disabled: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    enabled_tuple = tuple(str(value) for value in enabled)
    disabled_tuple = tuple(str(value) for value in disabled)
    canonical = tuple(PRIMARY_MODALITY_GROUPS)
    if (
        enabled_tuple
        != tuple(group for group in canonical if group in set(enabled_tuple))
        or disabled_tuple
        != tuple(group for group in canonical if group in set(disabled_tuple))
        or not enabled_tuple
        or set(enabled_tuple).intersection(disabled_tuple)
        or set(enabled_tuple).union(disabled_tuple) != set(canonical)
    ):
        raise ValueError("checkpoint modalities are not an ordered nine-source partition")
    return enabled_tuple, disabled_tuple


def _contract_hash(contract: Mapping[str, Any]) -> str:
    body = {str(key): value for key, value in contract.items() if key != "deployment_contract_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deployment_semantics(
    payload: Mapping[str, Any], *, expected_role: str
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, bool]:
    """Validate the acquisition/inference boundary before Dataset construction."""

    format_version = payload.get("format_version")
    reconstructed_v2 = (
        format_version == RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION
    )
    if format_version not in {
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
    }:
        raise ValueError("unsupported deployment checkpoint format")
    if expected_role != "distilled_student":
        raise ValueError("deployment_student.pt must use distilled_student role")
    if payload.get("checkpoint_kind") != "deployment_student_only":
        raise ValueError("deployment checkpoint kind is invalid")
    if payload.get("teacher_required") is not False:
        raise ValueError("deployment checkpoint unexpectedly requires a teacher")
    if payload.get("reconstructors_required") is not reconstructed_v2:
        raise ValueError("deployment format/reconstructor requirement mismatch")
    contract = _mapping(payload.get("deployment_contract"), "deployment contract")
    if contract.get("deployment_contract_hash") != _contract_hash(contract):
        raise ValueError("deployment contract hash is missing or invalid")
    if list(contract.get("primary_modality_order", ())) != list(
        PRIMARY_MODALITY_GROUPS
    ):
        raise ValueError("deployment primary modality order is incompatible")
    if reconstructed_v2:
        enabled, disabled = _ordered_partition(
            contract.get("retained_input_groups", ()),
            contract.get("removed_groups", ()),
        )
        if (
            tuple(contract.get("student_input_groups", ())) != enabled
            or tuple(contract.get("disabled_at_student_boundary", ())) != disabled
        ):
            raise ValueError("deployment V1 compatibility group aliases mismatch")
    else:
        enabled, disabled = _ordered_partition(
            contract.get("student_input_groups", ()),
            contract.get("disabled_at_student_boundary", ()),
        )
    declared_deployable = contract.get(
        "deployable_latent_reconstruction", False
    )
    if not isinstance(declared_deployable, bool):
        raise ValueError("deployment reconstruction flag must be boolean")
    if declared_deployable is not reconstructed_v2:
        raise ValueError("deployment format/inference reconstruction mismatch")

    inference_mode = (
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE
        if reconstructed_v2
        else LEGACY_DISTILLED_INFERENCE_MODE
    )
    checkpoint_model_id = (
        "distilled_student_latent_reconstruction"
        if reconstructed_v2
        else "distilled_student"
    )
    # Old V1 artifacts predate the explicit fields. They remain readable, but
    # any field they do declare must agree with the inferred V1 semantics.
    strict = reconstructed_v2
    checks = {
        "inference_mode": inference_mode,
        "experiment_model_id": checkpoint_model_id,
    }
    for field, expected in checks.items():
        if strict or field in contract:
            if contract.get(field) != expected:
                raise ValueError(f"deployment contract {field} mismatch")
        if strict or field in payload:
            if payload.get(field) != expected:
                raise ValueError(f"deployment payload {field} mismatch")
    if (
        strict or "reconstructors_required_for_deployment" in contract
    ) and contract.get(
        "reconstructors_required_for_deployment", reconstructed_v2
    ) is not reconstructed_v2:
        raise ValueError("deployment reconstructor contract mismatch")

    expected_group_fields = {
        "retained_input_groups": enabled,
        "removed_groups": disabled,
        "acquired_source_groups": enabled,
        "native_encoder_groups": enabled,
        "reconstructed_source_groups": disabled if reconstructed_v2 else (),
        "reconstructed_groups": disabled if reconstructed_v2 else (),
        "reconstructed_fusion_groups": disabled if reconstructed_v2 else (),
    }
    for field, expected in expected_group_fields.items():
        if strict or field in contract:
            if tuple(contract.get(field, ())) != expected:
                raise ValueError(f"deployment contract {field} mismatch")
    if reconstructed_v2:
        if (
            contract.get("student_state_layout")
            != "pruned_retained_encoders_fusion_single_task_head"
            or list(contract.get("deployment_task_heads", ()))
            != [payload.get("task")]
        ):
            raise ValueError("deployment pruned state semantics mismatch")
        if contract.get(
            "reconstruction_target_space"
        ) != LATENT_RECONSTRUCTION_TARGET_SPACE:
            raise ValueError("deployment reconstruction target-space mismatch")
        if contract.get(
            "reconstruction_mask_semantics"
        ) != LATENT_RECONSTRUCTION_MASK_SEMANTICS:
            raise ValueError("deployment reconstruction mask semantics mismatch")
    return (
        enabled,
        disabled,
        inference_mode,
        checkpoint_model_id,
        reconstructed_v2,
    )


def checkpoint_source_partition(
    payload: Mapping[str, Any], *, expected_role: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read only the cryptographically checked acquisition boundary metadata."""

    if payload.get("engine_format_version") == ENGINE_FORMAT_VERSION:
        provenance = _mapping(payload.get("config_provenance"), "config provenance")
        subset = _mapping(provenance.get("modality_subset"), "modality subset")
        return _ordered_partition(
            subset.get("enabled_groups", ()), subset.get("disabled_groups", ())
        )
    if payload.get("format_version") in {
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
    }:
        enabled, disabled, _, _, _ = _deployment_semantics(
            payload, expected_role=expected_role
        )
        return enabled, disabled
    raise ValueError("checkpoint is neither an OT2 supervised best nor deployment student")


def inspect_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_task: str,
    expected_config_hash: str,
    expected_model_config: Any,
    expected_role: str,
    expected_model_id: Optional[str] = None,
) -> FrozenCheckpointContract:
    """Validate role, seed, threshold, architecture, and modality boundary."""

    if expected_role not in MODEL_ROLES:
        raise ValueError(f"model_role must be one of {sorted(MODEL_ROLES)}")
    if payload.get("task") != expected_task:
        raise ValueError("checkpoint task differs from the requested task")
    threshold = _threshold(payload)

    if payload.get("engine_format_version") == ENGINE_FORMAT_VERSION:
        if payload.get("checkpoint_kind") != "best":
            raise ValueError("supervised evaluation requires checkpoint_kind=best")
        if expected_role == "distilled_student":
            raise ValueError("distilled_student requires deployment_student.pt")
        provenance = _mapping(payload.get("config_provenance"), "config provenance")
        if provenance.get("config_hash") != expected_config_hash:
            raise ValueError("checkpoint/config hash mismatch")
        if _normalized(provenance.get("resolved_model_config")) != _normalized(
            expected_model_config
        ):
            raise ValueError("checkpoint resolved model architecture mismatch")
        subset = _mapping(provenance.get("modality_subset"), "modality subset")
        enabled, disabled = _ordered_partition(
            subset.get("enabled_groups", ()), subset.get("disabled_groups", ())
        )
        training = _mapping(payload.get("training_config"), "training config")
        if tuple(training.get("disabled_groups", ())) != disabled:
            raise ValueError("checkpoint training and provenance modality masks differ")
        try:
            seed = int(training["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint training seed is missing or invalid") from error
        if expected_role == "full_teacher" and disabled:
            raise ValueError("full_teacher checkpoint disables a modality")
        if expected_role == "cheap_baseline" and not disabled:
            raise ValueError("cheap_baseline checkpoint is not an acquisition subset")
        state = _mapping(payload.get("model_state"), "model state")
        return FrozenCheckpointContract(
            checkpoint_kind="supervised_best",
            model_role=expected_role,
            seed=seed,
            threshold=threshold,
            enabled_groups=enabled,
            disabled_groups=disabled,
            state=state,
            inference_mode=(
                "native_full_single_pass"
                if expected_role == "full_teacher"
                else "native_subset_single_pass_missing_tokens"
            ),
            checkpoint_model_id=(
                str(expected_model_id) if expected_model_id is not None else expected_role
            ),
            reconstructors_required=False,
        )

    if payload.get("format_version") in {
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
    }:
        enabled, disabled, inference_mode, checkpoint_model_id, reconstructed = (
            _deployment_semantics(payload, expected_role=expected_role)
        )
        if (
            expected_model_id is not None
            and str(expected_model_id) != checkpoint_model_id
        ):
            raise ValueError("requested model_id differs from deployment contract")
        if _normalized(payload.get("resolved_model_config")) != _normalized(
            expected_model_config
        ):
            raise ValueError("deployment resolved model architecture mismatch")
        provenance = _mapping(payload.get("provenance"), "deployment provenance")
        if provenance.get("config_hash") != expected_config_hash:
            raise ValueError("deployment/config hash mismatch")
        try:
            seed = int(provenance["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("deployment seed is missing or invalid") from error
        state = _mapping(payload.get("student_state"), "student state")
        return FrozenCheckpointContract(
            checkpoint_kind=(
                "distilled_reconstructed_deployment"
                if reconstructed
                else "distilled_deployment"
            ),
            model_role=expected_role,
            seed=seed,
            threshold=threshold,
            enabled_groups=enabled,
            disabled_groups=disabled,
            state=state,
            inference_mode=inference_mode,
            checkpoint_model_id=checkpoint_model_id,
            reconstructors_required=reconstructed,
        )

    raise ValueError("checkpoint is neither an OT2 supervised best nor deployment student")


def _retinal_caps(config: Any) -> dict[str, int]:
    return {
        "retinal_photography": int(config.dataset.retinal_max_assets["photography"]),
        "retinal_oct": int(config.dataset.retinal_max_assets["oct"]),
        "retinal_octa": int(config.dataset.retinal_max_assets["octa"]),
        "retinal_flio": int(config.dataset.retinal_max_assets["flio"]),
    }


def _validate_deployment_input_hashes(
    payload: Mapping[str, Any], config_path: Path, config: Any, dataset: OT2V2Dataset
) -> None:
    provenance = _mapping(payload.get("provenance"), "deployment provenance")
    files = _mapping(provenance.get("input_files"), "deployment input files")
    paths = {
        "config_yaml": config_path,
        "master_csv": config.dataset.master_csv,
        "master_metadata_json": config.dataset.master_metadata_json,
        "scaler_json": config.dataset.scaler_json,
        "clinical_schema_json": config.dataset.clinical_schema_json,
        "environment_schema_json": config.dataset.environment_schema_json,
        "wearable_schema_json": config.dataset.wearable_schema_json,
        "ecg_schema_json": config.dataset.ecg_schema_json,
    }
    if "clinical_data" in dataset.loaded_source_groups:
        paths["clinical_features_csv"] = dataset.clinical_features_path
    for name, path in paths.items():
        entry = _mapping(files.get(name), f"deployment input file {name}")
        if entry.get("sha256") != _sha256_file(Path(path)):
            raise ValueError(f"deployment/current input file hash mismatch: {name}")
    counts = _mapping(provenance.get("split_counts"), "deployment split counts")
    if int(counts.get(dataset.index.iloc[0]["split"], -1)) != len(dataset):
        raise ValueError("deployment/current evaluation split count mismatch")


def _locked_absolute_path(cell: Mapping[str, Any], field: str) -> Path:
    raw = str(cell.get(field) or "").strip()
    candidate = Path(raw).expanduser()
    if not raw or not candidate.is_absolute():
        raise ValueError(f"locked test {field} must be an absolute path")
    return candidate.resolve()


def _locked_digest(cell: Mapping[str, Any], field: str) -> str:
    digest = str(cell.get(field) or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"locked test {field} is not a lowercase SHA-256 digest")
    return digest


def _config_dataset_identity_from_verified_bytes(
    config_bytes: bytes, config_path: Path
) -> Mapping[str, Any]:
    """Safely extract only pre-deserialization identity fields from locked YAML."""

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("the frozen evaluator requires a .yaml or .yml config")
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise RuntimeError("YAML preflight requires PyYAML") from error
    try:
        raw = yaml.safe_load(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("locked config is not valid safe YAML") from error
    root = _mapping(raw, "locked config")
    dataset = _mapping(root.get("dataset"), "locked config dataset")

    def resolve_config_path(field: str) -> Path:
        value = str(dataset.get(field) or "").strip()
        if not value:
            raise ValueError(f"locked config dataset.{field} is missing")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return candidate.resolve()

    task = str(dataset.get("task") or "").strip()
    if not task:
        raise ValueError("locked config dataset.task is missing")
    return {
        "task": task,
        "master_path": resolve_config_path("master_csv"),
        "scaler_path": resolve_config_path("scaler_json"),
    }


def _preflight_locked_test(args: argparse.Namespace) -> Mapping[str, Any]:
    """Authorize immutable paths and hashes before any model/data deserialization."""

    if args.selection_lock is None:
        raise ValueError("test evaluation requires --selection-lock")
    lock_path = Path(args.selection_lock).expanduser().resolve()
    lock = _mapping(
        _json_without_duplicate_keys(lock_path, "selection lock"),
        "selection lock",
    )
    if lock.get("schema_version") == SUPERVISED_FINAL_LOCK_VERSION:
        return validate_supervised_final_lock_preflight(
            lock_path,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_path=args.out,
            model_id=args.model_id,
            model_role=args.model_role,
            seed=int(args.seed),
        )
    if lock.get("schema_version") != SELECTION_LOCK_VERSION:
        raise ValueError("selection lock schema is incompatible")
    if (
        lock.get("analysis_role") != "validation_cost_selection"
        or lock.get("selection_split") != "val"
    ):
        raise ValueError("test requires a validation cost-selection lock")
    task = str(lock.get("task") or "").strip()
    if not task:
        raise ValueError("selection lock task is missing")
    arms = _mapping(lock.get("locked_arms"), "locked arms")
    if args.model_id not in arms:
        raise ValueError("model_id is not authorized by the selection lock")
    arm = _mapping(arms[args.model_id], f"locked arm {args.model_id}")
    if arm.get("model_role") != args.model_role:
        raise ValueError("model role differs from the selection lock")
    seeds = _mapping(arm.get("seeds"), f"locked seeds for {args.model_id}")
    seed_key = str(int(args.seed))
    if seed_key not in seeds:
        raise ValueError("seed is not authorized by the selection lock")
    cell = _mapping(seeds[seed_key], f"locked cell {args.model_id}/{seed_key}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    for name, cli_path in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
    ):
        locked_path = _locked_absolute_path(cell, f"{name}_path")
        if locked_path != cli_path:
            raise ValueError(f"locked test {name}_path mismatch")
        expected_digest = _locked_digest(cell, f"{name}_sha256")
        if _sha256_file(cli_path) != expected_digest:
            raise ValueError(f"locked test {name}_sha256 mismatch")

    # Parse only after the exact config bytes have matched the lock. No model
    # checkpoint has been deserialized and no tabular Dataset has been opened.
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != _locked_digest(
        cell, "config_sha256"
    ):
        raise ValueError("locked config changed during preflight")
    identity = _config_dataset_identity_from_verified_bytes(
        config_bytes, config_path
    )
    if identity["task"] != task:
        raise ValueError("locked config task differs from selection lock")
    for name in ("master", "scaler"):
        config_value = Path(identity[f"{name}_path"])
        locked_value = _locked_absolute_path(cell, f"{name}_path")
        if config_value != locked_value:
            raise ValueError(f"locked config {name}_path mismatch")
        if _sha256_file(config_value) != _locked_digest(
            cell, f"{name}_sha256"
        ):
            raise ValueError(f"locked test {name}_sha256 mismatch")
    return {
        "path": str(lock_path),
        "sha256": _sha256_file(lock_path),
        "task": task,
        "model_id": str(args.model_id),
        "model_role": str(args.model_role),
        "seed": int(args.seed),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "master_path": str(identity["master_path"]),
        "scaler_path": str(identity["scaler_path"]),
        "stage": "pre_deserialization_lock_authorized",
    }


def authorize_locked_test(
    lock_path: Path,
    *,
    task: str,
    model_id: str,
    role: str,
    seed: int,
    threshold: float,
    checkpoint_path: Path,
    config_path: Path,
    master_path: Path,
    scaler_path: Path,
) -> Mapping[str, Any]:
    """Authorize exactly one immutable cell; final analyzer replays selection."""

    lock = _mapping(
        _json_without_duplicate_keys(lock_path, "selection lock"), "selection lock"
    )
    if lock.get("schema_version") == SUPERVISED_FINAL_LOCK_VERSION:
        raise ValueError(
            "supervised final locks require authorize_supervised_final_lock with "
            "master metadata and schema bindings"
        )
    if lock.get("schema_version") != SELECTION_LOCK_VERSION:
        raise ValueError("selection lock schema is incompatible")
    if lock.get("analysis_role") != "validation_cost_selection" or lock.get(
        "selection_split"
    ) != "val":
        raise ValueError("test requires a validation cost-selection lock")
    if lock.get("task") != task:
        raise ValueError("selection lock task mismatch")
    arms = _mapping(lock.get("locked_arms"), "locked arms")
    if model_id not in arms:
        raise ValueError("model_id is not authorized by the selection lock")
    arm = _mapping(arms[model_id], f"locked arm {model_id}")
    if arm.get("model_role") != role:
        raise ValueError("model role differs from the selection lock")
    seeds = _mapping(arm.get("seeds"), f"locked seeds for {model_id}")
    if str(seed) not in seeds:
        raise ValueError("seed is not authorized by the selection lock")
    cell = _mapping(seeds[str(seed)], f"locked cell {model_id}/{seed}")
    expected = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "config_sha256": _sha256_file(config_path),
        "master_sha256": _sha256_file(master_path),
        "scaler_sha256": _sha256_file(scaler_path),
    }
    paths = {
        "checkpoint": checkpoint_path.resolve(),
        "config": config_path.resolve(),
        "master": master_path.resolve(),
        "scaler": scaler_path.resolve(),
    }
    for name, resolved in paths.items():
        if _locked_absolute_path(cell, f"{name}_path") != resolved:
            raise ValueError(f"locked test {name}_path mismatch")
    for key, digest in expected.items():
        if cell.get(key) != digest:
            raise ValueError(f"locked test {key} mismatch")
    try:
        locked_threshold = float(cell.get("threshold_from_validation"))
    except (TypeError, ValueError) as error:
        raise ValueError("locked validation threshold is invalid") from error
    if not math.isclose(locked_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("checkpoint threshold differs from the validation lock")
    return {
        "path": str(lock_path),
        "sha256": _sha256_file(lock_path),
        "model_id": model_id,
        "model_role": role,
        "seed": seed,
        "validation_threshold": locked_threshold,
    }


def _validate_state(state: Mapping[str, Any], model: Task2V2Model) -> None:
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            f"checkpoint state key mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"checkpoint tensor shape/type mismatch: {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint tensor contains non-finite values: {name}")
    model.load_state_dict(state, strict=True)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def run_frozen_evaluation(args: argparse.Namespace) -> Mapping[str, Any]:
    preflight_lock: Optional[Mapping[str, Any]] = None
    if args.split == "test":
        # This gate intentionally precedes config parsing, checkpoint
        # deserialization, Dataset construction, and every clinical CSV read.
        preflight_lock = _preflight_locked_test(args)
    config_path, config = load_supervised_config(args.config)
    validate_model_v2_lineage(config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if preflight_lock is not None:
        if (
            str(config_path) != preflight_lock["config_path"]
            or str(checkpoint_path) != preflight_lock["checkpoint_path"]
            or config.dataset.task != preflight_lock["task"]
            or str(config.dataset.master_csv.resolve())
            != preflight_lock["master_path"]
            or str(config.dataset.scaler_json.resolve())
            != preflight_lock["scaler_path"]
        ):
            raise ValueError("fully parsed config differs from locked preflight identity")
    checkpoint = _safe_torch_load(checkpoint_path)

    # Read the declared acquisition boundary before Dataset construction.  This
    # prevents even the clinical-wide CSV from being opened for an omitted
    # clinical modality, while the scaler still supplies its schema dimension.
    enabled_groups, _ = checkpoint_source_partition(
        checkpoint, expected_role=args.model_role
    )
    # Disabled participant artifacts, including very large FLIO archives, are
    # therefore never opened by this evaluation Dataset.
    dataset = OT2V2Dataset(
        config.dataset.master_csv,
        config.dataset.scaler_json,
        task=config.dataset.task,
        artifact_root=config.dataset.artifact_root,
        split=args.split,
        master_metadata_json=config.dataset.master_metadata_json,
        environment_schema_json=config.dataset.environment_schema_json,
        max_retinal_assets_per_group=_retinal_caps(config),
        require_y=True,
        loaded_source_groups=enabled_groups,
    )
    declared_model_config = build_model_config(
        config, dataset.model_feature_dimensions
    )
    contract = inspect_checkpoint(
        checkpoint,
        expected_task=config.dataset.task,
        expected_config_hash=config.config_hash,
        expected_model_config=declared_model_config,
        expected_role=args.model_role,
        expected_model_id=args.model_id,
    )
    if tuple(dataset.loaded_source_groups) != contract.enabled_groups:
        raise RuntimeError("Dataset loaded groups differ from checkpoint acquisition contract")
    if set(dataset.loaded_source_groups).intersection(contract.disabled_groups):
        raise RuntimeError("Dataset loaded a disabled/reconstructed source group")
    if checkpoint.get("engine_format_version") == ENGINE_FORMAT_VERSION:
        _validate_checkpoint_input_file_hashes(checkpoint, config)
        if "clinical_data" in contract.enabled_groups:
            _validate_clinical_feature_bytes(checkpoint, dataset)
        _validate_evaluation_cohort_provenance(checkpoint, dataset, args.split)
        _validate_resolved_model_config(checkpoint, declared_model_config)
    else:
        _validate_deployment_input_hashes(checkpoint, config_path, config, dataset)

    if int(args.seed) != contract.seed:
        raise ValueError(
            f"requested seed {args.seed} differs from checkpoint seed {contract.seed}"
        )
    lock_audit: Optional[Mapping[str, Any]] = None
    if args.split == "test":
        lock_path = Path(args.selection_lock).expanduser().resolve()
        assert preflight_lock is not None
        if preflight_lock.get("schema_version") == SUPERVISED_FINAL_LOCK_VERSION:
            lock_audit = authorize_supervised_final_lock(
                lock_path,
                task=config.dataset.task,
                model_id=args.model_id,
                model_role=args.model_role,
                seed=contract.seed,
                threshold=contract.threshold,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                master_path=config.dataset.master_csv,
                master_metadata_path=config.dataset.master_metadata_json,
                scaler_path=config.dataset.scaler_json,
                schema_paths={
                    "clinical_schema": config.dataset.clinical_schema_json,
                    "environment_schema": config.dataset.environment_schema_json,
                    "wearable_schema": config.dataset.wearable_schema_json,
                    "ecg_schema": config.dataset.ecg_schema_json,
                },
            )
        else:
            lock_audit = authorize_locked_test(
                lock_path,
                task=config.dataset.task,
                model_id=args.model_id,
                role=args.model_role,
                seed=contract.seed,
                threshold=contract.threshold,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                master_path=config.dataset.master_csv,
                scaler_path=config.dataset.scaler_json,
            )
    elif args.selection_lock is not None:
        raise ValueError("validation evaluation must not consume a test selection lock")

    # Initialization-only weights are set to None because the complete learned
    # state is loaded below.  This makes frozen inference safe on offline nodes.
    runtime_model_config = dataclasses.replace(
        declared_model_config,
        retinal_weights=None,
        retinal_weights_path=None,
    )
    native_model = Task2V2Model(runtime_model_config)
    if contract.reconstructors_required:
        model, reloaded_payload = load_deployment_inference_model(
            native_model,
            checkpoint_path,
            expected_task=config.dataset.task,
        )
        # The strict loader performs an independent on-disk read.  Recheck the
        # exact dispatch fields against the already inspected bytes so a future
        # loader cannot silently reinterpret the artifact.
        if (
            reloaded_payload.get("format_version")
            != RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION
            or reloaded_payload.get("experiment_model_id")
            != contract.checkpoint_model_id
            or reloaded_payload.get("inference_mode") != contract.inference_mode
        ):
            raise ValueError("reloaded deployment semantics differ from inspection")
    else:
        model = native_model
        _validate_state(contract.state, model)
    device = resolve_device(args.device)
    set_deterministic_seed(contract.seed, deterministic=True)
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=task2_v2_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )
    result = evaluate_binary(
        model,
        loader,
        device=device,
        task=config.dataset.task,
        threshold=contract.threshold,
        use_bf16_amp=bool(args.bf16),
        expected_split=args.split,
        disabled_groups=contract.disabled_groups,
    )

    output = Path(args.out).expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    supervised_one_time_test = bool(
        args.split == "test"
        and preflight_lock is not None
        and preflight_lock.get("schema_version") == SUPERVISED_FINAL_LOCK_VERSION
    )
    if (supervised_one_time_test or not args.overwrite) and (
        output.exists() or metadata_path.exists()
    ):
        raise FileExistsError(
            f"refusing to overwrite frozen evaluation evidence: {output}"
        )
    save_prediction_csv(
        result, output, task=config.dataset.task, split=args.split
    )
    metadata = {
        "schema_version": EVALUATION_FORMAT_VERSION,
        "analysis_role": (
            "locked_confirmatory_test"
            if args.split == "test"
            else "validation_evidence_reconstruction"
        ),
        "task": config.dataset.task,
        "split": args.split,
        "model_id": args.model_id,
        "model_role": args.model_role,
        "seed": contract.seed,
        "checkpoint_kind": contract.checkpoint_kind,
        "checkpoint_model_id": contract.checkpoint_model_id,
        "experiment_model_id": contract.checkpoint_model_id,
        "inference_mode": contract.inference_mode,
        "reconstructors_required": contract.reconstructors_required,
        "evaluation_capability": (
            LATENT_RECONSTRUCTION_EVALUATION_CAPABILITY
            if contract.reconstructors_required
            else EVALUATION_FORMAT_VERSION
        ),
        "enabled_groups": list(contract.enabled_groups),
        "disabled_groups": list(contract.disabled_groups),
        "retained_input_groups": list(contract.enabled_groups),
        "removed_groups": list(contract.disabled_groups),
        "acquired_source_groups": list(contract.enabled_groups),
        "native_encoder_groups": list(contract.enabled_groups),
        "reconstructed_groups": (
            list(contract.disabled_groups)
            if contract.reconstructors_required
            else []
        ),
        "reconstruction_target_space": (
            LATENT_RECONSTRUCTION_TARGET_SPACE
            if contract.reconstructors_required
            else None
        ),
        "reconstruction_mask_semantics": (
            LATENT_RECONSTRUCTION_MASK_SEMANTICS
            if contract.reconstructors_required
            else None
        ),
        "dataset_loaded_groups": list(dataset.loaded_source_groups),
        "reconstructed_groups_loaded_from_dataset": [],
        "disabled_participant_artifacts_opened": False,
        "reconstructed_participant_artifacts_opened": False,
        "threshold_from_validation": contract.threshold,
        "n_participants": len(result.person_ids),
        "metrics": result.metrics,
        "source_files": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
            },
            "config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "master": {
                "path": str(config.dataset.master_csv),
                "sha256": _sha256_file(config.dataset.master_csv),
            },
            "scaler": {
                "path": str(config.dataset.scaler_json),
                "sha256": _sha256_file(config.dataset.scaler_json),
            },
        },
        "selection_lock": lock_audit,
        "pre_deserialization_selection_lock": preflight_lock,
        "predictions": {
            "path": str(output),
            "sha256": _sha256_file(output),
        },
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "bf16": bool(args.bf16),
            "retinal_initialization_during_checkpoint_load": "none",
            "model_runtime": (
                "pruned_two_pass_deployment_wrapper"
                if contract.reconstructors_required
                else "task2_v2_native_model"
            ),
        },
    }
    _atomic_json(metadata, metadata_path)
    return {"predictions": output, "metadata": metadata_path, **metadata}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one immutable OT2 V2 checkpoint without retraining."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-role", required=True, choices=sorted(MODEL_ROLES))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--selection-lock", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", "--num-workers", dest="workers", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.seed < 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("seed/workers must be non-negative and batch-size positive")
    result = run_frozen_evaluation(args)
    print(
        json.dumps(
            {
                "predictions": str(result["predictions"]),
                "metadata": str(result["metadata"]),
                "task": result["task"],
                "split": result["split"],
                "model_id": result["model_id"],
                "seed": result["seed"],
                "n_participants": result["n_participants"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
