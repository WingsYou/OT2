#!/usr/bin/env python3
"""Freeze one validation-selected supervised checkpoint for one-time test use.

This is the cost-independent counterpart to the cost-selection lock.  Version
one is deliberately narrow: it authorizes exactly one ``full_teacher`` model,
trained with seed 42, for one task and one output path.  The lock can be made
only from a validation-screen run whose checkpoint, threshold, data lineage,
and full-retinal sidecar inventory are internally consistent.

The frozen evaluator imports the preflight helpers below so authorization and
all byte-level hashes are checked before checkpoint deserialization or Dataset
construction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from ot2.task2_v2_config import MODEL_SCHEMA_VERSION, canonical_json_hash
    from ot2.task2_v2_engine import ENGINE_FORMAT_VERSION
    from ot2.task2_v2_model import PRIMARY_MODALITY_GROUPS
except ModuleNotFoundError:  # Direct execution from the ot2 directory.
    from task2_v2_config import MODEL_SCHEMA_VERSION, canonical_json_hash  # type: ignore
    from task2_v2_engine import ENGINE_FORMAT_VERSION  # type: ignore
    from task2_v2_model import PRIMARY_MODALITY_GROUPS  # type: ignore


SUPERVISED_FINAL_LOCK_VERSION = "ot2-supervised-final-lock-v2-1"
SUPERVISED_FINAL_LOCK_ROLE = "validation_supervised_final_model_lock"
LOCKED_MODEL_ID = "full_teacher"
LOCKED_MODEL_ROLE = "full_teacher"
LOCKED_SEED = 42

_SOURCE_NAMES = (
    "config",
    "checkpoint",
    "run_summary",
    "validation_predictions",
    "master",
    "master_metadata",
    "scaler",
    "clinical_schema",
    "environment_schema",
    "wearable_schema",
    "ecg_schema",
    "retinal_sidecar_summary",
)
_DATASET_PATH_FIELDS = {
    "master": "master_csv",
    "master_metadata": "master_metadata_json",
    "scaler": "scaler_json",
    "clinical_schema": "clinical_schema_json",
    "environment_schema": "environment_schema_json",
    "wearable_schema": "wearable_schema_json",
    "ecg_schema": "ecg_schema_json",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    return _mapping(value, label)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _absolute_path(value: Any, label: str) -> Path:
    text = str(value or "").strip()
    candidate = Path(text).expanduser()
    if not text or not candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return candidate.resolve()


def _source_record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"locked source does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _verify_source_record(value: Any, label: str) -> Path:
    record = _mapping(value, label)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _absolute_path(record.get("path"), f"{label}.path")
    expected = _digest(record.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{label} sha256 mismatch")
    return path


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{label} must be a .yaml or .yml file")
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise RuntimeError("YAML final-lock validation requires PyYAML") from error
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid safe YAML") from error
    return _mapping(value, label)


def _resolve_from_config(value: Any, config_path: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _config_identity(config_path: Path) -> Mapping[str, Any]:
    raw = _load_yaml(config_path, "supervised config")
    if raw.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("supervised config schema_version is incompatible")
    dataset = _mapping(raw.get("dataset"), "supervised config dataset")
    training = _mapping(raw.get("training"), "supervised config training")
    paths = {
        name: _resolve_from_config(dataset.get(field), config_path, f"dataset.{field}")
        for name, field in _DATASET_PATH_FIELDS.items()
    }
    try:
        seed = int(raw.get("seed"))
    except (TypeError, ValueError) as error:
        raise ValueError("supervised config seed is invalid") from error
    return {
        "raw": raw,
        "config_hash": canonical_json_hash(raw),
        "task": str(dataset.get("task") or "").strip(),
        "seed": seed,
        "evaluation_phase": str(
            training.get("evaluation_phase", "validation_screen")
        ).strip().lower(),
        "paths": paths,
    }


def _threshold(value: Any, label: str) -> float:
    selected = _mapping(value, label)
    try:
        result = float(selected.get("threshold"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.threshold is invalid") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label}.threshold must be finite and in [0, 1]")
    if str(selected.get("method") or "").strip().lower() not in {"fixed", "youden"}:
        raise ValueError(f"{label} is not validation-derived")
    for key in ("selection_split", "threshold_selection_split", "source_split"):
        split = str(selected.get(key) or "").strip().lower()
        if split and split not in {"val", "validation"}:
            raise ValueError(f"{label} source is not validation")
    return result


def _safe_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "secure final-lock creation requires torch.load(..., weights_only=True)"
        ) from error
    return _mapping(payload, "checkpoint")


def _file_entry_matches(
    entry: Any, *, expected_path: Path, expected_sha256: str, label: str
) -> None:
    record = _mapping(entry, label)
    if Path(str(record.get("path") or "")).expanduser().resolve() != expected_path:
        raise ValueError(f"{label} path mismatch")
    if str(record.get("sha256") or "") != expected_sha256:
        raise ValueError(f"{label} sha256 mismatch")


def _validate_validation_predictions(
    path: Path, *, task: str, threshold: float
) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"person_id", "task", "split", "threshold"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("validation predictions have an incompatible schema")
        count = 0
        for row in reader:
            count += 1
            if str(row.get("task") or "") != task:
                raise ValueError("validation predictions task mismatch")
            if str(row.get("split") or "").strip().lower() not in {"val", "validation"}:
                raise ValueError("validation predictions contain a non-validation row")
            try:
                observed = float(row.get("threshold", ""))
            except ValueError as error:
                raise ValueError("validation predictions contain an invalid threshold") from error
            if not math.isclose(observed, threshold, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("validation predictions threshold mismatch")
    if count <= 0:
        raise ValueError("validation predictions are empty")
    return count


def _retinal_audit(
    summary: Mapping[str, Any],
    master_metadata: Mapping[str, Any],
    config_raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    integer_fields = (
        "n_successful_input_rows",
        "n_selected",
        "n_omitted_by_cap",
        "n_processed",
        "n_cached",
        "n_failed",
    )
    try:
        counts = {name: int(summary.get(name)) for name in integer_fields}
    except (TypeError, ValueError) as error:
        raise ValueError("retinal sidecar summary has invalid counts") from error
    if counts["n_successful_input_rows"] <= 0:
        raise ValueError("retinal full sidecar inventory is empty")
    if counts["n_omitted_by_cap"] != 0 or counts["n_failed"] != 0:
        raise ValueError("retinal sidecar summary is not a failure-free full inventory")
    if counts["n_selected"] != counts["n_successful_input_rows"]:
        raise ValueError("retinal sidecar summary omitted successful input assets")
    if counts["n_processed"] + counts["n_cached"] != counts["n_selected"]:
        raise ValueError("retinal sidecar processing counts do not cover the full inventory")
    if summary.get("canonical_artifacts_modified") is not False:
        raise ValueError("retinal sidecar summary does not preserve canonical artifacts")
    model_view_hash = str(summary.get("model_view_config_hash") or "").strip()
    if not model_view_hash:
        raise ValueError("retinal sidecar summary lacks model_view_config_hash")
    sidecar_config = _mapping(
        summary.get("model_view_config"), "retinal sidecar model_view_config"
    )
    sidecar_caps_raw = _mapping(
        sidecar_config.get("selection_caps_per_person_modality"),
        "retinal sidecar selection caps",
    )
    dataset = _mapping(config_raw.get("dataset"), "supervised config dataset")
    retinal = _mapping(dataset.get("retinal"), "supervised config retinal")
    dataset_caps_raw = _mapping(retinal.get("max_assets"), "supervised config retinal caps")
    modalities = ("photography", "oct", "octa", "flio")
    try:
        sidecar_caps = {name: int(sidecar_caps_raw[name]) for name in modalities}
        dataset_caps = {name: int(dataset_caps_raw[name]) for name in modalities}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("retinal sidecar/config caps are incomplete or invalid") from error
    if sidecar_caps != dataset_caps:
        raise ValueError("model retinal caps differ from full-sidecar selection caps")

    indexes = _mapping(master_metadata.get("source_indexes"), "master source_indexes")
    retinal_index = _mapping(indexes.get("retinal"), "master retinal source index")
    if int(retinal_index.get("n_successful_assets", -1)) != counts["n_selected"]:
        raise ValueError("master retinal asset count differs from full sidecar summary")
    if str(retinal_index.get("model_view_config_hash") or "") != model_view_hash:
        raise ValueError("master retinal model-view hash differs from sidecar summary")
    preprocessing = _mapping(
        master_metadata.get("preprocessing_config"), "master preprocessing_config"
    )
    adapter_hashes = _mapping(
        preprocessing.get("source_index_adapter_hashes", {}),
        "master source_index_adapter_hashes",
    )
    if str(adapter_hashes.get("retinal") or "") != model_view_hash:
        raise ValueError("master preprocessing contract does not bind the retinal sidecar")
    return {
        **counts,
        "model_view_config_hash": model_view_hash,
        "selection_caps_per_person_modality": sidecar_caps,
    }


def _lock_content_hash(payload: Mapping[str, Any]) -> str:
    body = {str(key): value for key, value in payload.items() if key != "lock_content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_supervised_final_lock(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    run_summary_path: str | Path,
    retinal_sidecar_summary_path: str | Path,
    authorized_output_path: str | Path,
    lock_path: str | Path,
    model_id: str = LOCKED_MODEL_ID,
    model_role: str = LOCKED_MODEL_ROLE,
    seed: int = LOCKED_SEED,
) -> Mapping[str, Any]:
    """Validate validation-only evidence and atomically write one final lock."""

    if model_id != LOCKED_MODEL_ID or model_role != LOCKED_MODEL_ROLE:
        raise ValueError("supervised final-lock V1 supports only full_teacher")
    if int(seed) != LOCKED_SEED:
        raise ValueError("supervised final-lock V1 supports only seed 42")
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run_summary = Path(run_summary_path).expanduser().resolve()
    retinal_summary_path = Path(retinal_sidecar_summary_path).expanduser().resolve()
    authorized_output = Path(authorized_output_path).expanduser().resolve()
    authorized_metadata = authorized_output.with_suffix(
        authorized_output.suffix + ".metadata.json"
    )
    destination = Path(lock_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite final lock: {destination}")
    if authorized_output.exists() or authorized_metadata.exists():
        raise FileExistsError("authorized test output already exists; prior test evidence detected")
    if checkpoint.parent != run_summary.parent:
        raise ValueError("checkpoint and run_summary must come from the same run directory")

    identity = _config_identity(config)
    if not identity["task"]:
        raise ValueError("supervised config task is missing")
    if identity["seed"] != LOCKED_SEED:
        raise ValueError("supervised final-lock config must prespecify seed 42")
    if identity["evaluation_phase"] != "validation_screen":
        raise ValueError("supervised final lock requires training.evaluation_phase=validation_screen")

    source_paths = dict(identity["paths"])
    for name, path in source_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"config source {name} does not exist: {path}")
    checkpoint_payload = _safe_checkpoint(checkpoint)
    if checkpoint_payload.get("engine_format_version") != ENGINE_FORMAT_VERSION:
        raise ValueError("checkpoint engine format is incompatible")
    if checkpoint_payload.get("checkpoint_kind") != "best":
        raise ValueError("final lock requires the validation-selected best checkpoint")
    if checkpoint_payload.get("task") != identity["task"]:
        raise ValueError("checkpoint task differs from config")
    checkpoint_threshold = _threshold(
        checkpoint_payload.get("selected_threshold"), "checkpoint selected_threshold"
    )
    provenance = _mapping(
        checkpoint_payload.get("config_provenance"), "checkpoint config_provenance"
    )
    if provenance.get("config_hash") != identity["config_hash"]:
        raise ValueError("checkpoint/config hash mismatch")
    if Path(str(provenance.get("config_path") or "")).expanduser().resolve() != config:
        raise ValueError("checkpoint config_path mismatch")
    runtime = _mapping(provenance.get("runtime"), "checkpoint config runtime")
    if Path(str(runtime.get("output_dir") or "")).expanduser().resolve() != checkpoint.parent:
        raise ValueError("checkpoint is not in its provenance-bound run directory")
    subset = _mapping(provenance.get("modality_subset"), "checkpoint modality_subset")
    if tuple(subset.get("enabled_groups", ())) != tuple(PRIMARY_MODALITY_GROUPS):
        raise ValueError("full_teacher checkpoint does not enable all modality groups")
    if tuple(subset.get("disabled_groups", ())) != ():
        raise ValueError("full_teacher checkpoint disables a modality group")
    training = _mapping(checkpoint_payload.get("training_config"), "checkpoint training_config")
    if int(training.get("seed", -1)) != LOCKED_SEED:
        raise ValueError("checkpoint seed differs from locked seed 42")
    if str(training.get("evaluation_phase") or "").strip().lower() != "validation_screen":
        raise ValueError("checkpoint was not produced by validation_screen training")
    data_provenance = _mapping(
        checkpoint_payload.get("data_provenance"), "checkpoint data_provenance"
    )
    if data_provenance.get("test_role") != "not_iterated_during_validation_screen":
        raise ValueError("checkpoint contains evidence that test was not held untouched")

    run = _read_json(run_summary, "run summary")
    if run.get("engine_format_version") != ENGINE_FORMAT_VERSION:
        raise ValueError("run summary engine format is incompatible")
    if run.get("task") != identity["task"]:
        raise ValueError("run summary task differs from config")
    if str(run.get("evaluation_phase") or "").strip().lower() != "validation_screen":
        raise ValueError("run summary is not a validation_screen run")
    if run.get("test_evaluated") is not False or run.get("test_metrics") is not None:
        raise ValueError("run summary contains evidence of prior test evaluation")
    run_threshold = _threshold(run.get("threshold_selection"), "run threshold_selection")
    if not math.isclose(run_threshold, checkpoint_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("run-summary and checkpoint validation thresholds differ")
    run_config_provenance = _mapping(
        run.get("config_provenance"), "run summary config_provenance"
    )
    if run_config_provenance.get("config_hash") != identity["config_hash"]:
        raise ValueError("run-summary/config hash mismatch")
    run_data_provenance = _mapping(
        run.get("data_provenance"), "run summary data_provenance"
    )
    if run_data_provenance.get("test_role") != "not_iterated_during_validation_screen":
        raise ValueError("run summary does not certify an untouched test split")

    validation_predictions_raw = str(run.get("validation_predictions") or "").strip()
    if not validation_predictions_raw:
        raise ValueError("run summary lacks validation_predictions")
    validation_predictions = Path(validation_predictions_raw).expanduser()
    if not validation_predictions.is_absolute():
        validation_predictions = run_summary.parent / validation_predictions
    validation_predictions = validation_predictions.resolve()
    if not validation_predictions.is_file():
        raise FileNotFoundError(
            f"validation predictions do not exist: {validation_predictions}"
        )
    validation_count = _validate_validation_predictions(
        validation_predictions,
        task=str(identity["task"]),
        threshold=checkpoint_threshold,
    )
    checkpoint_split_counts = _mapping(
        data_provenance.get("split_counts"), "checkpoint split_counts"
    )
    run_split_counts = _mapping(
        run_data_provenance.get("split_counts"), "run summary split_counts"
    )
    if (
        int(checkpoint_split_counts.get("val", -1)) != validation_count
        or int(run_split_counts.get("val", -1)) != validation_count
    ):
        raise ValueError("validation prediction count differs from checkpoint/run lineage")

    known_test_outputs = set(run_summary.parent.glob("*test*predictions*.csv"))
    known_test_outputs.update(run_summary.parent.glob("*test*predictions*.csv.metadata.json"))
    if known_test_outputs:
        raise FileExistsError(
            "run directory already contains test-prediction evidence: "
            + ", ".join(str(path) for path in sorted(known_test_outputs))
        )

    source_records = {
        "config": _source_record(config),
        "checkpoint": _source_record(checkpoint),
        "run_summary": _source_record(run_summary),
        "validation_predictions": _source_record(validation_predictions),
        **{
            name: _source_record(Path(path))
            for name, path in source_paths.items()
        },
        "retinal_sidecar_summary": _source_record(retinal_summary_path),
    }
    checkpoint_files = _mapping(data_provenance.get("files"), "checkpoint data files")
    for name, path in source_paths.items():
        _file_entry_matches(
            checkpoint_files.get(_DATASET_PATH_FIELDS[name]),
            expected_path=Path(path),
            expected_sha256=source_records[name]["sha256"],
            label=f"checkpoint data file {_DATASET_PATH_FIELDS[name]}",
        )

    master_metadata = _read_json(Path(source_paths["master_metadata"]), "master metadata")
    retinal_summary = _read_json(retinal_summary_path, "retinal sidecar summary")
    retinal_audit = _retinal_audit(
        retinal_summary, master_metadata, _mapping(identity["raw"], "config raw")
    )

    prior_test_audit = {
        "config_evaluation_phase": "validation_screen",
        "run_summary_evaluation_phase": "validation_screen",
        "run_summary_test_evaluated": False,
        "run_summary_test_metrics": None,
        "checkpoint_test_role": "not_iterated_during_validation_screen",
        "run_directory_test_prediction_files": [],
        "authorized_output_existed_at_lock_time": False,
        "authorized_metadata_existed_at_lock_time": False,
    }
    payload: dict[str, Any] = {
        "schema_version": SUPERVISED_FINAL_LOCK_VERSION,
        "analysis_role": SUPERVISED_FINAL_LOCK_ROLE,
        "selection_split": "val",
        "authorized_split": "test",
        "task": identity["task"],
        "model_id": LOCKED_MODEL_ID,
        "model_role": LOCKED_MODEL_ROLE,
        "seed": LOCKED_SEED,
        "threshold_from_validation": checkpoint_threshold,
        "validation_participants": validation_count,
        "source_files": source_records,
        "retinal_full_sidecar_audit": retinal_audit,
        "authorized_output": {
            "predictions_path": str(authorized_output),
            "metadata_path": str(authorized_metadata),
        },
        "prior_test_evidence_audit": prior_test_audit,
    }
    payload["lock_content_sha256"] = _lock_content_hash(payload)
    _atomic_json(payload, destination)
    return payload


def validate_supervised_final_lock_preflight(
    lock_path: str | Path,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    model_id: str,
    model_role: str,
    seed: int,
) -> Mapping[str, Any]:
    """Authorize and hash-check a supervised test cell before data loading."""

    lock_file = Path(lock_path).expanduser().resolve()
    lock = _read_json(lock_file, "supervised final lock")
    required = {
        "schema_version",
        "analysis_role",
        "selection_split",
        "authorized_split",
        "task",
        "model_id",
        "model_role",
        "seed",
        "threshold_from_validation",
        "validation_participants",
        "source_files",
        "retinal_full_sidecar_audit",
        "authorized_output",
        "prior_test_evidence_audit",
        "lock_content_sha256",
    }
    if set(lock) != required:
        raise ValueError("supervised final lock has missing or unexpected fields")
    if lock.get("schema_version") != SUPERVISED_FINAL_LOCK_VERSION:
        raise ValueError("supervised final lock schema is incompatible")
    if (
        lock.get("analysis_role") != SUPERVISED_FINAL_LOCK_ROLE
        or lock.get("selection_split") != "val"
        or lock.get("authorized_split") != "test"
    ):
        raise ValueError("supervised final lock is not validation-derived test authorization")
    if lock.get("lock_content_sha256") != _lock_content_hash(lock):
        raise ValueError("supervised final lock content hash mismatch")
    if (
        model_id != LOCKED_MODEL_ID
        or model_role != LOCKED_MODEL_ROLE
        or int(seed) != LOCKED_SEED
        or lock.get("model_id") != model_id
        or lock.get("model_role") != model_role
        or int(lock.get("seed", -1)) != int(seed)
    ):
        raise ValueError("requested model role/id/seed is not authorized by supervised final lock")

    sources = _mapping(lock.get("source_files"), "supervised final lock source_files")
    if set(sources) != set(_SOURCE_NAMES):
        raise ValueError("supervised final lock source_files are incomplete")
    # Authorize all paths from the lock before opening any of the source files.
    locked_paths = {
        name: _absolute_path(
            _mapping(sources[name], f"locked source {name}").get("path"),
            f"locked source {name}.path",
        )
        for name in _SOURCE_NAMES
    }
    cli_config = Path(config_path).expanduser().resolve()
    cli_checkpoint = Path(checkpoint_path).expanduser().resolve()
    cli_output = Path(output_path).expanduser().resolve()
    if locked_paths["config"] != cli_config:
        raise ValueError("supervised final lock config path mismatch")
    if locked_paths["checkpoint"] != cli_checkpoint:
        raise ValueError("supervised final lock checkpoint path mismatch")
    output = _mapping(lock.get("authorized_output"), "authorized_output")
    if set(output) != {"predictions_path", "metadata_path"}:
        raise ValueError("supervised final lock authorized_output is incompatible")
    locked_output = _absolute_path(output.get("predictions_path"), "authorized predictions path")
    locked_metadata = _absolute_path(output.get("metadata_path"), "authorized metadata path")
    expected_metadata = cli_output.with_suffix(cli_output.suffix + ".metadata.json")
    if locked_output != cli_output or locked_metadata != expected_metadata:
        raise ValueError("requested output path is not authorized by supervised final lock")
    if cli_output.exists() or expected_metadata.exists():
        raise FileExistsError("authorized test output already exists; refusing repeated evaluation")
    prior_run_outputs = set(
        locked_paths["run_summary"].parent.glob("*test*predictions*.csv")
    )
    prior_run_outputs.update(
        locked_paths["run_summary"].parent.glob(
            "*test*predictions*.csv.metadata.json"
        )
    )
    if prior_run_outputs:
        raise FileExistsError(
            "locked run directory now contains prior test-prediction evidence"
        )

    for name in _SOURCE_NAMES:
        verified = _verify_source_record(sources[name], f"locked source {name}")
        if verified != locked_paths[name]:
            raise ValueError(f"locked source {name} changed during preflight")

    identity = _config_identity(cli_config)
    if identity["task"] != lock.get("task"):
        raise ValueError("supervised final lock task differs from config")
    if identity["seed"] != LOCKED_SEED or identity["evaluation_phase"] != "validation_screen":
        raise ValueError("locked config is not the seed-42 validation-screen config")
    for name, path in identity["paths"].items():
        if Path(path) != locked_paths[name]:
            raise ValueError(f"locked config {name} path mismatch")

    retinal_summary = _read_json(locked_paths["retinal_sidecar_summary"], "retinal sidecar summary")
    master_metadata = _read_json(locked_paths["master_metadata"], "master metadata")
    observed_retinal = _retinal_audit(
        retinal_summary, master_metadata, _mapping(identity["raw"], "config raw")
    )
    if dict(observed_retinal) != dict(lock.get("retinal_full_sidecar_audit", {})):
        raise ValueError("retinal full-sidecar audit differs from lock")
    try:
        locked_threshold = float(lock.get("threshold_from_validation"))
    except (TypeError, ValueError) as error:
        raise ValueError("supervised final lock threshold is invalid") from error
    if not math.isfinite(locked_threshold) or not 0.0 <= locked_threshold <= 1.0:
        raise ValueError("supervised final lock threshold must be finite and in [0, 1]")
    run = _read_json(locked_paths["run_summary"], "run summary")
    if (
        run.get("task") != lock.get("task")
        or str(run.get("evaluation_phase") or "").strip().lower()
        != "validation_screen"
        or run.get("test_evaluated") is not False
        or run.get("test_metrics") is not None
    ):
        raise ValueError("locked run summary contains prior-test or task-mismatch evidence")
    run_threshold = _threshold(run.get("threshold_selection"), "run threshold_selection")
    if not math.isclose(run_threshold, locked_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("locked run-summary threshold differs from final lock")
    validation_count = _validate_validation_predictions(
        locked_paths["validation_predictions"],
        task=str(lock["task"]),
        threshold=locked_threshold,
    )
    if validation_count != int(lock.get("validation_participants", -1)):
        raise ValueError("validation prediction count differs from supervised final lock")
    prior = _mapping(lock.get("prior_test_evidence_audit"), "prior_test_evidence_audit")
    expected_prior = {
        "config_evaluation_phase": "validation_screen",
        "run_summary_evaluation_phase": "validation_screen",
        "run_summary_test_evaluated": False,
        "run_summary_test_metrics": None,
        "checkpoint_test_role": "not_iterated_during_validation_screen",
        "run_directory_test_prediction_files": [],
        "authorized_output_existed_at_lock_time": False,
        "authorized_metadata_existed_at_lock_time": False,
    }
    if dict(prior) != expected_prior:
        raise ValueError("supervised final lock records prior test evaluation")

    return {
        "path": str(lock_file),
        "sha256": _sha256_file(lock_file),
        "schema_version": SUPERVISED_FINAL_LOCK_VERSION,
        "task": str(lock["task"]),
        "model_id": model_id,
        "model_role": model_role,
        "seed": int(seed),
        "validation_threshold": locked_threshold,
        "checkpoint_path": str(cli_checkpoint),
        "config_path": str(cli_config),
        "master_path": str(locked_paths["master"]),
        "master_metadata_path": str(locked_paths["master_metadata"]),
        "scaler_path": str(locked_paths["scaler"]),
        "authorized_output_path": str(cli_output),
        "stage": "pre_deserialization_supervised_final_lock_authorized",
    }


def authorize_supervised_final_lock(
    lock_path: str | Path,
    *,
    task: str,
    model_id: str,
    model_role: str,
    seed: int,
    threshold: float,
    checkpoint_path: Path,
    config_path: Path,
    master_path: Path,
    master_metadata_path: Path,
    scaler_path: Path,
    schema_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    """Recheck parsed runtime identity and checkpoint threshold against a lock."""

    lock_file = Path(lock_path).expanduser().resolve()
    lock = _read_json(lock_file, "supervised final lock")
    if lock.get("schema_version") != SUPERVISED_FINAL_LOCK_VERSION:
        raise ValueError("supervised final lock schema is incompatible")
    expected_scalars = {
        "task": task,
        "model_id": model_id,
        "model_role": model_role,
        "seed": int(seed),
    }
    for field, expected in expected_scalars.items():
        if lock.get(field) != expected:
            raise ValueError(f"supervised final lock {field} mismatch")
    locked_threshold = float(lock.get("threshold_from_validation"))
    if not math.isclose(locked_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("checkpoint threshold differs from supervised final lock")
    sources = _mapping(lock.get("source_files"), "supervised final lock source_files")
    runtime_paths = {
        "checkpoint": checkpoint_path.resolve(),
        "config": config_path.resolve(),
        "master": master_path.resolve(),
        "master_metadata": master_metadata_path.resolve(),
        "scaler": scaler_path.resolve(),
        **{name: path.resolve() for name, path in schema_paths.items()},
    }
    for name, path in runtime_paths.items():
        record = _mapping(sources.get(name), f"locked source {name}")
        if _absolute_path(record.get("path"), f"locked source {name}.path") != path:
            raise ValueError(f"supervised final lock {name} path mismatch")
        if _sha256_file(path) != _digest(record.get("sha256"), f"locked source {name}.sha256"):
            raise ValueError(f"supervised final lock {name} sha256 mismatch")
    return {
        "path": str(lock_file),
        "sha256": _sha256_file(lock_file),
        "schema_version": SUPERVISED_FINAL_LOCK_VERSION,
        "model_id": model_id,
        "model_role": model_role,
        "seed": int(seed),
        "validation_threshold": locked_threshold,
    }


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--retinal-sidecar-summary", required=True, type=Path)
    parser.add_argument("--authorized-output", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-id", default=LOCKED_MODEL_ID, choices=(LOCKED_MODEL_ID,))
    parser.add_argument("--model-role", default=LOCKED_MODEL_ROLE, choices=(LOCKED_MODEL_ROLE,))
    parser.add_argument("--seed", default=LOCKED_SEED, type=int, choices=(LOCKED_SEED,))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = create_supervised_final_lock(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_summary_path=args.run_summary,
        retinal_sidecar_summary_path=args.retinal_sidecar_summary,
        authorized_output_path=args.authorized_output,
        lock_path=args.out,
        model_id=args.model_id,
        model_role=args.model_role,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "lock": str(Path(args.out).expanduser().resolve()),
                "task": payload["task"],
                "model_id": payload["model_id"],
                "model_role": payload["model_role"],
                "seed": payload["seed"],
                "threshold_from_validation": payload["threshold_from_validation"],
                "authorized_output": payload["authorized_output"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
