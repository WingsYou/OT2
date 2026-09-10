#!/usr/bin/env python3
"""Cost-aware validation selection and locked test non-inferiority for OT2 V2.

This command operates only on already-generated, participant-level prediction
CSVs.  It deliberately has two phases:

``validation-select``
    Verify a multi-seed validation manifest and an explicit acquisition-cost
    inventory, estimate a cost/performance Pareto frontier, and write a
    deterministic selection lock.  A row marked ``test`` is a hard error.

``test-confirm``
    Reconstruct and verify the validation decision before opening the test
    manifest, then evaluate only the locked teacher/candidate pair.  A row
    marked ``val`` is a hard error, as is an unlocked alternative model.

Non-inferiority deltas are degradation-positive: teacher minus candidate for
AUROC, AUPRC, and sensitivity; candidate minus teacher for fixed-bin expected
calibration error.  Confidence bounds use a paired hierarchical bootstrap:
participants are resampled within outcome class and training seeds are
resampled, while teacher and candidate always receive the same draws.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ANALYSIS_FORMAT_VERSION = "ot2-cost-noninferiority-analysis-v2-1"
SELECTION_LOCK_VERSION = "ot2-cost-selection-lock-v2-1"
PROCEDURE_FORMAT_VERSION = "ot2-cost-noninferiority-procedure-v2-1"

MODEL_ROLES = ("full_teacher", "cheap_baseline", "distilled_student")
PRIMARY_METRICS = ("auroc", "auprc", "sensitivity", "calibration_error")
BENEFIT_METRICS = frozenset({"auroc", "auprc", "sensitivity"})
LOSS_METRICS = frozenset({"calibration_error"})
REPORT_METRICS = PRIMARY_METRICS + ("specificity", "brier", "log_loss")
AVAILABILITY_COLUMNS = (
    "has_clinical",
    "has_retinal",
    "has_environment",
    "has_cgm",
    "has_activity",
    "has_ecg",
)
HASH_FIELDS = (
    "predictions_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "run_summary_sha256",
    "master_sha256",
    "scaler_sha256",
)
ARTIFACT_FIELDS = (
    ("predictions_path", "predictions_sha256"),
    ("checkpoint_path", "checkpoint_sha256"),
    ("config_path", "config_sha256"),
    ("run_summary_path", "run_summary_sha256"),
    ("master_path", "master_sha256"),
    ("scaler_path", "scaler_sha256"),
)
MANIFEST_COLUMNS = (
    "model_id",
    "model_role",
    "seed",
    "split",
    "task",
    "predictions_path",
    "predictions_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "config_path",
    "config_sha256",
    "run_summary_path",
    "run_summary_sha256",
    "master_path",
    "master_sha256",
    "scaler_path",
    "scaler_sha256",
    "acquisition_items",
    "threshold_source_split",
)
COST_COLUMNS = (
    "item_id",
    "acquisition_cost",
    "currency",
    "cost_basis",
    "cost_version",
)
REQUIRED_PREDICTION_COLUMNS = (
    "person_id",
    "task",
    "split",
    "y_true",
    "y_probability",
    "threshold",
)
OPTIONAL_PREDICTION_COLUMNS = ("y_logit", "y_pred")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True)
class Procedure:
    margins: Mapping[str, float]
    calibration_bin_edges: tuple[float, ...]
    confidence_level: float
    bootstrap_samples: int
    bootstrap_seed: int
    paired_id_policy: str
    prespecified_seeds: tuple[int, ...]
    minimum_paired_participants: int
    minimum_paired_per_class: int
    maximum_missing_fraction: float
    subgroup_minimum_participants: int
    subgroup_minimum_per_class: int
    source_path: Path
    source_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROCEDURE_FORMAT_VERSION,
            "noninferiority_margins": dict(self.margins),
            "calibration_bin_edges": list(self.calibration_bin_edges),
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "paired_id_policy": self.paired_id_policy,
            "prespecified_seeds": list(self.prespecified_seeds),
            "minimum_paired_participants": self.minimum_paired_participants,
            "minimum_paired_per_class": self.minimum_paired_per_class,
            "maximum_missing_fraction": self.maximum_missing_fraction,
            "subgroup_minimum_participants": self.subgroup_minimum_participants,
            "subgroup_minimum_per_class": self.subgroup_minimum_per_class,
        }


@dataclasses.dataclass(frozen=True)
class CostInventory:
    costs: Mapping[str, Decimal]
    currency: str
    cost_basis: str
    cost_version: str
    source_path: Path
    source_sha256: str


@dataclasses.dataclass
class Experiment:
    split: str
    task: str
    model_ids: tuple[str, ...]
    teacher_id: str
    roles: Mapping[str, str]
    seeds: tuple[int, ...]
    acquisition_items: Mapping[str, tuple[str, ...]]
    total_costs: Mapping[str, Decimal]
    person_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: Mapping[tuple[str, int], np.ndarray]
    thresholds: Mapping[tuple[str, int], float]
    lineage: Mapping[tuple[str, int], Mapping[str, Any]]
    manifest_rows: tuple[Mapping[str, Any], ...]
    pairing_audit: pd.DataFrame
    participant_ids_sha256: str
    subgroups: Mapping[str, np.ndarray]
    subgroup_source: Mapping[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_without_duplicate_keys(path: Path, label: str) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=build_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error


def _stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**32
    )


def _hash_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _strict_columns(frame: pd.DataFrame, expected: Sequence[str], label: str) -> None:
    actual = list(frame.columns)
    missing = [name for name in expected if name not in actual]
    extra = [name for name in actual if name not in expected]
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def _read_exact_csv(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    _strict_columns(frame, columns, label)
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")
    for column in columns:
        values = frame[column].astype(str)
        if any(value != value.strip() or not value for value in values.tolist()):
            raise ValueError(f"{label} column {column!r} has empty or padded values")
    return frame


def _resolve_artifact(raw: str, parent: Path, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _prepare_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite analysis outputs; pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )


def load_procedure(path: str | Path) -> Procedure:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"procedure JSON does not exist: {resolved}")
    raw = _json_without_duplicate_keys(resolved, "procedure JSON")
    if not isinstance(raw, Mapping):
        raise ValueError("procedure JSON must be an object")
    expected = {
        "schema_version",
        "noninferiority_margins",
        "calibration_bin_edges",
        "confidence_level",
        "bootstrap_samples",
        "bootstrap_seed",
        "paired_id_policy",
        "prespecified_seeds",
        "minimum_paired_participants",
        "minimum_paired_per_class",
        "maximum_missing_fraction",
        "subgroup_minimum_participants",
        "subgroup_minimum_per_class",
    }
    if set(raw) != expected:
        raise ValueError(
            "procedure JSON schema mismatch: "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    if raw["schema_version"] != PROCEDURE_FORMAT_VERSION:
        raise ValueError("procedure JSON has the wrong schema_version")
    margins_raw = raw["noninferiority_margins"]
    if not isinstance(margins_raw, Mapping) or set(margins_raw) != set(PRIMARY_METRICS):
        raise ValueError(
            "noninferiority_margins must contain exactly " + ", ".join(PRIMARY_METRICS)
        )
    margins: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        if isinstance(margins_raw[metric], bool):
            raise ValueError(f"margin {metric} must be numeric, not boolean")
        value = float(margins_raw[metric])
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"margin {metric} must be finite and in [0, 1)")
        margins[metric] = value

    edges_raw = raw["calibration_bin_edges"]
    if not isinstance(edges_raw, list) or len(edges_raw) < 3:
        raise ValueError("calibration_bin_edges must contain at least 0, one cut, and 1")
    if any(isinstance(value, bool) for value in edges_raw):
        raise ValueError("calibration_bin_edges cannot contain booleans")
    edges = tuple(float(value) for value in edges_raw)
    if (
        not all(math.isfinite(value) for value in edges)
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or any(right <= left for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError(
            "calibration_bin_edges must be finite, strictly increasing, and span [0, 1]"
        )

    if isinstance(raw["confidence_level"], bool):
        raise ValueError("confidence_level must be numeric, not boolean")
    confidence = float(raw["confidence_level"])
    if not math.isfinite(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")

    def positive_integer(name: str, *, minimum: int = 1) -> int:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    bootstrap_samples = positive_integer("bootstrap_samples", minimum=100)
    bootstrap_seed = raw["bootstrap_seed"]
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    policy = str(raw["paired_id_policy"])
    if policy != "error":
        raise ValueError(
            "paired_id_policy must be error for the primary multi-seed confirmation; "
            "missing paired IDs are rejected rather than silently intersected"
        )
    seeds_raw = raw["prespecified_seeds"]
    if (
        not isinstance(seeds_raw, list)
        or len(seeds_raw) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds_raw)
        or any(value < 0 for value in seeds_raw)
        or len(set(seeds_raw)) != len(seeds_raw)
        or list(sorted(seeds_raw)) != seeds_raw
    ):
        raise ValueError(
            "prespecified_seeds must be a sorted list of exactly five unique "
            "non-negative integer seeds"
        )
    prespecified_seeds = tuple(int(value) for value in seeds_raw)
    minimum_paired = positive_integer("minimum_paired_participants", minimum=2)
    minimum_per_class = positive_integer("minimum_paired_per_class")
    if 2 * minimum_per_class > minimum_paired:
        raise ValueError(
            "minimum_paired_participants must be at least twice "
            "minimum_paired_per_class"
        )
    if isinstance(raw["maximum_missing_fraction"], bool):
        raise ValueError("maximum_missing_fraction must be numeric, not boolean")
    missing_fraction = float(raw["maximum_missing_fraction"])
    if not math.isfinite(missing_fraction) or not 0.0 <= missing_fraction < 1.0:
        raise ValueError("maximum_missing_fraction must be finite and in [0, 1)")
    if missing_fraction != 0.0:
        raise ValueError("maximum_missing_fraction must be 0 for the required error policy")
    subgroup_minimum = positive_integer("subgroup_minimum_participants", minimum=2)
    subgroup_per_class = positive_integer("subgroup_minimum_per_class")
    if 2 * subgroup_per_class > subgroup_minimum:
        raise ValueError(
            "subgroup_minimum_participants must be at least twice "
            "subgroup_minimum_per_class"
        )

    return Procedure(
        margins=margins,
        calibration_bin_edges=edges,
        confidence_level=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(bootstrap_seed),
        paired_id_policy=policy,
        prespecified_seeds=prespecified_seeds,
        minimum_paired_participants=minimum_paired,
        minimum_paired_per_class=minimum_per_class,
        maximum_missing_fraction=missing_fraction,
        subgroup_minimum_participants=subgroup_minimum,
        subgroup_minimum_per_class=subgroup_per_class,
        source_path=resolved,
        source_sha256=_sha256_file(resolved),
    )


def load_cost_inventory(path: str | Path) -> CostInventory:
    resolved = Path(path).expanduser().resolve()
    frame = _read_exact_csv(resolved, COST_COLUMNS, "acquisition-cost CSV")
    if frame["item_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["item_id"].duplicated(False), "item_id"].unique())
        raise ValueError(f"acquisition-cost item_id values must be unique: {duplicates}")
    costs: dict[str, Decimal] = {}
    for row in frame.to_dict("records"):
        item = row["item_id"]
        if not SAFE_ID.fullmatch(item) or "|" in item:
            raise ValueError(f"invalid acquisition-cost item_id {item!r}")
        try:
            value = Decimal(row["acquisition_cost"])
        except InvalidOperation as error:
            raise ValueError(f"invalid acquisition cost for {item!r}") from error
        if not value.is_finite() or value < 0:
            raise ValueError(f"acquisition cost for {item!r} must be finite and non-negative")
        costs[item] = value
    metadata: dict[str, str] = {}
    for column in ("currency", "cost_basis", "cost_version"):
        values = sorted(frame[column].unique().tolist())
        if len(values) != 1:
            raise ValueError(f"acquisition-cost {column} must be constant; found {values}")
        metadata[column] = values[0]
    return CostInventory(
        costs=costs,
        currency=metadata["currency"],
        cost_basis=metadata["cost_basis"],
        cost_version=metadata["cost_version"],
        source_path=resolved,
        source_sha256=_sha256_file(resolved),
    )


def _parse_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise ValueError(f"manifest seed must be an integer, got {value!r}") from error
    if str(seed) != value:
        raise ValueError(f"manifest seed must use canonical integer text, got {value!r}")
    return seed


def _parse_items(value: str, known: Mapping[str, Decimal]) -> tuple[str, ...]:
    items = tuple(value.split("|"))
    if any(not item for item in items) or len(items) != len(set(items)):
        raise ValueError("acquisition_items must be non-empty and duplicate-free")
    if tuple(sorted(items)) != items:
        raise ValueError("acquisition_items must be lexically sorted and pipe-delimited")
    unknown = sorted(set(items) - set(known))
    if unknown:
        raise ValueError(f"acquisition_items lack explicit costs: {unknown}")
    return items


def _read_prediction_csv(path: Path, *, split: str, task: str) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(path, dtype={"person_id": str, "task": str, "split": str})
    actual = set(frame.columns)
    required = set(REQUIRED_PREDICTION_COLUMNS)
    allowed = required | set(OPTIONAL_PREDICTION_COLUMNS)
    if required - actual or actual - allowed:
        raise ValueError(
            f"prediction CSV schema mismatch for {path}: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - allowed)}"
        )
    if frame.empty:
        raise ValueError(f"prediction CSV cannot be empty: {path}")
    identifiers = frame["person_id"]
    if identifiers.isna().any():
        raise ValueError(f"prediction CSV has missing person_id: {path}")
    identifiers = identifiers.astype(str)
    if any(
        not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
        for value in identifiers.tolist()
    ):
        raise ValueError(f"prediction CSV has empty or padded person_id: {path}")
    if identifiers.duplicated().any():
        duplicate = identifiers[identifiers.duplicated(False)].iloc[0]
        raise ValueError(f"prediction CSV has duplicate person_id {duplicate!r}: {path}")
    if set(frame["task"].astype(str)) != {task}:
        raise ValueError(f"prediction CSV task does not match manifest task {task!r}: {path}")
    observed_splits = {str(value).strip().lower() for value in frame["split"]}
    if observed_splits != {split}:
        raise ValueError(
            f"prediction CSV contains split(s) {sorted(observed_splits)}, expected only {split}"
        )

    for column in ("y_true", "y_probability", "threshold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    labels = frame["y_true"].to_numpy(dtype=np.float64)
    probabilities = frame["y_probability"].to_numpy(dtype=np.float64)
    thresholds = frame["threshold"].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"prediction labels must be finite binary 0/1: {path}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"prediction probabilities must be finite and in [0, 1]: {path}")
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError(f"prediction thresholds must be finite and in [0, 1]: {path}")
    unique_thresholds = np.unique(thresholds)
    if len(unique_thresholds) != 1:
        raise ValueError(f"prediction CSV must contain one frozen threshold: {path}")
    threshold = float(unique_thresholds[0])

    if "y_pred" in frame:
        predictions = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy(dtype=np.float64)
        expected = (probabilities >= threshold).astype(np.float64)
        if not np.array_equal(predictions, expected):
            raise ValueError(f"y_pred is inconsistent with probability/threshold: {path}")
    if "y_logit" in frame:
        logits = pd.to_numeric(frame["y_logit"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError(f"y_logit must be finite: {path}")
        sigmoid = np.empty_like(logits)
        positive = logits >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        sigmoid[~positive] = exponent / (1.0 + exponent)
        if not np.allclose(sigmoid, probabilities, rtol=1e-6, atol=1e-8):
            raise ValueError(f"y_logit is inconsistent with y_probability: {path}")
    frame["person_id"] = identifiers
    frame["y_true"] = labels.astype(np.int64)
    frame["y_probability"] = probabilities
    return frame, threshold


def _strict_boolean(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be a strict boolean, got {value!r}")


def _load_locked_subgroups(
    master_path: Path,
    *,
    expected_sha256: str,
    split: str,
    task: str,
    person_ids: Sequence[str],
    labels: np.ndarray,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Join prespecified subgroup fields from the already hash-audited master."""

    if _sha256_file(master_path) != expected_sha256:
        raise ValueError("master changed before subgroup join")
    master = pd.read_csv(master_path, dtype={"person_id": str}, keep_default_na=False)
    required = {
        "person_id",
        "task",
        "split",
        "y",
        "clinical_site",
        *AVAILABILITY_COLUMNS,
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(
            "audited master lacks prespecified subgroup fields; "
            f"missing={missing}. Subgroups are never inferred from predictions."
        )
    master_ids = master["person_id"].astype(str)
    if any(
        not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
        for value in master_ids.tolist()
    ):
        raise ValueError("audited master has empty or padded person_id values")
    if master_ids.duplicated().any():
        raise ValueError("audited master person_id must be unique for subgroup joining")
    indexed = master.set_index(master_ids)
    absent = sorted(set(person_ids) - set(indexed.index))
    if absent:
        raise ValueError(f"prediction IDs are absent from audited master: {absent[:20]}")
    cohort = indexed.loc[list(person_ids)]
    observed_task = set(cohort["task"].astype(str))
    if observed_task != {task}:
        raise ValueError(
            f"audited master task differs from prediction task: {sorted(observed_task)}"
        )
    observed_splits = {
        "val" if str(value).strip().lower() == "validation" else str(value).strip().lower()
        for value in cohort["split"]
    }
    if observed_splits != {split}:
        raise ValueError(
            f"audited master split differs from prediction split: {sorted(observed_splits)}"
        )
    master_labels = pd.to_numeric(cohort["y"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(master_labels).all() or not np.isin(master_labels, [0.0, 1.0]).all():
        raise ValueError("audited master subgroup cohort has invalid binary labels")
    if not np.array_equal(master_labels.astype(np.int64), labels):
        raise ValueError("prediction labels differ from audited master labels")

    sites = []
    patterns = []
    for _, row in cohort.iterrows():
        site = str(row["clinical_site"]).strip()
        sites.append(site if site else "__missing__")
        bits = [
            "1" if _strict_boolean(row[column], f"master:{column}") else "0"
            for column in AVAILABILITY_COLUMNS
        ]
        patterns.append("".join(bits))
    return (
        {
            "clinical_site": np.asarray(sites, dtype=object),
            "availability_pattern": np.asarray(patterns, dtype=object),
        },
        {
            "master_path": str(master_path),
            "master_sha256": expected_sha256,
            "availability_columns_in_bit_order": list(AVAILABILITY_COLUMNS),
            "clinical_site_missing_level": "__missing__",
        },
    )


def load_experiment(
    manifest_path: str | Path,
    costs: CostInventory,
    procedure: Procedure,
    *,
    expected_split: str,
    required_model_ids: Optional[set[str]] = None,
    require_all_roles: bool = True,
) -> Experiment:
    """Load, hash-verify, and globally pair every prediction artifact."""

    if expected_split not in {"val", "test"}:
        raise ValueError("expected_split must be val or test")
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    manifest = _read_exact_csv(resolved_manifest, MANIFEST_COLUMNS, "prediction manifest")
    manifest_parent = resolved_manifest.parent
    declared_splits = {str(value).strip().lower() for value in manifest["split"]}
    if declared_splits != {expected_split}:
        if expected_split == "val":
            raise ValueError(
                "validation selection rejects any non-val prediction input before opening it"
            )
        raise ValueError(
            "test confirmation requires only test prediction input before opening it"
        )
    declared_model_ids = set(manifest["model_id"].astype(str))
    if required_model_ids is not None and declared_model_ids != set(required_model_ids):
        raise ValueError(
            "test manifest must contain exactly the locked teacher/candidate models; "
            f"expected={sorted(required_model_ids)}, observed={sorted(declared_model_ids)}"
        )
    records: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    prediction_frames: dict[tuple[str, int], pd.DataFrame] = {}
    thresholds: dict[tuple[str, int], float] = {}

    for raw in manifest.to_dict("records"):
        model_id = raw["model_id"]
        if not SAFE_ID.fullmatch(model_id):
            raise ValueError(f"invalid model_id {model_id!r}")
        role = raw["model_role"]
        if role not in MODEL_ROLES:
            raise ValueError(f"invalid model_role {role!r}")
        split = raw["split"].lower()
        if split != expected_split:
            if expected_split == "val":
                raise ValueError("validation selection rejects any non-val prediction input")
            raise ValueError("test confirmation requires only test prediction input")
        if raw["threshold_source_split"].lower() not in {"val", "validation"}:
            raise ValueError("every decision threshold must be declared validation-derived")
        seed = _parse_seed(raw["seed"])
        for field in HASH_FIELDS:
            if not SHA256.fullmatch(raw[field]):
                raise ValueError(f"manifest {field} must be a lowercase SHA-256 hex digest")
        items = _parse_items(raw["acquisition_items"], costs.costs)

        record: dict[str, Any] = dict(raw)
        record["seed"] = seed
        record["split"] = split
        record["acquisition_items_tuple"] = items
        for path_field, hash_field in ARTIFACT_FIELDS:
            artifact = _resolve_artifact(
                raw[path_field], manifest_parent, f"manifest {path_field}"
            )
            if artifact not in hash_cache:
                hash_cache[artifact] = _sha256_file(artifact)
            digest = hash_cache[artifact]
            if digest != raw[hash_field]:
                raise ValueError(
                    f"manifest hash mismatch for {path_field}: expected {raw[hash_field]}, "
                    f"observed {digest}, path={artifact}"
                )
            record[path_field] = str(artifact)
        key = (model_id, seed)
        if key in prediction_frames:
            raise ValueError(f"prediction manifest duplicates model/seed cell {key}")
        frame, threshold = _read_prediction_csv(
            Path(record["predictions_path"]), split=split, task=raw["task"]
        )
        prediction_frames[key] = frame
        thresholds[key] = threshold
        records.append(record)

    tasks = sorted({record["task"] for record in records})
    if len(tasks) != 1:
        raise ValueError(f"prediction manifest must contain one task, found {tasks}")
    task = tasks[0]
    model_ids = tuple(sorted({record["model_id"] for record in records}))
    if required_model_ids is not None and set(model_ids) != set(required_model_ids):
        raise ValueError(
            "test manifest must contain exactly the locked teacher/candidate models; "
            f"expected={sorted(required_model_ids)}, observed={list(model_ids)}"
        )
    role_by_model: dict[str, str] = {}
    items_by_model: dict[str, tuple[str, ...]] = {}
    seed_sets: dict[str, set[int]] = {}
    for record in records:
        model = record["model_id"]
        role = record["model_role"]
        items = record["acquisition_items_tuple"]
        if model in role_by_model and role_by_model[model] != role:
            raise ValueError(f"model {model!r} changes role across manifest rows")
        if model in items_by_model and items_by_model[model] != items:
            raise ValueError(f"model {model!r} changes acquisition_items across rows")
        role_by_model[model] = role
        items_by_model[model] = items
        seed_sets.setdefault(model, set()).add(record["seed"])
    teacher_ids = sorted(
        model for model, role in role_by_model.items() if role == "full_teacher"
    )
    if len(teacher_ids) != 1:
        raise ValueError(f"exactly one full_teacher model is required, found {teacher_ids}")
    if require_all_roles:
        missing_roles = sorted(set(MODEL_ROLES) - set(role_by_model.values()))
        if missing_roles:
            raise ValueError(f"validation comparison is missing model roles: {missing_roles}")
    unique_seed_sets = {tuple(sorted(values)) for values in seed_sets.values()}
    if len(unique_seed_sets) != 1:
        raise ValueError(f"every model must have the same seed set: {seed_sets}")
    seeds = next(iter(unique_seed_sets))
    if tuple(seeds) != procedure.prespecified_seeds:
        raise ValueError(
            "manifest seed set differs from prespecified_seeds: "
            f"observed={list(seeds)}, prespecified={list(procedure.prespecified_seeds)}"
        )
    expected_cells = {(model, seed) for model in model_ids for seed in seeds}
    if set(prediction_frames) != expected_cells:
        raise ValueError("prediction manifest is not a complete model-by-seed grid")

    master_hashes = sorted({record["master_sha256"] for record in records})
    scaler_hashes = sorted({record["scaler_sha256"] for record in records})
    if len(master_hashes) != 1 or len(scaler_hashes) != 1:
        raise ValueError("all compared cells must share one master and one scaler hash")

    # Reject contradictory outcomes before checking exact set equality so a
    # mislabeled participant is never reported merely as a pairing mismatch.
    label_registry: dict[str, int] = {}
    for key, frame in prediction_frames.items():
        for person_id, label in zip(frame["person_id"], frame["y_true"]):
            label = int(label)
            if person_id in label_registry and label_registry[person_id] != label:
                raise ValueError(f"outcome label conflict for person_id {person_id!r}")
            label_registry[person_id] = label

    id_sets = {key: set(frame["person_id"].tolist()) for key, frame in prediction_frames.items()}
    first_key = sorted(id_sets)[0]
    first_ids = id_sets[first_key]
    mismatches = []
    for key in sorted(id_sets):
        if id_sets[key] != first_ids:
            mismatches.append(
                {
                    "cell": f"{key[0]}:{key[1]}",
                    "missing": len(first_ids - id_sets[key]),
                    "extra": len(id_sets[key] - first_ids),
                }
            )
    if mismatches:
        raise ValueError(
            "paired_id_policy=error requires identical participant IDs; "
            f"mismatches={mismatches}"
        )
    paired_ids = set(first_ids)
    ordered_ids = tuple(sorted(paired_ids))
    if len(ordered_ids) < procedure.minimum_paired_participants:
        raise ValueError(
            f"paired cohort n={len(ordered_ids)} is below "
            f"minimum_paired_participants={procedure.minimum_paired_participants}"
        )
    labels = np.asarray([label_registry[value] for value in ordered_ids], dtype=np.int64)
    class_counts = {label: int(np.sum(labels == label)) for label in (0, 1)}
    if any(count < procedure.minimum_paired_per_class for count in class_counts.values()):
        raise ValueError(
            f"paired class counts {class_counts} are below "
            f"minimum_paired_per_class={procedure.minimum_paired_per_class}"
        )

    probabilities: dict[tuple[str, int], np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    paired_hash = _hash_lines(ordered_ids)
    for key in sorted(prediction_frames):
        frame = prediction_frames[key].set_index("person_id")
        dropped = sorted(id_sets[key] - paired_ids)
        missing_fraction = len(dropped) / len(frame)
        if missing_fraction > procedure.maximum_missing_fraction + 1e-15:
            raise ValueError(
                f"cell {key} missing fraction {missing_fraction:.6g} exceeds "
                f"maximum_missing_fraction={procedure.maximum_missing_fraction}"
            )
        aligned = frame.loc[list(ordered_ids)]
        if not np.array_equal(aligned["y_true"].to_numpy(dtype=np.int64), labels):
            raise RuntimeError(f"internal label alignment failure for cell {key}")
        probabilities[key] = aligned["y_probability"].to_numpy(dtype=np.float64)
        audit_rows.append(
            {
                "split": expected_split,
                "model_id": key[0],
                "model_role": role_by_model[key[0]],
                "seed": key[1],
                "paired_id_policy": procedure.paired_id_policy,
                "original_n": len(frame),
                "paired_n": len(ordered_ids),
                "dropped_n": len(dropped),
                "missing_fraction": missing_fraction,
                "original_ids_sha256": _hash_lines(id_sets[key]),
                "paired_ids_sha256": paired_hash,
                "dropped_ids_sha256": _hash_lines(dropped),
                "n_negative": class_counts[0],
                "n_positive": class_counts[1],
            }
        )

    totals = {
        model: sum((costs.costs[item] for item in items), Decimal(0))
        for model, items in items_by_model.items()
    }
    subgroup_record = records[0]
    subgroups, subgroup_source = _load_locked_subgroups(
        Path(subgroup_record["master_path"]),
        expected_sha256=subgroup_record["master_sha256"],
        split=expected_split,
        task=task,
        person_ids=ordered_ids,
        labels=labels,
    )
    lineage: dict[tuple[str, int], Mapping[str, Any]] = {}
    normalized_records: list[Mapping[str, Any]] = []
    for record in sorted(records, key=lambda item: (item["model_id"], item["seed"])):
        key = (record["model_id"], record["seed"])
        lineage[key] = {
            name: record[name]
            for name in (
                "checkpoint_path",
                "checkpoint_sha256",
                "config_path",
                "config_sha256",
                "run_summary_path",
                "run_summary_sha256",
                "master_path",
                "master_sha256",
                "scaler_path",
                "scaler_sha256",
            )
        }
        normalized_records.append(
            {
                name: (
                    "|".join(record["acquisition_items_tuple"])
                    if name == "acquisition_items"
                    else record[name]
                )
                for name in MANIFEST_COLUMNS
            }
        )

    return Experiment(
        split=expected_split,
        task=task,
        model_ids=model_ids,
        teacher_id=teacher_ids[0],
        roles=role_by_model,
        seeds=tuple(seeds),
        acquisition_items=items_by_model,
        total_costs=totals,
        person_ids=ordered_ids,
        labels=labels,
        probabilities=probabilities,
        thresholds=thresholds,
        lineage=lineage,
        manifest_rows=tuple(normalized_records),
        pairing_audit=pd.DataFrame(audit_rows),
        participant_ids_sha256=paired_hash,
        subgroups=subgroups,
        subgroup_source=subgroup_source,
        manifest_path=resolved_manifest,
        manifest_sha256=_sha256_file(resolved_manifest),
    )


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, edges: Sequence[float]
) -> float:
    cuts = np.asarray(edges[1:-1], dtype=np.float64)
    bins = np.searchsorted(cuts, probabilities, side="right")
    total = 0.0
    for index in range(len(edges) - 1):
        members = bins == index
        if members.any():
            total += float(members.mean()) * abs(
                float(probabilities[members].mean()) - float(labels[members].mean())
            )
    return float(total)


def _tie_aware_ranking_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Return exact trapezoidal AUROC and non-interpolated average precision."""

    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ranking metrics require both outcome classes")
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_positive = (labels[order] == 1).astype(np.float64)
    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.cumsum(1.0 - sorted_positive)
    group_end = np.ones(len(labels), dtype=bool)
    if len(labels) > 1:
        group_end[:-1] = sorted_probabilities[:-1] != sorted_probabilities[1:]
    endpoint_tp = cumulative_tp[group_end]
    endpoint_fp = cumulative_fp[group_end]
    previous_tp = np.concatenate(([0.0], endpoint_tp[:-1]))
    previous_fp = np.concatenate(([0.0], endpoint_fp[:-1]))
    auroc = float(
        np.sum((endpoint_fp - previous_fp) * (endpoint_tp + previous_tp))
        / (2.0 * positive_count * negative_count)
    )
    positive_in_group = endpoint_tp - previous_tp
    endpoint_positions = np.flatnonzero(group_end).astype(np.float64) + 1.0
    average_precision = float(
        np.sum((endpoint_tp / endpoint_positions) * positive_in_group) / positive_count
    )
    return auroc, average_precision


def binary_metrics(
    labels: Any,
    probabilities: Any,
    *,
    threshold: float,
    calibration_bin_edges: Sequence[float],
) -> dict[str, float]:
    raw_truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if raw_truth.shape != probs.shape or raw_truth.size == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if not np.isfinite(raw_truth).all() or not np.isin(raw_truth, [0.0, 1.0]).all():
        raise ValueError("labels must contain finite binary 0/1 values")
    truth = raw_truth.astype(np.int64)
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("metrics require both binary outcome classes")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    edges = tuple(float(value) for value in calibration_bin_edges)
    if (
        len(edges) < 3
        or not all(math.isfinite(value) for value in edges)
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or any(right <= left for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError("calibration_bin_edges must strictly span [0, 1]")
    predicted = probs >= float(threshold)
    positive = truth == 1
    negative = truth == 0
    auroc, auprc = _tie_aware_ranking_metrics(truth, probs)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": float(predicted[positive].mean()),
        "calibration_error": _expected_calibration_error(
            truth, probs, edges
        ),
        "specificity": float((~predicted[negative]).mean()),
        "brier": float(np.mean((probs - truth) ** 2)),
        "log_loss": float(
            -np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
        ),
    }


def degradation_delta(metric: str, teacher: float, candidate: float) -> float:
    if metric in BENEFIT_METRICS:
        return float(teacher) - float(candidate)
    if metric in LOSS_METRICS:
        return float(candidate) - float(teacher)
    raise ValueError(f"unsupported non-inferiority metric {metric!r}")


def _metrics_and_summary(
    experiment: Experiment, procedure: Procedure, costs: CostInventory
) -> tuple[pd.DataFrame, Mapping[str, Mapping[str, float]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    for model in experiment.model_ids:
        seed_metrics: list[Mapping[str, float]] = []
        for seed in experiment.seeds:
            values = binary_metrics(
                experiment.labels,
                experiment.probabilities[(model, seed)],
                threshold=experiment.thresholds[(model, seed)],
                calibration_bin_edges=procedure.calibration_bin_edges,
            )
            seed_metrics.append(values)
            rows.append(
                {
                    "split": experiment.split,
                    "model_id": model,
                    "model_role": experiment.roles[model],
                    "seed": str(seed),
                    "aggregation": "seed",
                    "n_participants": len(experiment.labels),
                    "n_negative": int(np.sum(experiment.labels == 0)),
                    "n_positive": int(np.sum(experiment.labels == 1)),
                    "threshold": experiment.thresholds[(model, seed)],
                    "acquisition_cost": _decimal_text(experiment.total_costs[model]),
                    "currency": costs.currency,
                    "cost_basis": costs.cost_basis,
                    **values,
                }
            )
        means = {
            metric: float(np.mean([values[metric] for values in seed_metrics]))
            for metric in REPORT_METRICS
        }
        summary[model] = means
        rows.append(
            {
                "split": experiment.split,
                "model_id": model,
                "model_role": experiment.roles[model],
                "seed": "",
                "aggregation": "mean_across_seeds",
                "n_participants": len(experiment.labels),
                "n_negative": int(np.sum(experiment.labels == 0)),
                "n_positive": int(np.sum(experiment.labels == 1)),
                "threshold": "",
                "acquisition_cost": _decimal_text(experiment.total_costs[model]),
                "currency": costs.currency,
                "cost_basis": costs.cost_basis,
                **means,
            }
        )
    return pd.DataFrame(rows), summary


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        if len(members) == 0:
            raise ValueError("stratified bootstrap requires both outcome classes")
        pieces.append(rng.choice(members, size=len(members), replace=True))
    combined = np.concatenate(pieces)
    return combined[rng.permutation(len(combined))]


def paired_hierarchical_bootstrap(
    experiment: Experiment,
    procedure: Procedure,
    candidate_id: str,
    *,
    participant_mask: Optional[np.ndarray] = None,
    confidence_level: Optional[float] = None,
    analysis_context: str = "overall",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return interval records and tidy replicate-level degradation deltas."""

    if candidate_id not in experiment.model_ids:
        raise ValueError(f"candidate model {candidate_id!r} is not in the experiment")
    teacher = experiment.teacher_id
    if participant_mask is None:
        selected_indices = np.arange(len(experiment.labels), dtype=np.int64)
    else:
        mask = np.asarray(participant_mask, dtype=bool).reshape(-1)
        if mask.shape != experiment.labels.shape:
            raise ValueError("participant_mask must align to the experiment cohort")
        selected_indices = np.flatnonzero(mask)
    selected_labels = experiment.labels[selected_indices]
    if set(np.unique(selected_labels)) != {0, 1}:
        raise ValueError("bootstrap analysis subset must contain both outcome classes")
    interval_confidence = (
        procedure.confidence_level
        if confidence_level is None
        else float(confidence_level)
    )
    if not 0.5 < interval_confidence < 1.0:
        raise ValueError("bootstrap confidence_level must lie in (0.5, 1)")
    observed: dict[str, list[float]] = {metric: [] for metric in PRIMARY_METRICS}
    for seed in experiment.seeds:
        teacher_metrics = binary_metrics(
            selected_labels,
            experiment.probabilities[(teacher, seed)][selected_indices],
            threshold=experiment.thresholds[(teacher, seed)],
            calibration_bin_edges=procedure.calibration_bin_edges,
        )
        candidate_metrics = binary_metrics(
            selected_labels,
            experiment.probabilities[(candidate_id, seed)][selected_indices],
            threshold=experiment.thresholds[(candidate_id, seed)],
            calibration_bin_edges=procedure.calibration_bin_edges,
        )
        for metric in PRIMARY_METRICS:
            observed[metric].append(
                degradation_delta(metric, teacher_metrics[metric], candidate_metrics[metric])
            )
    point = {metric: float(np.mean(values)) for metric, values in observed.items()}

    rng = np.random.default_rng(
        _stable_seed(
            procedure.bootstrap_seed,
            experiment.split,
            candidate_id,
            analysis_context,
        )
    )
    replicates = {
        metric: np.empty(procedure.bootstrap_samples, dtype=np.float64)
        for metric in PRIMARY_METRICS
    }
    for iteration in range(procedure.bootstrap_samples):
        participant_indices = _stratified_indices(selected_labels, rng)
        sampled_labels = selected_labels[participant_indices]
        seed_positions = rng.integers(0, len(experiment.seeds), size=len(experiment.seeds))
        iteration_values: dict[str, list[float]] = {
            metric: [] for metric in PRIMARY_METRICS
        }
        for position in seed_positions:
            seed = experiment.seeds[int(position)]
            teacher_metrics = binary_metrics(
                sampled_labels,
                experiment.probabilities[(teacher, seed)][selected_indices][
                    participant_indices
                ],
                threshold=experiment.thresholds[(teacher, seed)],
                calibration_bin_edges=procedure.calibration_bin_edges,
            )
            candidate_metrics = binary_metrics(
                sampled_labels,
                experiment.probabilities[(candidate_id, seed)][selected_indices][
                    participant_indices
                ],
                threshold=experiment.thresholds[(candidate_id, seed)],
                calibration_bin_edges=procedure.calibration_bin_edges,
            )
            for metric in PRIMARY_METRICS:
                iteration_values[metric].append(
                    degradation_delta(
                        metric, teacher_metrics[metric], candidate_metrics[metric]
                    )
                )
        for metric in PRIMARY_METRICS:
            replicates[metric][iteration] = float(np.mean(iteration_values[metric]))

    alpha = 1.0 - interval_confidence
    interval_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        values = replicates[metric]
        lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        # ``higher`` avoids an interpolated upper tail that is slightly less
        # conservative, especially after subgroup multiplicity adjustment.
        one_sided_upper = float(
            np.quantile(values, interval_confidence, method="higher")
        )
        margin = float(procedure.margins[metric])
        interval_rows.append(
            {
                "split": experiment.split,
                "reference_model_id": teacher,
                "candidate_model_id": candidate_id,
                "candidate_model_role": experiment.roles[candidate_id],
                "metric": metric,
                "direction": (
                    "teacher_minus_candidate"
                    if metric in BENEFIT_METRICS
                    else "candidate_minus_teacher"
                ),
                "point_degradation": point[metric],
                "two_sided_ci_lower": float(lower),
                "two_sided_ci_upper": float(upper),
                "one_sided_upper_bound": one_sided_upper,
                "confidence_level": interval_confidence,
                "noninferiority_margin": margin,
                "noninferior": bool(one_sided_upper <= margin),
                "margin_minus_upper_bound": margin - one_sided_upper,
                "bootstrap_samples": procedure.bootstrap_samples,
                "expected_upper_tail_replicates": (
                    procedure.bootstrap_samples * (1.0 - interval_confidence)
                ),
                "n_seeds": len(experiment.seeds),
                "n_participants": len(selected_labels),
                "analysis_context": analysis_context,
                "bootstrap_design": "paired_participant_stratified_and_seed_resampling",
            }
        )
        for iteration, value in enumerate(values):
            replicate_rows.append(
                {
                    "split": experiment.split,
                    "candidate_model_id": candidate_id,
                    "analysis_context": analysis_context,
                    "metric": metric,
                    "bootstrap_iteration": iteration,
                    "degradation_delta": float(value),
                }
            )
    return interval_rows, replicate_rows


def subgroup_stability_analysis(
    experiment: Experiment,
    procedure: Procedure,
    candidate_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Evaluate prespecified master-derived subgroups with Bonferroni bounds.

    Clinical-site and availability-pattern levels below the prespecified sample
    minima are retained in the coverage CSV as non-evaluable; they never vanish
    silently and are never pooled into an outcome-informed replacement group.
    """

    candidates = tuple(str(value) for value in candidate_ids)
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("subgroup candidate_ids must be non-empty and unique")
    unknown = sorted(set(candidates) - set(experiment.model_ids))
    if unknown:
        raise ValueError(f"unknown subgroup candidate models: {unknown}")
    coverage_rows: list[dict[str, Any]] = []
    evaluable: list[tuple[str, str, np.ndarray]] = []
    type_status: dict[str, str] = {}
    for subgroup_type in ("clinical_site", "availability_pattern"):
        values = experiment.subgroups[subgroup_type]
        n_evaluable = 0
        for level in sorted({str(value) for value in values.tolist()}):
            mask = values == level
            n = int(mask.sum())
            n_negative = int(np.sum(experiment.labels[mask] == 0))
            n_positive = int(np.sum(experiment.labels[mask] == 1))
            is_evaluable = (
                n >= procedure.subgroup_minimum_participants
                and n_negative >= procedure.subgroup_minimum_per_class
                and n_positive >= procedure.subgroup_minimum_per_class
            )
            if is_evaluable:
                reason = "meets_prespecified_minima"
                evaluable.append((subgroup_type, level, mask))
                n_evaluable += 1
            else:
                reason = "below_prespecified_participant_or_class_minimum"
            coverage_rows.append(
                {
                    "split": experiment.split,
                    "subgroup_type": subgroup_type,
                    "subgroup_value": level,
                    "n_participants": n,
                    "n_negative": n_negative,
                    "n_positive": n_positive,
                    "minimum_participants": procedure.subgroup_minimum_participants,
                    "minimum_per_class": procedure.subgroup_minimum_per_class,
                    "evaluable": is_evaluable,
                    "status": reason,
                    "subgroup_source_master_sha256": experiment.subgroup_source[
                        "master_sha256"
                    ],
                }
            )
        type_status[subgroup_type] = (
            "evaluated" if n_evaluable else "no_level_met_prespecified_minima"
        )

    # One selected test candidate is the confirmatory case.  Validation may
    # screen several candidates, so the simultaneous family also includes the
    # number of candidates to avoid optimistic subgroup gating during choice.
    family_size = max(
        1, len(coverage_rows) * len(PRIMARY_METRICS) * len(candidates)
    )
    alpha = 1.0 - procedure.confidence_level
    simultaneous_confidence = 1.0 - alpha / float(family_size)
    interval_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for subgroup_type, level, mask in evaluable:
            context = f"subgroup:{subgroup_type}:{level}"
            rows, replicates = paired_hierarchical_bootstrap(
                experiment,
                procedure,
                candidate,
                participant_mask=mask,
                confidence_level=simultaneous_confidence,
                analysis_context=context,
            )
            for row in rows:
                row.update(
                    {
                        "subgroup_type": subgroup_type,
                        "subgroup_value": level,
                        "multiplicity_method": "bonferroni_one_sided",
                        "multiplicity_family_size": family_size,
                        "nominal_confidence_level": procedure.confidence_level,
                        "simultaneous_confidence_level": simultaneous_confidence,
                    }
                )
            for row in replicates:
                row.update(
                    {
                        "subgroup_type": subgroup_type,
                        "subgroup_value": level,
                    }
                )
            interval_rows.extend(rows)
            replicate_rows.extend(replicates)
    gate_by_candidate = {
        candidate: bool(
            all(
                row["noninferior"]
                for row in interval_rows
                if row["candidate_model_id"] == candidate
            )
        )
        for candidate in candidates
    }
    coverage_columns = (
        "split",
        "subgroup_type",
        "subgroup_value",
        "n_participants",
        "n_negative",
        "n_positive",
        "minimum_participants",
        "minimum_per_class",
        "evaluable",
        "status",
        "subgroup_source_master_sha256",
    )
    interval_columns = (
        "split",
        "reference_model_id",
        "candidate_model_id",
        "candidate_model_role",
        "metric",
        "direction",
        "point_degradation",
        "two_sided_ci_lower",
        "two_sided_ci_upper",
        "one_sided_upper_bound",
        "confidence_level",
        "noninferiority_margin",
        "noninferior",
        "margin_minus_upper_bound",
        "bootstrap_samples",
        "expected_upper_tail_replicates",
        "n_seeds",
        "n_participants",
        "analysis_context",
        "bootstrap_design",
        "subgroup_type",
        "subgroup_value",
        "multiplicity_method",
        "multiplicity_family_size",
        "nominal_confidence_level",
        "simultaneous_confidence_level",
    )
    bootstrap_columns = (
        "split",
        "candidate_model_id",
        "analysis_context",
        "metric",
        "bootstrap_iteration",
        "degradation_delta",
        "subgroup_type",
        "subgroup_value",
    )
    return {
        "coverage": pd.DataFrame(coverage_rows, columns=coverage_columns),
        "intervals": pd.DataFrame(interval_rows, columns=interval_columns),
        "bootstrap": pd.DataFrame(replicate_rows, columns=bootstrap_columns),
        "gate_by_candidate": gate_by_candidate,
        "type_status": type_status,
        "evaluable_level_count": len(evaluable),
        "multiplicity_family_size": family_size,
        "simultaneous_confidence_level": simultaneous_confidence,
    }


def pareto_table(
    experiment: Experiment,
    summary: Mapping[str, Mapping[str, float]],
    costs: CostInventory,
) -> pd.DataFrame:
    """Return the validation-only cost/performance non-dominated frontier."""

    rows: list[dict[str, Any]] = []
    tolerance = 1e-15
    for model in experiment.model_ids:
        dominated_by: list[str] = []
        for other in experiment.model_ids:
            if other == model:
                continue
            no_worse = experiment.total_costs[other] <= experiment.total_costs[model]
            no_worse = no_worse and all(
                summary[other][metric] >= summary[model][metric] - tolerance
                for metric in BENEFIT_METRICS
            )
            no_worse = no_worse and all(
                summary[other][metric] <= summary[model][metric] + tolerance
                for metric in LOSS_METRICS
            )
            strictly_better = experiment.total_costs[other] < experiment.total_costs[model]
            strictly_better = strictly_better or any(
                summary[other][metric] > summary[model][metric] + tolerance
                for metric in BENEFIT_METRICS
            )
            strictly_better = strictly_better or any(
                summary[other][metric] < summary[model][metric] - tolerance
                for metric in LOSS_METRICS
            )
            if no_worse and strictly_better:
                dominated_by.append(other)
        rows.append(
            {
                "split": experiment.split,
                "model_id": model,
                "model_role": experiment.roles[model],
                "acquisition_items": "|".join(experiment.acquisition_items[model]),
                "acquisition_cost": _decimal_text(experiment.total_costs[model]),
                "currency": costs.currency,
                "cost_basis": costs.cost_basis,
                **{f"mean_{metric}": summary[model][metric] for metric in PRIMARY_METRICS},
                "is_pareto": not dominated_by,
                "dominated_by": "|".join(sorted(dominated_by)),
            }
        )
    return pd.DataFrame(rows)


def _validation_analysis(
    experiment: Experiment,
    procedure: Procedure,
    costs: CostInventory,
) -> Mapping[str, Any]:
    if experiment.split != "val":
        raise ValueError("selection analysis may only run on validation predictions")
    metrics_frame, summary = _metrics_and_summary(experiment, procedure, costs)
    pareto = pareto_table(experiment, summary, costs)
    interval_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for model in experiment.model_ids:
        rows, replicates = paired_hierarchical_bootstrap(experiment, procedure, model)
        interval_rows.extend(rows)
        replicate_rows.extend(replicates)
    intervals = pd.DataFrame(interval_rows)
    bootstrap = pd.DataFrame(replicate_rows)
    nonteacher_models = tuple(
        model for model in experiment.model_ids if model != experiment.teacher_id
    )
    subgroup = subgroup_stability_analysis(
        experiment, procedure, nonteacher_models
    )
    all_noninferior = {
        model: bool(intervals.loc[intervals["candidate_model_id"] == model, "noninferior"].all())
        for model in experiment.model_ids
    }
    minimum_slack = {
        model: float(
            intervals.loc[
                intervals["candidate_model_id"] == model, "margin_minus_upper_bound"
            ].min()
        )
        for model in experiment.model_ids
    }
    pareto_ids = set(pareto.loc[pareto["is_pareto"], "model_id"].astype(str))
    teacher_cost = experiment.total_costs[experiment.teacher_id]
    eligible = [
        model
        for model in experiment.model_ids
        if model != experiment.teacher_id
        and all_noninferior[model]
        and subgroup["gate_by_candidate"][model]
        and model in pareto_ids
        and experiment.total_costs[model] < teacher_cost
    ]
    if eligible:
        eligible.sort(
            key=lambda model: (
                experiment.total_costs[model],
                -minimum_slack[model],
                model,
            )
        )
        selected = eligible[0]
        status = "lower_cost_validation_noninferior_pareto_model_selected"
    else:
        selected = experiment.teacher_id
        status = "full_teacher_retained_no_eligible_lower_cost_candidate"

    decision = {
        "selection_rule": (
            "among validation Pareto models with cost strictly below the full teacher and "
            "all aggregate and evaluable prespecified-subgroup one-sided upper degradation "
            "bounds at or below their margins, minimize "
            "explicit acquisition cost; break an exact cost tie by larger minimum margin "
            "slack, then lexical model_id; otherwise retain the full teacher"
        ),
        "teacher_model_id": experiment.teacher_id,
        "selected_model_id": selected,
        "selected_model_role": experiment.roles[selected],
        "selection_status": status,
        "pareto_model_ids": sorted(pareto_ids),
        "eligible_lower_cost_model_ids": eligible,
        "all_metrics_noninferior": all_noninferior,
        "subgroup_stability_gate": {
            experiment.teacher_id: True,
            **dict(subgroup["gate_by_candidate"]),
        },
        "subgroup_type_status": dict(subgroup["type_status"]),
        "subgroup_evaluable_level_count": subgroup["evaluable_level_count"],
        "subgroup_multiplicity_method": "bonferroni_one_sided",
        "subgroup_multiplicity_family_size": subgroup["multiplicity_family_size"],
        "minimum_margin_slack": minimum_slack,
        "model_costs": {
            model: _decimal_text(experiment.total_costs[model])
            for model in experiment.model_ids
        },
        "paired_participant_ids_sha256": experiment.participant_ids_sha256,
        "paired_n": len(experiment.person_ids),
        "n_negative": int(np.sum(experiment.labels == 0)),
        "n_positive": int(np.sum(experiment.labels == 1)),
        "seeds": list(experiment.seeds),
        "primary_metrics": list(PRIMARY_METRICS),
    }
    fingerprint_payload = {
        "procedure": procedure.as_dict(),
        "decision": decision,
        "metric_summary": {
            model: {metric: summary[model][metric] for metric in REPORT_METRICS}
            for model in experiment.model_ids
        },
        "noninferiority": interval_rows,
        "subgroup_coverage": subgroup["coverage"].to_dict("records"),
        "subgroup_noninferiority": subgroup["intervals"].to_dict("records"),
        "thresholds": {
            model: {
                str(seed): experiment.thresholds[(model, seed)] for seed in experiment.seeds
            }
            for model in experiment.model_ids
        },
    }
    return {
        "metrics": metrics_frame,
        "metric_summary": summary,
        "pareto": pareto,
        "intervals": intervals,
        "bootstrap": bootstrap,
        "subgroup_coverage": subgroup["coverage"],
        "subgroup_intervals": subgroup["intervals"],
        "subgroup_bootstrap": subgroup["bootstrap"],
        "decision": decision,
        "validation_fingerprint": _canonical_hash(fingerprint_payload),
    }


def _source_record(path: Path, digest: Optional[str] = None) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest or _sha256_file(path.resolve())}


def _arm_lock_payload(experiment: Experiment, model: str) -> Mapping[str, Any]:
    seed_entries: dict[str, Mapping[str, Any]] = {}
    for seed in experiment.seeds:
        manifest_row = next(
            row
            for row in experiment.manifest_rows
            if row["model_id"] == model and int(row["seed"]) == seed
        )
        seed_entries[str(seed)] = {
            "threshold_from_validation": experiment.thresholds[(model, seed)],
            **dict(experiment.lineage[(model, seed)]),
            "validation_predictions_path": manifest_row["predictions_path"],
            "validation_predictions_sha256": manifest_row["predictions_sha256"],
        }
    return {
        "model_role": experiment.roles[model],
        "acquisition_items": list(experiment.acquisition_items[model]),
        "acquisition_cost": _decimal_text(experiment.total_costs[model]),
        "seeds": seed_entries,
    }


def run_validation_selection(
    manifest_path: str | Path,
    cost_path: str | Path,
    procedure_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    paths = {
        "metrics": output / "validation_metrics.csv",
        "pareto": output / "validation_pareto.csv",
        "intervals": output / "validation_noninferiority.csv",
        "bootstrap": output / "validation_bootstrap_deltas.csv",
        "pairing": output / "validation_pairing_audit.csv",
        "subgroup_coverage": output / "validation_subgroup_coverage.csv",
        "subgroup_intervals": output / "validation_subgroup_noninferiority.csv",
        "subgroup_bootstrap": output / "validation_subgroup_bootstrap_deltas.csv",
        "lock": output / "selection_lock.json",
        "metadata": output / "validation_analysis.json",
    }
    _prepare_outputs(paths.values(), overwrite=overwrite)
    procedure = load_procedure(procedure_path)
    costs = load_cost_inventory(cost_path)
    experiment = load_experiment(
        manifest_path, costs, procedure, expected_split="val", require_all_roles=True
    )
    analysis = _validation_analysis(experiment, procedure, costs)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(analysis["metrics"], paths["metrics"])
    _atomic_csv(analysis["pareto"], paths["pareto"])
    _atomic_csv(analysis["intervals"], paths["intervals"])
    _atomic_csv(analysis["bootstrap"], paths["bootstrap"])
    _atomic_csv(experiment.pairing_audit, paths["pairing"])
    _atomic_csv(analysis["subgroup_coverage"], paths["subgroup_coverage"])
    _atomic_csv(analysis["subgroup_intervals"], paths["subgroup_intervals"])
    _atomic_csv(analysis["subgroup_bootstrap"], paths["subgroup_bootstrap"])

    selected = str(analysis["decision"]["selected_model_id"])
    teacher = experiment.teacher_id
    validation_artifacts = {
        name: _source_record(paths[name])
        for name in (
            "metrics",
            "pareto",
            "intervals",
            "bootstrap",
            "pairing",
            "subgroup_coverage",
            "subgroup_intervals",
            "subgroup_bootstrap",
        )
    }
    lock = {
        "schema_version": SELECTION_LOCK_VERSION,
        "analysis_format_version": ANALYSIS_FORMAT_VERSION,
        "analysis_role": "validation_cost_selection",
        "selection_split": "val",
        "task": experiment.task,
        "validation_fingerprint": analysis["validation_fingerprint"],
        "decision": analysis["decision"],
        "procedure": procedure.as_dict(),
        "cost_metadata": {
            "currency": costs.currency,
            "cost_basis": costs.cost_basis,
            "cost_version": costs.cost_version,
        },
        "source_files": {
            "validation_manifest": _source_record(
                experiment.manifest_path, experiment.manifest_sha256
            ),
            "acquisition_costs": _source_record(costs.source_path, costs.source_sha256),
            "procedure": _source_record(procedure.source_path, procedure.source_sha256),
        },
        "validation_outputs": validation_artifacts,
        "locked_arms": {
            model: _arm_lock_payload(experiment, model)
            for model in sorted({teacher, selected})
        },
        "all_validation_model_ids": list(experiment.model_ids),
    }
    _atomic_json(lock, paths["lock"])
    metadata = {
        "schema_version": ANALYSIS_FORMAT_VERSION,
        "status": "complete",
        "analysis_role": "validation_cost_selection",
        "split": "val",
        "task": experiment.task,
        "validation_fingerprint": analysis["validation_fingerprint"],
        "decision": analysis["decision"],
        "selection_lock": _source_record(paths["lock"]),
        "outputs": {
            **validation_artifacts,
            "selection_lock": _source_record(paths["lock"]),
        },
    }
    _atomic_json(metadata, paths["metadata"])
    return {"lock": lock, "metadata": metadata, "paths": paths}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_source_record(value: Any, label: str) -> Path:
    record = _mapping(value, label)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    digest = str(record["sha256"])
    if not SHA256.fullmatch(digest):
        raise ValueError(f"{label} has an invalid SHA-256 digest")
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError(f"{label} file is missing or its hash changed: {path}")
    return path


def _load_and_reconstruct_lock(
    lock_path: str | Path,
) -> tuple[Mapping[str, Any], Procedure, CostInventory, Experiment, Mapping[str, Any]]:
    resolved = Path(lock_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"selection lock does not exist: {resolved}")
    lock = _mapping(
        _json_without_duplicate_keys(resolved, "selection lock"), "selection lock"
    )
    required = {
        "schema_version",
        "analysis_format_version",
        "analysis_role",
        "selection_split",
        "task",
        "validation_fingerprint",
        "decision",
        "procedure",
        "cost_metadata",
        "source_files",
        "validation_outputs",
        "locked_arms",
        "all_validation_model_ids",
    }
    if set(lock) != required:
        raise ValueError("selection lock has missing or unexpected fields")
    if lock["schema_version"] != SELECTION_LOCK_VERSION:
        raise ValueError("selection lock has the wrong schema_version")
    if lock["analysis_format_version"] != ANALYSIS_FORMAT_VERSION:
        raise ValueError("selection lock analysis format is incompatible")
    if lock["analysis_role"] != "validation_cost_selection" or lock["selection_split"] != "val":
        raise ValueError("test confirmation requires a validation-only selection lock")
    sources = _mapping(lock["source_files"], "selection lock source_files")
    if set(sources) != {"validation_manifest", "acquisition_costs", "procedure"}:
        raise ValueError("selection lock source_files are incomplete")
    validation_manifest_path = _verify_source_record(
        sources["validation_manifest"], "locked validation manifest"
    )
    cost_path = _verify_source_record(sources["acquisition_costs"], "locked acquisition costs")
    procedure_path = _verify_source_record(sources["procedure"], "locked procedure")
    outputs = _mapping(lock["validation_outputs"], "selection lock validation_outputs")
    if set(outputs) != {
        "metrics",
        "pareto",
        "intervals",
        "bootstrap",
        "pairing",
        "subgroup_coverage",
        "subgroup_intervals",
        "subgroup_bootstrap",
    }:
        raise ValueError("selection lock validation_outputs are incomplete")
    for name, record in outputs.items():
        _verify_source_record(record, f"locked validation output {name}")

    procedure = load_procedure(procedure_path)
    if procedure.as_dict() != lock["procedure"]:
        raise ValueError("locked procedure payload differs from its source file")
    costs = load_cost_inventory(cost_path)
    cost_metadata = _mapping(lock["cost_metadata"], "selection lock cost_metadata")
    expected_cost_metadata = {
        "currency": costs.currency,
        "cost_basis": costs.cost_basis,
        "cost_version": costs.cost_version,
    }
    if dict(cost_metadata) != expected_cost_metadata:
        raise ValueError("locked cost metadata differs from acquisition-cost source")

    validation = load_experiment(
        validation_manifest_path,
        costs,
        procedure,
        expected_split="val",
        require_all_roles=True,
    )
    reconstructed = _validation_analysis(validation, procedure, costs)
    if reconstructed["validation_fingerprint"] != lock["validation_fingerprint"]:
        raise ValueError("selection lock validation fingerprint does not reproduce")
    if reconstructed["decision"] != lock["decision"]:
        raise ValueError("selection lock decision was not deterministically derived from validation")
    if lock["task"] != validation.task:
        raise ValueError("selection lock task differs from validation manifest")
    if list(validation.model_ids) != lock["all_validation_model_ids"]:
        raise ValueError("selection lock validation model inventory differs from source")
    decision = _mapping(lock["decision"], "selection lock decision")
    expected_locked_models = {
        str(decision["teacher_model_id"]),
        str(decision["selected_model_id"]),
    }
    expected_locked_arms = {
        model: _arm_lock_payload(validation, model)
        for model in sorted(expected_locked_models)
    }
    if lock["locked_arms"] != expected_locked_arms:
        raise ValueError(
            "selection lock arm/checkpoint lineage was not derived from validation"
        )
    return lock, procedure, costs, validation, reconstructed


def _validate_test_against_lock(
    test: Experiment,
    validation: Experiment,
    lock: Mapping[str, Any],
) -> None:
    if test.task != validation.task or test.task != lock["task"]:
        raise ValueError("test task differs from the locked validation task")
    if tuple(test.seeds) != tuple(validation.seeds):
        raise ValueError("test seed set differs from the locked validation seed set")
    if set(test.person_ids).intersection(validation.person_ids):
        raise ValueError("validation and test participant IDs overlap")
    locked_arms = _mapping(lock["locked_arms"], "selection lock locked_arms")
    if set(locked_arms) != set(test.model_ids):
        raise ValueError("test model IDs differ from locked arms")
    for model in test.model_ids:
        arm = _mapping(locked_arms[model], f"locked arm {model}")
        if arm.get("model_role") != test.roles[model]:
            raise ValueError(f"test model role differs from lock for {model}")
        if arm.get("acquisition_items") != list(test.acquisition_items[model]):
            raise ValueError(f"test acquisition items differ from lock for {model}")
        if str(arm.get("acquisition_cost")) != _decimal_text(test.total_costs[model]):
            raise ValueError(f"test acquisition cost differs from lock for {model}")
        locked_seeds = _mapping(arm.get("seeds"), f"locked seeds for {model}")
        if set(locked_seeds) != {str(seed) for seed in test.seeds}:
            raise ValueError(f"test seed inventory differs from lock for {model}")
        for seed in test.seeds:
            locked = _mapping(locked_seeds[str(seed)], f"locked arm {model} seed {seed}")
            if not math.isclose(
                float(locked.get("threshold_from_validation")),
                test.thresholds[(model, seed)],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"test threshold differs from validation lock for {model}/{seed}")
            for field in (
                "checkpoint_sha256",
                "config_sha256",
                "run_summary_sha256",
                "master_sha256",
                "scaler_sha256",
            ):
                if locked.get(field) != test.lineage[(model, seed)][field]:
                    raise ValueError(f"test {field} differs from validation lock for {model}/{seed}")


def run_test_confirmation(
    manifest_path: str | Path,
    selection_lock_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    paths = {
        "metrics": output / "test_metrics.csv",
        "intervals": output / "test_noninferiority.csv",
        "bootstrap": output / "test_bootstrap_deltas.csv",
        "pairing": output / "test_pairing_audit.csv",
        "subgroup_coverage": output / "test_subgroup_coverage.csv",
        "subgroup_intervals": output / "test_subgroup_noninferiority.csv",
        "subgroup_bootstrap": output / "test_subgroup_bootstrap_deltas.csv",
        "results": output / "test_results.json",
    }
    _prepare_outputs(paths.values(), overwrite=overwrite)

    # Reproduce the validation-only decision before reading a test artifact.
    lock, procedure, costs, validation, _ = _load_and_reconstruct_lock(
        selection_lock_path
    )
    decision = _mapping(lock["decision"], "selection lock decision")
    teacher = str(decision["teacher_model_id"])
    selected = str(decision["selected_model_id"])
    required_models = {teacher, selected}
    test = load_experiment(
        manifest_path,
        costs,
        procedure,
        expected_split="test",
        required_model_ids=required_models,
        require_all_roles=False,
    )
    _validate_test_against_lock(test, validation, lock)
    metrics, _ = _metrics_and_summary(test, procedure, costs)
    intervals_rows, bootstrap_rows = paired_hierarchical_bootstrap(
        test, procedure, selected
    )
    intervals = pd.DataFrame(intervals_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    subgroup = subgroup_stability_analysis(test, procedure, [selected])
    if selected == teacher:
        confirmatory_decision = "reference_selected_no_noninferiority_claim"
        all_noninferior: Optional[bool] = None
        subgroup_gate: Optional[bool] = None
    else:
        all_noninferior = bool(intervals["noninferior"].all())
        subgroup_raw = bool(subgroup["gate_by_candidate"][selected])
        all_types_evaluable = all(
            status == "evaluated" for status in subgroup["type_status"].values()
        )
        subgroup_gate = subgroup_raw if (not subgroup_raw or all_types_evaluable) else None
        if not all_noninferior:
            confirmatory_decision = "locked_candidate_not_confirmed_noninferior"
        elif subgroup_gate is False:
            confirmatory_decision = (
                "aggregate_noninferior_but_prespecified_subgroup_stability_failed"
            )
        elif subgroup_gate is None:
            confirmatory_decision = (
                "aggregate_noninferior_but_subgroup_gate_not_fully_evaluable"
            )
        else:
            confirmatory_decision = (
                "locked_candidate_noninferior_and_evaluable_subgroups_stable"
            )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(metrics, paths["metrics"])
    _atomic_csv(intervals, paths["intervals"])
    _atomic_csv(bootstrap, paths["bootstrap"])
    _atomic_csv(test.pairing_audit, paths["pairing"])
    _atomic_csv(subgroup["coverage"], paths["subgroup_coverage"])
    _atomic_csv(subgroup["intervals"], paths["subgroup_intervals"])
    _atomic_csv(subgroup["bootstrap"], paths["subgroup_bootstrap"])
    output_records = {
        name: _source_record(paths[name])
        for name in (
            "metrics",
            "intervals",
            "bootstrap",
            "pairing",
            "subgroup_coverage",
            "subgroup_intervals",
            "subgroup_bootstrap",
        )
    }
    results = {
        "schema_version": ANALYSIS_FORMAT_VERSION,
        "status": "complete",
        "analysis_role": "locked_confirmatory_test_no_selection",
        "split": "test",
        "task": test.task,
        "selection_lock": _source_record(Path(selection_lock_path).expanduser().resolve()),
        "test_manifest": _source_record(test.manifest_path, test.manifest_sha256),
        "teacher_model_id": teacher,
        "selected_model_id": selected,
        "validation_selection_status": decision["selection_status"],
        "confirmatory_decision": confirmatory_decision,
        "all_prespecified_metrics_noninferior": all_noninferior,
        "prespecified_subgroup_stability_gate": subgroup_gate,
        "subgroup_type_status": dict(subgroup["type_status"]),
        "subgroup_evaluable_level_count": subgroup["evaluable_level_count"],
        "subgroup_multiplicity_method": "bonferroni_one_sided",
        "subgroup_multiplicity_family_size": subgroup["multiplicity_family_size"],
        "subgroup_simultaneous_confidence_level": subgroup[
            "simultaneous_confidence_level"
        ],
        "noninferiority_margins": dict(procedure.margins),
        "calibration_bin_edges": list(procedure.calibration_bin_edges),
        "confidence_level": procedure.confidence_level,
        "bootstrap_samples": procedure.bootstrap_samples,
        "bootstrap_design": "paired_participant_stratified_and_seed_resampling",
        "paired_id_policy": procedure.paired_id_policy,
        "paired_n": len(test.person_ids),
        "n_negative": int(np.sum(test.labels == 0)),
        "n_positive": int(np.sum(test.labels == 1)),
        "paired_participant_ids_sha256": test.participant_ids_sha256,
        "seeds": list(test.seeds),
        "acquisition_cost": {
            "teacher": _decimal_text(test.total_costs[teacher]),
            "selected": _decimal_text(test.total_costs[selected]),
            "saving_vs_teacher": _decimal_text(
                test.total_costs[teacher] - test.total_costs[selected]
            ),
            "currency": costs.currency,
            "cost_basis": costs.cost_basis,
            "cost_version": costs.cost_version,
            "selected_acquisition_items": list(test.acquisition_items[selected]),
        },
        "outputs": output_records,
    }
    _atomic_json(results, paths["results"])
    return {"results": results, "paths": paths}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    selection = subparsers.add_parser(
        "validation-select", help="build a validation-only Pareto/selection lock"
    )
    selection.add_argument("--manifest", required=True, type=Path)
    selection.add_argument("--acquisition-costs", required=True, type=Path)
    selection.add_argument("--procedure", required=True, type=Path)
    selection.add_argument("--out-dir", required=True, type=Path)
    selection.add_argument("--overwrite", action="store_true")

    confirmation = subparsers.add_parser(
        "test-confirm", help="confirm only the locked model on test"
    )
    confirmation.add_argument("--manifest", required=True, type=Path)
    confirmation.add_argument("--selection-lock", required=True, type=Path)
    confirmation.add_argument("--out-dir", required=True, type=Path)
    confirmation.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "validation-select":
        result = run_validation_selection(
            args.manifest,
            args.acquisition_costs,
            args.procedure,
            args.out_dir,
            overwrite=args.overwrite,
        )
        summary = {
            "status": "complete",
            "analysis_role": "validation_cost_selection",
            "selection_lock": str(result["paths"]["lock"]),
            "selected_model_id": result["lock"]["decision"]["selected_model_id"],
            "selection_status": result["lock"]["decision"]["selection_status"],
        }
    else:
        result = run_test_confirmation(
            args.manifest,
            args.selection_lock,
            args.out_dir,
            overwrite=args.overwrite,
        )
        summary = {
            "status": "complete",
            "analysis_role": "locked_confirmatory_test_no_selection",
            "test_results": str(result["paths"]["results"]),
            "selected_model_id": result["results"]["selected_model_id"],
            "confirmatory_decision": result["results"]["confirmatory_decision"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_FORMAT_VERSION",
    "COST_COLUMNS",
    "MANIFEST_COLUMNS",
    "PRIMARY_METRICS",
    "PROCEDURE_FORMAT_VERSION",
    "SELECTION_LOCK_VERSION",
    "binary_metrics",
    "build_arg_parser",
    "degradation_delta",
    "load_cost_inventory",
    "load_experiment",
    "load_procedure",
    "main",
    "paired_hierarchical_bootstrap",
    "pareto_table",
    "run_test_confirmation",
    "run_validation_selection",
]
