#!/usr/bin/env python3
"""Summarize one frozen-seed OT2 benchmark across MoCA and both CVD tasks.

This is a report-only command.  It reads validation metrics and the threshold
selected during validation from each training ``run_summary.json``.  Optional
test metrics are accepted only from ``locked_confirmatory_test`` metadata
written by ``evaluate_frozen_checkpoint_v2.py`` and must reuse that exact
validation threshold.  The command never ranks checkpoints, tunes a threshold,
or reads test predictions to make a selection.
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

try:
    from ot2.task2_v2_config import PRIMARY_MODALITIES
except ModuleNotFoundError:  # Support direct execution from the ot2 directory.
    from task2_v2_config import PRIMARY_MODALITIES  # type: ignore


SCHEMA_VERSION = "ot2-three-task-benchmark-v1"
FROZEN_EVALUATION_VERSION = "ot2-frozen-evaluation-v2-1"
FULL_MODEL_SEED = 42
TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TASK_LABELS = {
    "moca_binary": "MoCA < 26",
    "cvd_broad_binary": "CVD broad",
    "cvd_strict_binary": "CVD strict",
}
METRICS = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "log_loss",
    "brier",
)
CSV_COLUMNS = (
    "task",
    "task_label",
    "seed",
    "split",
    "n",
    "prevalence",
    *METRICS,
    "threshold",
    "threshold_source",
    "evidence_role",
    "benchmark_scope",
)
SINGLE_SEED_WARNING = (
    "Single-seed benchmark: uncertainty across random initializations is not "
    "measured by this report."
)
TEST_GUARD_WARNING = (
    "Test metrics are report-only locked confirmation. Models and decision "
    "thresholds must be selected on validation before test is opened."
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_count(value: Any, label: str) -> int:
    number = _finite_float(value, label)
    result = int(number)
    if result <= 0 or number != result:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _seed(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _normalized_path(value: Any, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    return str(Path(raw).expanduser().resolve())


def _source_record(value: Any, label: str) -> dict[str, str]:
    record = _mapping(value, label)
    path = _normalized_path(record.get("path"), label)
    digest = str(record.get("sha256") or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} SHA256 is missing or invalid")
    return {"path": path, "sha256": digest}


def _same_source(
    validation: Mapping[str, str], test: Mapping[str, str], label: str
) -> None:
    if dict(validation) != dict(test):
        raise ValueError(f"validation/test {label} lineage mismatch")


def _metric_row(
    metrics_value: Any,
    *,
    task: str,
    split: str,
    seed: int,
    threshold: float,
    evidence_role: str,
) -> dict[str, Any]:
    metrics = _mapping(metrics_value, f"{task} {split} metrics")
    observed_threshold = _finite_float(
        metrics.get("threshold"), f"{task} {split} metrics.threshold"
    )
    if not math.isclose(observed_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{task} {split} metric threshold differs from validation threshold"
        )
    n = _positive_count(metrics.get("n"), f"{task} {split} metrics.n")
    prevalence = _finite_float(
        metrics.get("prevalence"), f"{task} {split} metrics.prevalence"
    )
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"{task} {split} prevalence must lie in [0, 1]")
    row: dict[str, Any] = {
        "task": task,
        "task_label": TASK_LABELS[task],
        "seed": seed,
        "split": split,
        "n": n,
        "prevalence": prevalence,
        "threshold": threshold,
        "threshold_source": "validation",
        "evidence_role": evidence_role,
        "benchmark_scope": "single_seed",
    }
    for name in METRICS:
        value = _finite_float(metrics.get(name), f"{task} {split} metrics.{name}")
        if name not in {"log_loss"} and not 0.0 <= value <= 1.0:
            raise ValueError(f"{task} {split} metrics.{name} must lie in [0, 1]")
        if name == "log_loss" and value < 0.0:
            raise ValueError(f"{task} {split} metrics.log_loss cannot be negative")
        row[name] = value
    return row


def _resolve_validation_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    return path / "run_summary.json" if path.is_dir() else path


def _resolve_test_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.metadata.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"test directory must contain exactly one *.metadata.json: {path}"
            )
        return candidates[0]
    if path.name.endswith(".metadata.json"):
        return path
    sidecar = Path(str(path) + ".metadata.json")
    return sidecar if sidecar.is_file() else path


def _validate_validation(
    task: str, summary_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _read_json(summary_path, f"{task} validation run summary")
    if str(summary.get("task") or "").strip() != task:
        raise ValueError(f"validation task mismatch for {task}")
    if str(summary.get("evaluation_phase") or "").strip().lower() != "validation_screen":
        raise ValueError(f"{task} must come from evaluation_phase=validation_screen")
    if summary.get("test_evaluated") is not False or summary.get("test_metrics") is not None:
        raise ValueError(f"{task} validation run must not contain test evaluation")

    threshold_selection = _mapping(
        summary.get("threshold_selection"), f"{task} threshold selection"
    )
    method = str(threshold_selection.get("method") or "").strip().lower()
    if method not in {"fixed", "youden"}:
        raise ValueError(f"{task} threshold method is not validation-compatible")
    threshold = _finite_float(
        threshold_selection.get("threshold"), f"{task} validation threshold"
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{task} validation threshold must lie in [0, 1]")

    config = _mapping(summary.get("config_provenance"), f"{task} config provenance")
    runtime = _mapping(config.get("runtime"), f"{task} runtime provenance")
    seed = _seed(runtime.get("seed"), f"{task} seed")
    if seed != FULL_MODEL_SEED:
        raise ValueError(
            f"{task} full-model benchmark seed must be {FULL_MODEL_SEED}"
        )
    config_hash = str(config.get("config_hash") or "").strip()
    if not config_hash:
        raise ValueError(f"{task} config hash is missing")
    config_path = _normalized_path(config.get("config_path"), f"{task} config")
    subset = _mapping(config.get("modality_subset"), f"{task} modality subset")
    enabled_groups = tuple(str(value) for value in subset.get("enabled_groups", ()))
    disabled_groups = tuple(str(value) for value in subset.get("disabled_groups", ()))
    if enabled_groups != tuple(PRIMARY_MODALITIES) or disabled_groups:
        raise ValueError(
            f"{task} benchmark must enable all nine canonical modalities"
        )

    data = _mapping(summary.get("data_provenance"), f"{task} data provenance")
    if str(data.get("task") or "").strip() != task:
        raise ValueError(f"{task} data provenance task mismatch")
    split_counts = _mapping(data.get("split_counts"), f"{task} split counts")
    files = _mapping(data.get("files"), f"{task} input files")
    master = _source_record(files.get("master_csv"), f"{task} master")
    scaler = _source_record(files.get("scaler_json"), f"{task} scaler")
    scaler_provenance = _mapping(
        summary.get("scaler_provenance"), f"{task} scaler provenance"
    )
    _same_source(
        scaler,
        _source_record(scaler_provenance, f"{task} scaler provenance"),
        "scaler",
    )

    row = _metric_row(
        summary.get("validation_metrics"),
        task=task,
        split="val",
        seed=seed,
        threshold=threshold,
        evidence_role="validation_screen",
    )
    if row["n"] != _positive_count(split_counts.get("val"), f"{task} val split count"):
        raise ValueError(f"{task} validation n differs from data provenance")
    test_n = _positive_count(split_counts.get("test"), f"{task} test split count")
    lineage = {
        "task": task,
        "seed": seed,
        "threshold": threshold,
        "threshold_method": method,
        "config_hash": config_hash,
        "config_path": config_path,
        "master": master,
        "scaler": scaler,
        "enabled_groups": list(enabled_groups),
        "disabled_groups": list(disabled_groups),
        "expected_test_n": test_n,
        "run_summary_path": str(summary_path),
        "run_summary_sha256": _sha256(summary_path),
        "best_checkpoint_path": str((summary_path.parent / "best_model.pt").resolve()),
    }
    return row, lineage


def _validate_test(
    task: str, metadata_path: Path, validation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(metadata_path, f"{task} locked test metadata")
    if metadata.get("schema_version") != FROZEN_EVALUATION_VERSION:
        raise ValueError(f"{task} frozen test schema is incompatible")
    if metadata.get("analysis_role") != "locked_confirmatory_test":
        raise ValueError(f"{task} test is not locked confirmatory evidence")
    if str(metadata.get("split") or "").strip().lower() != "test":
        raise ValueError(f"{task} frozen evaluation split must be test")
    if str(metadata.get("task") or "").strip() != task:
        raise ValueError(f"locked test task mismatch for {task}")
    if metadata.get("model_id") != "full_teacher" or metadata.get("model_role") != "full_teacher":
        raise ValueError(f"{task} locked test must evaluate the full_teacher model")
    test_seed = _seed(metadata.get("seed"), f"{task} test seed")
    if test_seed != validation["seed"]:
        raise ValueError(f"{task} validation/test seed mismatch")
    test_threshold = _finite_float(
        metadata.get("threshold_from_validation"), f"{task} test threshold"
    )
    if not math.isclose(
        test_threshold, validation["threshold"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{task} test threshold differs from validation threshold")

    lock = _mapping(metadata.get("selection_lock"), f"{task} selection lock")
    if (
        _seed(lock.get("seed"), f"{task} selection-lock seed") != test_seed
        or str(lock.get("model_id") or "") != str(metadata.get("model_id") or "")
        or str(lock.get("model_role") or "") != str(metadata.get("model_role") or "")
    ):
        raise ValueError(f"{task} frozen evaluation/selection-lock identity mismatch")
    locked_threshold = _finite_float(
        lock.get("validation_threshold"), f"{task} selection-lock threshold"
    )
    if not math.isclose(locked_threshold, test_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{task} selection-lock threshold mismatch")
    lock_sha = str(lock.get("sha256") or "").strip().lower()
    if len(lock_sha) != 64 or any(
        character not in "0123456789abcdef" for character in lock_sha
    ):
        raise ValueError(f"{task} selection-lock SHA256 is invalid")
    lock_path = Path(_normalized_path(lock.get("path"), f"{task} selection lock"))
    if lock_path.is_file() and _sha256(lock_path) != lock_sha:
        raise ValueError(f"{task} selection-lock SHA256 mismatch")

    preflight = _mapping(
        metadata.get("pre_deserialization_selection_lock"),
        f"{task} pre-deserialization selection lock",
    )
    for key, expected in (
        ("task", task),
        ("model_id", str(metadata.get("model_id") or "")),
        ("model_role", str(metadata.get("model_role") or "")),
        ("seed", test_seed),
    ):
        if preflight.get(key) != expected:
            raise ValueError(f"{task} preflight {key} mismatch")

    if list(metadata.get("enabled_groups", ())) != validation["enabled_groups"]:
        raise ValueError(f"{task} validation/test enabled modalities mismatch")
    if list(metadata.get("disabled_groups", ())) != validation["disabled_groups"]:
        raise ValueError(f"{task} validation/test disabled modalities mismatch")

    source_files = _mapping(metadata.get("source_files"), f"{task} test source files")
    checkpoint = _source_record(source_files.get("checkpoint"), f"{task} checkpoint")
    config = _source_record(source_files.get("config"), f"{task} config")
    master = _source_record(source_files.get("master"), f"{task} master")
    scaler = _source_record(source_files.get("scaler"), f"{task} scaler")
    if checkpoint["path"] != validation["best_checkpoint_path"]:
        raise ValueError(f"{task} test checkpoint is not the validation best checkpoint")
    if config["path"] != validation["config_path"]:
        raise ValueError(f"{task} validation/test config path mismatch")
    _same_source(validation["master"], master, "master")
    _same_source(validation["scaler"], scaler, "scaler")
    for key, record in (
        ("checkpoint_path", checkpoint),
        ("config_path", config),
        ("master_path", master),
        ("scaler_path", scaler),
    ):
        if _normalized_path(preflight.get(key), f"{task} preflight {key}") != record["path"]:
            raise ValueError(f"{task} preflight/source {key} mismatch")

    predictions = _source_record(metadata.get("predictions"), f"{task} test predictions")
    prediction_path = Path(predictions["path"])
    if prediction_path.is_file() and _sha256(prediction_path) != predictions["sha256"]:
        raise ValueError(f"{task} test prediction SHA256 mismatch")
    row = _metric_row(
        metadata.get("metrics"),
        task=task,
        split="test",
        seed=test_seed,
        threshold=test_threshold,
        evidence_role="locked_confirmatory_test",
    )
    declared_n = _positive_count(metadata.get("n_participants"), f"{task} n_participants")
    if row["n"] != declared_n or row["n"] != validation["expected_test_n"]:
        raise ValueError(f"{task} locked test n differs from declared cohort")
    test_lineage = {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "selection_lock_path": str(lock_path),
        "selection_lock_sha256": lock_sha,
        "checkpoint": checkpoint,
        "config": config,
        "master": master,
        "scaler": scaler,
        "predictions": predictions,
        "model_id": str(metadata.get("model_id") or ""),
        "model_role": str(metadata.get("model_role") or ""),
    }
    return row, test_lineage


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _markdown(rows: Sequence[Mapping[str, Any]], *, benchmark_name: str) -> str:
    lines = [
        f"# {benchmark_name}",
        "",
        "**Scope: single-seed benchmark.** " + SINGLE_SEED_WARNING,
        "",
        TEST_GUARD_WARNING,
        "",
        "| Task | Split | Seed | N | Prev. | AUROC | AUPRC | Bal. acc. | Sens. | Spec. | Log loss | Brier | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {split} | {seed} | {n} | {prevalence:.4f} | "
            "{auroc:.4f} | {auprc:.4f} | {balanced_accuracy:.4f} | "
            "{sensitivity:.4f} | {specificity:.4f} | {log_loss:.4f} | "
            "{brier:.4f} | {threshold:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "- Every threshold in this table comes from validation.",
            "- Test rows, when present, are loaded only from locked frozen-evaluation metadata.",
            "- This command performs no model ranking, checkpoint selection, or threshold tuning.",
            "- One seed is not enough to quantify training variability; run a prespecified multi-seed analysis for final inference.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_three_task_benchmark(
    validation_summaries: Mapping[str, str | Path],
    out_dir: str | Path,
    *,
    test_evaluations: Optional[Mapping[str, str | Path]] = None,
    benchmark_name: str = "Full-assets three-task benchmark",
) -> dict[str, Path]:
    if set(validation_summaries) != set(TASKS):
        raise ValueError(f"validation inputs must contain exactly: {list(TASKS)}")
    tests = dict(test_evaluations or {})
    unknown_tests = sorted(set(tests) - set(TASKS))
    if unknown_tests:
        raise ValueError(f"unknown test tasks: {unknown_tests}")

    rows: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {}
    for task in TASKS:
        summary_path = _resolve_validation_path(validation_summaries[task])
        validation_row, validation_lineage = _validate_validation(task, summary_path)
        rows.append(validation_row)
        task_lineage: dict[str, Any] = {"validation": validation_lineage}
        if task in tests:
            metadata_path = _resolve_test_path(tests[task])
            test_row, test_lineage = _validate_test(
                task, metadata_path, validation_lineage
            )
            rows.append(test_row)
            task_lineage["test"] = test_lineage
        lineage[task] = task_lineage

    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "three_task_benchmark_v1.csv"
    json_path = destination / "three_task_benchmark_v1.json"
    markdown_path = destination / "three_task_benchmark_v1.md"
    _atomic_text(csv_path, _csv_text(rows))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "single_seed_three_task_benchmark_reporting",
        "benchmark_name": benchmark_name,
        "benchmark_scope": "single_seed_per_task",
        "tasks": list(TASKS),
        "test_tasks_included": [task for task in TASKS if task in tests],
        "model_selection_performed": False,
        "threshold_selection_performed": False,
        "threshold_source": "validation",
        "warnings": [SINGLE_SEED_WARNING, TEST_GUARD_WARNING],
        "lineage": lineage,
        "rows": rows,
    }
    _atomic_text(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(markdown_path, _markdown(rows, benchmark_name=benchmark_name))
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def _assignments(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        task, separator, path = value.partition("=")
        task = task.strip()
        path = path.strip()
        if not separator or task not in TASKS or not path:
            raise ValueError(f"{label} must use TASK=PATH with TASK in {list(TASKS)}")
        if task in result:
            raise ValueError(f"duplicate {label} input for {task}")
        result[task] = Path(path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        action="append",
        required=True,
        metavar="TASK=RUN_SUMMARY_OR_DIR",
        help="Repeat once for each of the three tasks.",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        metavar="TASK=METADATA_OR_PREDICTIONS_OR_DIR",
        help="Optional locked-test evaluator output; repeat per available task.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark-name", default="Full-assets three-task benchmark"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = summarize_three_task_benchmark(
        _assignments(args.validation, "--validation"),
        args.out_dir,
        test_evaluations=_assignments(args.test, "--test"),
        benchmark_name=str(args.benchmark_name).strip()
        or "Full-assets three-task benchmark",
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_COLUMNS",
    "FROZEN_EVALUATION_VERSION",
    "FULL_MODEL_SEED",
    "METRICS",
    "SCHEMA_VERSION",
    "SINGLE_SEED_WARNING",
    "TASKS",
    "TEST_GUARD_WARNING",
    "build_arg_parser",
    "summarize_three_task_benchmark",
]
