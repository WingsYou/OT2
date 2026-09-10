#!/usr/bin/env python3
"""Create a guarded, single-seed V0 validation-ablation feedback summary.

The command consumes an already trained full-model ``run_summary.json`` and
the matching validation-only modality-importance evidence.  It never trains,
selects on test, or makes a causal claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from ot2.analyze_modality_importance_v2 import IMPORTANCE_FORMAT_VERSION
    from ot2.task2_v2_config import PRIMARY_MODALITIES
except ModuleNotFoundError:  # Support ``python ot2/summarize_...py``.
    from analyze_modality_importance_v2 import IMPORTANCE_FORMAT_VERSION  # type: ignore
    from task2_v2_config import PRIMARY_MODALITIES  # type: ignore


FEEDBACK_FORMAT_VERSION = "ot2-feedback-v0-1"
REQUIRED_METRICS = ("auroc", "auprc", "log_loss")
BENEFIT_METRICS = frozenset({"auroc", "auprc"})
LOSS_METRICS = frozenset({"log_loss"})
PRELIMINARY_WARNING = (
    "Single-seed preliminary validation result; it requires multi-seed and "
    "locked-test confirmation before any deployment or modality-removal claim."
)
NONCAUSAL_WARNING = (
    "Ablation measures reliance of this fitted model on this validation "
    "cohort; it is not a causal estimate of clinical utility."
)
MISSINGNESS_WARNING = (
    "Because the model uses explicit source presence and learned missing tokens, "
    "frozen ablation combines dependence on measured content with dependence on "
    "the cohort's acquisition/missingness pattern; it is not content-only importance."
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _mapping(payload, label)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _extract_seed(run_summary: Mapping[str, Any]) -> int:
    config = _mapping(run_summary.get("config_provenance"), "config provenance")
    runtime = _mapping(config.get("runtime"), "config runtime provenance")
    value = runtime.get("seed")
    if isinstance(value, bool):
        raise ValueError("run seed must be an integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("run seed must be an integer") from exc
    if seed < 0 or seed != value:
        raise ValueError("run seed must be a non-negative integer")
    return seed


def _validate_lineage(
    *,
    run_dir: Path,
    run_summary: Mapping[str, Any],
    importance_dir: Path,
    metadata: Mapping[str, Any],
    summary_path: Path,
) -> Dict[str, Any]:
    if str(run_summary.get("evaluation_phase") or "").strip().lower() != (
        "validation_screen"
    ):
        raise ValueError("V0 feedback requires a validation_screen training run")
    if run_summary.get("test_evaluated") is not False:
        raise ValueError("V0 feedback refuses a run that evaluated test")
    if run_summary.get("test_metrics") is not None:
        raise ValueError("V0 feedback refuses a run containing test metrics")

    if metadata.get("schema_version") != IMPORTANCE_FORMAT_VERSION:
        raise ValueError("importance metadata has the wrong schema version")
    if str(metadata.get("split") or "").strip().lower() not in {
        "val",
        "validation",
    }:
        raise ValueError("V0 feedback requires validation-only importance")
    if metadata.get("analysis_role") != "validation_modality_selection":
        raise ValueError("importance metadata is not validation selection evidence")
    if metadata.get("checkpoint_kind") != "best":
        raise ValueError("importance evidence must use the best validation checkpoint")

    methods = metadata.get("methods")
    if not isinstance(methods, list) or "ablation" not in methods:
        raise ValueError("importance metadata must declare grouped ablation")
    modalities = metadata.get("primary_modalities")
    if modalities != list(PRIMARY_MODALITIES):
        raise ValueError(
            "importance metadata must declare all nine canonical modalities in order"
        )

    run_config = _mapping(
        run_summary.get("config_provenance"), "run config provenance"
    )
    config_hash = str(run_config.get("config_hash") or "").strip()
    if not config_hash or config_hash != str(metadata.get("config_hash") or ""):
        raise ValueError("run/importance config lineage mismatch")
    subset = _mapping(run_config.get("modality_subset"), "run modality subset")
    if subset.get("disabled_groups") != []:
        raise ValueError("V0 nine-modality feedback requires a full-source model")
    if subset.get("enabled_groups") != list(PRIMARY_MODALITIES):
        raise ValueError("run provenance does not enable all nine modalities")

    task = str(run_summary.get("task") or "").strip()
    if not task or task != str(metadata.get("task") or "").strip():
        raise ValueError("run/importance task mismatch")
    data = _mapping(run_summary.get("data_provenance"), "run data provenance")
    if str(data.get("task") or "").strip() != task:
        raise ValueError("run data provenance task mismatch")

    seed = _extract_seed(run_summary)
    metadata_seed = metadata.get("seed")
    if isinstance(metadata_seed, bool) or not isinstance(metadata_seed, int):
        raise ValueError("importance seed must be an integer")
    if seed != metadata_seed:
        raise ValueError("run/importance seed mismatch")

    checkpoint = (run_dir / "best_model.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint does not exist: {checkpoint}")
    declared_checkpoint_path = Path(
        str(metadata.get("checkpoint_path") or "")
    ).expanduser()
    if not declared_checkpoint_path.is_absolute():
        declared_checkpoint_path = (importance_dir / declared_checkpoint_path).resolve()
    else:
        declared_checkpoint_path = declared_checkpoint_path.resolve()
    if declared_checkpoint_path != checkpoint:
        raise ValueError("importance checkpoint path does not match run_dir/best_model.pt")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != str(metadata.get("checkpoint_sha256") or ""):
        raise ValueError("importance checkpoint SHA256 does not match the run checkpoint")

    output_hashes = _mapping(
        metadata.get("output_sha256"), "importance output hashes"
    )
    if str(output_hashes.get(summary_path.name) or "") != _sha256(summary_path):
        raise ValueError("importance summary SHA256 is missing or mismatched")
    return {
        "task": task,
        "seed": seed,
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
    }


def _wide_ablation_table(
    summary: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> pd.DataFrame:
    required_columns = {
        "method",
        "group",
        "metric",
        "baseline_metric",
        "mean_perturbed_metric",
        "mean_degradation_delta",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "delta_definition",
    }
    missing_columns = sorted(required_columns - set(summary.columns))
    if missing_columns:
        raise ValueError(f"importance summary is missing columns: {missing_columns}")
    selected = summary.loc[
        summary["method"].astype(str).eq("ablation")
        & summary["metric"].astype(str).isin(REQUIRED_METRICS)
    ].copy()
    expected_pairs = {
        (group, metric)
        for group in PRIMARY_MODALITIES
        for metric in REQUIRED_METRICS
    }
    observed_pairs = list(
        zip(selected["group"].astype(str), selected["metric"].astype(str))
    )
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("importance summary contains duplicate ablation metric rows")
    if set(observed_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(observed_pairs))
        extra = sorted(set(observed_pairs) - expected_pairs)
        raise ValueError(
            "importance summary must contain ablation rows for all nine modalities "
            f"and required metrics (missing={missing}, extra={extra})"
        )

    metadata_baseline = _mapping(
        metadata.get("baseline_metrics"), "importance baseline metrics"
    )
    by_pair = {
        (str(row.group), str(row.metric)): row
        for row in selected.itertuples(index=False)
    }
    rows: list[Dict[str, Any]] = []
    for group in PRIMARY_MODALITIES:
        output: Dict[str, Any] = {
            "modality": group,
            "task": lineage["task"],
            "split": "val",
            "method": "ablation",
            "seed": int(lineage["seed"]),
            "n_participants": int(metadata.get("n_participants")),
            "config_hash": lineage["config_hash"],
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "evidence_status": "exploratory_not_selection_eligible",
            "selection_eligible": False,
            "causal_interpretation_allowed": False,
        }
        for metric in REQUIRED_METRICS:
            record = by_pair[(group, metric)]
            baseline = _finite_float(
                record.baseline_metric, f"{group}/{metric} baseline"
            )
            ablated = _finite_float(
                record.mean_perturbed_metric, f"{group}/{metric} ablated"
            )
            delta = _finite_float(
                record.mean_degradation_delta, f"{group}/{metric} delta"
            )
            lower = _finite_float(record.ci_lower, f"{group}/{metric} CI lower")
            upper = _finite_float(record.ci_upper, f"{group}/{metric} CI upper")
            confidence = _finite_float(
                record.confidence_level, f"{group}/{metric} confidence level"
            )
            expected_definition = (
                "baseline_minus_perturbed"
                if metric in BENEFIT_METRICS
                else "perturbed_minus_baseline"
            )
            if str(record.delta_definition) != expected_definition:
                raise ValueError(f"{group}/{metric} has the wrong delta direction")
            recomputed = baseline - ablated if metric in BENEFIT_METRICS else ablated - baseline
            if not math.isclose(delta, recomputed, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{group}/{metric} delta is inconsistent with metrics")
            metadata_value = _finite_float(
                metadata_baseline.get(metric), f"metadata baseline {metric}"
            )
            if not math.isclose(baseline, metadata_value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{group}/{metric} baseline differs from metadata")
            if lower > upper or not 0.0 < confidence < 1.0:
                raise ValueError(f"{group}/{metric} confidence interval is invalid")
            output[f"baseline_{metric}"] = baseline
            output[f"ablated_{metric}"] = ablated
            output[f"delta_{metric}"] = delta
            output[f"{metric}_ci_lower"] = lower
            output[f"{metric}_ci_upper"] = upper
            output[f"{metric}_confidence_level"] = confidence
        rows.append(output)
    result = pd.DataFrame(rows).sort_values(
        ["delta_auroc", "modality"], ascending=[False, True], kind="stable"
    )
    result.insert(0, "auroc_drop_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown(table: pd.DataFrame, lineage: Mapping[str, Any]) -> str:
    lines = [
        f"# V0 modality-ablation feedback: `{lineage['task']}`",
        "",
        f"> **Preliminary:** {PRELIMINARY_WARNING}",
        "",
        f"> **Non-causal:** {NONCAUSAL_WARNING}",
        "",
        f"> **Missingness:** {MISSINGNESS_WARNING}",
        "",
        "Sorted by validation AUROC degradation (largest first). Positive AUROC/AUPRC "
        "delta means worse after ablation; positive log-loss delta also means worse.",
        "",
        "| Rank | Modality | AUROC full | AUROC ablated | AUROC drop (95% CI) | "
        "AUPRC full | AUPRC ablated | AUPRC drop (95% CI) | "
        "Log loss full | Log loss ablated | Log-loss increase (95% CI) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| {rank} | `{modality}` | {b_auroc:.4f} | {a_auroc:.4f} | "
            "{d_auroc:.4f} [{l_auroc:.4f}, {u_auroc:.4f}] | "
            "{b_auprc:.4f} | {a_auprc:.4f} | "
            "{d_auprc:.4f} [{l_auprc:.4f}, {u_auprc:.4f}] | "
            "{b_ll:.4f} | {a_ll:.4f} | {d_ll:.4f} [{l_ll:.4f}, {u_ll:.4f}] |".format(
                rank=row.auroc_drop_rank,
                modality=row.modality,
                b_auroc=row.baseline_auroc,
                a_auroc=row.ablated_auroc,
                d_auroc=row.delta_auroc,
                l_auroc=row.auroc_ci_lower,
                u_auroc=row.auroc_ci_upper,
                b_auprc=row.baseline_auprc,
                a_auprc=row.ablated_auprc,
                d_auprc=row.delta_auprc,
                l_auprc=row.auprc_ci_lower,
                u_auprc=row.auprc_ci_upper,
                b_ll=row.baseline_log_loss,
                a_ll=row.ablated_log_loss,
                d_ll=row.delta_log_loss,
                l_ll=row.log_loss_ci_lower,
                u_ll=row.log_loss_ci_upper,
            )
        )
    lines.extend(
        [
            "",
            "## Evidence scope",
            "",
            f"- Split: validation only",
            f"- Seed: `{lineage['seed']}` (one seed)",
            f"- Config hash: `{lineage['config_hash']}`",
            f"- Checkpoint SHA256: `{lineage['checkpoint_sha256']}`",
            "- This ranking is feedback for the next experiment, not evidence that a "
            "modality can already be removed.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_feedback(
    run_dir: str | Path,
    importance_dir: str | Path,
    out_dir: str | Path,
) -> Dict[str, Path]:
    run_root = Path(run_dir).expanduser().resolve()
    importance_root = Path(importance_dir).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve()
    run_summary_path = run_root / "run_summary.json"
    importance_summary_path = importance_root / "importance_summary.csv"
    metadata_path = importance_root / "importance_metadata.json"
    run_summary = _read_json(run_summary_path, "run summary")
    metadata = _read_json(metadata_path, "importance metadata")
    if not importance_summary_path.is_file():
        raise FileNotFoundError(
            f"importance summary does not exist: {importance_summary_path}"
        )
    summary = pd.read_csv(importance_summary_path)
    lineage = _validate_lineage(
        run_dir=run_root,
        run_summary=run_summary,
        importance_dir=importance_root,
        metadata=metadata,
        summary_path=importance_summary_path,
    )
    table = _wide_ablation_table(summary, metadata=metadata, lineage=lineage)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "feedback_v0_modality_ablation.csv"
    json_path = destination / "feedback_v0_summary.json"
    markdown_path = destination / "feedback_v0_report.md"
    _atomic_text(csv_path, table.to_csv(index=False))
    payload = {
        "schema_version": FEEDBACK_FORMAT_VERSION,
        "analysis_role": "single_seed_preliminary_validation_ablation_feedback",
        "evidence_status": "exploratory_not_selection_eligible",
        "selection_eligible": False,
        "task": lineage["task"],
        "split": "val",
        "method": "ablation",
        "seed": lineage["seed"],
        "n_modalities": len(table),
        "sort": "descending_delta_auroc_then_modality",
        "warnings": [
            PRELIMINARY_WARNING,
            NONCAUSAL_WARNING,
            MISSINGNESS_WARNING,
        ],
        "lineage": dict(lineage),
        "inputs": {
            "run_summary": {
                "path": str(run_summary_path),
                "sha256": _sha256(run_summary_path),
            },
            "importance_summary": {
                "path": str(importance_summary_path),
                "sha256": _sha256(importance_summary_path),
            },
            "importance_metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
        },
        "rows": table.to_dict(orient="records"),
    }
    _atomic_text(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(markdown_path, _markdown(table, lineage))
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--importance-dir", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: IMPORTANCE_DIR/feedback_v0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir or (args.importance_dir / "feedback_v0")
    outputs = summarize_feedback(args.run_dir, args.importance_dir, out_dir)
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
    "FEEDBACK_FORMAT_VERSION",
    "MISSINGNESS_WARNING",
    "NONCAUSAL_WARNING",
    "PRELIMINARY_WARNING",
    "REQUIRED_METRICS",
    "summarize_feedback",
]
