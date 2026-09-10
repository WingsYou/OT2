"""Synthetic report/runner fixtures; these are not study results."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_clinical_basic_information_v1 import catalog_fixture
from clinical_abcd_plan_v1 import build_abcd_plan, CONDITIONS, CONTRASTS, COHORTS
from summarize_clinical_abcd_v1 import METRICS, FILES, sha256


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reseal(directory):
    path = Path(directory) / "abcd_metadata.json"
    meta = json.loads(path.read_text())
    meta["output_sha256"] = {n: sha256(Path(directory) / n) for n in FILES}
    path.write_text(json.dumps(meta))


def write_result(directory, task="moca_binary", split="val"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    schema, catalog = catalog_fixture()
    schema["task"] = task
    plan = build_abcd_plan(schema, catalog)
    (directory / "abcd_plan.json").write_text(json.dumps(plan))
    (directory / "abcd_conditions.json").write_text(json.dumps([plan["conditions"][n] for n in CONDITIONS]))
    counts, metrics, summary, intervals = [], [], [], []
    for scope in COHORTS:
        context = {"task": task, "split": split, "cohort_scope": scope, "n_evaluated": 8,
                   "n_positive": 4, "n_negative": 4, "prevalence": 0.5, "cohort_mask_sha256": "8" * 64}
        counts.append(context)
        for name, level in zip(CONDITIONS, (0, 0.03, 0.10, 0.15)):
            metrics.append({**context, "condition": name, "threshold": 0.6, "n_without_any_input": 0,
                            **{m: 0.7 + level if m in ("brier", "log_loss") else 0.7 - level for m in METRICS}})
        for comp, (ref, deleted) in CONTRASTS.items():
            a, b = (next(r for r in metrics if r["condition"] == n and r["cohort_scope"] == scope) for n in (ref, deleted))
            for metric in METRICS:
                delta = b[metric] - a[metric] if metric in ("brier", "log_loss") else a[metric] - b[metric]
                row = {**context, "comparison": comp, "reference_condition": ref, "perturbed_condition": deleted,
                       "threshold_from_validation": 0.6, "metric": metric, "baseline_metric": a[metric],
                       "mean_perturbed_metric": b[metric], "mean_degradation_delta": delta,
                       "ci_lower": delta - 0.01, "ci_upper": delta + 0.01, "ci_status": "ok"}
                summary.append(row)
                intervals.append({**row, "point_estimate": delta})
    write_csv(directory / "abcd_metrics.csv", metrics)
    write_csv(directory / "abcd_summary.csv", summary)
    write_csv(directory / "abcd_bootstrap_intervals.csv", intervals)
    write_csv(directory / "abcd_cohort_counts.csv", counts)
    write_csv(directory / "abcd_paired_predictions.csv", [
        {"task": task, "split": split, "person_id": f"PRIVATE_SYNTHETIC_ID_{i}", "y": i % 2,
         "threshold": 0.6, **{f"{n}_probability": 0.5 for n in CONDITIONS}} for i in range(8)])
    meta = {"schema_version": "ot2-clinical-abcd-v1", "task": task, "split": split,
            "analysis_role": "synthetic validation" if split == "val" else "synthetic exploratory test",
            "training_seed": 42, "bootstrap_seed": 42, "bootstrap_samples": 16,
            "conditions": list(CONDITIONS), "plan_sha256": plan["plan_sha256"], "n_participants": 8,
            "threshold": 0.6, "threshold_source": "saved_full_teacher_validation_threshold",
            "baseline_metrics": {m: 0.7 for m in METRICS}, "code_sha256": {"synthetic.py": "0" * 64},
            "checkpoint_path": "synthetic-test-checkpoint", "checkpoint_sha256": "1" * 64,
            "config_hash": "2" * 64, "config_sha256": "3" * 64,
            "clinical_schema_sha256": "4" * 64, "clinical_catalog_sha256": "5" * 64,
            "source_token_cache": {"file_sha256": "6" * 64, "reused": True, "identity": {
                "task": task, "split": split, "n_participants": 8, "checkpoint_sha256": "1" * 64,
                "cohort_person_ids_sha256": "7" * 64}}, "inference_checks": {"synthetic_fixture": True},
            "output_sha256": {name: sha256(directory / name) for name in FILES}}
    (directory / "abcd_metadata.json").write_text(json.dumps(meta))
    return meta
