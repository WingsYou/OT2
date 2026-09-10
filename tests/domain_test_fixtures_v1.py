"""Synthetic metadata/results for tests; never scientific study results."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_domain_groups_v1 import build_domain_plan, DEFAULT_GROUPS
from summarize_domain_ablation_v1 import METRICS, FILES, sha256


def catalog_fixture(task="moca_binary"):
    rows = []
    for source, table, count, kind in [
        ("age", "participants", 1, "numeric"), ("cage", "observation", 1, "numeric"),
        ("brthyy", "observation", 1, "numeric"), ("years_of_education", "observation", 1, "numeric"),
        ("pxed1", "observation", 8, "categorical_concept_presence"),
        ("cl_maristat", "observation", 5, "categorical_concept_presence"),
        ("pxep1", "observation", 2, "categorical_concept_presence"),
        ("pxep2", "observation", 2, "categorical_concept_presence"),
        ("pxep3", "observation", 2, "categorical_concept_presence"),
        ("import_glucose", "measurement", 1, "numeric"),
        ("lbscat_a1c", "measurement", 1, "numeric"),
        ("eos_ds", "observation", 1, "numeric"),
        ("bmi_vsorres", "measurement", 1, "numeric"),
        ("mhoccur_rvo", "observation", 2, "categorical_concept_presence"),
        ("mhoccur_rvo", "condition", 1, "numeric"),
        ("rtfl_od", "observation", 1, "categorical_concept_presence"),
        ("via1", "observation", 2, "categorical_concept_presence"),
        ("ces7", "observation", 2, "categorical_concept_presence"),
        ("cestl", "observation", 1, "numeric"),
        ("cm_slp", "observation", 2, "categorical_concept_presence"),
        ("bp1_sysbp_vsorres", "measurement", 1, "numeric"),
    ]:
        for i in range(count):
            rows.append({"feature_name": f"{table}__{source}__c{i}", "source_key": source,
                         "table": table, "value_kind": kind})
    return {"task": task, "n_features": len(rows), "feature_names": [r["feature_name"] for r in rows]}, rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_result(directory, task="moca_binary", groups=DEFAULT_GROUPS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    plan = build_domain_plan(*catalog_fixture(task))
    (directory / "domain_plan.json").write_text(json.dumps(plan))
    conditions = [{**plan["groups"][g], "group": g} for g in groups]
    (directory / "domain_conditions.json").write_text(json.dumps(conditions))
    rows, intervals = [], []
    for group in groups:
        for scope in ("all_participants", "native_present_only", "any_target_information_present"):
            for metric in METRICS:
                bad = metric in ("log_loss", "brier")
                delta = 0.08 if bad else 0.03
                row = {"task": task, "split": "val", "group": group, "cohort_scope": scope,
                       "metric": metric, "baseline_metric": 0.7, "mean_perturbed_metric": 0.7 + delta if bad else 0.7 - delta,
                       "mean_degradation_delta": delta, "ci_lower": delta - 0.01, "ci_upper": delta + 0.01,
                       "ci_status": "ok", "n_evaluated": 8, "n_positive": 4, "n_negative": 4,
                       "prevalence": 0.5, "n_without_any_input": 0}
                rows.append(row)
                intervals.append({**row, "point_estimate": delta})
    write_csv(directory / "domain_summary.csv", rows)
    write_csv(directory / "domain_bootstrap_intervals.csv", intervals)
    (directory / "domain_paired_predictions.csv").write_text("person_id,y\nPRIVATE_SYNTHETIC_ID,1\n")
    (directory / "domain_cohort_counts.csv").write_text("task,n\nsynthetic,8\n")
    meta = {"schema_version": "ot2-domain-ablation-v1", "task": task, "split": "val",
            "training_seed": 42, "bootstrap_seed": 42, "bootstrap_samples": 16,
            "groups": list(groups), "plan_sha256": plan["plan_sha256"], "n_participants": 8,
            "threshold": 0.6, "threshold_source": "synthetic test threshold",
            "baseline_metrics": {m: 0.7 for m in METRICS}, "code_sha256": {"synthetic.py": "0" * 64},
            "checkpoint_path": "synthetic-test-checkpoint", "checkpoint_sha256": "1" * 64,
            "config_hash": "2" * 64, "config_sha256": "3" * 64,
            "clinical_schema_sha256": "4" * 64, "clinical_catalog_sha256": "5" * 64,
            "source_token_cache": {"file_sha256": "6" * 64, "identity": {"cohort_person_ids_sha256": "7" * 64}},
            "inference_checks": {"synthetic_fixture": True},
            "output_sha256": {name: sha256(directory / name) for name in FILES}}
    (directory / "domain_metadata.json").write_text(json.dumps(meta))
    return meta
