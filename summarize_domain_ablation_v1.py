#!/usr/bin/env python3
"""Build a self-contained, aggregate-only report using the Python standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TASK_LABELS = {"moca_binary": "MoCA", "cvd_broad_binary": "CVD broad", "cvd_strict_binary": "CVD strict"}
METRICS = ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "log_loss", "brier")
FILES = ("domain_plan.json", "domain_conditions.json", "domain_summary.csv",
         "domain_bootstrap_intervals.csv", "domain_paired_predictions.csv", "domain_cohort_counts.csv")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        name = f.name
        f.write(content)
    os.replace(name, path)


def _same(a, b):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def load_task(directory, expected_task=None):
    directory = Path(directory)
    meta_path = directory / "domain_metadata.json"
    meta = json.loads(meta_path.read_text())
    task = meta["task"]
    if meta.get("schema_version") != "ot2-domain-ablation-v1" or meta.get("split") != "val":
        raise ValueError("Unexpected report schema or non-validation results")
    if expected_task and task != expected_task:
        raise ValueError("Task identity mismatch")
    if meta.get("training_seed") != 42:
        raise ValueError("Expected frozen training seed 42")
    if set(meta["output_sha256"]) != set(FILES):
        raise ValueError("Incomplete result hash manifest")
    for filename in FILES:
        if sha256(directory / filename) != meta["output_sha256"][filename]:
            raise ValueError(f"Result hash mismatch: {filename}")
    plan = json.loads((directory / "domain_plan.json").read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if digest != plan["plan_sha256"] or digest != meta["plan_sha256"] or plan["task"] != task:
        raise ValueError("Clinical plan hash/task mismatch")
    groups = meta["groups"]
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Empty or duplicate ablation groups")
    conditions = json.loads((directory / "domain_conditions.json").read_text())
    if [d["group"] for d in conditions] != groups:
        raise ValueError("Condition/group mismatch")
    for d in conditions:
        planned = plan["groups"][d["group"]]
        for key in ("indices", "feature_names", "feature_count", "disabled_modalities"):
            if d[key] != planned[key]:
                raise ValueError(f"Applied condition differs from plan: {d['group']}: {key}")
    with (directory / "domain_summary.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (directory / "domain_bootstrap_intervals.csv").open(newline="") as f:
        intervals = list(csv.DictReader(f))
    def index_records(records):
        indexed = {}
        for row in records:
            key = (row["group"], row["cohort_scope"], row["metric"])
            if key in indexed or row["task"] != task or row["split"] != "val" or key[0] not in groups:
                raise ValueError("Duplicate or misidentified result row")
            indexed[key] = row
        return indexed
    indexed, ci = index_records(rows), index_records(intervals)
    if indexed.keys() != ci.keys() or set(r["group"] for r in rows) != set(groups):
        raise ValueError("Incomplete metrics or bootstrap intervals")
    for group in groups:
        scopes = {r["cohort_scope"] for r in rows if r["group"] == group}
        if "all_participants" not in scopes or "any_target_information_present" not in scopes:
            raise ValueError("Required participant cohorts missing")
        for scope in scopes:
            if {m for g, s, m in indexed if (g, s) == (group, scope)} != set(METRICS):
                raise ValueError("Missing metric in cohort")
    for key, row in indexed.items():
        for a, b in (("mean_degradation_delta", "point_estimate"), ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper")):
            if not _same(row[a], ci[key][b]):
                raise ValueError("Summary/interval disagreement")
        baseline, perturbed = float(row["baseline_metric"]), float(row["mean_perturbed_metric"])
        expected = perturbed - baseline if key[2] in ("brier", "log_loss") else baseline - perturbed
        if not _same(row["mean_degradation_delta"], expected):
            raise ValueError("Incorrect degradation sign")
        if key[1] == "all_participants":
            if int(row["n_evaluated"]) != meta["n_participants"] or not _same(row["baseline_metric"], meta["baseline_metrics"][key[2]]):
                raise ValueError("Baseline/cohort mismatch")
    return {"meta": meta, "plan": plan, "conditions": conditions, "rows": rows,
            "indexed": indexed, "metadata_sha256": sha256(meta_path), "directory": str(directory)}


def fmt(value, signed=False):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "NA"
    return format(value, "+.4f" if signed else ".4f") if math.isfinite(value) else "NA"


def effect(row):
    if row is None:
        return "Not available"
    return f"{fmt(row['mean_degradation_delta'], True)} [{fmt(row['ci_lower'], True)}, {fmt(row['ci_upper'], True)}]"


def esc(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def task_details(data):
    meta, plan = data["meta"], data["plan"]
    lines = [f"## {TASK_LABELS.get(meta['task'], meta['task'])}: detailed results", "",
             f"N = {meta['n_participants']}; fixed threshold = {meta['threshold']:.8f}; bootstrap samples = {meta['bootstrap_samples']}.", "",
             "### All-participant metrics", "",
             "| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |",
             "|---|---|---:|---:|---|---|"]
    for group in meta["groups"]:
        for metric in METRICS:
            row = data["indexed"][(group, "all_participants", metric)]
            lines.append(f"| {plan['groups'][group]['label']} | {metric} | {fmt(row['baseline_metric'])} | {fmt(row['mean_perturbed_metric'])} | {effect(row)} | {row['ci_status']} |")
    lines += ["", "### Availability strata (paired AUROC comparisons)", "",
              "`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.", "",
              "| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |",
              "|---|---|---:|---|---:|---:|---:|---|"]
    for row in data["rows"]:
        if row["metric"] == "auroc":
            lines.append(f"| {plan['groups'][row['group']]['label']} | {row['cohort_scope']} | {row['n_evaluated']} | {row['n_positive']} / {row['n_negative']} | {row['n_without_any_input']} | {fmt(row['baseline_metric'])} | {fmt(row['mean_perturbed_metric'])} | {effect(row)} |")
    lines += ["", "### Exact Clinical source scope", "",
              "Counts below are encoded input columns, not independent examinations. Every encoding of a listed source is removed. The full ordered encoded-column list and indices are preserved in domain_plan.json and domain_conditions.json.", ""]
    for group in meta["groups"]:
        d = plan["groups"][group]
        lines += [f"**{d['label']}** — {d['feature_count']} columns from {d['source_key_count']} sources; disabled branches: {', '.join(d['disabled_modalities'])}.", "",
                  "Removed source keys: " + (", ".join(f"`{s}`" for s in d["source_keys"]) or "None") + ".", ""]
        if d["absent_source_keys"]:
            lines += ["Allowlisted but absent from this trained task (not reintroduced): " + ", ".join(f"`{s}`" for s in d["absent_source_keys"]) + ".", ""]
    lines += ["### Provenance and verification", "", "| Item | Value |", "|---|---|"]
    provenance = {"checkpoint": meta["checkpoint_path"], "checkpoint SHA256": meta["checkpoint_sha256"],
                  "config hash": meta["config_hash"], "config file SHA256": meta["config_sha256"],
                  "Clinical schema SHA256": meta["clinical_schema_sha256"],
                  "Clinical catalog SHA256": meta["clinical_catalog_sha256"], "deletion plan SHA256": meta["plan_sha256"],
                  "metadata SHA256": data["metadata_sha256"], "threshold origin": meta["threshold_source"],
                  "training seed": meta["training_seed"], "bootstrap seed": meta["bootstrap_seed"],
                  "source cache SHA256": meta["source_token_cache"]["file_sha256"],
                  "evaluation cohort hash": meta["source_token_cache"]["identity"]["cohort_person_ids_sha256"],
                  "master config hash": meta["source_token_cache"]["identity"].get("master_config_hash", "unavailable"),
                  "scaler config hash": meta["source_token_cache"]["identity"].get("scaler_config_hash", "unavailable"),
                  "resolved model config hash": meta["source_token_cache"]["identity"].get("model_config_hash", "unavailable"),
                  "result files": "All six file hashes verified before report generation",
                  **meta["inference_checks"]}
    for k, v in provenance.items():
        lines.append(f"| {esc(k)} | {esc(v)} |")
    return lines + [""]


def render_report(data_by_task, expected_tasks, errors, status):
    complete = not errors and len(data_by_task) == len(expected_tasks) and status.get("state", "complete") == "complete"
    lines = ["# Domain ablation summary", "", f"**Overall status: {'COMPLETE' if complete else 'INCOMPLETE / FAILED'}**", "",
             "Frozen seed-42 models; validation split; joint removal of model branches and prespecified Clinical information. This file contains aggregate results and field names, without participant-level predictions or identifiers.", ""]
    if status:
        lines += [f"Runner state: {esc(status.get('state'))}; phase: {esc(status.get('phase'))}; updated UTC: {esc(status.get('updated_utc', 'unavailable'))}.", ""]
        if status.get("error"):
            lines += [f"Runner error: {esc(status['error'])}", ""]
    lines += ["| Task | Result status |", "|---|---|"]
    for task in expected_tasks:
        lines.append(f"| {TASK_LABELS.get(task, task)} | {'Verified' if task in data_by_task else esc(errors.get(task, 'Not completed'))} |")
    if errors.get("consistency"):
        lines += ["", "Consistency error: " + esc(errors["consistency"])]
    lines += ["", "## AUROC degradation across tasks", "",
              "Positive = AUROC decreased after deletion; negative = AUROC increased. Values are absolute AUROC differences, followed by pointwise 95% paired bootstrap intervals. 0.0100 equals one AUROC percentage point.", ""]
    groups = []
    for data in data_by_task.values():
        for g in data["meta"]["groups"]:
            if g not in groups:
                groups.append(g)
    lines += ["| Deletion | " + " | ".join(TASK_LABELS.get(t, t) for t in expected_tasks) + " |",
              "|---|" + "---|" * len(expected_tasks)]
    for group in groups:
        example = next(d for d in data_by_task.values() if group in d["meta"]["groups"])
        cells = []
        for task in expected_tasks:
            data = data_by_task.get(task)
            row = data["indexed"].get((group, "all_participants", "auroc")) if data else None
            cells.append(effect(row))
        lines.append("| " + example["plan"]["groups"][group]["label"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Descriptive findings", ""]
    for task, data in data_by_task.items():
        candidates = [r for r in data["rows"] if r["metric"] == "auroc" and r["cohort_scope"] == "all_participants"
                      and math.isfinite(float(r["mean_degradation_delta"]))]
        if candidates:
            largest = max(candidates, key=lambda r: float(r["mean_degradation_delta"]))
            label = data["plan"]["groups"][largest["group"]]["label"]
            supported = [data["plan"]["groups"][r["group"]]["label"] for r in candidates if float(r["ci_lower"]) > 0]
            lines += [f"- {TASK_LABELS.get(task, task)}: largest observed AUROC degradation among the selected conditions: {label}, {effect(largest)}. Conditions with a pointwise interval entirely above zero: {', '.join(supported) or 'none'}. This is descriptive validation evidence."]
    lines += ["", "## Full-input baseline", "", "| Task | N | Prevalence | AUROC | AUPRC | Sensitivity | Specificity | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, data in data_by_task.items():
        m = data["meta"]
        row = data["indexed"][(m["groups"][0], "all_participants", "auroc")]
        lines.append(f"| {TASK_LABELS.get(task, task)} | {m['n_participants']} | {fmt(row['prevalence'])} | " + " | ".join(fmt(m["baseline_metrics"][x]) for x in ("auroc", "auprc", "sensitivity", "specificity", "brier")) + " |")
    lines += ["", "## Interpretation and scope", "",
              "- AUROC/AUPRC/recall/balanced-accuracy degradation = full minus deleted; Brier/log-loss degradation = deleted minus full. Positive always means worse.",
              "- Intervals crossing zero indicate unresolved direction, not equivalence or noninferiority. Intervals are not adjusted for multiple comparisons and exclude training-seed uncertainty.",
              "- Full and deleted predictions use identical people within each reported cohort and the original full-model validation threshold. Threshold metrics can change substantially under input removal.",
              "- These are fitted-model joint information-removal effects; they are neither causal effects nor exclusive device contributions. Single-group effects need not add up to a joint effect.",
              "- Missing-input performance does not demonstrate reconstruction or the performance achievable by retraining."]
    if data_by_task:
        first = next(iter(data_by_task.values()))
        lines += ["- " + note for note in first["plan"]["scope_notes"]]
        lines += ["", "## Shared code snapshot", "", "| File | SHA256 |", "|---|---|"]
        for name, digest in first["meta"]["code_sha256"].items():
            lines.append(f"| {name} | {digest} |")
    for task in expected_tasks:
        if task in data_by_task:
            lines += [""] + task_details(data_by_task[task])
    return "\n".join(lines) + "\n", complete


def write_report(root, tasks=TASKS, single_directory=None):
    root = Path(root)
    status_path = root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() and single_directory is None else {}
    data, errors = {}, {}
    for task in tasks:
        path = Path(single_directory) if single_directory else root / task / "seed42/val_domain"
        try:
            data[task] = load_task(path, task)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[task] = f"Not verified: {type(exc).__name__}: {exc}"
    snapshots = {json.dumps(d["meta"]["code_sha256"], sort_keys=True) for d in data.values()}
    group_sets = {tuple(d["meta"]["groups"]) for d in data.values()}
    bootstrap_specs = {(d["meta"]["bootstrap_samples"], d["meta"]["bootstrap_seed"]) for d in data.values()}
    if len(snapshots) > 1 or len(group_sets) > 1 or len(bootstrap_specs) > 1:
        errors["consistency"] = "Tasks differ in code snapshots, selected groups, or bootstrap settings; do not compare as one run."
    report, complete = render_report(data, tuple(tasks), errors, status)
    atomic_write(root / "SUMMARY.md", report)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    status_path = args.root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    tasks = args.tasks or status.get("tasks") or TASKS
    complete = write_report(args.root, tasks)
    print(f"REPORT: {args.root / 'SUMMARY.md'}; {'complete' if complete else 'incomplete'}")
    return 0 if complete or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
