#!/usr/bin/env python3
"""One aggregate report for retraining progress and validation modality reliance."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from basic_retrain_contract_v1 import (TASKS, TASK_LABELS, GROUPS, SCOPES, METRICS, read_json, sha256,
                                       verify_prepared, verify_training, verify_files, atomic_text)


def num(x):
    return float(x) if x not in (None, "") else math.nan


def same(a, b):
    a, b = num(a), num(b)
    return math.isnan(a) and math.isnan(b) or math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9)


def fmt(x, signed=False):
    x = num(x)
    return format(x, "+.4f" if signed else ".4f") if math.isfinite(x) else "NA"


def effect(r):
    return f"{fmt(r['mean_degradation_delta'], True)} [{fmt(r['ci_lower'], True)}, {fmt(r['ci_upper'], True)}]"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def load_importance(root, task):
    root = Path(root)
    inputs = root / "inputs" / task
    prep = verify_prepared(inputs)
    trained = verify_training(root / "models" / task / "seed42", sha256(inputs / "PREPARED.json"))
    directory = root / "importance" / task / "seed42/val"
    meta = read_json(directory / "importance_metadata.json")
    expected = {"importance_summary.csv", "importance_bootstrap_intervals.csv",
                "importance_paired_predictions.csv", "importance_cohort_counts.csv"}
    if meta["schema_version"] != "ot2-basic-retrain-importance-v1" or meta["task"] != task or meta["split"] != "val" or meta["clinical_width"] != 23:
        raise ValueError("Unexpected analysis identity")
    if meta["groups"] != {g: list(m) for g, m in GROUPS.items()} or set(meta["output_sha256"]) != expected:
        raise ValueError("Incomplete analysis conditions/files")
    verify_files(directory, meta["output_sha256"])
    if meta["checkpoint_sha256"] != trained["output_sha256"]["best_model.pt"] or meta["code_sha256"] != trained["code_sha256"] or meta["config_hash"] != trained["config_hash"]:
        raise ValueError("Training/importance model or code mismatch")
    if meta["prepared_sha256"] != sha256(inputs / "PREPARED.json") or meta["trained_sha256"] != sha256(root / "models" / task / "seed42/TRAINED.json"):
        raise ValueError("Training/importance receipt mismatch")
    if not same(meta["threshold"], trained["threshold"]) or meta["basic_plan_sha256"] != prep["basic_plan_sha256"]:
        raise ValueError("Training/importance threshold or Clinical plan mismatch")
    for metric in METRICS:
        if not same(meta["baseline_metrics"][metric], trained["validation_metrics"][metric]):
            raise ValueError("Importance baseline differs from best-checkpoint validation metrics")
    def read(name):
        with (directory / name).open(newline="") as f:
            return list(csv.DictReader(f))
    def index(rows):
        out = {}
        for row in rows:
            key = (row["group"], row["cohort_scope"], row["metric"])
            if key in out or row["task"] != task or row["split"] != "val":
                raise ValueError("Duplicate/misidentified importance row")
            out[key] = row
        return out
    summary = index(read("importance_summary.csv"))
    intervals = index(read("importance_bootstrap_intervals.csv"))
    required = {(g, s, m) for g in GROUPS for s in SCOPES for m in METRICS}
    if set(summary) != required or set(intervals) != required:
        raise ValueError("Incomplete cohort/metric coverage")
    repeated = {}
    for key, r in summary.items():
        ci = intervals[key]
        for a, b in (("mean_degradation_delta", "point_estimate"), ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper")):
            if not same(r[a], ci[b]):
                raise ValueError("Summary/bootstrap interval disagreement")
        ref, deleted = num(r["baseline_metric"]), num(r["mean_perturbed_metric"])
        expected_delta = deleted - ref if key[2] in ("brier", "log_loss") else ref - deleted
        if not same(expected_delta, r["mean_degradation_delta"]) or not same(r["threshold_from_validation"], meta["threshold"]):
            raise ValueError("Incorrect degradation/threshold")
        if key[1] == "all_participants" and (int(r["n_evaluated"]) != meta["n_participants"] or not same(ref, meta["baseline_metrics"][key[2]])):
            raise ValueError("Incorrect baseline/cohort")
        tag = (key[0], r["cohort_mask_sha256"], key[2])
        if tag in repeated and any(not same(r[k], repeated[tag][k]) for k in ("mean_degradation_delta", "ci_lower", "ci_upper")):
            raise ValueError("Identical cohorts have different estimates")
        repeated[tag] = r
    return {"meta": meta, "trained": trained, "prepared": prep,
            "plan": read_json(inputs / "basic_plan.json"), "rows": summary}


def write_report(root, tasks=TASKS):
    root = Path(root)
    state = read_json(root / "RUN_STATUS.json") if (root / "RUN_STATUS.json").exists() else {}
    data, trained, errors = {}, {}, {}
    for task in tasks:
        try:
            trained[task] = verify_training(root / "models" / task / "seed42", sha256(root / "inputs" / task / "PREPARED.json"))
            data[task] = load_importance(root, task)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[task] = f"{type(exc).__name__}: {exc}"
    specs = {(json.dumps(d["meta"]["code_sha256"], sort_keys=True), d["meta"]["bootstrap_samples"], d["meta"]["bootstrap_seed"]) for d in data.values()}
    if len(specs) > 1:
        errors["consistency"] = "Mixed code/bootstrap settings across tasks"
    complete = not errors and len(data) == len(tasks) and state.get("state", "complete") == "complete"
    lines = ["# Basic Clinical retraining and modality importance", "", f"**Overall status: {'COMPLETE' if complete else 'INCOMPLETE / RUNNING / FAILED'}**", "",
             "New seed-42 task models trained with only the agreed 23 Clinical columns plus the other eight modalities. Original splits and retained train-fitted scaler statistics are preserved. No test inference is included.", "",
             "All nonclinical preprocessing, model settings and ImageNet retinal initialization follow the source experiment. The previous full-Clinical task checkpoint is not loaded. Clinical input width is physically 23 during training and inference.", "",
             "This aggregate report contains no participant identifiers or predictions.", "",
             f"Runner: {esc(state.get('state', 'standalone'))}; phase: {esc(state.get('phase', 'report'))}; updated UTC: {esc(state.get('updated_utc', 'unavailable'))}.", ""]
    if state.get("error"):
        lines += ["Runner error: " + esc(state["error"]), ""]
    lines += ["| Task | Training | Importance |", "|---|---|---|"]
    for task in tasks:
        lines.append(f"| {TASK_LABELS[task]} | {'Verified' if task in trained else 'Not complete'} | {'Verified' if task in data else esc(errors.get(task, 'Not complete'))} |")
    if "consistency" in errors:
        lines += ["", errors["consistency"]]
    lines += ["", "## New-model validation baseline", "", "| Task | Best epoch | Epochs | Threshold | AUROC | AUPRC | Sensitivity | Specificity | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, t in trained.items():
        lines.append(f"| {TASK_LABELS[task]} | {t['best_epoch']} | {t['epochs_completed']} | {fmt(t['threshold'])} | " + " | ".join(fmt(t["validation_metrics"][m]) for m in ("auroc", "auprc", "sensitivity", "specificity", "brier")) + " |")
    lines += ["", "## AUROC degradation after deletion", "",
              "Positive = new model performs worse after removal. These are absolute AUROC differences with pointwise paired 95% bootstrap intervals, not percentages of importance.", "",
              "| Removed input | " + " | ".join(TASK_LABELS[t] for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
    for group in GROUPS:
        cells = [effect(data[t]["rows"][(group, "all_participants", "auroc")]) if t in data else "Pending" for t in tasks]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += ["", "`clinical_data` means only the 23 Basic Clinical columns. `nonclinical_all` removes the other eight together and leaves Basic Clinical; it tests whether the newly trained model collectively uses the other modalities.", "",
              "- Clinical information reduction does not guarantee every remaining modality is used, or that reconstruction has been learned.",
              "- All deletion conditions use the new model's full-input validation threshold. AUROC, probability errors and threshold-based metrics should be interpreted together.",
              "- Identical participant masks reuse identical bootstrap intervals. Intervals exclude training-seed uncertainty and are not adjusted for multiplicity. Crossing zero does not establish equivalence or noninferiority.",
              "- Leave-one-out measures this newly fitted model's reliance under missing inputs; it is not leave-one-modality-out retraining or a causal effect.", ""]
    for task, d in data.items():
        meta = d["meta"]
        lines += [f"## {TASK_LABELS[task]} details", "", f"Validation N={meta['n_participants']}; bootstrap samples={meta['bootstrap_samples']}; threshold={meta['threshold']:.8f}.", "",
                  "| Deletion | Metric | Baseline | After deletion | Degradation [95% CI] |", "|---|---|---:|---:|---|"]
        for group in GROUPS:
            for metric in METRICS:
                r = d["rows"][(group, "all_participants", metric)]
                lines.append(f"| {group} | {metric} | {fmt(r['baseline_metric'])} | {fmt(r['mean_perturbed_metric'])} | {effect(r)} |")
        lines += ["", "### Availability strata", "", "| Deletion | Cohort | N | Positive / negative | No remaining input | AUROC degradation [95% CI] |", "|---|---|---:|---|---:|---|"]
        for group in GROUPS:
            for scope in SCOPES:
                r = d["rows"][(group, scope, "auroc")]
                lines.append(f"| {group} | {scope} | {r['n_evaluated']} | {r['n_positive']} / {r['n_negative']} | {r['n_without_any_input']} | {effect(r)} |")
        lines += ["", "### Exact retained Clinical columns", "", "| Encoded column | Table | Source key |", "|---|---|---|"]
        for i in d["plan"]["retained_indices_zero_based"]:
            r = d["plan"]["source_identities"][i]
            lines.append(f"| {r['feature_name']} | {r['table']} | {r['source_key']} |")
        lines += ["", "### Provenance", "", "| Item | Value |", "|---|---|"]
        for key in ("checkpoint_sha256", "config_hash", "prepared_sha256", "trained_sha256", "basic_plan_sha256", "threshold_source", "source_token_cache", "output_sha256"):
            lines.append(f"| {key} | {esc(json.dumps(meta[key], sort_keys=True))} |")
    if data:
        lines += ["", "## Code snapshot", "", "| File | SHA256 |", "|---|---|"]
        for name, digest in next(iter(data.values()))["meta"]["code_sha256"].items():
            lines.append(f"| {name} | {digest} |")
    atomic_text(root / "SUMMARY.md", "\n".join(lines) + "\n")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    state = read_json(args.root / "RUN_STATUS.json")
    raise SystemExit(0 if write_report(args.root, state["tasks"]) else 1)


if __name__ == "__main__":
    main()
