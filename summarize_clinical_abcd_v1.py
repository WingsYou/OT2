#!/usr/bin/env python3
"""Verify ABCD outputs and write one aggregate-only Markdown report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from clinical_abcd_plan_v1 import CONDITIONS, CONTRASTS, COHORTS, LABELS, NONCLINICAL, plan_hash

TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TASK_LABELS = dict(zip(TASKS, ("MoCA", "CVD broad", "CVD strict")))
METRICS = ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "log_loss", "brier")
FILES = ("abcd_plan.json", "abcd_conditions.json", "abcd_metrics.csv", "abcd_summary.csv",
         "abcd_bootstrap_intervals.csv", "abcd_paired_predictions.csv", "abcd_cohort_counts.csv")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        name = f.name
        f.write(content)
    os.replace(name, path)


def number(value):
    return float(value) if value not in (None, "") else math.nan


def same(a, b):
    a, b = number(a), number(b)
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)


def fmt(value, signed=False):
    value = number(value)
    return format(value, "+.4f" if signed else ".4f") if math.isfinite(value) else "NA"


def effect(row):
    return f"{fmt(row['mean_degradation_delta'], True)} [{fmt(row['ci_lower'], True)}, {fmt(row['ci_upper'], True)}]"


def esc(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def load_task(directory, expected_task=None, expected_split="val"):
    directory = Path(directory)
    meta = json.loads((directory / "abcd_metadata.json").read_text())
    task, split = meta["task"], meta["split"]
    if meta["schema_version"] != "ot2-clinical-abcd-v1" or split not in ("val", "test") or split != expected_split:
        raise ValueError("Unexpected ABCD schema/split")
    if (expected_task and task != expected_task) or meta["training_seed"] != 42 or meta["conditions"] != list(CONDITIONS):
        raise ValueError("Task/seed/condition identity mismatch")
    if set(meta["output_sha256"]) != set(FILES):
        raise ValueError("Incomplete output hash manifest")
    for name in FILES:
        if sha256(directory / name) != meta["output_sha256"][name]:
            raise ValueError(f"Output hash mismatch: {name}")
    plan = json.loads((directory / "abcd_plan.json").read_text())
    if plan_hash({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"] or plan["plan_sha256"] != meta["plan_sha256"]:
        raise ValueError("Plan hash mismatch")
    if plan["task"] != task or plan["n_retained"] != 23:
        raise ValueError("Basic Clinical scope mismatch")
    conditions = json.loads((directory / "abcd_conditions.json").read_text())
    if conditions != [plan["conditions"][n] for n in CONDITIONS]:
        raise ValueError("Applied conditions differ from plan")
    for d in conditions:
        name = d["condition"]
        if d["disabled_modalities"] != (list(NONCLINICAL) if name in ("B", "D") else []):
            raise ValueError("Incorrect nonclinical deletion scope")
        if d["masked_indices"] != (plan["masked_indices_zero_based"] if name in ("C", "D") else []):
            raise ValueError("Incorrect Clinical deletion scope")
        if d["n_clinical_columns_retained"] != (23 if name in ("C", "D") else plan["original_width"]):
            raise ValueError("Incorrect Clinical column count")
    identity = meta["source_token_cache"]["identity"]
    if identity["task"] != task or identity["split"] != split or identity["n_participants"] != meta["n_participants"] or identity["checkpoint_sha256"] != meta["checkpoint_sha256"]:
        raise ValueError("Cache provenance mismatch")
    if meta["threshold_source"] != "saved_full_teacher_validation_threshold":
        raise ValueError("Threshold must originate from validation")

    def records(name):
        with (directory / name).open(newline="") as f:
            rows = list(csv.DictReader(f))
        if any(r["task"] != task or r["split"] != split for r in rows):
            raise ValueError("Misidentified result row")
        return rows

    def indexed(rows, fields):
        result = {tuple(r[f] for f in fields): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate result rows")
        return result

    metrics = indexed(records("abcd_metrics.csv"), ("condition", "cohort_scope"))
    rows = records("abcd_summary.csv")
    summary = indexed(rows, ("comparison", "cohort_scope", "metric"))
    ci = indexed(records("abcd_bootstrap_intervals.csv"), ("comparison", "cohort_scope", "metric"))
    counts = indexed(records("abcd_cohort_counts.csv"), ("cohort_scope",))
    expected_keys = {(c, s, m) for c in CONTRASTS for s in COHORTS for m in METRICS}
    if set(metrics) != {(n, s) for n in CONDITIONS for s in COHORTS} or set(summary) != expected_keys or set(ci) != expected_keys or set(counts) != {(s,) for s in COHORTS}:
        raise ValueError("Missing or unexpected conditions/cohorts/metrics")
    for (name, scope), row in metrics.items():
        count = counts[(scope,)]
        if int(row["n_evaluated"]) != int(count["n_evaluated"]) or not same(row["threshold"], meta["threshold"]):
            raise ValueError("Condition cohort/threshold mismatch")
        if not 0 <= int(row["n_without_any_input"]) <= int(row["n_evaluated"]):
            raise ValueError("Invalid missing-input count")
        if scope == "all_participants" and int(row["n_evaluated"]) != meta["n_participants"]:
            raise ValueError("All-participant cohort size mismatch")
        for metric in METRICS:
            number(row[metric])
            if name == "A" and scope == "all_participants" and not same(row[metric], meta["baseline_metrics"][metric]):
                raise ValueError("Full-input baseline mismatch")
    repeated = {}
    for key, row in summary.items():
        comparison, scope, metric = key
        ref, deleted = CONTRASTS[comparison]
        if row["reference_condition"] != ref or row["perturbed_condition"] != deleted:
            raise ValueError("Comparison direction mismatch")
        if not same(row["threshold_from_validation"], meta["threshold"]):
            raise ValueError("Comparison threshold mismatch")
        for field, other in (("mean_degradation_delta", "point_estimate"), ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper")):
            if not same(row[field], ci[key][other]):
                raise ValueError("Summary/interval disagreement")
        a, b = metrics[(ref, scope)][metric], metrics[(deleted, scope)][metric]
        delta = number(b) - number(a) if metric in ("brier", "log_loss") else number(a) - number(b)
        if not same(row["baseline_metric"], a) or not same(row["mean_perturbed_metric"], b) or not same(row["mean_degradation_delta"], delta):
            raise ValueError("Incorrect absolute metric or degradation sign")
        if int(row["n_evaluated"]) != int(counts[(scope,)]["n_evaluated"]) or row["cohort_mask_sha256"] != counts[(scope,)]["cohort_mask_sha256"]:
            raise ValueError("Comparison cohort mismatch")
        repeat_key = (comparison, row["cohort_mask_sha256"], metric)
        if repeat_key in repeated and any(not same(row[k], repeated[repeat_key][k]) for k in ("mean_degradation_delta", "ci_lower", "ci_upper")):
            raise ValueError("Identical cohort has inconsistent intervals")
        repeated[repeat_key] = row
    predictions = records("abcd_paired_predictions.csv")
    if len(predictions) != meta["n_participants"] or len({r["person_id"] for r in predictions}) != len(predictions):
        raise ValueError("Paired prediction participant count/uniqueness mismatch")
    for row in predictions:
        if row["y"] not in ("0", "1") or not same(row["threshold"], meta["threshold"]):
            raise ValueError("Paired prediction label/threshold mismatch")
        if any(not 0 <= number(row[f"{n}_probability"]) <= 1 for n in CONDITIONS):
            raise ValueError("Invalid condition probability")
    return {"meta": meta, "plan": plan, "metrics": metrics, "summary": summary,
            "counts": counts, "metadata_sha256": sha256(directory / "abcd_metadata.json")}


def write_report(root, tasks=TASKS, single_directory=None, split="val"):
    root = Path(root)
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    status_path = root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() and single_directory is None else {}
    data, errors = {}, {}
    for task in tasks:
        path = Path(single_directory) if single_directory else root / task / "seed42" / f"{split}_abcd"
        try:
            data[task] = load_task(path, task, expected_split=split)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[task] = f"Not verified: {type(exc).__name__}: {exc}"
    specs = {(json.dumps(d["meta"]["code_sha256"], sort_keys=True), d["meta"]["bootstrap_samples"], d["meta"]["bootstrap_seed"]) for d in data.values()}
    if len(specs) > 1 or (status and (status.get("split") != split or tuple(status.get("tasks", ())) != tuple(tasks))):
        errors["consistency"] = "Different code/bootstrap settings or requested split/tasks; do not combine as one run."
    complete = not errors and len(data) == len(tasks) and status.get("state", "complete") == "complete"
    role = "Validation analysis" if split == "val" else "Exploratory test analysis (test has already been inspected)"
    lines = ["# Clinical A/B/C/D summary", "", f"**Overall status: {'COMPLETE' if complete else 'INCOMPLETE / FAILED'}**", "",
             f"{role}; frozen seed-42 models. All four predictions are recomputed and paired on the same people. No training or threshold optimization.", "",
             "This report contains aggregate results and feature names, without participant identifiers or predictions.", ""]
    if status:
        lines += [f"Runner: {esc(status.get('state'))}; phase: {esc(status.get('phase'))}; updated UTC: {esc(status.get('updated_utc'))}.", ""]
        if status.get("error"):
            lines += [f"Runner error: {esc(status['error'])}", ""]
    for key, error in errors.items():
        lines += [f"- {esc(key)}: {esc(error)}"]
    lines += ["", "## Conditions", "", "| ID | Clinical | Other eight modalities |", "|---|---|---|",
              "| A | Full, as trained | Native availability |", "| B | Full, as trained | All disabled |",
              "| C | Agreed 23 basic columns | Native availability |", "| D | Agreed 23 basic columns | All disabled |", "",
              "Disabled in B/D: " + ", ".join(f"`{m}`" for m in NONCLINICAL) + ". ECG removal includes waveform and machine-report values. Clinical values are preserved in A/B, including any Clinical records related to other modalities.", "",
              "## Main comparison", "", "| Task | N | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for task in tasks:
        if task in data:
            d = data[task]
            aucs = [fmt(d["metrics"][(n, "all_participants")]["auroc"]) for n in CONDITIONS]
            effects = [effect(d["summary"][(c, "all_participants", "auroc")]) for c in CONTRASTS]
            lines.append(f"| {TASK_LABELS.get(task, task)} | {d['meta']['n_participants']} | " + " | ".join(aucs + effects) + " |")
    lines += ["", "Positive A−B/C−D means removing the other eight modalities worsened AUROC. A−B measures their collective contribution with full Clinical; C−D measures it with basic Clinical. 0.0100 equals one AUROC percentage point.", "",
              "- These are input-removal controls on the existing multimodal model. B/D are not separately trained Clinical-only models.",
              "- A/C retain naturally available nonclinical inputs; no complete-case restriction is used for the primary cohort.",
              "- Pointwise 95% paired bootstrap intervals stratify participants by outcome; identical participant masks reuse identical intervals. They exclude training-seed uncertainty and multiplicity adjustment.",
              "- An interval crossing zero does not establish equivalence/noninferiority. These results alone do not demonstrate reconstruction or the performance achievable after retraining.",
              "- All conditions retain the original validation-selected threshold. Positive degradation means reference minus deleted for benefit metrics, and deleted minus reference for Brier/log loss.", ""]
    for task, d in data.items():
        m, plan = d["meta"], d["plan"]
        lines += [f"## {TASK_LABELS.get(task, task)}", "", f"Threshold: {m['threshold']:.8f}; bootstrap samples: {m['bootstrap_samples']}.", "",
                  "### All-participant metrics", "", "| Metric | A | B | C | D | A − B degradation [95% CI] | C − D degradation [95% CI] |",
                  "|---|---:|---:|---:|---:|---|---|"]
        for metric in METRICS:
            values = [fmt(d["metrics"][(n, "all_participants")][metric]) for n in CONDITIONS]
            effects = [effect(d["summary"][(c, "all_participants", metric)]) for c in CONTRASTS]
            lines.append(f"| {metric} | " + " | ".join(values + effects) + " |")
        lines += ["", "### Availability strata", "", "All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.", "",
                  "| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |",
                  "|---|---:|---|---:|---:|---:|---:|---|---|---|"]
        for scope in COHORTS:
            c = d["counts"][(scope,)]
            values = [fmt(d["metrics"][(n, scope)]["auroc"]) for n in CONDITIONS]
            effects = [effect(d["summary"][(comp, scope, "auroc")]) for comp in CONTRASTS]
            missing = "/".join(d["metrics"][(n, scope)]["n_without_any_input"] for n in CONDITIONS)
            lines.append(f"| {scope} | {c['n_evaluated']} | {c['n_positive']} / {c['n_negative']} | " + " | ".join(values + effects + [missing]) + " |")
        lines += ["", "### Exact basic Clinical scope", "", f"A/B retain all {plan['original_width']} trained Clinical columns. C/D retain {plan['n_retained']} and mask the other {plan['n_masked']}; both values and observed masks are cleared, with the original input width/scalers preserved.", "",
                  "Basic groups: " + ", ".join(f"{k}: {v}" for k, v in plan["group_counts"].items()) + ".", "",
                  "| Retained encoded feature | Source table | Source key |", "|---|---|---|"]
        for index in plan["retained_indices_zero_based"]:
            s = plan["source_identities"][index]
            lines.append(f"| {esc(s['feature_name'])} | {esc(s['table'])} | {esc(s['source_key'])} |")
        lines += ["", "### Provenance", "", "| Item | Value |", "|---|---|"]
        items = {key: m[key] for key in ("analysis_role", "checkpoint_path", "checkpoint_sha256", "config_hash", "config_sha256", "clinical_schema_sha256", "clinical_catalog_sha256", "plan_sha256", "threshold_source", "training_seed", "bootstrap_seed")}
        items.update(metadata_sha256=d["metadata_sha256"], basic_plan_sha256=plan["basic_plan_sha256"],
                     source_cache=json.dumps(m["source_token_cache"], sort_keys=True),
                     output_hashes=json.dumps(m["output_sha256"], sort_keys=True), **m["inference_checks"])
        for key, value in items.items():
            lines.append(f"| {esc(key)} | {esc(value)} |")
    if data:
        lines += ["", "## Shared code snapshot", "", "| File | SHA256 |", "|---|---|"]
        for name, digest in next(iter(data.values()))["meta"]["code_sha256"].items():
            lines.append(f"| {esc(name)} | {esc(digest)} |")
    atomic_write(root / "SUMMARY.md", "\n".join(lines) + "\n")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    args = parser.parse_args()
    status_path = args.root / "RUN_STATUS.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    complete = write_report(args.root, args.tasks or state.get("tasks") or TASKS, split=args.split or state.get("split", "val"))
    print(f"REPORT: {args.root / 'SUMMARY.md'}; {'complete' if complete else 'incomplete'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
