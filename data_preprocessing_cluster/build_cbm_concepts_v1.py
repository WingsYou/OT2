#!/usr/bin/env python3
"""Build observed clinical concept supervision; never add it to model inputs.

This independent, standard-library-only builder reads the raw OMOP tables.
It does not use outcomes, impute unknown answers, or modify an existing model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tempfile

DEFAULT_SPEC = Path(__file__).resolve().parent / "configs" / "cbm_concepts_v1.json"
EXPECTED = (
    ("bmi", "continuous", "measurement", (("bmi_vsorres", 4245997),)),
    ("systolic_bp", "continuous", "measurement", (("bp1_sysbp_vsorres", 3004249), ("bp2_sysbp_vsorres", 3004249))),
    ("diastolic_bp", "continuous", "measurement", (("bp1_diabp_vsorres", 3012888), ("bp2_diabp_vsorres", 3012888))),
    ("hba1c", "continuous", "measurement", (("import_hba1c", 3004410),)),
    ("hdl_cholesterol", "continuous", "measurement", (("import_hdl_cholesterol", 3007070),)),
    ("ldl_cholesterol", "continuous", "measurement", (("import_ldl_cholesterol", 3028288),)),
    ("photopic_logmar_right", "continuous", "measurement", (("viaodpscore", 2005200052),)),
    ("photopic_logmar_left", "continuous", "measurement", (("viaospscore", 2005200053),)),
    ("photopic_log_contrast_right", "continuous", "measurement", (("plcsodlog", 2005200155),)),
    ("photopic_log_contrast_left", "continuous", "measurement", (("plcsoslog", 2005200156),)),
    ("self_reported_activity", "categorical", "observation", (("dmlact", 44808782),)),
    ("restless_sleep", "categorical", "observation", (("ces7", 1761695),)),
)
CONCEPT_NAMES = tuple(row[0] for row in EXPECTED)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def person_id(value):
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") and value[:-2].isdigit() else value


def source_code(value):
    return str(value or "").split(",", 1)[0].strip().lower()


def integer(value):
    try:
        number = float(str(value or "").strip())
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def unit_key(value):
    normalized = "".join(str(value or "").strip().lower().split())
    return "" if normalized in {"nan", "none", "null", "nat"} else normalized


def load_spec(path=None):
    path = Path(path or DEFAULT_SPEC)
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != "ot2-cbm-concept-spec-v1":
        raise ValueError("Unsupported concept specification")
    concepts = spec.get("concepts", [])
    if [c.get("name") for c in concepts] != list(CONCEPT_NAMES):
        raise ValueError("Concept names/order must match the audited 12-concept allowlist")
    for c, (name, kind, table, sources) in zip(concepts, EXPECTED):
        observed = [(s.get("table"), s.get("source_key"), s.get("concept_id")) for s in c.get("sources", [])]
        if c.get("type") != kind or observed != [(table, key, cid) for key, cid in sources]:
            raise ValueError(f"Forbidden or mismatched source/OMOP identity for {name}")
        if kind == "categorical":
            categories = [0, 1, 2] if name == "self_reported_activity" else [0, 1, 2, 3]
            if c.get("categories") != categories or len(c.get("category_labels", [])) != len(categories):
                raise ValueError(f"Invalid categories for {name}")
            expected_mapping = ({"2005200314": (2, [10]), "2005200315": (1, [5]), "2005200316": (0, [0])}
                                if name == "self_reported_activity" else
                                {str(2005200141 + k): (k, [k]) for k in range(4)})
            mapping = c.get("response_mapping", {})
            actual = {key: (v.get("category"), v.get("raw_values")) for key, v in mapping.items()}
            if actual != expected_mapping:
                raise ValueError(f"Invalid response mapping for {name}")
    return spec, path


def record_date(row, table):
    """Require documented date; datetime may supply it, never participant order."""
    raw_date = str(row.get(f"{table}_date", "") or "").strip()
    raw_datetime = str(row.get(f"{table}_datetime", "") or "").strip()
    parsed = []
    for raw in (raw_date, raw_datetime):
        if raw.lower() in {"", "nan", "nat", "none", "null"}:
            continue
        try:
            if len(raw) == 10:
                parsed.append(date.fromisoformat(raw))
            else:
                parsed.append(datetime.fromisoformat(raw.replace("Z", "+00:00")).date())
        except ValueError:
            raise ValueError("invalid_date") from None
    if not parsed:
        raise ValueError("missing_date")
    if len(set(parsed)) != 1:
        raise ValueError("conflicting_date_and_datetime")
    return parsed[0].isoformat()


def numeric_value(row, c):
    raw = str(row.get("value_as_number", "") or "").strip()
    if raw.lower() in {"", "nan", "none", "null"}:
        raise ValueError("missing_numeric")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError("invalid_numeric") from None
    if not math.isfinite(value):
        raise ValueError("nonfinite_numeric")
    raw_response = str(row.get("value_as_concept_id", "") or "").strip()
    response_id = integer(raw_response)
    if response_id is None and raw_response.lower() not in {"", "nan", "none", "null"}:
        raise ValueError("invalid_numeric_response_id")
    if response_id not in {None, 0}:
        raise ValueError("categorical_response_in_numeric_field")
    raw_operator = str(row.get("operator_concept_id", "") or "").strip()
    operator_id = integer(raw_operator)
    if operator_id is None and raw_operator.lower() not in {"", "nan", "none", "null"}:
        raise ValueError("invalid_numeric_operator")
    if operator_id not in {None, 0, 4172703}:
        # 4172703 is Equals; inequalities are censored values, not exact labels.
        raise ValueError("nonexact_numeric_operator")
    if any(sign in str(row.get("value_source_value", "")) for sign in ("<", ">", "≤", "≥")):
        raise ValueError("censored_numeric_value")
    minimum = c.get("minimum")
    if minimum is not None and (value < minimum or (value == minimum and not c.get("minimum_inclusive", True))):
        raise ValueError("invalid_numeric_domain")
    uid = integer(row.get("unit_concept_id"))
    raw_unit = unit_key(row.get("unit_source_value"))
    allowed_ids = c.get("unit_concept_ids", [])
    allowed_units = [unit_key(x) for x in c.get("unit_aliases", [])]
    if uid is None and str(row.get("unit_concept_id", "") or "").strip() not in {"", "nan"}:
        raise ValueError("invalid_unit_concept")
    if uid not in {None, 0} and uid not in allowed_ids:
        raise ValueError("unexpected_unit_concept")
    if raw_unit and raw_unit not in allowed_units:
        raise ValueError("unexpected_unit_source")
    documented_blank = not raw_unit and uid in {None, 0}
    if documented_blank and not c.get("allow_documented_blank_unit", False):
        raise ValueError("unverifiable_blank_unit")
    return value, documented_blank


def categorical_value(row, c):
    response = integer(row.get("value_as_concept_id"))
    mapping = c["response_mapping"].get(str(response))
    if mapping is None:
        raise ValueError("unknown_or_refused_response")
    for field in ("value_as_number", "value_as_string"):
        raw = str(row.get(field, "") or "").strip()
        if raw.lower() in {"", "nan", "none", "null"}:
            continue
        # Raw OMOP numeric response encodings must agree with the response ID.
        try:
            numeric = float(raw)
        except ValueError:
            if field == "value_as_number":
                raise ValueError("invalid_response_numeric") from None
            continue  # Descriptive value_as_string is not a numeric code.
        if not math.isfinite(numeric) or numeric not in mapping["raw_values"]:
            raise ValueError("conflicting_response_encoding")
    return mapping["category"]


def _audit_markdown(schema, values, splits):
    lines = ["# CBM concept label audit", "", "These are observed supervision labels, not prediction results or model inputs.", "",
             f"Cohort: **{schema['n_participants']}** participants. Missing/invalid labels remain empty; no imputation.",
             "", "Temporal rule: one valid calendar date per participant-concept. Same-date repeated numeric values are averaged within each source, then BP1/BP2 source means are averaged equally. Multiple dates or conflicting categorical answers are excluded. Eyes remain separate.",
             "", "These labels are recorded clinical measurements and questionnaire responses; their dates can differ across sources. A single date within each concept does not guarantee eligibility at a prediction time or establish that a label precedes the outcome. Prospective claims require an explicit prediction-time eligibility audit.",
             "", "| Concept | Type / unit | Observed / cohort | Coverage | Observed range or category counts |", "|---|---|---:|---:|---|"]
    for c in schema["concepts"]:
        observed = [row[c["name"]] for row in values if row[c["name"]] is not None]
        if c["type"] == "continuous":
            detail = f"min {min(observed):.6g}; median {statistics.median(observed):.6g}; max {max(observed):.6g}" if observed else "unavailable"
        else:
            counts = Counter(observed)
            detail = "; ".join(f"{label}: {counts[k]}" for k, label in zip(c["categories"], c["category_labels"]))
        lines.append(f"| {c['name']} | {c['type']} / {c['unit']} | {len(observed)} / {len(values)} | {len(observed)/len(values):.1%} | {detail} |")
    lines += ["", "## Coverage by provided split", "", "| Split | Participants | Concept | Observed |", "|---|---:|---|---:|"]
    for split in sorted(set(splits.values())):
        subset = [row for row in values if splits[row["person_id"]] == split]
        for c in schema["concepts"]:
            lines.append(f"| {split} | {len(subset)} | {c['name']} | {sum(row[c['name']] is not None for row in subset)} |")
    lines += ["", "## Quality exclusions", "", "Counts below describe candidate raw records; a participant may have multiple records. Participant-concept conflicts invalidate that entire concept label.", "", "| Concept | Reason | Count |", "|---|---|---:|"]
    for name, reasons in schema["qc"]["record_counts_by_concept"].items():
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {name} | {reason} | {count} |")
    for name, reasons in schema["qc"]["participant_exclusions_by_concept"].items():
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {name} | participant: {reason} | {count} |")
    if schema.get("deferred_candidates"):
        lines += ["", "## Deferred candidates", "",
                  "The following audited candidates are documented but are not columns in concept_labels.csv and do not enter the strict bottleneck. Earlier feasibility counts stored in the specification describe a separate recommended train/val cohort, not this run's task cohort.",
                  "", "| Candidate | Reason |", "|---|---|"]
        for candidate in schema["deferred_candidates"]:
            lines.append(f"| {candidate['name']} | {candidate['deferred_reason']} |")
    lines += ["", "## Definitions and provenance", "", "- Exact source keys and OMOP IDs are enforced; MoCA/CVD outcome definitions and administrative/acquisition fields are outside the allowlist.",
              "- Unknown/refused questionnaire answers are missing, never normal. The previous zero-filled one-hot tables are not used.",
              "- BMI and BP may use source-specific documented units when both raw unit fields are blank. Such assumptions are counted explicitly. No unit conversion, BMI recomputation or clinical threshold diagnosis is performed.",
              "- Photopic LogMAR uses viaodpscore/viaospscore. The similarly named plog fields are letter counts and are not substituted.",
              "- Restless sleep is the CES-D item ces7, not total CES-D-10 or an overall sleep-quality score.",
              f"- Labels SHA256: `{schema['label_sha256']}`", f"- Specification SHA256: `{schema['specification_sha256']}`", ""]
    return "\n".join(lines)


def build_labels(measurement_csv, observation_csv, person_ids, out_dir, *, spec_path=None, split_by_person=None):
    """Write raw-scale labels and immutable metadata in the exact supplied ID order.

    No task outcomes are accepted. ``split_by_person`` is metadata for coverage,
    not a rule for selecting values. The CLI selects train/val IDs by default.
    """
    specification_path = Path(spec_path or DEFAULT_SPEC)
    specification_hash = sha256(specification_path)
    spec, specification_path = load_spec(specification_path)
    ids = [person_id(value) for value in person_ids]
    if not ids or any(not value or value.lower() in {"nan", "none", "null"} for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Cohort requires unique, non-empty person IDs")
    selected = set(ids)
    splits = {pid: "unspecified" for pid in ids}
    if split_by_person is not None:
        normalized = {person_id(k): str(v).strip().lower() for k, v in split_by_person.items()}
        if set(normalized) != selected or any(v not in {"train", "val", "test", "unspecified"} for v in normalized.values()):
            raise ValueError("Split metadata must match cohort IDs with valid split values")
        splits = normalized
    paths = {"measurement": Path(measurement_csv), "observation": Path(observation_csv)}
    hashes_before = {table: sha256(path) for table, path in paths.items()}
    by_source = {table: {} for table in paths}
    counts = {c["name"]: Counter() for c in spec["concepts"]}
    excluded = {c["name"]: Counter() for c in spec["concepts"]}
    records = defaultdict(list)
    dates = defaultdict(set)
    poisoned = defaultdict(set)
    for c in spec["concepts"]:
        for source in c["sources"]:
            by_source[source["table"]][source["source_key"]] = (c, source["concept_id"])
    for table, path in paths.items():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"person_id", f"{table}_source_value", f"{table}_concept_id", "value_as_number"}
            if table == "measurement":
                required |= {"unit_concept_id", "unit_source_value"}
            else:
                required.add("value_as_concept_id")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Raw {table} table lacks required columns: {sorted(required-set(reader.fieldnames or []))}")
            if not ({f"{table}_date", f"{table}_datetime"} & set(reader.fieldnames or [])):
                raise ValueError(f"Raw {table} table lacks measurement dates")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Duplicate {table} header names")
            for row in reader:
                pid = person_id(row["person_id"])
                key = source_code(row[f"{table}_source_value"])
                if pid not in selected or key not in by_source[table]:
                    continue
                c, expected_id = by_source[table][key]
                name = c["name"]
                pair = (pid, name)
                counts[name]["candidate_records"] += 1
                if integer(row[f"{table}_concept_id"]) != expected_id:
                    counts[name]["source_concept_id_mismatch"] += 1
                    poisoned[pair].add("source_concept_id_mismatch")
                    continue
                try:
                    when = record_date(row, table)
                except ValueError as exc:
                    reason = str(exc)
                    counts[name][reason] += 1
                    poisoned[pair].add(reason)
                    continue
                dates[pair].add(when)
                try:
                    if c["type"] == "continuous":
                        value, assumed = numeric_value(row, c)
                        if assumed:
                            counts[name]["documented_blank_unit_records"] += 1
                    else:
                        value = categorical_value(row, c)
                except ValueError as exc:
                    reason = str(exc)
                    counts[name][reason] += 1
                    # Ordinary missing/refused values are excluded individually;
                    # inconsistent encodings/units invalidate the whole target.
                    if reason not in {"missing_numeric", "unknown_or_refused_response"}:
                        poisoned[pair].add(reason)
                    continue
                records[pair].append((key, value))
                counts[name]["valid_candidate_records"] += 1
    if hashes_before != {table: sha256(path) for table, path in paths.items()}:
        raise RuntimeError("Raw source tables changed during label construction")
    if specification_hash != sha256(specification_path):
        raise RuntimeError("Concept specification changed during label construction")
    values = []
    for pid in ids:
        row = {"person_id": pid}
        for c in spec["concepts"]:
            name = c["name"]
            pair = (pid, name)
            reasons = set(poisoned[pair])
            if len(dates[pair]) > 1:
                reasons.add("multiple_calendar_dates")
            available = records[pair]
            if c["type"] == "categorical" and len({value for _, value in available}) > 1:
                reasons.add("conflicting_categorical_answers")
            result = None
            if reasons:
                excluded[name].update(reasons)
            elif available:
                if c["type"] == "categorical":
                    result = available[0][1]
                else:
                    per_source = defaultdict(list)
                    for key, value in available:
                        per_source[key].append(value)
                    result = statistics.mean(statistics.mean(part) for part in per_source.values())
            else:
                excluded[name]["no_valid_observed_label"] += 1
            row[name] = result
        values.append(row)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=["person_id", *CONCEPT_NAMES], lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    label_bytes = csv_buffer.getvalue().encode()
    schema = {
        "schema_version": "ot2-cbm-concepts-v1", "concept_names": list(CONCEPT_NAMES),
        "concepts": spec["concepts"], "deferred_candidates": spec.get("deferred_candidates", []), "labels_file": "concept_labels.csv",
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "source_sha256": hashes_before, "source_paths": {table: str(path.resolve()) for table, path in paths.items()},
        "specification_sha256": specification_hash, "builder_sha256": sha256(Path(__file__)),
        "definition_source": spec.get("definition_source"),
        "cohort_person_ids_sha256": canonical_hash({"person_ids": ids}),
        "n_participants": len(ids), "split_counts": dict(sorted(Counter(splits.values()).items())),
        "temporal_policy": spec["temporal_policy"], "supervision_only": True,
        "missing_value": "empty CSV cell; missing labels are never imputed",
        "qc": {"record_counts_by_concept": {k: dict(v) for k, v in counts.items()},
               "participant_exclusions_by_concept": {k: dict(v) for k, v in excluded.items()},
               "observed_persons_by_concept": {name: sum(row[name] is not None for row in values) for name in CONCEPT_NAMES}},
    }
    artifacts = {"concept_labels.csv": label_bytes,
                 "concept_schema.json": (json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
                 "CONCEPT_AUDIT.md": _audit_markdown(schema, values, splits).encode()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in artifacts):
        raise FileExistsError("Concept artifacts already exist; choose a fresh output directory")
    for name, content in artifacts.items():
        with tempfile.NamedTemporaryFile(dir=out_dir, prefix=f".{name}.", delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(out_dir / name)
    return schema


def load_cohort(path, allowed_splits=("train", "val")):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "person_id" not in (reader.fieldnames or []):
            raise ValueError("Cohort CSV must contain person_id")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate cohort header names")
        rows = list(reader)
        has_split = "split" in reader.fieldnames
    seen, selected, splits = set(), [], {}
    for row in rows:
        pid = person_id(row["person_id"])
        if not pid or pid in seen:
            raise ValueError("Cohort CSV has missing or duplicate person IDs")
        seen.add(pid)
        split = str(row.get("split", "unspecified")).strip().lower()
        if split not in {"train", "val", "test", "unspecified"}:
            raise ValueError("Cohort CSV has an invalid split")
        if has_split and split not in allowed_splits:
            continue
        selected.append(pid)
        splits[pid] = split
    if not selected:
        raise ValueError("No participants in the requested cohort splits")
    return selected, splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-csv", type=Path, required=True)
    parser.add_argument("--observation-csv", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    args = parser.parse_args(argv)
    ids, splits = load_cohort(args.cohort_csv, args.splits)
    schema = build_labels(args.measurement_csv, args.observation_csv, ids, args.out_dir,
                          spec_path=args.spec, split_by_person=splits)
    print(json.dumps({"out_dir": str(args.out_dir), "n_participants": schema["n_participants"],
                      "label_sha256": schema["label_sha256"], "observed_persons": schema["qc"]["observed_persons_by_concept"]}, indent=2))


if __name__ == "__main__":
    main()
