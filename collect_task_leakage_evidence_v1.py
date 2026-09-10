#!/usr/bin/env python3
"""Collect aggregate preprocessing evidence without patient rows or model tensors.

Run on the cluster with its standard-library Python; no torch/GPU is required.
Original files are never changed. JSON excerpts are intentionally redacted and
must not be passed back to the training pipeline as replacement artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
MAX_FILE_BYTES = 32 * 1024 * 1024
FORMAT = "ot2-task-leakage-evidence-v1"

SCHEMA_FIELDS = set("""
preprocessing_version config_hash preprocessing_config task task_config
task_config_hash temporal_policy_hash temporal_estimand fit_split
fit_person_ids_hash min_train_persons n_all_persons n_train_persons
n_raw_clinical_persons raw_clinical_person_ids_hash n_features feature_names
dropped_train_constant_features raw_scale missing_value temporal_audit
""".split())
PREPROCESSING_FIELDS = set("""
preprocessing_version builder task task_config_hash fit_split fit_person_ids_hash
min_train_persons include_condition include_procedure temporal_policy_hash
temporal_estimand feature_names numeric_feature_key event_absence_fill
numeric_missing_fill clinical_data_quality_rules cohort_role clinical_schema_hash
source_index_config_hashes source_index_adapter_hashes path_semantics
allow_missing_artifacts environment_coverage_qc input_bindings
""".split())
TEMPORAL_AUDIT_FIELDS = set("""
preprocessing_version task task_config_hash temporal_policy_hash temporal_estimand
n_rows n_people included_rows excluded_rows status_counts by_table
retained_label_offset_days
""".split())
MASTER_FIELDS = set("""
preprocessing_version config_hash preprocessing_config task n_rows
label_positive_count split_counts coverage_by_split environment_coverage_tier_by_split
participant_flag_vs_artifact_mismatches fit_split_authority clinical_schema_hash
input_bindings path_semantics
""".split())
SCALER_FIELDS = set("""
scaler_format_version preprocessing_version scaler_config_hash fit_split
fit_population source_config_hashes scaler_config_hash_semantics
""".split())
FIT_FIELDS = set("""
n_participants_tsv n_official_train_participants fit_person_ids_hash hash_semantics
""".split())
SCALER_GROUP_FIELDS = set("""
method feature_order n_train_rows n_train_participants_with_rows
fit_participant_ids_hash n_train_artifacts_loaded
n_official_train_ids_without_loaded_artifact config_binding
""".split())
RUN_FIELDS = set("""
engine_format_version task best_epoch best_score selection_metric
selection_direction epochs_completed evaluation_phase test_evaluated
""".split())
RUN_CONFIG_FIELDS = set("""
config_hash config_path engine_format_version schema_version resolved_model_config
runtime modality_subset trainable_parameters
""".split())
RUN_DATA_FIELDS = set("""
task artifact_root lineage files clinical_features_csv split_counts
split_person_ids_sha256 model_feature_dimensions test_role
""".split())
RUN_SCALER_FIELDS = SCALER_FIELDS | {"path", "sha256"}
CATALOG_COLUMNS = set("""
feature_name table source_key concept_id unit unit_concept_id response_concept_id
value_kind aggregation train_nonmissing_persons train_prevalence
""".split())
AUDIT_COLUMNS = {"table", "rule", "column", "rows", "people"}
TEMPORAL_COLUMNS = {"table", "temporal_status", "rows"}
PRIVATE_KEY = re.compile(r"(?:person|patient|participant|subject)_?ids?$", re.I)
PRIVATE_CONTAINERS = {
    "rows", "records", "participants", "patients", "subjects", "predictions",
    "paired_predictions", "labels", "y", "values", "by_person", "per_person",
    "by_participant", "per_participant", "by_patient", "per_patient",
    "raw_clinical_person_ids", "train_person_ids", "validation_person_ids",
    "test_person_ids", "input_files", "source_files", "data_contract_audit",
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_fields(value: Any, fields: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key in fields} if isinstance(value, dict) else {}


def redact(value: Any) -> Any:
    """Defense in depth for selected metadata blocks; retain hashes and counts."""
    if isinstance(value, dict):
        return {
            key: redact(item) for key, item in value.items()
            if not PRIVATE_KEY.search(key)
            and not (key.lower() in PRIVATE_CONTAINERS and isinstance(item, (dict, list)))
            and key.lower() not in {"y", "label_value", "person_id", "patient_id"}
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def summarize_json(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "schema":
        result = select_fields(payload, SCHEMA_FIELDS)
    elif kind == "master":
        result = select_fields(payload, MASTER_FIELDS)
    elif kind == "temporal":
        result = select_fields(payload, TEMPORAL_AUDIT_FIELDS)
    elif kind == "scaler":
        result = select_fields(payload, SCALER_FIELDS)
        result["fit_population"] = select_fields(payload.get("fit_population"), FIT_FIELDS)
        groups = payload.get("scalers", {})
        result["scalers"] = {
            name: select_fields(group, SCALER_GROUP_FIELDS)
            for name, group in groups.items()
            if name in {"clinical", "ecg_numeric", "ecg_waveform", "environment", "cgm", "activity"}
        } if isinstance(groups, dict) else {}
    elif kind == "run":
        result = select_fields(payload, RUN_FIELDS)
        result["config_provenance"] = select_fields(payload.get("config_provenance"), RUN_CONFIG_FIELDS)
        result["data_provenance"] = select_fields(payload.get("data_provenance"), RUN_DATA_FIELDS)
        scaler = select_fields(payload.get("scaler_provenance"), RUN_SCALER_FIELDS)
        scaler["fit_population"] = select_fields(scaler.get("fit_population"), FIT_FIELDS)
        result["scaler_provenance"] = scaler
    elif kind == "task_config":
        # This is a code configuration, not a participant-data artifact.
        result = payload
    elif kind == "label_metadata":
        result = select_fields(payload, {"preprocessing_version", "task_config_hash", "temporal_policy_hash", "temporal_estimand", "task_specs", "summaries"})
    else:
        raise ValueError(f"Unsupported JSON evidence kind: {kind}")
    if "preprocessing_config" in result:
        result["preprocessing_config"] = select_fields(result["preprocessing_config"], PREPROCESSING_FIELDS)
    if "temporal_audit" in result:
        result["temporal_audit"] = select_fields(result["temporal_audit"], TEMPORAL_AUDIT_FIELDS)
    return redact(result)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence source escapes its configured root")
    return path


def _csv_excerpt(raw: bytes, kind: str) -> bytes:
    allowed = {"catalog": CATALOG_COLUMNS, "audit": AUDIT_COLUMNS, "temporal_csv": TEMPORAL_COLUMNS}[kind]
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    required = {"feature_name"} if kind == "catalog" else ({"rule", "rows"} if kind == "audit" else {"temporal_status", "rows"})
    if not required.issubset(fields) or len(fields) != len(set(fields)) or set(fields) - allowed:
        raise ValueError("CSV is not the expected aggregate-only column schema")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has a different width than its aggregate header")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def evidence_specs(out_root: Path, code_root: Path) -> list[tuple[Path, str, str, str]]:
    specs: list[tuple[Path, str, str, str]] = []
    for task in TASKS:
        clinical = f"clinical/{task}"
        for stem, kind, suffix in (
            ("clinical_schema", "schema", "json"),
            ("clinical_feature_catalog", "catalog", "csv"),
            ("clinical_exclusion_audit", "audit", "csv"),
            ("clinical_temporal_audit", "temporal", "json"),
            ("clinical_temporal_audit_overall", "temporal_csv", "csv"),
            ("clinical_temporal_audit_by_table", "temporal_csv", "csv"),
        ):
            relative = f"{clinical}/{stem}_{task}.{suffix}"
            specs.append((out_root, relative, f"artifacts/{relative}", kind))
        for relative, kind in (
            (f"scalers/train_scalers_{task}.json", "scaler"),
            (f"model_full_assets_v1/inputs/master_{task}.metadata.json", "master"),
            (f"model_full_assets_v1/models/{task}/seed42/run_summary.json", "run"),
        ):
            specs.append((out_root, relative, f"artifacts/{relative}", kind))
        for stem, kind, suffix in (
            ("label_temporal_audit", "temporal", "json"),
            ("label_temporal_audit_overall", "temporal_csv", "csv"),
            ("label_temporal_audit_by_table", "temporal_csv", "csv"),
        ):
            relative = f"labels/{stem}_{task}.{suffix}"
            specs.append((out_root, relative, f"artifacts/{relative}", kind))
    specs.append((out_root, "labels/label_build_metadata.json", "artifacts/labels/label_build_metadata.json", "label_metadata"))
    specs.append((code_root, "configs/downstream_tasks.json", "code/configs/downstream_tasks.json", "task_config"))
    for short in ("moca", "cvd_broad", "cvd_strict"):
        relative = f"configs/task2_v2_{short}_full_assets_v1.yaml"
        specs.append((code_root, relative, f"code/{relative}", "yaml"))
    return specs


def _clinical_header_evidence(out_root: Path, task: str, run: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    record: dict[str, Any] = {"root": "output", "kind": "clinical_header", "task": task}
    try:
        provenance = run.get("data_provenance", {})
        declared = provenance.get("clinical_features_csv", {}) if isinstance(provenance, dict) else {}
        declared_path = declared.get("path") if isinstance(declared, dict) else None
        if isinstance(declared_path, str) and declared_path:
            candidate = Path(declared_path)
            if candidate.is_absolute():
                candidate = candidate.resolve()
                # Allow a copied output tree to be audited with --out-root.
                if candidate.is_relative_to(out_root):
                    relative = candidate.relative_to(out_root)
                else:
                    relative = candidate.relative_to(Path("/data/pzhao3/ot2_output"))
            else:
                relative = candidate
            record["binding_source"] = "run_summary.data_provenance.clinical_features_csv.path"
            if isinstance(declared.get("sha256"), str):
                record["run_declared_clinical_csv_sha256"] = declared["sha256"]
        else:
            relative = Path(f"clinical/{task}/clinical_features_{task}_wide.csv")
            if not _safe_file(out_root, str(relative)).is_file():
                relative = Path(f"clinical/{task}/clinical_features_{task}.csv")
            record["binding_source"] = "deterministic_task_filename_fallback"
        record["source_relative_path"] = str(relative)
        path = _safe_file(out_root, str(relative))
        if not path.is_file():
            record["status"] = "missing"
            return record, None
        before = path.stat()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if not header or len(header) != len(set(header)) or "person_id" not in header:
            raise ValueError("Clinical CSV header must contain unique columns including person_id")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Clinical CSV changed during header/hash collection")
        record.update(original_size_bytes=after.st_size, original_sha256=digest.hexdigest())
        if "run_declared_clinical_csv_sha256" in record:
            record["matches_run_declared_sha256"] = record["original_sha256"] == record["run_declared_clinical_csv_sha256"]
        content = _json_bytes({
            "task": task, "columns": header,
            "feature_names": [column for column in header if column != "person_id"],
            "n_features": len(header) - 1,
            "collection": "Header only; full file was read as opaque bytes for SHA-256, never included or parsed as participant records.",
        })
        record.update(status="collected", archive_path=f"artifacts/clinical/{task}/clinical_wide_header.json", excerpt_sha256=hashlib.sha256(content).hexdigest())
        return record, content
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        record.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
        return record, None


def collect_evidence(out_root: Path, code_root: Path, out: Path) -> dict[str, Any]:
    out_root, code_root, out = out_root.resolve(), code_root.resolve(), out.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new --out: {out}")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {out.parent}")
    members: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    run_payloads: dict[str, dict[str, Any]] = {}
    for root, relative, member_name, kind in evidence_specs(out_root, code_root):
        record: dict[str, Any] = {"source_relative_path": relative, "root": "code" if root == code_root else "output", "kind": kind}
        records.append(record)
        try:
            path = _safe_file(root, relative)
            if not path.is_file():
                record["status"] = "missing"
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Evidence file exceeds 32 MiB metadata limit")
            raw = path.read_bytes()
            if len(raw) > MAX_FILE_BYTES:
                raise ValueError("Evidence file exceeds 32 MiB metadata limit")
            record.update(original_size_bytes=len(raw), original_sha256=hashlib.sha256(raw).hexdigest())
            if kind in {"catalog", "audit", "temporal_csv"}:
                content = _csv_excerpt(raw, kind)
            elif kind == "yaml":
                raw.decode("utf-8")
                content = raw
            else:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if kind == "run":
                    task = Path(relative).parts[-3]
                    run_payloads[task] = payload
                record["original_canonical_json_sha256"] = canonical_json_sha256(payload)
                record["original_declared_hashes"] = {
                    key: value for key, value in payload.items()
                    if (key.endswith("_hash") or key.endswith("_sha256")) and isinstance(value, str)
                }
                if isinstance(payload.get("preprocessing_config"), dict):
                    record["original_preprocessing_config_sha256"] = canonical_json_sha256(payload["preprocessing_config"])
                if "scaler_config_hash" in payload:
                    record["original_scaler_payload_without_hash_sha256"] = canonical_json_sha256({k: v for k, v in payload.items() if k != "scaler_config_hash"})
                if kind == "task_config":
                    record["original_without_temporal_estimand_sha256"] = canonical_json_sha256({k: v for k, v in payload.items() if k != "temporal_estimand"})
                content = _json_bytes(summarize_json(payload, kind))
            record.update(status="collected", archive_path=member_name, excerpt_sha256=hashlib.sha256(content).hexdigest())
            members[member_name] = content
        except (OSError, UnicodeError, ValueError, TypeError, csv.Error) as exc:
            record.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
    for task in TASKS:
        record, content = _clinical_header_evidence(out_root, task, run_payloads.get(task, {}))
        records.append(record)
        if content is not None:
            members[record["archive_path"]] = content
    manifest = {
        "schema_version": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": list(TASKS),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": records,
        "counts": {status: sum(row["status"] == status for row in records) for status in ("collected", "missing", "rejected")},
        "interpretation": "Original hashes describe cluster source bytes before redaction; excerpts are for audit only and cannot replace training artifacts. Missing temporal audit files alone do not prove leakage.",
    }
    members["manifest.json"] = _json_bytes(manifest)
    members["README.txt"] = (
        "Aggregate preprocessing evidence; no clinical wide tables, participant rows, labels, "
        "per-person audits, predictions, tensors, caches or checkpoints are collected.\n"
        "JSON files are allowlisted excerpts. Original byte/canonical hashes are in manifest.json; "
        "never replace production files with these excerpts. CSVs must match aggregate-only headers.\n"
        "Current code config is separate from recorded training provenance; compare the hashes.\n"
    ).encode("utf-8")
    # Exclusive creation prevents both accidental replacement and a concurrent overwrite.
    with out.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for name, content in sorted(members.items()):
                info = tarfile.TarInfo(name=name)
                info.size, info.mode, info.mtime = len(content), 0o600, 0
                archive.addfile(info, io.BytesIO(content))
    return {"out": str(out), "archive_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), **manifest["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("/data/pzhao3/ot2_output"))
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=Path("task_leakage_evidence_v1.tar.gz"))
    args = parser.parse_args()
    try:
        result = collect_evidence(args.out_root, args.code_root, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Collection failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
