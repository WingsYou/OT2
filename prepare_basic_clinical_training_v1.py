#!/usr/bin/env python3
"""Project existing train-fitted Clinical artifacts to the fixed 23-column view.

Labels, participant splits and all nonclinical artifacts are preserved. No
statistics, vocabulary or feature choice is fitted on validation/test data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import tempfile

from basic_retrain_contract_v1 import (TASKS, MODALITIES, canonical_hash, sha256, read_json,
                                       write_json, verify_prepared, atomic_text)
from clinical_basic_information_v1 import build_basic_clinical_plan
from task2_v2_config import load_model_v2_config


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Empty/duplicate CSV header: {path}")
        rows = list(reader)
    if any(None in r or None in r.values() for r in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return columns, rows


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_config(path):
    path = Path(path).resolve()
    if path.suffix == ".json":
        return read_json(path)
    import yaml
    return yaml.safe_load(path.read_text())


def resolve(path, base):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def present(value):
    text = str(value).strip()
    if text.lower() in ("", "nan", "na", "n/a", "null", "none"):
        return False
    if not math.isfinite(float(text)):
        raise ValueError("Clinical features must be finite or missing")
    return True


def strict_bool(value):
    s = str(value).strip().lower()
    if s not in ("true", "false", "1", "0"):
        raise ValueError(f"Invalid Clinical availability flag: {s}")
    return s in ("true", "1")


def prepare(config_path, experiment_root, *, check_only=False):
    config_path, experiment_root = Path(config_path).resolve(), Path(experiment_root).resolve()
    load_model_v2_config(config_path, validate_files=True)
    raw = read_config(config_path)
    cfg = copy.deepcopy(raw)
    dataset = cfg["dataset"]
    task = dataset["task"]
    if task not in TASKS or cfg["seed"] != 42 or tuple(cfg["model"]["source_order"]) != MODALITIES:
        raise ValueError("Expected a current three-task seed-42, nine-modality config")
    if cfg["training"].get("disabled_groups", []) or cfg["training"].get("initialization_checkpoint"):
        raise ValueError("Source config must enable all modalities with no previous task checkpoint")
    if cfg["training"].get("evaluation_phase") != "validation_screen":
        raise ValueError("Expected validation-only model selection")
    for name in ("master_csv", "master_metadata_json", "scaler_json", "clinical_schema_json",
                 "environment_schema_json", "wearable_schema_json", "ecg_schema_json", "artifact_root"):
        dataset[name] = str(resolve(dataset[name], config_path.parent))
    retinal = cfg["model"]["retinal"]
    if retinal.get("weights_path"):
        retinal["weights_path"] = str(resolve(retinal["weights_path"], config_path.parent))
    source_paths = {k: Path(dataset[k]) for k in ("master_csv", "master_metadata_json", "scaler_json", "clinical_schema_json",
                                                "environment_schema_json", "wearable_schema_json", "ecg_schema_json")}
    schema, scaler, metadata = (read_json(source_paths[k]) for k in ("clinical_schema_json", "scaler_json", "master_metadata_json"))
    catalog_path = source_paths["clinical_schema_json"].parent / f"clinical_feature_catalog_{task}.csv"
    catalog_columns, catalog = read_csv(catalog_path)
    plan = build_basic_clinical_plan(schema, catalog)
    if schema.get("task") != task or schema.get("fit_split") != "train" or scaler.get("fit_split") != "recommended_split=train":
        raise ValueError("Clinical schema/scaler must match the task and official train fit")
    if scaler["scaler_config_hash"] != canonical_hash({k: v for k, v in scaler.items() if k != "scaler_config_hash"}):
        raise ValueError("Source scaler hash mismatch (sanitized evidence is not a training scaler)")
    if schema["config_hash"] != canonical_hash(schema["preprocessing_config"]):
        raise ValueError("Source Clinical schema config hash mismatch")
    if metadata["config_hash"] != canonical_hash(metadata["preprocessing_config"]) or metadata["clinical_schema_hash"] != canonical_hash(schema):
        raise ValueError("Source master metadata hash/schema mismatch")
    original_scaler = scaler["scalers"]["clinical"]
    if original_scaler["feature_order"] != plan["feature_order"] or set(original_scaler.get("features", {})) != set(plan["feature_order"]):
        raise ValueError("Source scaler must contain full fitted statistics in schema order")
    if schema.get("fit_person_ids_hash") != scaler["fit_population"]["fit_person_ids_hash"]:
        raise ValueError("Clinical/scaler official training participant hash mismatch")
    master_columns, master_rows = read_csv(source_paths["master_csv"])
    if not master_rows or len({r["person_id"] for r in master_rows}) != len(master_rows):
        raise ValueError("Master must contain unique participants")
    if any(r["task"] != task or r["split"] not in ("train", "val", "test") for r in master_rows):
        raise ValueError("Source master task/split mismatch")
    declared = {r["clinical_features_path"].strip() for r in master_rows}
    if len(declared) != 1 or "" in declared:
        raise ValueError("Expected exactly one Clinical table")
    clinical_path = resolve(next(iter(declared)), Path(dataset["artifact_root"]))
    columns, rows = read_csv(clinical_path)
    if columns != ["person_id", *plan["feature_order"]] or len({r["person_id"] for r in rows}) != len(rows):
        raise ValueError("Clinical CSV header/participant identity mismatch")
    retained = plan["retained_feature_names"]
    projected = [{key: r[key] for key in ("person_id", *retained)} for r in rows]
    lookup = {r["person_id"]: r for r in projected}
    if not set(r["person_id"] for r in master_rows).issubset(lookup):
        raise ValueError("Clinical table is missing master participants")
    # No label values are consulted to choose columns or estimate scalers.
    source_paths.update(source_config=config_path, clinical_catalog=catalog_path, clinical_features=clinical_path)
    source_sha = {k: {"path": str(p), "sha256": sha256(p)} for k, p in source_paths.items()}
    final = experiment_root / "inputs" / task
    new_schema = copy.deepcopy(schema)
    derivation = {"builder": Path(__file__).name, "basic_plan_sha256": plan["plan_sha256"],
                  "source_schema_sha256": source_sha["clinical_schema_json"]["sha256"],
                  "source_scaler_sha256": source_sha["scaler_json"]["sha256"],
                  "source_clinical_csv_sha256": source_sha["clinical_features"]["sha256"],
                  "operation": "fixed_column_projection_without_refitting_or_outcome_selection"}
    new_schema.update(feature_names=retained, n_features=23)
    new_schema["preprocessing_config"] = {**new_schema["preprocessing_config"], "feature_names": retained,
                                           "builder": Path(__file__).name, "basic_projection": derivation}
    new_schema["config_hash"] = canonical_hash(new_schema["preprocessing_config"])
    new_scaler = copy.deepcopy(scaler)
    new_scaler["scalers"]["clinical"]["feature_order"] = retained
    new_scaler["scalers"]["clinical"]["features"] = {n: original_scaler["features"][n] for n in retained}
    new_scaler["source_config_hashes"]["clinical"].update(config_hash=new_schema["config_hash"], schema_hash=canonical_hash(new_schema))
    new_scaler["basic_projection"] = derivation
    new_scaler.pop("scaler_config_hash")
    new_scaler["scaler_config_hash"] = canonical_hash(new_scaler)
    new_master = copy.deepcopy(master_rows)
    changed_availability = 0
    for row in new_master:
        basic_present = strict_bool(row["has_clinical"]) and any(present(lookup[row["person_id"]][n]) for n in retained)
        changed_availability += int(strict_bool(row["has_clinical"]) != basic_present)
        row["has_clinical"] = "True" if basic_present else "False"
        row["clinical_features_path"] = str(final / "clinical_basic.csv")
        if "available_modality_count" in row:
            row["available_modality_count"] = str(sum(strict_bool(row[k]) for k in ("has_clinical", "has_retinal", "has_environment", "has_cgm", "has_activity", "has_ecg")))
    allowed_changes = {"clinical_features_path", "has_clinical", "available_modality_count"}
    preserved_columns = [c for c in master_columns if c not in allowed_changes]
    assert all(all(a[c] == b[c] for c in preserved_columns) for a, b in zip(master_rows, new_master))
    new_meta = copy.deepcopy(metadata)
    new_meta["clinical_schema_hash"] = canonical_hash(new_schema)
    new_meta["preprocessing_config"].update(builder=Path(__file__).name, clinical_schema_hash=canonical_hash(new_schema),
                                          basic_projection={**derivation, "parent_master_config_hash": metadata["config_hash"]})
    # Preserve parent fingerprints as parent provenance; bind the derived view explicitly.
    new_meta["config_hash"] = canonical_hash(new_meta["preprocessing_config"])
    for split in ("train", "val", "test"):
        if split in new_meta.get("coverage_by_split", {}):
            new_meta["coverage_by_split"][split]["has_clinical"] = sum(strict_bool(r["has_clinical"]) for r in new_master if r["split"] == split)
    new_meta.setdefault("notes", []).append("Derived 23-column basic Clinical view; parent source_files/input_bindings describe the source experiment. Labels/splits/nonclinical paths preserved exactly; no scaler refit.")
    cfg["output_dir"] = str(experiment_root / "models" / task / "seed42")
    for key, filename in (("master_csv", "master.csv"), ("master_metadata_json", "master.metadata.json"),
                          ("scaler_json", "scalers_basic.json"), ("clinical_schema_json", "clinical_schema_basic.json")):
        dataset[key] = str(final / filename)
    counts = {s: sum(r["split"] == s for r in master_rows) for s in ("train", "val", "test")}
    if metadata.get("n_rows") != len(master_rows) or metadata.get("split_counts") != counts:
        raise ValueError("Source master row/split counts disagree with metadata")
    receipt = {"schema_version": "ot2-basic-retrain-inputs-v1", "task": task, "clinical_width": 23,
               "original_clinical_width": plan["original_width"], "basic_plan_sha256": plan["plan_sha256"],
               "source_files": source_sha, "split_counts": counts, "n_clinical_availability_changed": changed_availability,
               "preserved_master_columns": preserved_columns,
               "preserved_master_content_sha256": canonical_hash([{c: r[c] for c in preserved_columns} for r in master_rows]),
               "retained_scaler_statistics_copied_exactly": True, "nonclinical_scalers_unchanged": True,
               "initialization_checkpoint": None, "test_evaluation": "not_requested", "output_sha256": {}}
    if check_only:
        return receipt
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        old = verify_prepared(final)
        if old["source_files"] != source_sha:
            raise ValueError("Prepared inputs came from different source files; use a new experiment root")
        return old
    with tempfile.TemporaryDirectory(prefix=f".{task}-", dir=final.parent) as temporary:
        temp = Path(temporary) / "ready"
        temp.mkdir()
        write_csv(temp / "clinical_basic.csv", ["person_id", *retained], projected)
        write_csv(temp / "clinical_feature_catalog_basic.csv", catalog_columns, [catalog[i] for i in plan["retained_indices_zero_based"]])
        write_csv(temp / "master.csv", master_columns, new_master)
        for name, payload in (("clinical_schema_basic.json", new_schema), ("scalers_basic.json", new_scaler),
                              ("master.metadata.json", new_meta), ("basic_plan.json", plan)):
            write_json(temp / name, payload)
        # Use the supervised loader's required suffix AND YAML serialization.
        # Merely renaming JSON turns e.g. 5e-05 into a string in PyYAML 1.1.
        import yaml
        atomic_text(temp / "train_config.yaml", yaml.safe_dump(cfg, sort_keys=True))
        receipt["output_sha256"] = {p.name: sha256(p) for p in sorted(temp.iterdir())}
        write_json(temp / "PREPARED.json", receipt)
        os.rename(temp, final)
    verify_prepared(final)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.config, args.root, check_only=args.check_only)
    print(json.dumps({k: result[k] for k in ("task", "clinical_width", "original_clinical_width", "split_counts", "n_clinical_availability_changed")}, indent=2))


if __name__ == "__main__":
    main()
