"""Small synthetic artifacts for contract tests; never research results."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_clinical_basic_information_v1 import catalog_fixture
from basic_retrain_contract_v1 import (TASKS, MODALITIES, GROUPS, SCOPES, METRICS, canonical_hash,
                                       sha256, read_json, write_json, code_hashes)
from prepare_basic_clinical_training_v1 import write_csv, prepare


def source_fixture(root, task=TASKS[0]):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    schema, catalog = catalog_fixture()
    names = schema["feature_names"]
    fit_hash = canonical_hash({"person_ids": ["p0", "p1"]})
    schema.update(task=task, fit_split="train", fit_person_ids_hash=fit_hash, task_config_hash="task-hash",
                  preprocessing_config={"task": task, "feature_names": names, "builder": "synthetic"})
    schema["config_hash"] = canonical_hash(schema["preprocessing_config"])
    write_json(root / "clinical_schema.json", schema)
    write_csv(root / f"clinical_feature_catalog_{task}.csv", list(catalog[0]), catalog)
    values = []
    for i in range(6):
        values.append({"person_id": f"p{i}", **{n: str((i + j) / 10) for j, n in enumerate(names)}})
    values[0][names[0]] = "0"  # Observed zero must stay observed.
    for name in names[:23]:
        values[1][name] = ""  # Has nonbasic Clinical records but no retained input.
    write_csv(root / "clinical.csv", ["person_id", *names], values)
    group_names = {"clinical": names, "environment": ["temperature"], "activity": ["steps"],
                   "cgm": ["glucose"], "ecg_waveform": ["I"], "ecg_numeric": ["Rate"]}
    scaler = {"scaler_format_version": "aireadi_ot2_train_scalers_v2", "preprocessing_version": "aireadi-ot2-v2",
              "fit_split": "recommended_split=train", "fit_population": {"fit_person_ids_hash": fit_hash},
              "scalers": {g: {"feature_order": cols, "method": "synthetic",
                   "features": {n: {"center": j / 20, "scale": 1.0 + j, "scale_applied": True} for j, n in enumerate(cols)}} for g, cols in group_names.items()},
              "source_config_hashes": {"clinical": {"task": task, "task_config_hash": "task-hash", "config_hash": schema["config_hash"], "schema_hash": canonical_hash(schema)}}}
    for source in ("environment", "wearable", "ecg"):
        payload = {"config_hash": f"{source}-hash"}
        if source == "environment":
            payload.update(feature_names=["temperature"], default_model_excluded_feature_names=[])
        write_json(root / f"{source}.json", payload)
        scaler["source_config_hashes"][source] = {"config_hash": payload["config_hash"]}
    scaler["scaler_config_hash"] = canonical_hash(scaler)
    write_json(root / "scaler.json", scaler)
    master = []
    for i in range(6):
        master.append({"person_id": f"p{i}", "task": task, "task_config_hash": "task-hash",
                       "split": ("train", "val", "test")[i // 2], "y": str(i % 2), "label_value": str(20 + i),
                       "clinical_features_path": str(root / "clinical.csv"), "has_clinical": "True", "has_retinal": "False",
                       "has_retinal_flio": "False", "has_retinal_oct": "False", "has_retinal_octa": "False", "has_retinal_photography": "False",
                       "has_environment": "False", "has_ecg": "False", "has_activity": "False", "has_cgm": "False",
                       "environment_path": "", "wearable_path": "", "ecg_path": "", "retinal_asset_paths_json": "[]",
                       "available_modality_count": "1"})
    write_csv(root / "master.csv", list(master[0]), master)
    meta_cfg = {"task": task, "task_config_hash": "task-hash", "clinical_schema_hash": canonical_hash(schema),
                "source_index_config_hashes": {s: f"{s}-hash" for s in ("environment", "wearable", "ecg")}}
    metadata = {"task": task, "n_rows": 6, "split_counts": {"train": 2, "val": 2, "test": 2},
                "preprocessing_config": meta_cfg, "config_hash": canonical_hash(meta_cfg), "clinical_schema_hash": canonical_hash(schema),
                "coverage_by_split": {s: {"has_clinical": 2} for s in ("train", "val", "test")}}
    write_json(root / "master.metadata.json", metadata)
    raw = {"schema_version": "ot2-model-v2-1", "seed": 42, "output_dir": str(root / "old_model"),
           "dataset": {"task": task, "master_csv": str(root / "master.csv"), "master_metadata_json": str(root / "master.metadata.json"),
                       "clinical_schema_json": str(root / "clinical_schema.json"), "scaler_json": str(root / "scaler.json"),
                       **{f"{s}_schema_json": str(root / f"{s}.json") for s in ("environment", "wearable", "ecg")},
                       "artifact_root": str(root), "strict_lineage": True, "allow_missing_modalities": True,
                       "retinal": {"deterministic_selection": True, "max_assets": {"photography": 1, "oct": 1, "octa": 1, "flio": 1}}},
           "model": {"source_order": list(MODALITIES), "task_heads": {task: 1}, "d_model": 8, "dropout": 0.0,
                     "longitudinal": {"patch_size_slots": 2, "layers": 1, "heads": 2, "ff_dim": 16},
                     "ecg": {"patch_size_samples": 2, "layers": 1, "heads": 2, "ff_dim": 16},
                     "retinal": {"backbone": "small_cnn", "backbone_dim": 8, "normalization": "none", "weights": None, "frozen": False},
                     "fusion": {"layers": 1, "heads": 2, "ff_dim": 16}},
           "training": {"evaluation_phase": "validation_screen", "initialization_checkpoint": None, "disabled_groups": [],
                        "batch_size": 2, "gradient_accumulation_steps": 1, "max_epochs": 1, "learning_rate": 5e-5,
                        "weight_decay": 0.01, "early_stopping_patience": 1, "selection_metric": "auroc",
                        "num_workers": 0, "bf16": False, "device": "cpu", "threshold_method": "youden",
                        "scheduler": {"name": "cosine_with_linear_warmup", "warmup_ratio": 0.0}},
           "importance": {"methods": ["ablation"], "bootstrap_samples": 16}}
    write_json(root / "source_config.json", raw)
    return root / "source_config.json"


def prepared_fixture(root, task=TASKS[0]):
    config = source_fixture(Path(root) / "sources" / task, task)
    prepare(config, root)
    return Path(root) / "inputs" / task


def trained_fixture(root, task=TASKS[0]):
    root = Path(root)
    inputs = root / "inputs" / task
    output = root / "models" / task / "seed42"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("best_model.pt", "last_model.pt", "validation_predictions.csv", "run_summary.json"):
        (output / name).write_text("synthetic fixture, not a study result")
    from task2_v2_config import load_model_v2_config
    config = load_model_v2_config(inputs / "train_config.yaml")
    receipt = {"schema_version": "ot2-basic-retrain-trained-v1", "task": task, "clinical_width": 23,
               "initialization_checkpoint": None, "test_evaluated": False, "threshold": 0.6,
               "prepared_sha256": sha256(inputs / "PREPARED.json"), "best_epoch": 1, "epochs_completed": 1,
               "config_hash": config.config_hash, "validation_metrics": {m: 0.7 for m in METRICS},
               "code_sha256": code_hashes(), "output_sha256": {p.name: sha256(p) for p in output.iterdir()}}
    write_json(output / "TRAINED.json", receipt)
    return receipt


def importance_fixture(root, task=TASKS[0]):
    root = Path(root)
    output = root / "importance" / task / "seed42/val"
    output.mkdir(parents=True, exist_ok=False)
    prepared = read_json(root / "inputs" / task / "PREPARED.json")
    trained = read_json(root / "models" / task / "seed42/TRAINED.json")
    rows, intervals = [], []
    for group in GROUPS:
        for scope in SCOPES:
            for metric in METRICS:
                deleted = 0.73 if metric in ("brier", "log_loss") else 0.67
                row = {"task": task, "split": "val", "group": group, "cohort_scope": scope, "metric": metric,
                       "n_evaluated": 2, "n_positive": 1, "n_negative": 1, "n_without_any_input": 0,
                       "cohort_mask_sha256": "8" * 64, "threshold_from_validation": 0.6, "baseline_metric": 0.7,
                       "mean_perturbed_metric": deleted, "mean_degradation_delta": 0.03, "ci_lower": 0.01, "ci_upper": 0.05}
                rows.append(row)
                intervals.append({**row, "point_estimate": 0.03})
    write_csv(output / "importance_summary.csv", list(rows[0]), rows)
    write_csv(output / "importance_bootstrap_intervals.csv", list(intervals[0]), intervals)
    (output / "importance_paired_predictions.csv").write_text("PRIVATE_SYNTHETIC_ID\n")
    (output / "importance_cohort_counts.csv").write_text("synthetic counts\n")
    meta = {"schema_version": "ot2-basic-retrain-importance-v1", "task": task, "split": "val", "clinical_width": 23,
            "groups": {g: list(m) for g, m in GROUPS.items()}, "n_participants": 2,
            "checkpoint_sha256": trained["output_sha256"]["best_model.pt"], "code_sha256": trained["code_sha256"],
            "config_hash": trained["config_hash"], "prepared_sha256": sha256(root / "inputs" / task / "PREPARED.json"),
            "trained_sha256": sha256(root / "models" / task / "seed42/TRAINED.json"), "threshold": 0.6,
            "basic_plan_sha256": prepared["basic_plan_sha256"], "baseline_metrics": trained["validation_metrics"],
            "threshold_source": "this_retrained_models_validation_selected_threshold", "bootstrap_samples": 16, "bootstrap_seed": 42,
            "source_token_cache": {"synthetic": True}, "output_sha256": {p.name: sha256(p) for p in output.iterdir()}}
    write_json(output / "importance_metadata.json", meta)


def complete_fixture(root, task=TASKS[0]):
    prepared_fixture(root, task)
    trained_fixture(root, task)
    importance_fixture(root, task)
