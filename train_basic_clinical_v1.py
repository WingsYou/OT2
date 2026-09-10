#!/usr/bin/env python3
"""Train a new 23-column Clinical multimodal model, using the existing engine."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from basic_retrain_contract_v1 import (MODALITIES, verify_prepared, sha256, read_json, write_json,
                                       code_hashes, assert_local_modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    receipt = verify_prepared(inputs)
    from train_task2_v2 import load_supervised_config, build_datasets, run_supervised_training
    from task2_v2_config import validate_model_v2_lineage
    assert_local_modules()
    path = inputs / "train_config.yaml"
    _, config = load_supervised_config(path)
    validate_model_v2_lineage(config)
    if config.dataset.task != receipt["task"] or config.seed != 42:
        raise ValueError("Prepared task/seed mismatch")
    if config.raw["training"].get("initialization_checkpoint") or config.raw["training"].get("disabled_groups"):
        raise ValueError("This study requires all modalities and no previous task checkpoint")
    schema = read_json(config.dataset.clinical_schema_json)
    if schema["n_features"] != 23 or len(schema["feature_names"]) != 23:
        raise ValueError("Training must use exactly 23 Clinical columns")
    if args.check_only:
        datasets = build_datasets(config)
        for ds in (datasets.train, datasets.validation, datasets.test):
            if ds.clinical_values_raw.shape[1] != 23 or ds.model_feature_dimensions["clinical_dim"] != 23 or tuple(ds.loaded_source_groups) != MODALITIES:
                raise ValueError("Dataset did not physically restrict Clinical inputs to 23 columns")
        print(f"CHECK_OK: {receipt['task']}; physical Clinical width 23; all nine sources; no model initialized, no test inference", flush=True)
        return
    if config.output_dir.exists():
        raise FileExistsError(f"Training directory already exists: {config.output_dir}; use the runner's continue stage")
    # Observe successful checkpoint saves to report epochs without modifying
    # the shared training engine or its optimization/selection behavior.
    trainer_module = importlib.import_module(run_supervised_training.__module__)
    engine = importlib.import_module(trainer_module.fit_evaluate_binary.__module__)
    original_save = engine.atomic_torch_save

    def logged_save(payload, destination):
        result = original_save(payload, destination)
        if Path(destination).name == "last_model.pt":
            metric = payload["validation_metrics"]
            print(f"[epoch][{receipt['task']}] {payload['epoch']}: validation AUROC={metric['auroc']:.4f}; "
                  f"AUPRC={metric['auprc']:.4f}; Brier={metric['brier']:.4f}", flush=True)
        return result

    engine.atomic_torch_save = logged_save
    try:
        result = run_supervised_training(path)
    finally:
        engine.atomic_torch_save = original_save
    if result.test is not None:
        raise RuntimeError("Unexpected test evaluation during training")
    summary = read_json(config.output_dir / "run_summary.json")
    if summary["data_provenance"]["model_feature_dimensions"]["clinical_dim"] != 23:
        raise RuntimeError("Trained model Clinical width mismatch")
    artifacts = [result.best_checkpoint, result.last_checkpoint, result.validation_predictions_csv,
                 config.output_dir / "run_summary.json"]
    trained = {"schema_version": "ot2-basic-retrain-trained-v1", "task": receipt["task"],
               "clinical_width": 23, "training_seed": 42, "initialization_checkpoint": None,
               "retinal_initialization": str(config.model.retinal_weights or config.model.retinal_weights_path or "none"),
               "test_evaluated": False, "prepared_sha256": sha256(inputs / "PREPARED.json"),
               "config_hash": config.config_hash, "best_epoch": result.best_epoch,
               "epochs_completed": result.epochs_completed, "threshold": result.threshold_selection.threshold,
               "validation_metrics": result.validation.metrics, "code_sha256": code_hashes(),
               "output_sha256": {p.name: sha256(p) for p in artifacts}}
    write_json(config.output_dir / "TRAINED.json", trained)
    print(f"TRAINED: {receipt['task']}; best epoch {result.best_epoch}; validation AUROC {result.validation.metrics['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
