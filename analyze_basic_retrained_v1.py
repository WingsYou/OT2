#!/usr/bin/env python3
"""Validation leave-one-out and family controls for the newly trained model."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import analyze_modality_importance_v2 as am
from basic_retrain_contract_v1 import (MODALITIES, GROUPS, SCOPES, verify_prepared, verify_training,
                                       sha256, read_json, write_json, code_hashes, assert_local_modules)
from train_task2_v2 import load_supervised_config, resolve_device


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def analyze(model, cache, *, task, threshold, batch_size, device, bf16, bootstrap_samples, seed=42):
    if cache.identity.get("task") != task or cache.identity.get("split") != "val":
        raise ValueError("Expected matching validation cache")
    if tuple(cache.identity["source_order"]) != MODALITIES or tuple(model.source_order) != MODALITIES:
        raise ValueError("Unexpected modality order")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    original = (tensor_hash(cache.source_tokens), tensor_hash(cache.source_present))
    options = dict(task=task, threshold=threshold, batch_size=batch_size, device=device, use_bf16_amp=bf16)
    baseline = am.evaluate_cached_condition(model, cache, **options)
    am._assert_cached_baseline_exact(cache, baseline)
    native = cache.source_present.cpu().numpy()
    summaries, intervals, predictions, counts = [], [], [], []
    for group, members in GROUPS.items():
        print(f"[importance][{task}] remove {group}: {','.join(members)}", flush=True)
        result = am.evaluate_cached_condition(model, cache, disabled_groups=members, **options)
        am._assert_aligned(baseline, result)
        indices = [MODALITIES.index(m) for m in members]
        any_present, all_present = native[:, indices].any(1), native[:, indices].all(1)
        remaining = native.copy()
        remaining[:, indices] = False
        has_remaining = remaining.any(1)
        cohorts = dict(zip(SCOPES, (np.ones(len(native), dtype=bool), any_present, all_present, any_present & has_remaining)))
        memo = {}
        for scope, mask in cohorts.items():
            labels = baseline.y_true[mask]
            count = {"task": task, "split": "val", "group": group, "cohort_scope": scope,
                     "n_total": len(native), "n_evaluated": len(labels), "n_positive": int((labels == 1).sum()),
                     "n_negative": int((labels == 0).sum()), "prevalence": float(labels.mean()) if len(labels) else float("nan"),
                     "n_native_present": int(any_present.sum()), "n_eligible": len(labels),
                     "n_without_any_input": int((~has_remaining & mask).sum()),
                     "cohort_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                     "status": "empty_cohort" if not len(labels) else "single_class_cohort" if len(np.unique(labels)) < 2 else "ok"}
            counts.append(count)
            key = mask.tobytes()
            if key not in memo:
                memo[key] = am._cohort_condition_rows(baseline, [result], mask=mask, counts=count,
                    method="ablation", group=group, threshold=threshold, bootstrap_samples=bootstrap_samples,
                    seed=am._stable_seed(seed, "basic_retrained_v1", group, count["cohort_mask_sha256"]))
            summary, ci = memo[key]
            summaries.extend({**r, **count} for r in summary)
            intervals.extend({**r, **count} for r in ci)
        for i, person_id in enumerate(baseline.person_ids):
            predictions.append({"task": task, "split": "val", "group": group, "person_id": person_id,
                                "y": int(baseline.y_true[i]), "threshold": threshold,
                                "baseline_probability": float(baseline.probabilities[i]),
                                "perturbed_probability": float(result.probabilities[i]),
                                "has_remaining_input": bool(has_remaining[i]),
                                **{f"in_{s}": bool(mask[i]) for s, mask in cohorts.items()}})
    if original != (tensor_hash(cache.source_tokens), tensor_hash(cache.source_present)):
        raise RuntimeError("Native cache was mutated")
    return {"summary": pd.DataFrame(summaries), "intervals": pd.DataFrame(intervals),
            "predictions": pd.DataFrame(predictions), "counts": pd.DataFrame(counts),
            "baseline_metrics": baseline.metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    assert_local_modules()
    inputs = args.inputs.resolve()
    prepared = verify_prepared(inputs)
    _, config = load_supervised_config(inputs / "train_config.yaml")
    trained = verify_training(config.output_dir, sha256(inputs / "PREPARED.json"))
    if trained["code_sha256"] != code_hashes():
        raise ValueError("Training and importance code snapshots differ")
    am.validate_model_v2_lineage(config)
    checkpoint_path = config.output_dir / "best_model.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    master_hash, scaler_hash = am._expected_lineage_hashes(config)
    threshold = am.validate_importance_checkpoint(checkpoint, expected_config_hash=config.config_hash,
        expected_task=config.dataset.task, expected_master_hash=master_hash, expected_scaler_hash=scaler_hash)
    am._validate_checkpoint_input_file_hashes(checkpoint, config)
    dataset = am.OT2V2Dataset(config.dataset.master_csv, config.dataset.scaler_json, task=config.dataset.task,
        artifact_root=config.dataset.artifact_root, split="val", master_metadata_json=config.dataset.master_metadata_json,
        environment_schema_json=config.dataset.environment_schema_json,
        max_retinal_assets_per_group={f"retinal_{k}": v for k, v in config.dataset.retinal_max_assets.items()}, require_y=True)
    am._validate_clinical_feature_bytes(checkpoint, dataset)
    am._validate_evaluation_cohort_provenance(checkpoint, dataset, "val")
    model_config = am._build_model_config(config, dataset)
    am._validate_resolved_model_config(checkpoint, model_config)
    if model_config.clinical_dim != 23 or list(dataset.scalers["clinical"].feature_order) != read_json(inputs / "basic_plan.json")["retained_feature_names"]:
        raise ValueError("Expected the exact trained Basic Clinical 23-column view")
    batch = config.training.batch_size
    workers = int(config.raw["training"].get("num_workers", 2))
    bf16 = bool(config.raw["training"].get("bf16", False))
    device = resolve_device(str(config.raw["training"].get("device", "auto")))
    am.set_deterministic_seed(42, deterministic=True)
    model = am.Task2V2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = {"checkpoint_sha256": checkpoint_hash, "config_hash": config.config_hash,
                "master_config_hash": master_hash, "scaler_config_hash": scaler_hash,
                "model_config_hash": am.canonical_json_hash(dataclasses.asdict(model_config)),
                "task": config.dataset.task, "split": "val", "source_order": list(MODALITIES),
                "encoder_batch_size": batch, "amp_mode": "bfloat16_autocast" if bf16 else "float32",
                "cohort_person_ids_sha256": am._person_ids_sha256(dataset.index["person_id"].tolist()),
                "n_participants": len(dataset)}
    loader = am._loader(dataset, batch_size=batch, num_workers=workers, device=torch.device(device))
    cache = am.build_or_load_source_token_cache(model, loader, frame=dataset.index, device=device,
        task=config.dataset.task, expected_split="val", identity=identity, cache_path=args.cache, use_bf16_amp=bf16)
    result = analyze(model, cache, task=config.dataset.task, threshold=threshold, batch_size=batch,
                     device=device, bf16=bf16, bootstrap_samples=args.bootstrap_samples)
    files = []
    for key, suffix in (("summary", "summary"), ("intervals", "bootstrap_intervals"),
                        ("predictions", "paired_predictions"), ("counts", "cohort_counts")):
        name = f"importance_{suffix}.csv"
        result[key].to_csv(out / name, index=False)
        files.append(name)
    metadata = {"schema_version": "ot2-basic-retrain-importance-v1", "task": config.dataset.task,
                "split": "val", "training_seed": 42, "bootstrap_seed": 42, "bootstrap_samples": args.bootstrap_samples,
                "clinical_width": 23, "groups": {k: list(v) for k, v in GROUPS.items()},
                "n_participants": len(dataset), "threshold": threshold,
                "threshold_source": "this_retrained_models_validation_selected_threshold",
                "baseline_metrics": result["baseline_metrics"], "checkpoint_sha256": checkpoint_hash,
                "config_hash": config.config_hash, "prepared_sha256": sha256(inputs / "PREPARED.json"),
                "trained_sha256": sha256(config.output_dir / "TRAINED.json"), "basic_plan_sha256": prepared["basic_plan_sha256"],
                "code_sha256": code_hashes(), "source_token_cache": {"identity": identity, "reused": cache.reused,
                    "path": str(args.cache.resolve()), "file_sha256": sha256(args.cache), "content_sha256": cache.content_sha256},
                "identical_cohort_intervals_reused": True,
                "output_sha256": {n: sha256(out / n) for n in files}}
    write_json(out / "importance_metadata.json", metadata)
    print(f"IMPORTANCE_COMPLETE: {config.dataset.task}", flush=True)


if __name__ == "__main__":
    main()
