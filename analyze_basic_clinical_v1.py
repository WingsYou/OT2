#!/usr/bin/env python3
"""Fresh validation inference: full, 23 basic, and no Clinical inputs.

All three conditions use the same frozen model, cohort, threshold and eight
non-clinical inputs. No previous prediction file or source cache is read.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from ot2 import analyze_modality_importance_v2 as am
    from ot2.clinical_basic_information_v1 import build_basic_clinical_plan
    from ot2.clinical_overlap_inference_v1 import (
        materialize_clinical_bank, encode_clinical_override, assert_native_clinical_equivalent,
    )
    from ot2.train_task2_v2 import load_supervised_config, resolve_device
except ModuleNotFoundError:
    import analyze_modality_importance_v2 as am
    from clinical_basic_information_v1 import build_basic_clinical_plan
    from clinical_overlap_inference_v1 import (
        materialize_clinical_bank, encode_clinical_override, assert_native_clinical_equivalent,
    )
    from train_task2_v2 import load_supervised_config, resolve_device


CONDITIONS = ("full_clinical", "basic_clinical", "no_clinical")
CODE_FILES = (
    "analyze_basic_clinical_v1.py", "clinical_basic_information_v1.py",
    "clinical_overlap_inference_v1.py", "analyze_modality_importance_v2.py",
    "task2_v2_config.py", "task2_v2_dataset.py", "task2_v2_model.py",
    "task2_v2_engine.py", "train_task2_v2.py", "fit_train_scalers_v2.py", "preprocessing_v2.py",
)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _assert_local_runtime_modules() -> None:
    """Reject a shadowing installed ot2 package from another code snapshot."""
    code_root = Path(__file__).resolve().parent
    expected = {Path(name).stem: name for name in CODE_FILES}
    for name, module in tuple(sys.modules.items()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in expected and getattr(module, "__file__", None):
            if Path(module.__file__).resolve() != code_root / expected[leaf]:
                raise RuntimeError(f"mixed runtime code version: {name} came from {module.__file__}")


@torch.no_grad()
def run_basic_clinical_analysis(
    model: torch.nn.Module, cache: am.SourceTokenCache,
    clinical_values: torch.Tensor, clinical_masks: torch.Tensor, plan: Mapping[str, Any],
    *, task: str, threshold: float, batch_size: int, device: str,
    use_bf16_amp: bool, bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    if cache.identity.get("split") != "val" or cache.identity.get("task") != task:
        raise ValueError("basic Clinical analysis requires the matching validation cache")
    if plan.get("task") != task or clinical_values.shape[1] != len(plan["feature_order"]):
        raise ValueError("basic Clinical plan does not match task/input width")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    original = (_tensor_hash(cache.source_tokens), _tensor_hash(cache.source_present))
    options = dict(device=device, batch_size=batch_size, use_bf16_amp=use_bf16_amp)
    untouched = encode_clinical_override(model, cache, clinical_values, clinical_masks, (), **options)
    assert_native_clinical_equivalent(cache, *untouched)
    basic_token, basic_present = encode_clinical_override(
        model, cache, clinical_values, clinical_masks, plan["masked_indices_zero_based"], **options,
    )
    evaluate_options = dict(task=task, threshold=threshold, **options)
    # Each call recomputes fusion/head predictions, including the no-Clinical control.
    results = {
        "full_clinical": am.evaluate_cached_condition(model, cache, **evaluate_options),
        "basic_clinical": am.evaluate_cached_condition(
            model, cache, clinical_token_override=basic_token,
            clinical_present_override=basic_present, **evaluate_options,
        ),
        "no_clinical": am.evaluate_cached_condition(
            model, cache, disabled_groups=("clinical_data",), **evaluate_options,
        ),
    }
    full = results["full_clinical"]
    am._assert_cached_baseline_exact(cache, full)
    for result in results.values():
        am._assert_aligned(full, result)
    presence = cache.source_present.cpu().numpy()
    ci = tuple(am.PRIMARY_MODALITIES).index("clinical_data")
    others = np.delete(presence, ci, axis=1).any(axis=1)
    basic_available = basic_present.cpu().numpy()
    # Identical recipient cohorts for all three conditions; no outcome selection.
    cohort_masks, counts = am._cohort_masks_and_counts(presence, full.y_true, "clinical_data")
    conditions = []
    for name in CONDITIONS:
        kept = plan["original_width"] if name == "full_clinical" else plan["n_retained"] if name == "basic_clinical" else 0
        conditions.append({
            "condition": name, "n_clinical_columns_retained": kept,
            "n_clinical_columns_removed": plan["original_width"] - kept,
            "disabled_modalities": ["clinical_data"] if name == "no_clinical" else [],
            "prediction_origin": "fresh_fusion_and_head_this_run",
        })
    availability = {
        "full_clinical": presence[:, ci], "basic_clinical": basic_available,
        "no_clinical": np.zeros(len(full.person_ids), dtype=bool),
    }
    predictions = []
    for i, person_id in enumerate(full.person_ids):
        row = {
            "task": task, "split": "val", "person_id": person_id, "y": int(full.y_true[i]),
            "threshold": threshold, "native_clinical_present": bool(presence[i, ci]),
            "basic_clinical_present": bool(basic_available[i]),
            "has_nonclinical_input": bool(others[i]),
            "n_observed_basic_columns": int(clinical_masks[i, plan["retained_indices_zero_based"]].sum()),
            **{f"in_{scope}": bool(mask[i]) for scope, mask in cohort_masks.items()},
        }
        for name, result in results.items():
            row.update({f"{name}_probability": float(result.probabilities[i]),
                        f"{name}_logit": float(result.logits[i]),
                        f"{name}_prediction": int(result.predictions[i])})
        predictions.append(row)
    metrics = []
    cohort_rows = []
    for count in counts:
        mask = cohort_masks[count["cohort_scope"]]
        cohort_rows.append({"task": task, **count,
                            "n_basic_present": int(basic_available[mask].sum()),
                            "n_without_basic_input": int((~basic_available[mask]).sum()),
                            "n_with_nonclinical_input": int(others[mask].sum())})
        for definition in conditions:
            name = definition["condition"]
            measured = am._cohort_metrics_matrix(full.y_true[mask], results[name].probabilities[mask], threshold=threshold)
            metrics.append({"task": task, "split": "val", **count,
                            "condition": name, "threshold": threshold,
                            "n_clinical_columns_retained": definition["n_clinical_columns_retained"],
                            "n_without_any_input": int((~(availability[name] | others))[mask].sum()),
                            **{metric: float(values[0]) for metric, values in measured.items()}})
    summaries, intervals = [], []
    contrasts = (
        ("full_vs_basic", "full_clinical", "basic_clinical"),
        ("full_vs_no_clinical", "full_clinical", "no_clinical"),
        ("basic_vs_no_clinical", "basic_clinical", "no_clinical"),
    )
    for comparison, reference, perturbed in contrasts:
        print(f"[basic][{task}] paired bootstrap: {comparison}", flush=True)
        for count in counts:
            scope = count["cohort_scope"]
            if comparison == "full_vs_no_clinical":
                parts = ("bootstrap", "ablation", "clinical_data")
                if scope != "all_participants":
                    parts += (scope,)
            else:
                parts = ("basic_clinical_bootstrap", comparison, scope)
            summary, interval = am._cohort_condition_rows(
                results[reference], [results[perturbed]], mask=cohort_masks[scope], counts=count,
                method="ablation", group="clinical_data", threshold=threshold,
                bootstrap_samples=bootstrap_samples, seed=am._stable_seed(seed, *parts),
            )
            context = {"task": task, "split": "val", "comparison": comparison,
                       "reference_condition": reference, "perturbed_condition": perturbed}
            summaries.extend({**row, **context} for row in summary)
            intervals.extend({**row, **context} for row in interval)
    if original != (_tensor_hash(cache.source_tokens), _tensor_hash(cache.source_present)):
        raise RuntimeError("native source-token cache was mutated")
    return {
        "metrics": pd.DataFrame(metrics), "summary": pd.DataFrame(summaries),
        "intervals": pd.DataFrame(intervals), "predictions": pd.DataFrame(predictions),
        "cohort_counts": pd.DataFrame(cohort_rows), "conditions": conditions,
        "baseline_metrics": full.metrics,
        "checks": {"untouched_clinical_equivalence": "atol=1e-6, rtol=1e-6; passed",
                   "full_forward_equivalence": "atol=1e-6, rtol=1e-6; passed",
                   "native_cache_unchanged": True, "values_and_masks_removed": True,
                   "all_three_predictions_recomputed": True},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clinical-catalog", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    _assert_local_runtime_modules()
    if args.batch_size <= 0 or args.num_workers < 0 or args.bootstrap_samples <= 0:
        raise ValueError("invalid batch size, worker count or bootstrap count")
    _, config = load_supervised_config(args.config)
    lineage = am.validate_model_v2_lineage(config)
    checkpoint_path = (args.checkpoint or config.output_dir / "best_model.pt").resolve()
    checkpoint_hash = am._sha256_file(checkpoint_path)
    master_hash, scaler_hash = am._expected_lineage_hashes(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    threshold = am.validate_importance_checkpoint(
        checkpoint, expected_config_hash=config.config_hash, expected_task=config.dataset.task,
        expected_master_hash=master_hash, expected_scaler_hash=scaler_hash,
    )
    am._validate_checkpoint_input_file_hashes(checkpoint, config)
    schema_path = config.dataset.clinical_schema_json
    catalog_path = (args.clinical_catalog or schema_path.parent / f"clinical_feature_catalog_{config.dataset.task}.csv").resolve()
    with catalog_path.open(newline="", encoding="utf-8-sig") as handle:
        plan = build_basic_clinical_plan(json.loads(schema_path.read_text()), list(csv.DictReader(handle)))
    dataset = am.OT2V2Dataset(
        config.dataset.master_csv, config.dataset.scaler_json, task=config.dataset.task,
        artifact_root=config.dataset.artifact_root, split="val",
        master_metadata_json=config.dataset.master_metadata_json,
        environment_schema_json=config.dataset.environment_schema_json,
        max_retinal_assets_per_group={f"retinal_{k}": v for k, v in config.dataset.retinal_max_assets.items()},
        require_y=True,
    )
    am._validate_clinical_feature_bytes(checkpoint, dataset)
    am._validate_evaluation_cohort_provenance(checkpoint, dataset, "val")
    if plan["task"] != config.dataset.task or tuple(plan["feature_order"]) != tuple(dataset.scalers["clinical"].feature_order):
        raise ValueError("basic Clinical plan differs from configured task/trained scaler order")
    model_config = am._build_model_config(config, dataset)
    am._validate_resolved_model_config(checkpoint, model_config)
    values, masks = materialize_clinical_bank(dataset)
    if args.check_only:
        print(json.dumps({"task": config.dataset.task, "status": "lineage_and_23_column_mapping_passed",
                          "n_val": len(dataset), "original_width": plan["original_width"],
                          "n_retained": plan["n_retained"], "n_masked": plan["n_masked"],
                          "n_basic_present": int(masks[:, plan["retained_indices_zero_based"]].any(dim=1).sum())}, indent=2))
        return 0
    requested = args.out_dir.expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Use a fresh output directory; refusing overwrite: {requested}")
    out_dir = requested.resolve()
    device = resolve_device(args.device)
    am.set_deterministic_seed(args.seed, deterministic=True)
    model = am.Task2V2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "basic_clinical_plan.json", plan)
    # Deliberately no external-cache option: this control is a fresh raw-input rerun.
    cache_path = out_dir / "source_token_cache_val.pt"
    identity = {
        "checkpoint_sha256": checkpoint_hash, "config_hash": config.config_hash,
        "master_config_hash": master_hash, "scaler_config_hash": scaler_hash,
        "model_config_hash": am.canonical_json_hash(dataclasses.asdict(model_config)),
        "task": config.dataset.task, "split": "val", "source_order": list(am.PRIMARY_MODALITIES),
        "encoder_batch_size": args.batch_size,
        "amp_mode": "bfloat16_autocast" if args.bf16 else "float32",
        "cohort_person_ids_sha256": am._person_ids_sha256(dataset.index["person_id"].tolist()),
        "n_participants": len(dataset),
    }
    print(f"[basic][{config.dataset.task}] Fresh raw-input encoding, n={len(dataset)}", flush=True)
    loader = am._loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=torch.device(device))
    cache = am.build_or_load_source_token_cache(
        model, loader, frame=dataset.index, device=device, task=config.dataset.task,
        expected_split="val", identity=identity, cache_path=cache_path, use_bf16_amp=args.bf16,
    )
    if cache.reused:
        raise RuntimeError("fresh rerun unexpectedly loaded an existing cache")
    result = run_basic_clinical_analysis(
        model, cache, values, masks, plan, task=config.dataset.task, threshold=threshold,
        batch_size=args.batch_size, device=device, use_bf16_amp=args.bf16,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    output_files = ["basic_clinical_plan.json", "basic_clinical_conditions.json"]
    _write_json(out_dir / "basic_clinical_conditions.json", result["conditions"])
    for key, suffix in (("metrics", "metrics"), ("summary", "summary"),
                        ("intervals", "bootstrap_intervals"), ("predictions", "paired_predictions"),
                        ("cohort_counts", "cohort_counts")):
        name = f"basic_clinical_{suffix}.csv"
        result[key].to_csv(out_dir / name, index=False)
        output_files.append(name)
    metadata = {
        "schema_version": "ot2-basic-clinical-inference-v1",
        "analysis_role": "validation_basic_clinical_sensitivity_analysis",
        "task": config.dataset.task, "split": "val", "training_seed": config.seed,
        "bootstrap_seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "config_hash": config.config_hash, "config_sha256": am._sha256_file(args.config),
        "lineage": lineage, "clinical_schema_sha256": am._sha256_file(schema_path),
        "clinical_catalog_sha256": am._sha256_file(catalog_path), "plan_sha256": plan["plan_sha256"],
        "threshold": threshold, "threshold_source": "saved_full_teacher_validation_threshold",
        "n_participants": len(dataset), "conditions": list(CONDITIONS),
        "baseline_metrics": result["baseline_metrics"], "inference_checks": result["checks"],
        "n_retained_basic_columns": plan["n_retained"],
        "source_token_cache": {"path": str(cache_path), "reused": False,
                               "raw_dataset_passes_this_run": 1,
                               "file_sha256": am._sha256_file(cache_path),
                               "content_sha256": cache.content_sha256, "identity": dict(cache.identity)},
        "scope": "Full, basic-23, and disabled Clinical; all eight other modalities, including ECG reports, remain unchanged.",
        "cohort_rule": "Identical native-presence-defined participant subsets in every condition and contrast; remaining input means any of the other eight modalities.",
        "limitations": [
            "Frozen-model input removal can be outside the training distribution; this is not retraining or reconstruction.",
            "Basic Clinical retains age, education, marital status and language; it is not removal of all structured data.",
            "Intervals are pointwise paired participant bootstrap intervals and omit between-training-seed uncertainty.",
        ],
        "output_sha256": {name: am._sha256_file(out_dir / name) for name in output_files},
        "code_sha256": {name: am._sha256_file(Path(__file__).parent / name) for name in CODE_FILES},
    }
    _write_json(out_dir / "basic_clinical_metadata.json", metadata)
    print(json.dumps({"task": config.dataset.task, "output": str(out_dir), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
