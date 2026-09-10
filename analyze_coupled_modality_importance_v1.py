#!/usr/bin/env python3
"""Validation ablation of frozen modality branches and related clinical inputs.

Profiles distinguish device acquisition records from independent shared-domain
measurements. Original data, scalers, checkpoints and native caches stay intact.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from ot2 import analyze_modality_importance_v2 as am
    from ot2.clinical_modality_overlap_v1 import build_overlap_plan, feature_indices_for_group
    from ot2.clinical_overlap_inference_v1 import (
        materialize_clinical_bank, encode_clinical_override,
        assert_native_clinical_equivalent,
    )
    from ot2.train_task2_v2 import load_supervised_config, resolve_device
except ModuleNotFoundError:
    import analyze_modality_importance_v2 as am
    from clinical_modality_overlap_v1 import build_overlap_plan, feature_indices_for_group
    from clinical_overlap_inference_v1 import (
        materialize_clinical_bank, encode_clinical_override,
        assert_native_clinical_equivalent,
    )
    from train_task2_v2 import load_supervised_config, resolve_device


FORMAT = "ot2-coupled-modality-ablation-v1"
PROFILES = ("branch_only", "direct_clinical", "domain_clinical")
DEFAULT_GROUPS = tuple(am.PRIMARY_MODALITIES) + ("retinal_all", "wearable_all")


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(array.dtype), tuple(array.shape))).encode())
    digest.update(array.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_coupled_analysis(
    model: torch.nn.Module,
    cache: am.SourceTokenCache,
    clinical_values: torch.Tensor,
    clinical_masks: torch.Tensor,
    plan: Mapping[str, Any],
    *,
    groups: Sequence[str],
    task: str,
    threshold: float,
    batch_size: int,
    device: str,
    use_bf16_amp: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if cache.identity.get("split") != "val":
        raise ValueError("coupled modality analysis is validation-only")
    if cache.identity.get("task") != task:
        raise ValueError("coupled task differs from native cache")
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("groups must be non-empty and unique")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for group in groups:
        am._target_modalities(group)
    if clinical_values.shape[1] != len(plan["feature_order"]):
        raise ValueError("clinical overlap plan does not match the input bank")

    encoder_options = dict(device=device, batch_size=batch_size, use_bf16_amp=use_bf16_amp)
    untouched = encode_clinical_override(
        model, cache, clinical_values, clinical_masks, (), **encoder_options
    )
    assert_native_clinical_equivalent(cache, *untouched)
    evaluate_options = dict(task=task, threshold=threshold, **encoder_options)
    baseline = am.evaluate_cached_condition(model, cache, **evaluate_options)
    am._assert_cached_baseline_exact(cache, baseline)
    original_token_hash = _tensor_hash(cache.source_tokens)
    original_presence_hash = _tensor_hash(cache.source_present)
    clinical_index = tuple(am.PRIMARY_MODALITIES).index("clinical_data")
    original_presence = cache.source_present.cpu().numpy().copy()

    overrides: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    summaries, intervals, predictions, cohort_counts, definitions = [], [], [], [], []
    for group in groups:
        members = am._target_modalities(group)
        index_sets = {
            profile: feature_indices_for_group(plan, members, profile)
            for profile in PROFILES
        }
        results, reused_from, by_indices = {}, {}, {}
        for profile, indices in index_sets.items():
            if indices in by_indices:
                source_profile = by_indices[indices]
                results[profile] = results[source_profile]
                reused_from[profile] = source_profile
                continue
            kwargs = {}
            if indices:
                if indices not in overrides:
                    overrides[indices] = encode_clinical_override(
                        model, cache, clinical_values, clinical_masks, indices,
                        **encoder_options,
                    )
                tokens, present = overrides[indices]
                kwargs = dict(clinical_token_override=tokens, clinical_present_override=present)
            results[profile] = am.evaluate_cached_condition(
                model, cache, disabled_groups=members, **evaluate_options, **kwargs
            )
            am._assert_aligned(baseline, results[profile])
            by_indices[indices] = profile
            reused_from[profile] = None

        # Shared cohorts across profiles: remaining input is assessed AFTER the
        # broadest clinical intervention, while target availability is native.
        joint_presence = original_presence.copy()
        broad_indices = index_sets["domain_clinical"]
        if broad_indices:
            joint_presence[:, clinical_index] = overrides[broad_indices][1].cpu().numpy()
        masks, counts = am._cohort_masks_and_counts(joint_presence, baseline.y_true, group)
        cohort_counts.extend({"task": task, **r} for r in counts)
        target_indices = [tuple(am.PRIMARY_MODALITIES).index(m) for m in members]
        target_present = original_presence[:, target_indices].any(axis=1)
        remaining = joint_presence.copy()
        remaining[:, target_indices] = False

        for profile, indices in index_sets.items():
            definitions.append({
                "group": group, "profile": profile, "disabled_modalities": list(members),
                "clinical_feature_indices": list(indices),
                "clinical_feature_names": [plan["feature_order"][i] for i in indices],
                "n_clinical_features_masked": len(indices),
                "predictions_reused_from_profile": reused_from[profile],
                "n_people_with_selected_feature_mask_true": (
                    int(clinical_masks[:, list(indices)].any(dim=1).sum()) if indices else 0
                ),
            })
            result = results[profile]
            for i, person_id in enumerate(baseline.person_ids):
                predictions.append({
                    "task": task, "group": group, "profile": profile,
                    "person_id": person_id, "y": int(baseline.y_true[i]),
                    "baseline_probability": float(baseline.probabilities[i]),
                    "perturbed_probability": float(result.probabilities[i]),
                    "baseline_logit": float(baseline.logits[i]),
                    "perturbed_logit": float(result.logits[i]),
                    "threshold": threshold,
                    "n_clinical_features_masked": len(indices),
                    "native_target_present": bool(target_present[i]),
                    "has_remaining_input_after_domain_cleanup": bool(remaining[i].any()),
                    **{f"in_{scope}": bool(mask[i]) for scope, mask in masks.items()},
                })

        contrasts = [
            ("full_vs_branch", "full", "branch_only"),
            ("full_vs_direct", "full", "direct_clinical"),
            ("full_vs_domain", "full", "domain_clinical"),
            ("extra_direct_clinical", "branch_only", "direct_clinical"),
            ("extra_domain_clinical", "branch_only", "domain_clinical"),
        ]
        all_results = {"full": baseline, **results}
        # Exact reuse for identical participant masks AND prediction pairs.
        # Incremental CIs are directly paired, never differences of CI limits.
        memo = {}
        for contrast, reference_name, perturbed_name in contrasts:
            reference, perturbed = all_results[reference_name], all_results[perturbed_name]
            for count in counts:
                scope = count["cohort_scope"]
                mask = masks[scope]
                key = (id(reference), id(perturbed), mask.tobytes())
                if key not in memo:
                    if contrast == "full_vs_branch":
                        components = ("bootstrap", "ablation", group)
                        if scope != "all_participants":
                            components += (scope,)
                    else:
                        components = ("coupled_bootstrap", group, contrast, scope)
                    memo[key] = am._cohort_condition_rows(
                        reference, [perturbed], mask=mask, counts=count,
                        method="ablation", group=group, threshold=threshold,
                        bootstrap_samples=bootstrap_samples,
                        seed=am._stable_seed(seed, *components),
                    )
                summary_rows, interval_rows = memo[key]
                context = {
                    "task": task, **count, "comparison": contrast,
                    "reference_profile": reference_name, "perturbed_profile": perturbed_name,
                    "n_clinical_features_masked": len(index_sets[perturbed_name]),
                    "cohort_remaining_input_basis": "domain_clinical_joint_intervention",
                }
                summaries.extend({**r, **context} for r in summary_rows)
                intervals.extend({**r, **context} for r in interval_rows)
        print(f"[coupled][{task}][{group}] clinical columns: "
              f"direct={len(index_sets['direct_clinical'])}, domain={len(broad_indices)}", flush=True)

    if original_token_hash != _tensor_hash(cache.source_tokens) or original_presence_hash != _tensor_hash(cache.source_present):
        raise RuntimeError("original source-token cache was mutated")
    return {
        "summary": pd.DataFrame(summaries), "intervals": pd.DataFrame(intervals),
        "predictions": pd.DataFrame(predictions), "cohort_counts": pd.DataFrame(cohort_counts),
        "conditions": definitions, "baseline_metrics": baseline.metrics,
        "clinical_inference": {
            "untouched_clinical_token_equivalence": "passed; atol=1e-6, rtol=1e-6",
            "original_native_cache_unchanged": True,
            "distinct_clinical_interventions_encoded": len(overrides),
            "nonclinical_encoders_called_for_clinical_interventions": False,
            "value_and_observed_mask_removed": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clinical-catalog", type=Path)
    parser.add_argument("--source-token-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=DEFAULT_GROUPS + ("retinal_wearable_all",), default=DEFAULT_GROUPS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.bootstrap_samples <= 0:
        raise ValueError("invalid batch size, worker count or bootstrap count")
    if len(args.groups) != len(set(args.groups)):
        raise ValueError("duplicate groups")
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
        catalog = list(csv.DictReader(handle))
    plan = build_overlap_plan(json.loads(schema_path.read_text()), catalog)
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
    if plan.get("task") != config.dataset.task:
        raise ValueError("clinical overlap plan task differs from configured task")
    if tuple(plan["feature_order"]) != tuple(dataset.scalers["clinical"].feature_order):
        raise ValueError("clinical overlap plan order differs from trained scaler order")
    model_config = am._build_model_config(config, dataset)
    am._validate_resolved_model_config(checkpoint, model_config)
    values, masks = materialize_clinical_bank(dataset)
    counts = {
        group: {profile: len(feature_indices_for_group(plan, am._target_modalities(group), profile))
                for profile in PROFILES}
        for group in args.groups
    }
    if args.check_only:
        print(json.dumps({"task": config.dataset.task, "status": "lineage_and_mapping_passed",
                          "n_val": len(dataset), "clinical_feature_counts": counts}, indent=2))
        return 0
    requested_out_dir = args.out_dir.expanduser()
    if requested_out_dir.exists() or requested_out_dir.is_symlink():
        raise FileExistsError(f"Use a new output directory; refusing overwrite: {requested_out_dir}")
    out_dir = requested_out_dir.resolve()
    device = resolve_device(args.device)
    am.set_deterministic_seed(args.seed, deterministic=True)
    model = am.Task2V2Model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    # Reserve atomically after validations, before expensive inference.
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "clinical_overlap_plan.json", plan)
    cache_path = (args.source_token_cache or out_dir / "source_token_cache_val.pt").expanduser().resolve()
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
    loader = am._loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=torch.device(device))
    cache = am.build_or_load_source_token_cache(
        model, loader, frame=dataset.index, device=device, task=config.dataset.task,
        expected_split="val", identity=identity, cache_path=cache_path, use_bf16_amp=args.bf16,
    )
    result = run_coupled_analysis(
        model, cache, values, masks, plan, groups=args.groups, task=config.dataset.task,
        threshold=threshold, batch_size=args.batch_size, device=device,
        use_bf16_amp=args.bf16, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    output_files = []
    for key, filename in [
        ("summary", "coupled_importance_summary.csv"),
        ("intervals", "coupled_importance_bootstrap_intervals.csv"),
        ("predictions", "coupled_importance_paired_predictions.csv"),
        ("cohort_counts", "coupled_importance_cohort_counts.csv"),
    ]:
        result[key].to_csv(out_dir / filename, index=False)
        output_files.append(filename)
    _write_json(out_dir / "coupled_conditions.json", result["conditions"])
    output_files += ["coupled_conditions.json", "clinical_overlap_plan.json"]
    metadata = {
        "schema_version": FORMAT, "analysis_role": "validation_joint_information_removal",
        "task": config.dataset.task, "split": "val", "seed": args.seed,
        "training_seed": config.seed, "bootstrap_seed": args.seed,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "config_hash": config.config_hash, "config_sha256": am._sha256_file(args.config),
        "lineage": lineage, "clinical_schema_sha256": am._sha256_file(schema_path),
        "clinical_catalog_sha256": am._sha256_file(catalog_path),
        "clinical_overlap_plan_sha256": plan["plan_sha256"],
        "threshold": threshold, "threshold_source": "saved_full_teacher_validation_threshold",
        "n_participants": len(dataset), "bootstrap_samples": args.bootstrap_samples,
        "baseline_metrics": result["baseline_metrics"], "clinical_inference": result["clinical_inference"],
        "groups": list(args.groups), "profiles": list(PROFILES),
        "source_token_cache": {"path": str(cache_path), "reused": cache.reused,
                               "file_sha256": am._sha256_file(cache_path),
                               "content_sha256": cache.content_sha256, "identity": dict(cache.identity)},
        "cohort_rule": "Same people in every profile and contrast; remaining input assessed after domain cleanup; target availability from native encoding.",
        "scope": "85 retinal acquisition flags, 27 shared ocular function/method features, 13 activity/sleep questionnaire features; all corresponding one-hot responses removed.",
        "limitations": [
            "Domain profiles remove independent clinical measurements as well as the modality branch; their effects are not unique modality importance or causal clinical utility.",
            "The additional 24 via1-via6 vision questionnaire features and laboratory measurements are outside this predefined intervention.",
            "Old model and all non-target inputs remain fixed, including the earlier EOS/time-policy definition; this is not a cleaned retraining experiment.",
            "Intervals are pointwise participant bootstrap intervals; they omit between-training-seed uncertainty.",
            "Ablation may be outside the training distribution; this does not evaluate reconstruction.",
        ],
        "output_sha256": {name: am._sha256_file(out_dir / name) for name in output_files},
        "code_sha256": {name: am._sha256_file(Path(__file__).parent / name) for name in (
            "analyze_coupled_modality_importance_v1.py", "analyze_modality_importance_v2.py",
            "clinical_modality_overlap_v1.py", "clinical_overlap_inference_v1.py",
        )},
    }
    _write_json(out_dir / "coupled_importance_metadata.json", metadata)
    print(json.dumps({"task": config.dataset.task, "output": str(out_dir), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
