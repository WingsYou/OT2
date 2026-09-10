#!/usr/bin/env python3
"""Fresh validation inference for joint modality and Clinical domain removal.

Every condition uses the same frozen model and validation participants.
No previous prediction file or source cache is read.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from ot2 import analyze_modality_importance_v2 as am
    from ot2.clinical_domain_groups_v1 import build_domain_plan, DEFAULT_GROUPS, MEMBERS
    from ot2.domain_ablation_engine_v1 import run_domain_analysis
    from ot2.clinical_overlap_inference_v1 import (
        materialize_clinical_bank,
    )
    from ot2.train_task2_v2 import load_supervised_config, resolve_device
except ModuleNotFoundError:
    import analyze_modality_importance_v2 as am
    from clinical_domain_groups_v1 import build_domain_plan, DEFAULT_GROUPS, MEMBERS
    from domain_ablation_engine_v1 import run_domain_analysis
    from clinical_overlap_inference_v1 import (
        materialize_clinical_bank,
    )
    from train_task2_v2 import load_supervised_config, resolve_device


CODE_FILES = (
    "analyze_domain_ablation_v1.py", "clinical_domain_groups_v1.py",
    "domain_ablation_engine_v1.py", "clinical_basic_information_v1.py", "clinical_modality_overlap_v1.py",
    "summarize_domain_ablation_v1.py", "run_domain_ablation_v1.py",
    "clinical_overlap_inference_v1.py", "analyze_modality_importance_v2.py",
    "task2_v2_config.py", "task2_v2_dataset.py", "task2_v2_model.py",
    "task2_v2_engine.py", "train_task2_v2.py", "fit_train_scalers_v2.py", "preprocessing_v2.py",
)


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clinical-catalog", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(MEMBERS), default=DEFAULT_GROUPS)
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
    if config.seed != 42:
        raise ValueError("This version requires a frozen training-seed-42 model")
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
        plan = build_domain_plan(json.loads(schema_path.read_text()), list(csv.DictReader(handle)))
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
        raise ValueError("domain Clinical plan differs from configured task/trained scaler order")
    model_config = am._build_model_config(config, dataset)
    am._validate_resolved_model_config(checkpoint, model_config)
    values, masks = materialize_clinical_bank(dataset)
    if len(set(args.groups)) != len(args.groups):
        raise ValueError("duplicate groups")
    if args.check_only:
        print(json.dumps({"task": config.dataset.task, "status": "lineage_and_domain_mapping_passed",
                          "n_val": len(dataset), "original_width": plan["original_width"],
                          "n_basic_columns": plan["n_basic_columns"], "plan_sha256": plan["plan_sha256"],
                          "removed_columns": {g: plan["groups"][g]["feature_count"] for g in args.groups}}, indent=2))
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
    _write_json(out_dir / "domain_plan.json", plan)
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
    print(f"[domain][{config.dataset.task}] Fresh raw-input encoding, n={len(dataset)}", flush=True)
    loader = am._loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=torch.device(device))
    cache = am.build_or_load_source_token_cache(
        model, loader, frame=dataset.index, device=device, task=config.dataset.task,
        expected_split="val", identity=identity, cache_path=cache_path, use_bf16_amp=args.bf16,
    )
    if cache.reused:
        raise RuntimeError("fresh rerun unexpectedly loaded an existing cache")
    result = run_domain_analysis(
        model, cache, values, masks, plan, groups=args.groups, task=config.dataset.task,
        threshold=threshold, batch_size=args.batch_size, device=device,
        use_bf16_amp=args.bf16, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    output_files = ["domain_plan.json", "domain_conditions.json"]
    _write_json(out_dir / "domain_conditions.json", result["conditions"])
    for key, suffix in (("summary", "summary"), ("intervals", "bootstrap_intervals"),
                        ("predictions", "paired_predictions"), ("cohort_counts", "cohort_counts")):
        name = f"domain_{suffix}.csv"
        result[key].to_csv(out_dir / name, index=False)
        output_files.append(name)
    metadata = {
        "schema_version": "ot2-domain-ablation-v1",
        "analysis_role": "validation_joint_information_removal",
        "task": config.dataset.task, "split": "val", "training_seed": config.seed,
        "bootstrap_seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "config_hash": config.config_hash, "config_sha256": am._sha256_file(args.config),
        "lineage": lineage, "clinical_schema_sha256": am._sha256_file(schema_path),
        "clinical_catalog_sha256": am._sha256_file(catalog_path), "plan_sha256": plan["plan_sha256"],
        "threshold": threshold, "threshold_source": "saved_full_teacher_validation_threshold",
        "n_participants": len(dataset), "groups": list(args.groups),
        "baseline_metrics": result["baseline_metrics"], "inference_checks": result["checks"],
        "n_retained_basic_columns": plan["n_basic_columns"],
        "source_token_cache": {"path": str(cache_path), "reused": False,
                               "raw_dataset_passes_this_run": 1,
                               "file_sha256": am._sha256_file(cache_path),
                               "content_sha256": cache.content_sha256, "identity": dict(cache.identity)},
        "scope_notes": plan["scope_notes"],
        "cohort_rule": "Pair baseline and perturbed predictions on the same people. Native target presence defines device strata; remaining input is assessed after Clinical cleanup. Families include any/all-target strata. any_target_information_present includes observed Clinical companions even if the target device is absent.",
        "limitations": [
            "Frozen-model input removal can be outside the training distribution; this is not retraining or reconstruction.",
            "Joint effects are model-dependent information-group effects, not exclusive modality importance or causal clinical effects.",
            "Intervals are pointwise 95% paired participant bootstrap intervals, without multiplicity adjustment or between-training-seed uncertainty; crossing zero does not establish noninferiority.",
        ],
        "output_sha256": {name: am._sha256_file(out_dir / name) for name in output_files},
        "code_sha256": {name: am._sha256_file(Path(__file__).parent / name) for name in CODE_FILES},
    }
    _write_json(out_dir / "domain_metadata.json", metadata)
    from summarize_domain_ablation_v1 import write_report
    if not write_report(out_dir, (config.dataset.task,), single_directory=out_dir):
        raise RuntimeError("Per-task report verification failed")
    print(json.dumps({"task": config.dataset.task, "output": str(out_dir), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
