"""Shared, standard-library-only contract for the basic Clinical retraining study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TASK_LABELS = dict(zip(TASKS, ("MoCA", "CVD broad", "CVD strict")))
MODALITIES = ("clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct",
              "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose")
GROUPS = {m: (m,) for m in MODALITIES}
GROUPS.update(retinal_all=MODALITIES[3:7], wearable_all=MODALITIES[7:], nonclinical_all=MODALITIES[1:])
METRICS = ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "log_loss", "brier")
SCOPES = ("all_participants", "any_target_present", "all_targets_present", "any_target_present_with_remaining_input")
CODE_FILES = (
    "basic_retrain_contract_v1.py", "prepare_basic_clinical_training_v1.py", "train_basic_clinical_v1.py",
    "analyze_basic_retrained_v1.py", "summarize_basic_retrain_v1.py", "run_basic_retrain_v1.py",
    "clinical_basic_information_v1.py", "analyze_modality_importance_v2.py", "task2_v2_config.py",
    "task2_v2_dataset.py", "task2_v2_model.py", "task2_v2_engine.py", "train_task2_v2.py",
    "fit_train_scalers_v2.py", "preprocessing_v2.py",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(content)
        name = f.name
    os.replace(name, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def code_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in CODE_FILES}


def assert_local_modules():
    import sys
    root = Path(__file__).resolve().parent
    expected = {Path(n).stem: n for n in CODE_FILES}
    for name, module in tuple(sys.modules.items()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in expected and getattr(module, "__file__", None):
            if Path(module.__file__).resolve() != root / expected[leaf]:
                raise RuntimeError(f"Mixed runtime code: {name}: {module.__file__}")


def verify_files(root, hashes):
    root = Path(root).resolve()
    for name, digest in hashes.items():
        p = (root / name).resolve(strict=True)
        if root not in p.parents or sha256(p) != digest:
            raise ValueError(f"Artifact hash/path mismatch: {name}")


def verify_prepared(path):
    path = Path(path)
    receipt = read_json(path / "PREPARED.json")
    if receipt["schema_version"] != "ot2-basic-retrain-inputs-v1" or receipt["clinical_width"] != 23:
        raise ValueError("Unexpected prepared-input contract")
    verify_files(path, receipt["output_sha256"])
    return receipt


def verify_training(path, prepared_hash=None):
    path = Path(path)
    receipt = read_json(path / "TRAINED.json")
    if receipt["schema_version"] != "ot2-basic-retrain-trained-v1" or receipt["clinical_width"] != 23:
        raise ValueError("Unexpected trained-model contract")
    if receipt["test_evaluated"] or receipt["initialization_checkpoint"] is not None:
        raise ValueError("Training must use train/validation and fresh task-model initialization")
    if prepared_hash and receipt["prepared_sha256"] != prepared_hash:
        raise ValueError("Training belongs to different prepared inputs")
    verify_files(path, receipt["output_sha256"])
    return receipt
