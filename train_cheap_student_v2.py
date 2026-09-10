#!/usr/bin/env python3
"""Train a deployment-safe cheap-modality OT2 V2 student.

The command deliberately separates three artifacts:

* a validation-selected, full-modality ``best_model.pt`` teacher;
* best/last *training* checkpoints containing the student and latent
  reconstructors (but never a duplicate teacher state); and
* ``deployment_student.pt``, which in deployable-reconstruction mode contains
  the pruned cheap-input student plus the small inference reconstructors.  That
  mode contains neither the teacher nor expensive native-encoder weights; the
  explicit legacy mode retains its historical student-only layout.

Training is validation-only and never requests a test batch.  Final test use
remains a separate post-lock operation.  Legacy artifacts retain their
existing evaluator path; opt-in reconstruction artifacts are loaded with
``load_deployment_inference_model`` so their reconstructors cannot be omitted.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from ot2.task2_v2_distill import (
        ExpensiveLatentReconstructor,
        LATENT_RECONSTRUCTION_MASK_SEMANTICS,
        LATENT_RECONSTRUCTION_TARGET_SPACE,
        LEGACY_DISTILLED_INFERENCE_MODE,
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE,
        Task2V2CheapStudentDistiller,
        Task2V2DeploymentStudent,
        Task2V2DistillationConfig,
    )
    from ot2.task2_v2_engine import (
        ENGINE_FORMAT_VERSION,
        EvaluationResult,
        atomic_torch_save,
        binary_classification_metrics,
        compute_train_pos_weight,
        evaluate_binary,
        metric_is_better,
        recursive_to_device,
        save_prediction_csv,
        select_validation_threshold,
        selection_metric_direction,
        set_deterministic_seed,
    )
    from ot2.task2_v2_model import (
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )
    from ot2.train_task2_v2 import (
        PRETRAIN_CHECKPOINT_FORMAT_VERSION,
        DatasetBundle,
        LoaderBundle,
        build_dataloaders,
        build_datasets,
        build_model_config,
        load_pretrained_student_checkpoint,
        load_supervised_config,
        pretraining_lineage_expectations,
        resolve_device,
    )
except ModuleNotFoundError:  # Direct execution from the ot2 directory.
    from task2_v2_distill import (  # type: ignore
        ExpensiveLatentReconstructor,
        LATENT_RECONSTRUCTION_MASK_SEMANTICS,
        LATENT_RECONSTRUCTION_TARGET_SPACE,
        LEGACY_DISTILLED_INFERENCE_MODE,
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE,
        Task2V2CheapStudentDistiller,
        Task2V2DeploymentStudent,
        Task2V2DistillationConfig,
    )
    from task2_v2_engine import (  # type: ignore
        ENGINE_FORMAT_VERSION,
        EvaluationResult,
        atomic_torch_save,
        binary_classification_metrics,
        compute_train_pos_weight,
        evaluate_binary,
        metric_is_better,
        recursive_to_device,
        save_prediction_csv,
        select_validation_threshold,
        selection_metric_direction,
        set_deterministic_seed,
    )
    from task2_v2_model import (  # type: ignore
        PRIMARY_MODALITY_GROUPS,
        Task2V2Model,
        Task2V2ModelConfig,
    )
    from train_task2_v2 import (  # type: ignore
        PRETRAIN_CHECKPOINT_FORMAT_VERSION,
        DatasetBundle,
        LoaderBundle,
        build_dataloaders,
        build_datasets,
        build_model_config,
        load_pretrained_student_checkpoint,
        load_supervised_config,
        pretraining_lineage_expectations,
        resolve_device,
    )


DISTILL_CHECKPOINT_FORMAT_VERSION = "ot2-task2-v2-cheap-distill-1"
DEPLOYMENT_CHECKPOINT_FORMAT_VERSION = "ot2-task2-v2-cheap-deployment-1"
RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION = (
    "ot2-task2-v2-cheap-deployment-2"
)
STUDENT_INITIALIZATIONS = frozenset({"random", "teacher_backbone", "pretrained"})


@dataclass(frozen=True)
class CheapStudentTrainingConfig:
    task: str
    cheap_groups: tuple[str, ...]
    max_epochs: int = 30
    gradient_accumulation_steps: int = 4
    grad_clip_norm: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_bf16_amp: bool = False
    use_pos_weight: bool = True
    selection_metric: str = "auroc"
    selection_min_delta: float = 0.0
    early_stopping_patience: int = 7
    threshold_method: str = "youden"
    fixed_threshold: float = 0.5
    seed: int = 42
    deterministic: bool = True
    evaluation_phase: str = "validation_screen"

    def __post_init__(self) -> None:
        unknown = sorted(set(self.cheap_groups) - set(PRIMARY_MODALITY_GROUPS))
        if unknown or not self.cheap_groups or len(set(self.cheap_groups)) != len(
            self.cheap_groups
        ):
            raise ValueError(
                "cheap_groups must be a non-empty, duplicate-free canonical subset; "
                f"unknown={unknown}"
            )
        if len(self.cheap_groups) == len(PRIMARY_MODALITY_GROUPS):
            raise ValueError("cheap_groups must omit at least one expensive modality")
        if self.max_epochs <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("max_epochs and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1]")
        if self.grad_clip_norm < 0 or self.selection_min_delta < 0:
            raise ValueError("gradient/selection tolerances cannot be negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.threshold_method not in {"fixed", "youden"}:
            raise ValueError("threshold_method must be fixed or youden")
        if not 0.0 <= self.fixed_threshold <= 1.0:
            raise ValueError("fixed_threshold must lie in [0, 1]")
        if self.evaluation_phase != "validation_screen":
            raise ValueError(
                "cheap-student training is validation-only; use the locked "
                "frozen-checkpoint evaluator for test"
            )
        selection_metric_direction(self.selection_metric)

    @property
    def expensive_groups(self) -> tuple[str, ...]:
        selected = set(self.cheap_groups)
        return tuple(group for group in PRIMARY_MODALITY_GROUPS if group not in selected)


@dataclass
class CheapStudentFitResult:
    best_epoch: int
    best_score: float
    epochs_completed: int
    stopped_early: bool
    pos_weight: Optional[float]
    selected_threshold: Mapping[str, Any]
    validation: EvaluationResult
    test: Optional[EvaluationResult]
    best_checkpoint: Path
    last_checkpoint: Path
    deployment_checkpoint: Path
    validation_predictions_csv: Path
    history: list[Dict[str, Any]]


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _deployment_architecture_signature(value: Any) -> Any:
    """Normalize architecture while excluding initialization-only weight sources."""

    normalized = _json_safe(value)
    if isinstance(normalized, Mapping):
        normalized = dict(normalized)
        normalized["retinal_weights"] = None
        normalized["retinal_weights_path"] = None
    return normalized


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_torch_load(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - unsupported old torch.
        raise RuntimeError(
            "secure checkpoint loading requires a PyTorch version with "
            "torch.load(..., weights_only=True) support"
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {resolved}")
    return dict(payload)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _state_dict_audit(
    state: Any, model: Task2V2Model, *, label: str
) -> Dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} is missing a mapping state dict")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{label} key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    audited: Dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}[{key!r}] is not a tensor")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"{label}[{key!r}] shape {tuple(value.shape)} != {tuple(target.shape)}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}[{key!r}] contains non-finite values")
        audited[key] = value
    return audited


def _is_expensive_native_encoder_key(
    key: str, cheap_groups: Sequence[str]
) -> bool:
    """Identify weights that are unnecessary under a fixed acquisition set."""

    cheap = set(str(group) for group in cheap_groups)
    if key.startswith("clinical_encoder."):
        return "clinical_data" not in cheap
    if key.startswith("ecg_encoder."):
        return "cardiac_ecg" not in cheap
    temporal_prefixes = {
        "environment": "temporal_encoders.environment.",
        "wearable_activity_monitor": (
            "temporal_encoders.wearable_activity_monitor."
        ),
        "wearable_blood_glucose": "temporal_encoders.wearable_blood_glucose.",
    }
    for group, prefix in temporal_prefixes.items():
        if key.startswith(prefix):
            return group not in cheap
    retinal = {
        "retinal_flio",
        "retinal_oct",
        "retinal_octa",
        "retinal_photography",
    }
    if key.startswith("retinal_encoder."):
        cheap_retinal = retinal.intersection(cheap)
        if not cheap_retinal:
            return True
        for group in retinal - cheap_retinal:
            if key.startswith(f"retinal_encoder.asset_pools.{group}.") or key.startswith(
                f"retinal_encoder.group_projections.{group}."
            ):
                return True
        # The shared backbone/view projection and cheap group pools are needed
        # whenever at least one retinal acquisition remains cheap.
        return False
    return False


def _deployment_student_state(
    model: Task2V2Model,
    cheap_groups: Sequence[str],
    *,
    task: str,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not _is_expensive_native_encoder_key(key, cheap_groups)
        and (
            not key.startswith("heads.")
            or key.startswith(f"heads.{task}.")
        )
    }


def prune_student_for_deployment(
    model: Task2V2Model,
    *,
    cheap_groups: Sequence[str],
    task: str,
) -> Task2V2Model:
    """Destructively retain only deployable native encoders, fusion, and one head.

    The returned module is still able to use the public ``encode_sources`` and
    ``predict_from_source_tokens`` APIs because omitted groups are disabled at
    the model boundary before attribute lookup.  Unlike a permissive
    ``strict=False`` load, omitted encoder modules are no longer registered and
    cannot appear in runtime ``named_modules()`` or ``state_dict()``.
    """

    cheap = set(str(group) for group in cheap_groups)
    unknown = cheap.difference(PRIMARY_MODALITY_GROUPS)
    if unknown or not cheap:
        raise ValueError(f"invalid deployment cheap groups: {sorted(unknown)}")
    if task not in model.heads:
        raise ValueError(f"deployment task head is missing: {task}")
    if "clinical_data" not in cheap:
        model._modules.pop("clinical_encoder", None)
    if "cardiac_ecg" not in cheap:
        model._modules.pop("ecg_encoder", None)
    for group in (
        "environment",
        "wearable_activity_monitor",
        "wearable_blood_glucose",
    ):
        if group not in cheap and group in model.temporal_encoders:
            del model.temporal_encoders[group]
    retinal_groups = {
        "retinal_flio",
        "retinal_oct",
        "retinal_octa",
        "retinal_photography",
    }
    cheap_retinal = retinal_groups.intersection(cheap)
    if not cheap_retinal:
        model._modules.pop("retinal_encoder", None)
    else:
        for group in retinal_groups.difference(cheap_retinal):
            if group in model.retinal_encoder.asset_pools:
                del model.retinal_encoder.asset_pools[group]
            if group in model.retinal_encoder.group_projections:
                del model.retinal_encoder.group_projections[group]
    for registered_task in tuple(model.heads):
        if registered_task != task:
            del model.heads[registered_task]
    return model


def _load_pruned_deployment_student_state(
    state: Any,
    model: Task2V2Model,
    *,
    cheap_groups: Sequence[str],
    task: str,
) -> Dict[str, torch.Tensor]:
    prune_student_for_deployment(model, cheap_groups=cheap_groups, task=task)
    audited = _state_dict_audit(
        state,
        model,
        label="pruned deployment student_state",
    )
    model.load_state_dict(audited, strict=True)
    return audited


def _reconstructor_state_audit(
    state: Any,
    reconstructors: torch.nn.ModuleDict,
    *,
    label: str,
) -> Dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} is missing a mapping state dict")
    expected = reconstructors.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{label} key mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    audited: Dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
            target.shape
        ):
            raise ValueError(f"{label}[{key!r}] tensor shape/type mismatch")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}[{key!r}] contains non-finite values")
        audited[key] = value
    return audited


def load_full_teacher_checkpoint(
    teacher: Task2V2Model,
    checkpoint_path: str | Path,
    *,
    expected_task: str,
    expected_config_hash: str,
    expected_model_config: Task2V2ModelConfig,
    expected_master_sha256: str,
    expected_scaler_sha256: str,
    expected_clinical_features_sha256: str,
    expected_data_file_sha256: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load a completed validation-selected full teacher with strict lineage."""

    path = Path(checkpoint_path).expanduser().resolve()
    if path.name != "best_model.pt" or not path.is_file():
        raise ValueError("teacher must be an existing validation-selected best_model.pt")
    payload = _safe_torch_load(path)
    if payload.get("engine_format_version") != ENGINE_FORMAT_VERSION:
        raise ValueError("teacher engine_format_version is unsupported")
    if payload.get("checkpoint_kind") != "best":
        raise ValueError("teacher checkpoint_kind must be best")
    if str(payload.get("task")) != expected_task:
        raise ValueError("teacher task does not match the requested distillation task")

    config_provenance = payload.get("config_provenance")
    data_provenance = payload.get("data_provenance")
    scaler_provenance = payload.get("scaler_provenance")
    if not all(
        isinstance(item, Mapping)
        for item in (config_provenance, data_provenance, scaler_provenance)
    ):
        raise ValueError("teacher is missing config/data/scaler provenance")
    if config_provenance.get("config_hash") != expected_config_hash:
        raise ValueError("teacher config hash differs from the current audited YAML")
    if _json_safe(config_provenance.get("resolved_model_config")) != _json_safe(
        expected_model_config
    ):
        raise ValueError("teacher resolved model architecture differs from the student")
    if data_provenance.get("task") != expected_task:
        raise ValueError("teacher data provenance task differs from the requested task")
    subset = config_provenance.get("modality_subset")
    if not isinstance(subset, Mapping):
        raise ValueError("teacher lacks modality_subset provenance")
    if list(subset.get("disabled_groups", ())) or set(
        subset.get("enabled_groups", ())
    ) != set(PRIMARY_MODALITY_GROUPS):
        raise ValueError("teacher is not a full nine-modality model")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping) or list(
        training_config.get("disabled_groups", ())
    ):
        raise ValueError("teacher training_config does not prove full-modality training")

    files = data_provenance.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("teacher data provenance lacks input file hashes")
    master = files.get("master_csv")
    if not isinstance(master, Mapping) or master.get("sha256") != expected_master_sha256:
        raise ValueError("teacher master CSV hash differs from the current input")
    for name, expected_hash in dict(expected_data_file_sha256 or {}).items():
        observed = files.get(name)
        if not isinstance(observed, Mapping) or observed.get("sha256") != expected_hash:
            raise ValueError(f"teacher {name} hash differs from the current input")
    if scaler_provenance.get("sha256") != expected_scaler_sha256:
        raise ValueError("teacher scaler hash differs from the current input")
    clinical_features = data_provenance.get("clinical_features_csv")
    if (
        not isinstance(clinical_features, Mapping)
        or clinical_features.get("read") is False
        or clinical_features.get("sha256")
        != expected_clinical_features_sha256
    ):
        raise ValueError(
            "teacher clinical wide CSV hash differs from the current input"
        )

    threshold = payload.get("selected_threshold")
    if not isinstance(threshold, Mapping):
        raise ValueError(
            "teacher lacks selected_threshold; training must have completed validation selection"
        )
    threshold_value = float(threshold.get("threshold", math.nan))
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("teacher validation-selected threshold is invalid")
    if threshold.get("method") not in {"fixed", "youden"}:
        raise ValueError("teacher validation-selected threshold method is invalid")
    if not isinstance(payload.get("validation_metrics_selected_threshold"), Mapping):
        raise ValueError("teacher lacks validation metrics at the selected threshold")
    selection_score = float(payload.get("selection_score", math.nan))
    if not math.isfinite(selection_score):
        raise ValueError("teacher validation selection score is non-finite")

    state = _state_dict_audit(payload.get("model_state"), teacher, label="teacher model_state")
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "epoch": int(payload.get("epoch", -1)),
        "selection_metric": payload.get("selection_metric"),
        "selection_score": selection_score,
        "selected_threshold": _json_safe(threshold),
        "config_hash": expected_config_hash,
        "master_csv_sha256": expected_master_sha256,
        "scaler_sha256": expected_scaler_sha256,
        "clinical_features_csv_sha256": expected_clinical_features_sha256,
        "full_modality_teacher": True,
    }


def initialize_student(
    student: Task2V2Model,
    teacher: Task2V2Model,
    *,
    method: str,
    task: str,
    pretrained_checkpoint: Optional[str | Path] = None,
    pretraining_lineage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize only the backbone; the downstream task head stays fresh."""

    method = str(method).strip().lower()
    if method not in STUDENT_INITIALIZATIONS:
        raise ValueError(f"student initialization must be one of {sorted(STUDENT_INITIALIZATIONS)}")
    head_before = {
        key: value.detach().clone()
        for key, value in student.state_dict().items()
        if key.startswith("heads.")
    }
    if method == "pretrained":
        if pretrained_checkpoint is None:
            raise ValueError("pretrained initialization requires --pretrained-checkpoint")
        audit = load_pretrained_student_checkpoint(
            student,
            pretrained_checkpoint,
            expected_task=task,
            expected_lineage=(
                pretraining_lineage
                if pretraining_lineage is not None
                else {}
            ),
        )
    elif method == "teacher_backbone":
        if pretrained_checkpoint is not None:
            raise ValueError("--pretrained-checkpoint is only valid with --student-init pretrained")
        selected = {
            key: value.detach().clone()
            for key, value in teacher.state_dict().items()
            if not key.startswith("heads.")
        }
        incompatibility = student.load_state_dict(selected, strict=False)
        expected_missing = sorted(
            key for key in student.state_dict() if key.startswith("heads.")
        )
        if sorted(incompatibility.missing_keys) != expected_missing or incompatibility.unexpected_keys:
            raise RuntimeError("teacher-backbone initialization produced an unexpected key mismatch")
        audit = {
            "kind": "validation_selected_teacher_backbone",
            "loaded_backbone_tensor_count": len(selected),
        }
    else:
        if pretrained_checkpoint is not None:
            raise ValueError("--pretrained-checkpoint is only valid with --student-init pretrained")
        audit = {"kind": "fresh_random_backbone"}
    head_after = {
        key: value
        for key, value in student.state_dict().items()
        if key.startswith("heads.")
    }
    if any(not torch.equal(value, head_after[key]) for key, value in head_before.items()):
        raise RuntimeError("student initialization unexpectedly changed a task head")
    return {
        **_json_safe(audit),
        "method": method,
        "task_head_initialization": "fresh_random_not_copied_from_teacher_or_pretraining",
        "pretraining_format_version": (
            PRETRAIN_CHECKPOINT_FORMAT_VERSION if method == "pretrained" else None
        ),
    }


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_distillation_epoch(
    distiller: Task2V2CheapStudentDistiller,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
    gradient_accumulation_steps: int,
    grad_clip_norm: float,
    use_bf16_amp: bool,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    """One official-train-only epoch; scheduler advances per optimizer step."""

    distiller.train()
    optimizer.zero_grad(set_to_none=True)
    batches = 0
    examples = 0
    optimizer_steps = 0
    component_names = (
        "supervised_bce",
        "temperature_scaled_logit_kd",
        "patient_embedding_mse",
        "expensive_latent_reconstruction",
    )
    component_weights = {
        "supervised_bce": float(distiller.config.supervised_weight),
        "temperature_scaled_logit_kd": float(distiller.config.logit_kd_weight),
        "patient_embedding_mse": float(
            distiller.config.patient_embedding_weight
        ),
        "expensive_latent_reconstruction": float(
            distiller.config.expensive_reconstruction_weight
        ),
    }
    epoch_numerators = {name: 0.0 for name in component_names}
    epoch_denominators = {name: 0 for name in component_names}
    count_metrics: Dict[str, float] = {}
    window_numerators: Dict[str, torch.Tensor] = {}
    window_denominators = {name: 0 for name in component_names}
    try:
        n_batches = len(loader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "train loader must define length so the final accumulation window is exact"
        ) from exc
    for batch_index, raw_batch in enumerate(loader, start=1):
        if not isinstance(raw_batch, Mapping):
            raise TypeError("train loader must yield mapping batches")
        if "split" in raw_batch and set(str(item) for item in raw_batch["split"]) != {"train"}:
            raise ValueError("distillation optimization may consume only split=train")
        batch = recursive_to_device(raw_batch, device, non_blocking=device.type == "cuda")
        labels = batch.get("y")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("training batch is missing tensor labels")
        with _autocast(device, use_bf16_amp):
            output = distiller(batch, pos_weight=pos_weight)
            batch_loss = output["loss"]
        if not bool(torch.isfinite(batch_loss.float())):
            raise FloatingPointError("non-finite distillation loss")
        numerators = _mapping(
            output.get("component_numerators"), "component_numerators"
        )
        denominators = _mapping(
            output.get("component_denominators"), "component_denominators"
        )
        if set(numerators) != set(component_names) or set(denominators) != set(
            component_names
        ):
            raise ValueError("distiller returned an incomplete component contract")
        for name in component_names:
            numerator = numerators[name]
            if not isinstance(numerator, torch.Tensor) or numerator.numel() != 1:
                raise ValueError(f"component numerator {name} must be scalar")
            denominator = int(denominators[name])
            if denominator < 0:
                raise ValueError(f"component denominator {name} cannot be negative")
            if denominator > 0 and not bool(torch.isfinite(numerator.float())):
                raise FloatingPointError(f"non-finite component numerator: {name}")
            window_numerators[name] = (
                numerator
                if name not in window_numerators
                else window_numerators[name] + numerator
            )
            window_denominators[name] += denominator
            epoch_numerators[name] += float(numerator.detach().float().cpu())
            epoch_denominators[name] += denominator
        last_batch = batch_index == n_batches
        if batch_index % gradient_accumulation_steps == 0 or last_batch:
            weighted_means = [
                window_numerators[name]
                / float(window_denominators[name])
                * component_weights[name]
                for name in component_names
                if component_weights[name] > 0
                and window_denominators[name] > 0
            ]
            if not weighted_means:
                raise ValueError(
                    "accumulation window has no eligible distillation target"
                )
            window_loss = torch.stack(weighted_means).sum()
            if not bool(torch.isfinite(window_loss.float())):
                raise FloatingPointError("non-finite accumulation-window loss")
            window_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in distiller.parameters() if p.requires_grad], grad_clip_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            optimizer_steps += 1
            window_numerators = {}
            window_denominators = {name: 0 for name in component_names}
        count = int(labels.numel())
        examples += count
        batches += 1
        for name, value in output["component_counts"].items():
            metric_name = f"count_{name}"
            count_metrics[metric_name] = count_metrics.get(metric_name, 0.0) + float(value)
        for group, counts in output["reconstruction_counts_by_modality"].items():
            metric_name = f"count_reconstruction_vectors_{group}"
            count_metrics[metric_name] = count_metrics.get(metric_name, 0.0) + float(
                counts["observed_teacher_vectors"]
            )
    if batches == 0 or examples == 0:
        raise ValueError("train loader yielded no batches")
    component_means = {
        name: (
            epoch_numerators[name] / float(epoch_denominators[name])
            if epoch_denominators[name] > 0
            else 0.0
        )
        for name in component_names
    }
    result = {
        "loss": sum(
            component_weights[name] * component_means[name]
            for name in component_names
        ),
        "examples": float(examples),
        "batches": float(batches),
        "optimizer_steps": float(optimizer_steps),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    result.update(component_means)
    result.update(count_metrics)
    result.update(
        {
            f"denominator_{name}": float(epoch_denominators[name])
            for name in component_names
        }
    )
    return result


def _scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(math.floor(total_steps * warmup_ratio))
    if warmup_ratio > 0 and total_steps > 0:
        warmup = max(1, warmup)

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        denominator = max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup + 1) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _training_checkpoint_payload(
    *,
    kind: str,
    distiller: Task2V2CheapStudentDistiller,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    training_config: CheapStudentTrainingConfig,
    distillation_config: Task2V2DistillationConfig,
    provenance: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    pos_weight: Optional[float],
) -> Dict[str, Any]:
    return {
        "format_version": DISTILL_CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": kind,
        "epoch": epoch,
        "task": training_config.task,
        "student_state": distiller.student.state_dict(),
        "reconstructor_state": distiller.expensive_reconstructors.state_dict(),
        "teacher_state_included": False,
        "training_config": _json_safe(training_config),
        "distillation_config": _json_safe(distillation_config),
        "deployment_contract": distiller.deployment_metadata(),
        "provenance": _json_safe(provenance),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "selection_metric": training_config.selection_metric,
        "selection_score": float(validation_metrics[training_config.selection_metric]),
        "selection_direction": selection_metric_direction(training_config.selection_metric),
        "pos_weight": pos_weight,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": [dict(item) for item in history],
    }


def _deployment_payload(
    *,
    distiller: Task2V2CheapStudentDistiller,
    model_config: Task2V2ModelConfig,
    task: str,
    threshold: Mapping[str, Any],
    provenance: Mapping[str, Any],
    best_epoch: int,
) -> Dict[str, Any]:
    deployable_reconstruction = bool(
        distiller.config.deployable_latent_reconstruction
    )
    complete_student_state = distiller.student.state_dict()
    student_state = (
        _deployment_student_state(
            distiller.student,
            distiller.cheap_groups,
            task=task,
        )
        if deployable_reconstruction
        else complete_student_state
    )
    payload: Dict[str, Any] = {
        "format_version": (
            RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION
            if deployable_reconstruction
            else DEPLOYMENT_CHECKPOINT_FORMAT_VERSION
        ),
        "checkpoint_kind": "deployment_student_only",
        "task": task,
        "experiment_model_id": distiller.deployment_metadata()[
            "experiment_model_id"
        ],
        "inference_mode": distiller.deployment_metadata()["inference_mode"],
        "best_epoch": int(best_epoch),
        "student_state": student_state,
        "resolved_model_config": _json_safe(model_config),
        "selected_threshold": _json_safe(threshold),
        "deployment_contract": distiller.deployment_metadata(),
        "provenance": _json_safe(provenance),
        "teacher_required": False,
        "reconstructors_required": deployable_reconstruction,
        "student_state_contract": {
            "pruned_expensive_native_encoders": deployable_reconstruction,
            "strict_pruned_runtime": deployable_reconstruction,
            "registered_native_encoder_groups": (
                list(distiller.cheap_groups)
                if deployable_reconstruction
                else list(PRIMARY_MODALITY_GROUPS)
            ),
            "registered_task_heads": (
                [task]
                if deployable_reconstruction
                else list(distiller.student.heads)
            ),
            "n_tensors_included": len(student_state),
            "n_tensors_in_full_student": len(complete_student_state),
            "excluded_expensive_groups": (
                list(distiller.expensive_groups)
                if deployable_reconstruction
                else []
            ),
            "shared_retinal_encoder_retained_only_if_needed_by_a_cheap_retinal_group": True,
        },
        "excluded_training_components": (
            ["teacher", "expensive_native_encoders"]
            if deployable_reconstruction
            else ["teacher", "expensive_latent_reconstructors"]
        ),
    }
    if deployable_reconstruction:
        payload["deployable_reconstructor_state"] = {
            key: value.detach().cpu()
            for key, value in distiller.expensive_reconstructors.state_dict().items()
        }
    return payload


def load_deployment_student_checkpoint(
    model: Task2V2Model,
    checkpoint_path: str | Path,
    *,
    expected_task: str,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Strictly load deployment weights and return the acquisition mask.

    For a deployable-reconstruction checkpoint this loads only the pruned
    cheap/fusion/head student state.  Use
    :func:`load_deployment_inference_model` to also construct and load the
    required latent reconstructors.
    """

    payload = _safe_torch_load(checkpoint_path)
    format_version = payload.get("format_version")
    supported_formats = {
        DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
        RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION,
    }
    if format_version not in supported_formats:
        raise ValueError("unsupported deployment checkpoint format")
    is_reconstructed_v2 = (
        format_version == RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION
    )
    if payload.get("checkpoint_kind") != "deployment_student_only":
        raise ValueError("checkpoint is not a student-only deployment artifact")
    if payload.get("task") != expected_task:
        raise ValueError("deployment task mismatch")
    if payload.get("teacher_required") is not False:
        raise ValueError("deployment artifact unexpectedly requires a teacher")
    if "teacher_state" in payload or "reconstructor_state" in payload:
        raise ValueError("deployment artifact embeds forbidden training-only state")
    if _deployment_architecture_signature(
        payload.get("resolved_model_config")
    ) != _deployment_architecture_signature(model.config):
        raise ValueError("deployment model architecture mismatch")
    contract = payload.get("deployment_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("deployment contract is missing")
    contract_body = {
        str(key): value
        for key, value in contract.items()
        if key != "deployment_contract_hash"
    }
    canonical_contract = json.dumps(
        contract_body, sort_keys=True, separators=(",", ":")
    )
    expected_contract_hash = hashlib.sha256(
        canonical_contract.encode("utf-8")
    ).hexdigest()
    if contract.get("deployment_contract_hash") != expected_contract_hash:
        raise ValueError("deployment contract hash is missing or invalid")
    if list(contract.get("primary_modality_order", ())) != list(
        PRIMARY_MODALITY_GROUPS
    ):
        raise ValueError("deployment primary modality order is incompatible")
    if is_reconstructed_v2:
        cheap = tuple(str(item) for item in contract.get("retained_input_groups", ()))
        expensive = tuple(str(item) for item in contract.get("removed_groups", ()))
    else:
        cheap = tuple(str(item) for item in contract.get("student_input_groups", ()))
        expensive = tuple(
            str(item)
            for item in contract.get("disabled_at_student_boundary", ())
        )
    deployable_reconstruction = contract.get(
        "deployable_latent_reconstruction", False
    )
    if not isinstance(deployable_reconstruction, bool):
        raise ValueError("deployment latent-reconstruction flag must be boolean")
    try:
        hidden_multiplier = int(contract.get("reconstructor_hidden_multiplier", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("deployment reconstructor hidden multiplier is invalid") from exc
    validated_contract = Task2V2DistillationConfig(
        task_name=expected_task,
        cheap_groups=cheap,
        expensive_groups=expensive,
        reconstructor_hidden_multiplier=hidden_multiplier,
        deployable_latent_reconstruction=deployable_reconstruction,
    )
    expected_model_id = (
        "distilled_student_latent_reconstruction"
        if deployable_reconstruction
        else "distilled_student"
    )
    expected_inference_mode = (
        RECONSTRUCTED_DISTILLED_INFERENCE_MODE
        if deployable_reconstruction
        else LEGACY_DISTILLED_INFERENCE_MODE
    )
    if is_reconstructed_v2 != deployable_reconstruction:
        raise ValueError(
            "deployment format and latent-reconstruction mode are inconsistent"
        )
    # V1 checkpoints created before these fields existed remain readable.  If
    # a V1 artifact does declare them, they are still checked fail-closed.
    strict_semantics = is_reconstructed_v2
    if (
        (strict_semantics or "experiment_model_id" in contract)
        and contract.get("experiment_model_id") != expected_model_id
    ) or (
        (strict_semantics or "experiment_model_id" in payload)
        and payload.get("experiment_model_id") != expected_model_id
    ):
        raise ValueError("deployment experiment_model_id is inconsistent")
    if (
        (strict_semantics or "inference_mode" in contract)
        and contract.get("inference_mode") != expected_inference_mode
    ) or (
        (strict_semantics or "inference_mode" in payload)
        and payload.get("inference_mode") != expected_inference_mode
    ):
        raise ValueError("deployment inference_mode is inconsistent")
    if (
        tuple(validated_contract.cheap_groups) != cheap
        or tuple(validated_contract.expensive_groups) != expensive
        or tuple(group for group in PRIMARY_MODALITY_GROUPS if group in set(cheap))
        != cheap
        or tuple(group for group in PRIMARY_MODALITY_GROUPS if group in set(expensive))
        != expensive
        or set(cheap).intersection(expensive)
        or set(cheap).union(expensive) != set(PRIMARY_MODALITY_GROUPS)
    ):
        raise ValueError(
            "deployment cheap/expensive groups are not an ordered partition"
        )
    if payload.get("reconstructors_required") is not deployable_reconstruction:
        raise ValueError(
            "deployment reconstructors_required disagrees with the hashed contract"
        )
    if (
        contract.get(
            "reconstructors_required_for_deployment",
            deployable_reconstruction,
        )
        is not deployable_reconstruction
    ):
        raise ValueError("deployment reconstructor contract is inconsistent")
    semantic_groups = {
        "retained_input_groups": cheap,
        "removed_groups": expensive,
        "acquired_source_groups": cheap,
        "native_encoder_groups": cheap,
        "reconstructed_source_groups": expensive if deployable_reconstruction else (),
        "reconstructed_groups": expensive if deployable_reconstruction else (),
        "reconstructed_fusion_groups": expensive if deployable_reconstruction else (),
    }
    for field, expected_groups in semantic_groups.items():
        if strict_semantics or field in contract:
            if tuple(contract.get(field, ())) != expected_groups:
                raise ValueError(f"deployment {field} is inconsistent")
    if strict_semantics and (
        tuple(contract.get("student_input_groups", ())) != cheap
        or tuple(contract.get("disabled_at_student_boundary", ())) != expensive
    ):
        raise ValueError("deployment V1 compatibility group aliases are inconsistent")
    if strict_semantics:
        if (
            contract.get("student_state_layout")
            != "pruned_retained_encoders_fusion_single_task_head"
            or list(contract.get("deployment_task_heads", ()))
            != [expected_task]
        ):
            raise ValueError("deployment pruned state semantics are inconsistent")
        if contract.get(
            "reconstruction_target_space"
        ) != LATENT_RECONSTRUCTION_TARGET_SPACE:
            raise ValueError("deployment reconstruction target space is inconsistent")
        if contract.get(
            "reconstruction_mask_semantics"
        ) != LATENT_RECONSTRUCTION_MASK_SEMANTICS:
            raise ValueError("deployment reconstruction mask semantics are inconsistent")

    if deployable_reconstruction:
        _load_pruned_deployment_student_state(
            payload.get("student_state"),
            model,
            cheap_groups=cheap,
            task=expected_task,
        )
        if not isinstance(payload.get("deployable_reconstructor_state"), Mapping):
            raise ValueError("deployment checkpoint lacks required reconstructors")
        state_contract = _mapping(
            payload.get("student_state_contract"), "deployment student state contract"
        )
        if state_contract.get("pruned_expensive_native_encoders") is not True:
            raise ValueError(
                "deployment state does not prove expensive native encoders were pruned"
            )
        if (
            state_contract.get("strict_pruned_runtime") is not True
            or tuple(state_contract.get("registered_native_encoder_groups", ()))
            != cheap
            or list(state_contract.get("registered_task_heads", ()))
            != [expected_task]
        ):
            raise ValueError("deployment pruned runtime contract is inconsistent")
    else:
        if "deployable_reconstructor_state" in payload:
            raise ValueError("legacy deployment unexpectedly embeds reconstructors")
        state = _state_dict_audit(
            payload.get("student_state"), model, label="student_state"
        )
        model.load_state_dict(state, strict=True)
    return payload, expensive


def load_deployment_inference_model(
    model: Task2V2Model,
    checkpoint_path: str | Path,
    *,
    expected_task: str,
) -> tuple[Task2V2DeploymentStudent, Mapping[str, Any]]:
    """Construct a teacher-free deployment module from one audited artifact."""

    payload, expensive = load_deployment_student_checkpoint(
        model, checkpoint_path, expected_task=expected_task
    )
    contract = _mapping(payload.get("deployment_contract"), "deployment contract")
    cheap = tuple(str(item) for item in contract["student_input_groups"])
    deployable = bool(contract.get("deployable_latent_reconstruction", False))
    hidden = int(contract.get("reconstructor_hidden_multiplier", 2))
    config = Task2V2DistillationConfig(
        task_name=expected_task,
        cheap_groups=cheap,
        expensive_groups=expensive,
        reconstructor_hidden_multiplier=hidden,
        deployable_latent_reconstruction=deployable,
    )
    reconstructors = torch.nn.ModuleDict(
        {
            group: ExpensiveLatentReconstructor(
                model.config.d_model,
                hidden,
                deployable=deployable,
            )
            for group in expensive
        }
    )
    if deployable:
        state = _reconstructor_state_audit(
            payload.get("deployable_reconstructor_state"),
            reconstructors,
            label="deployable_reconstructor_state",
        )
        reconstructors.load_state_dict(state, strict=True)
    # In legacy mode the modules are structurally present but never evaluated;
    # keeping one wrapper API avoids downstream conditionals.
    deployed = Task2V2DeploymentStudent(model, reconstructors, config)
    deployed.eval()
    return deployed, payload


def fit_cheap_student(
    distiller: Task2V2CheapStudentDistiller,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    test_loader: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    config: CheapStudentTrainingConfig,
    model_config: Task2V2ModelConfig,
    provenance: Mapping[str, Any],
    device: torch.device | str,
    train_labels: torch.Tensor,
) -> CheapStudentFitResult:
    """Optimize on train, select/threshold on val, and touch test only explicitly."""

    if tuple(config.cheap_groups) != tuple(distiller.cheap_groups):
        raise ValueError("training and distillation cheap-group contracts differ")
    if tuple(config.expensive_groups) != tuple(distiller.expensive_groups):
        raise ValueError("training and distillation expensive-group contracts differ")
    device = torch.device(device)
    set_deterministic_seed(config.seed, deterministic=config.deterministic)
    distiller.to(device)
    parameters = [parameter for parameter in distiller.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    try:
        batches_per_epoch = len(train_loader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("train_loader must define length for the cosine schedule") from exc
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    scheduler = _scheduler(
        optimizer,
        total_steps=steps_per_epoch * config.max_epochs,
        warmup_ratio=config.warmup_ratio,
    )
    pos_weight_tensor = (
        compute_train_pos_weight(train_labels, device=device)
        if config.use_pos_weight
        else None
    )
    pos_weight = (
        None if pos_weight_tensor is None else float(pos_weight_tensor.detach().cpu())
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_student_training.pt"
    last_path = output / "last_student_training.pt"
    deployment_path = output / "deployment_student.pt"
    validation_prediction_path = output / "validation_predictions.csv"
    history: list[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_epoch = 0
    no_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_distillation_epoch(
            distiller,
            train_loader,
            optimizer,
            device=device,
            pos_weight=pos_weight_tensor,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            grad_clip_norm=config.grad_clip_norm,
            use_bf16_amp=config.use_bf16_amp,
            scheduler=scheduler,
        )
        validation = evaluate_binary(
            distiller.deployment_model(),
            validation_loader,
            device=device,
            task=config.task,
            threshold=config.fixed_threshold,
            use_bf16_amp=config.use_bf16_amp,
            expected_split="val",
            disabled_groups=config.expensive_groups,
        )
        score = float(validation.metrics[config.selection_metric])
        record = {
            "epoch": epoch,
            "train": dict(train_metrics),
            "validation": dict(validation.metrics),
        }
        history.append(record)
        payload = _training_checkpoint_payload(
            kind="last",
            distiller=distiller,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            training_config=config,
            distillation_config=distiller.config,
            provenance=provenance,
            train_metrics=train_metrics,
            validation_metrics=validation.metrics,
            history=history,
            pos_weight=pos_weight,
        )
        atomic_torch_save(payload, last_path)
        if metric_is_better(
            score,
            best_score,
            metric=config.selection_metric,
            min_delta=config.selection_min_delta,
        ):
            best_score = score
            best_epoch = epoch
            no_improvement = 0
            payload["checkpoint_kind"] = "best"
            atomic_torch_save(payload, best_path)
        else:
            no_improvement += 1
            if no_improvement >= config.early_stopping_patience:
                stopped_early = True
                break
    if not best_path.is_file() or best_score is None:
        raise RuntimeError("distillation did not produce a validation-selected best student")

    best_payload = _safe_torch_load(best_path)
    student_state = _state_dict_audit(
        best_payload.get("student_state"), distiller.student, label="best student_state"
    )
    distiller.student.load_state_dict(student_state, strict=True)
    reconstructor_state = _reconstructor_state_audit(
        best_payload.get("reconstructor_state"),
        distiller.expensive_reconstructors,
        label="best reconstructor_state",
    )
    distiller.expensive_reconstructors.load_state_dict(
        reconstructor_state, strict=True
    )
    validation_fixed = evaluate_binary(
        distiller.deployment_model(),
        validation_loader,
        device=device,
        task=config.task,
        threshold=config.fixed_threshold,
        use_bf16_amp=config.use_bf16_amp,
        expected_split="val",
        disabled_groups=config.expensive_groups,
    )
    threshold = select_validation_threshold(
        validation_fixed.y_true,
        validation_fixed.probabilities,
        method=config.threshold_method,
        fixed_threshold=config.fixed_threshold,
        split="val",
    )
    threshold_payload = dataclasses.asdict(threshold)
    validation = evaluate_binary(
        distiller.deployment_model(),
        validation_loader,
        device=device,
        task=config.task,
        threshold=threshold.threshold,
        use_bf16_amp=config.use_bf16_amp,
        expected_split="val",
        disabled_groups=config.expensive_groups,
    )
    save_prediction_csv(
        validation,
        validation_prediction_path,
        task=config.task,
        split="val",
    )
    del test_loader  # A validation-only trainer must never iterate this object.
    test: Optional[EvaluationResult] = None
    (output / "test_predictions.csv").unlink(missing_ok=True)

    # The threshold belongs to the validation-selected *best* state.  Attaching
    # it to the last-epoch state would falsely imply that it was selected for
    # that different set of weights.
    enriched_best = _safe_torch_load(best_path)
    enriched_best["selected_threshold"] = threshold_payload
    enriched_best["validation_metrics_selected_threshold"] = validation.metrics
    enriched_best["test_evaluated"] = test is not None
    atomic_torch_save(enriched_best, best_path)
    deployment = _deployment_payload(
        distiller=distiller,
        model_config=model_config,
        task=config.task,
        threshold=threshold_payload,
        provenance=provenance,
        best_epoch=best_epoch,
    )
    atomic_torch_save(deployment, deployment_path)
    summary = {
        "format_version": DISTILL_CHECKPOINT_FORMAT_VERSION,
        "task": config.task,
        "experiment_model_id": distiller.deployment_metadata()[
            "experiment_model_id"
        ],
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "cheap_groups": list(config.cheap_groups),
        "expensive_groups": list(config.expensive_groups),
        "retained_input_groups": list(config.cheap_groups),
        "removed_groups": list(config.expensive_groups),
        "reconstructed_groups": (
            list(config.expensive_groups)
            if distiller.config.deployable_latent_reconstruction
            else []
        ),
        "deployable_latent_reconstruction": bool(
            distiller.config.deployable_latent_reconstruction
        ),
        "deployment_contract": distiller.deployment_metadata(),
        "selected_threshold": threshold_payload,
        "validation_metrics": validation.metrics,
        "validation_predictions": str(validation_prediction_path),
        "evaluation_phase": config.evaluation_phase,
        "test_evaluated": test is not None,
        "test_metrics": None if test is None else test.metrics,
        "pos_weight": pos_weight,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "deployment_checkpoint": str(deployment_path),
        "history": history,
        "provenance": _json_safe(provenance),
    }
    _atomic_json(summary, output / "run_summary.json")
    return CheapStudentFitResult(
        best_epoch=best_epoch,
        best_score=float(best_score),
        epochs_completed=len(history),
        stopped_early=stopped_early,
        pos_weight=pos_weight,
        selected_threshold=threshold_payload,
        validation=validation,
        test=test,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        deployment_checkpoint=deployment_path,
        validation_predictions_csv=validation_prediction_path,
        history=history,
    )


def _runtime_value(raw: Mapping[str, Any], key: str, override: Any, default: Any) -> Any:
    return raw.get(key, default) if override is None else override


def run_distillation(args: argparse.Namespace) -> CheapStudentFitResult | Dict[str, Any]:
    config_path, run_config = load_supervised_config(args.config)
    training_raw = dict(run_config.raw.get("training", {}))
    effective_seed = int(
        run_config.seed if getattr(args, "seed", None) is None else args.seed
    )
    if effective_seed < 0:
        raise ValueError("--seed must be non-negative")
    device = resolve_device(str(_runtime_value(training_raw, "device", args.device, "auto")))
    workers = int(
        _runtime_value(
            training_raw,
            "num_workers" if "num_workers" in training_raw else "workers",
            args.workers,
            0,
        )
    )
    bf16 = bool(_runtime_value(training_raw, "bf16", args.bf16, False))
    if bf16 and device.type not in {"cpu", "cuda"}:
        raise ValueError("bf16 is supported only on CPU/CUDA")
    set_deterministic_seed(effective_seed, deterministic=True)
    datasets = build_datasets(run_config)
    loaders = build_dataloaders(
        datasets,
        batch_size=int(
            run_config.training.batch_size
            if args.batch_size is None
            else args.batch_size
        ),
        workers=workers,
        seed=effective_seed,
        device=device,
    )
    model_config = build_model_config(run_config, datasets.feature_dimensions)
    teacher = Task2V2Model(model_config)
    teacher_data_paths = {
        "master_csv": run_config.dataset.master_csv,
        "master_metadata_json": run_config.dataset.master_metadata_json,
        "clinical_schema_json": run_config.dataset.clinical_schema_json,
        "environment_schema_json": run_config.dataset.environment_schema_json,
        "wearable_schema_json": run_config.dataset.wearable_schema_json,
        "ecg_schema_json": run_config.dataset.ecg_schema_json,
    }
    teacher_audit = load_full_teacher_checkpoint(
        teacher,
        args.teacher_checkpoint,
        expected_task=run_config.dataset.task,
        expected_config_hash=run_config.config_hash,
        expected_model_config=model_config,
        expected_master_sha256=_sha256_file(run_config.dataset.master_csv),
        expected_scaler_sha256=_sha256_file(run_config.dataset.scaler_json),
        expected_clinical_features_sha256=_sha256_file(
            datasets.train.clinical_features_path
        ),
        expected_data_file_sha256={
            name: _sha256_file(path) for name, path in teacher_data_paths.items()
        },
    )
    # Reseed so the fresh head is independent of incidental teacher construction.
    set_deterministic_seed(effective_seed + 1, deterministic=True)
    student = Task2V2Model(model_config)
    initialization = initialize_student(
        student,
        teacher,
        method=args.student_init,
        task=run_config.dataset.task,
        pretrained_checkpoint=args.pretrained_checkpoint,
        pretraining_lineage=pretraining_lineage_expectations(
            config_path, run_config, datasets.train
        ),
    )
    if len(set(args.cheap_groups)) != len(args.cheap_groups):
        raise ValueError("--cheap-groups cannot contain duplicates")
    cheap = tuple(
        group for group in PRIMARY_MODALITY_GROUPS if group in set(args.cheap_groups)
    )
    expensive = tuple(group for group in PRIMARY_MODALITY_GROUPS if group not in set(cheap))
    objective = Task2V2DistillationConfig(
        task_name=run_config.dataset.task,
        cheap_groups=cheap,
        expensive_groups=expensive,
        temperature=args.temperature,
        supervised_weight=args.supervised_weight,
        logit_kd_weight=args.logit_kd_weight,
        patient_embedding_weight=args.embedding_weight,
        expensive_reconstruction_weight=args.reconstruction_weight,
        reconstructor_hidden_multiplier=args.reconstructor_hidden_multiplier,
        deployable_latent_reconstruction=args.deployable_latent_reconstruction,
    )
    distiller = Task2V2CheapStudentDistiller(teacher, student, objective)
    training = CheapStudentTrainingConfig(
        task=run_config.dataset.task,
        cheap_groups=cheap,
        max_epochs=int(
            run_config.training.max_epochs
            if args.max_epochs is None
            else args.max_epochs
        ),
        gradient_accumulation_steps=int(
            run_config.training.gradient_accumulation_steps
            if args.gradient_accumulation_steps is None
            else args.gradient_accumulation_steps
        ),
        grad_clip_norm=float(training_raw.get("grad_clip_norm", 1.0)),
        learning_rate=float(
            run_config.training.learning_rate
            if args.learning_rate is None
            else args.learning_rate
        ),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else run_config.training.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        use_bf16_amp=bf16,
        use_pos_weight=bool(training_raw.get("use_pos_weight", True)),
        selection_metric=str(training_raw.get("selection_metric", run_config.training.selection_metric)),
        selection_min_delta=float(training_raw.get("selection_min_delta", 0.0)),
        early_stopping_patience=int(training_raw.get("early_stopping_patience", run_config.training.early_stopping_patience)),
        threshold_method=str(training_raw.get("threshold_method", "youden")),
        fixed_threshold=float(training_raw.get("fixed_threshold", 0.5)),
        seed=effective_seed,
        deterministic=bool(training_raw.get("deterministic", True)),
        evaluation_phase=args.evaluation_phase,
    )
    paths = {
        "config_yaml": config_path,
        "master_csv": run_config.dataset.master_csv,
        "master_metadata_json": run_config.dataset.master_metadata_json,
        "scaler_json": run_config.dataset.scaler_json,
        "clinical_schema_json": run_config.dataset.clinical_schema_json,
        "environment_schema_json": run_config.dataset.environment_schema_json,
        "wearable_schema_json": run_config.dataset.wearable_schema_json,
        "ecg_schema_json": run_config.dataset.ecg_schema_json,
        "clinical_features_csv": datasets.train.clinical_features_path,
    }
    provenance = {
        "teacher": teacher_audit,
        "student_initialization": initialization,
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "config_hash": run_config.config_hash,
        "resolved_model_config": _json_safe(model_config),
        "seed": effective_seed,
        "configured_seed": int(run_config.seed),
        "split_counts": {
            "train": len(datasets.train),
            "val": len(datasets.validation),
            "test": len(datasets.test),
        },
        "train_label_counts": {
            "negative": int((datasets.train.index["y"] == 0).sum()),
            "positive": int((datasets.train.index["y"] == 1).sum()),
            "source": "official_train_master_index_only",
        },
        "selection_policy": (
            "official train optimization; validation-only early stopping and threshold; "
            "test unopened unless evaluation_phase=final_test"
        ),
    }
    if args.dry_run:
        raw_batch = next(iter(loaders.train))
        batch = recursive_to_device(raw_batch, device)
        distiller.to(device)
        with _autocast(device, bf16):
            result = distiller(batch)
        result["loss"].backward()
        return {
            "status": "dry_run_ok",
            "task": training.task,
            "experiment_model_id": distiller.deployment_metadata()[
                "experiment_model_id"
            ],
            "cheap_groups": list(cheap),
            "expensive_groups": list(expensive),
            "retained_input_groups": list(cheap),
            "removed_groups": list(expensive),
            "reconstructed_groups": (
                list(expensive)
                if objective.deployable_latent_reconstruction
                else []
            ),
            "loss": float(result["loss"].detach().float().cpu()),
            "deployable_latent_reconstruction": bool(
                objective.deployable_latent_reconstruction
            ),
            "teacher": teacher_audit,
            "student_initialization": initialization,
            "validation_or_test_batches_iterated": False,
        }
    labels = torch.as_tensor(
        datasets.train.index["y"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    return fit_cheap_student(
        distiller,
        loaders.train,
        loaders.validation,
        loaders.test,
        output_dir=args.output_dir,
        config=training,
        model_config=model_config,
        provenance=provenance,
        device=device,
        train_labels=labels,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cheap-groups",
        "--retained-input-groups",
        dest="cheap_groups",
        nargs="+",
        required=True,
        choices=PRIMARY_MODALITY_GROUPS,
        help=(
            "Externally selected retained deployment inputs; every omitted "
            "canonical group is removed (legacy alias: --cheap-groups)"
        ),
    )
    parser.add_argument(
        "--student-init",
        choices=sorted(STUDENT_INITIALIZATIONS),
        default="teacher_backbone",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--logit-kd-weight", type=float, default=1.0)
    parser.add_argument("--embedding-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstructor-hidden-multiplier", type=int, default=2)
    parser.add_argument(
        "--deployable-latent-reconstruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject reconstructed expensive-source latent tokens into fusion at "
            "training and deployment as a separate experimental model arm "
            "(default: disabled; distilled_student remains auxiliary-only)."
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the YAML seed for a prespecified multi-seed replicate",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--evaluation-phase",
        choices=("validation_screen",),
        default="validation_screen",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _result_summary(result: CheapStudentFitResult) -> Dict[str, Any]:
    return {
        "status": "complete",
        "best_epoch": result.best_epoch,
        "best_score": result.best_score,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "pos_weight": result.pos_weight,
        "selected_threshold": result.selected_threshold,
        "validation_metrics": result.validation.metrics,
        "test_metrics": None if result.test is None else result.test.metrics,
        "test_evaluated": result.test is not None,
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
        "deployment_checkpoint": str(result.deployment_checkpoint),
        "validation_predictions": str(result.validation_predictions_csv),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_distillation(args)
    payload = result if isinstance(result, Mapping) else _result_summary(result)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheapStudentFitResult",
    "CheapStudentTrainingConfig",
    "DEPLOYMENT_CHECKPOINT_FORMAT_VERSION",
    "DISTILL_CHECKPOINT_FORMAT_VERSION",
    "RECONSTRUCTED_DEPLOYMENT_CHECKPOINT_FORMAT_VERSION",
    "build_arg_parser",
    "fit_cheap_student",
    "initialize_student",
    "load_deployment_inference_model",
    "load_deployment_student_checkpoint",
    "load_full_teacher_checkpoint",
    "prune_student_for_deployment",
    "main",
    "run_distillation",
    "train_distillation_epoch",
]
