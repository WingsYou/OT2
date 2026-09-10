#!/usr/bin/env python3
"""Reliable binary-task training/evaluation primitives for OT2 Model V2.

The engine is intentionally independent of the CLI and dataset builder.  It
accepts ordinary iterable loaders whose batches are mappings, and a model with
the :class:`task2_v2_model.Task2V2Model` calling convention.  The important
scientific safeguards live here rather than in a shell script:

* class weighting is estimated from the training loader only;
* model selection and early stopping use validation data only;
* decision thresholds are selected from the best model's validation outputs;
* the best checkpoint is unconditionally restored before final test scoring;
* both best and last checkpoints carry configuration/data/scaler provenance.

No command-line interface is defined in this module.  A future experiment CLI
can compose these functions without duplicating their selection logic.
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


ENGINE_FORMAT_VERSION = "ot2-task2-v2-engine-1"


@dataclasses.dataclass(frozen=True)
class BinaryTrainingConfig:
    """Training behavior that can affect model selection or reported results."""

    task: str
    max_epochs: int = 30
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    use_bf16_amp: bool = False
    use_pos_weight: bool = True
    selection_metric: str = "auroc"
    selection_min_delta: float = 0.0
    early_stopping_patience: int = 7
    threshold_method: str = "youden"
    fixed_threshold: float = 0.5
    seed: int = 42
    deterministic: bool = True
    # Fixed for the complete run.  This supports single-modality and
    # leave-one-modality-out retraining without mutating/censoring artifacts.
    disabled_groups: tuple[str, ...] = ()
    # ``validation_screen`` is mandatory for architecture/modality screening.
    # ``final_test`` is reserved for a prespecified, frozen final analysis.
    evaluation_phase: str = "final_test"

    def __post_init__(self) -> None:
        if not str(self.task).strip():
            raise ValueError("task must be a non-empty string")
        if int(self.max_epochs) <= 0:
            raise ValueError("max_epochs must be positive")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if float(self.grad_clip_norm) < 0:
            raise ValueError("grad_clip_norm cannot be negative")
        if float(self.selection_min_delta) < 0:
            raise ValueError("selection_min_delta cannot be negative")
        if int(self.early_stopping_patience) < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.threshold_method.lower() not in {"fixed", "youden"}:
            raise ValueError("threshold_method must be 'fixed' or 'youden'")
        if not 0.0 <= float(self.fixed_threshold) <= 1.0:
            raise ValueError("fixed_threshold must lie in [0, 1]")
        normalized_groups = tuple(str(value).strip() for value in self.disabled_groups)
        if any(not value for value in normalized_groups):
            raise ValueError("disabled_groups cannot contain empty names")
        if len(set(normalized_groups)) != len(normalized_groups):
            raise ValueError("disabled_groups cannot contain duplicates")
        if str(self.evaluation_phase).strip().lower() not in {
            "validation_screen",
            "final_test",
        }:
            raise ValueError(
                "evaluation_phase must be validation_screen or final_test"
            )
        # Validate the spelling now instead of silently selecting in the wrong
        # direction after an expensive run.
        selection_metric_direction(self.selection_metric)


@dataclasses.dataclass(frozen=True)
class CheckpointProvenance:
    """Immutable lineage recorded in every best/last checkpoint."""

    config: Mapping[str, Any]
    data: Mapping[str, Any]
    scaler: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("config", "data", "scaler"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} provenance must be a mapping")

    def as_checkpoint_fields(self) -> Dict[str, Dict[str, Any]]:
        return {
            "config_provenance": dict(self.config),
            "data_provenance": dict(self.data),
            "scaler_provenance": dict(self.scaler),
        }


@dataclasses.dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    method: str
    objective: float
    fallback_reason: Optional[str] = None


@dataclasses.dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    person_ids: list[str]
    y_true: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    threshold: float


@dataclasses.dataclass
class FitResult:
    best_epoch: int
    best_score: float
    epochs_completed: int
    stopped_early: bool
    pos_weight: Optional[float]
    threshold_selection: ThresholdSelection
    validation: EvaluationResult
    test: Optional[EvaluationResult]
    history: list[Dict[str, Any]]
    best_checkpoint: Path
    last_checkpoint: Path
    validation_predictions_csv: Path
    predictions_csv: Optional[Path]


def set_deterministic_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch and configure deterministic torch kernels."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def recursive_to_device(
    value: Any,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Any:
    """Move tensors in arbitrarily nested batches without touching metadata."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: recursive_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(
            *(recursive_to_device(item, device, non_blocking=non_blocking) for item in value)
        )
    if isinstance(value, tuple):
        return tuple(
            recursive_to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            recursive_to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    return value


def _as_1d_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    result = value.detach().to(dtype=torch.float32).reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if not torch.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain binary 0/1 labels")
    return result


def _batch_strings(value: Any) -> list[str]:
    if isinstance(value, torch.Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def collect_train_labels(
    train_loader: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "y",
) -> torch.Tensor:
    """Read labels from a train loader and reject an explicitly non-train batch.

    The split check is intentionally strict when batches expose ``split``.  It
    makes an accidentally concatenated train+validation loader fail loudly
    instead of leaking validation prevalence into ``pos_weight``.
    """

    labels: list[torch.Tensor] = []
    for batch in train_loader:
        if not isinstance(batch, Mapping):
            raise TypeError("train_loader must yield mapping batches")
        if "split" in batch:
            splits = {item.strip().lower() for item in _batch_strings(batch["split"])}
            if splits != {"train"}:
                raise ValueError(
                    "pos_weight input contains a non-train split: "
                    f"{sorted(splits)}"
                )
        if label_key not in batch:
            raise KeyError(f"training batch is missing label key {label_key!r}")
        labels.append(_as_1d_float_tensor(batch[label_key], name="train labels"))
    if not labels:
        raise ValueError("train_loader yielded no labels")
    return torch.cat(labels)


def compute_train_pos_weight(
    train_labels: Any,
    *,
    device: torch.device | str | None = None,
) -> Optional[torch.Tensor]:
    """Return ``n_negative / n_positive`` from *training labels only*.

    Single-class training data cannot define a meaningful finite class ratio;
    in that case the safe behavior is unweighted BCE (``None``), not zero or
    infinity.
    """

    labels = _as_1d_float_tensor(train_labels, name="train labels")
    n_positive = int((labels == 1).sum().item())
    n_negative = int((labels == 0).sum().item())
    if n_positive == 0 or n_negative == 0:
        return None
    return torch.tensor(
        float(n_negative) / float(n_positive),
        dtype=torch.float32,
        device=device,
    )


def selection_metric_direction(metric: str) -> str:
    """Return whether a validation selection metric is maximized or minimized."""

    name = str(metric).strip().lower()
    if name in {"loss", "log_loss", "brier"}:
        return "min"
    if name in {
        "auroc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    }:
        return "max"
    raise ValueError(f"unsupported selection metric {metric!r}")


def metric_is_better(
    candidate: float,
    incumbent: Optional[float],
    *,
    metric: str,
    min_delta: float = 0.0,
) -> bool:
    """NaN-safe improvement check with the metric's correct direction."""

    candidate = float(candidate)
    if incumbent is None:
        # Epoch one must produce a recoverable best checkpoint even if a tiny
        # validation pilot contains only one class and AUROC is undefined.
        return True
    incumbent = float(incumbent)
    if not math.isfinite(candidate):
        return False
    if not math.isfinite(incumbent):
        return True
    delta = float(min_delta)
    if selection_metric_direction(metric) == "min":
        return candidate < incumbent - delta
    return candidate > incumbent + delta


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def binary_classification_metrics(
    y_true: Any,
    *,
    logits: Any | None = None,
    probabilities: Any | None = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute ranking, calibration and threshold metrics for a binary task.

    AUROC/AUPRC and class-specific rates are ``nan`` when the necessary class
    is absent.  Log loss and Brier score remain defined for single-class data.
    """

    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        raise ValueError("y_true cannot be empty")
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain finite binary 0/1 labels")
    if (logits is None) == (probabilities is None):
        raise ValueError("provide exactly one of logits or probabilities")
    if probabilities is None:
        raw_logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if raw_logits.shape != labels.shape or not np.isfinite(raw_logits).all():
            raise ValueError("logits must be finite and match y_true")
        probs = _sigmoid_numpy(raw_logits)
    else:
        probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if probs.shape != labels.shape or not np.isfinite(probs).all():
            raise ValueError("probabilities must be finite and match y_true")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    log_loss = -np.mean(
        labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)
    )
    brier = np.mean((probs - labels) ** 2)
    predicted = (probs >= float(threshold)).astype(np.int64)
    truth = labels.astype(np.int64)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced = (
        0.5 * (sensitivity + specificity)
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )

    auroc = math.nan
    auprc = math.nan
    if np.unique(truth).size == 2:
        # scikit-learn is an explicit model dependency and provides correct
        # tie handling for both ranking metrics.
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(truth, probs))
        auprc = float(average_precision_score(truth, probs))

    return {
        "n": float(labels.size),
        "prevalence": float(labels.mean()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": float(balanced),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": float(threshold),
    }


def select_validation_threshold(
    y_true: Any,
    probabilities: Any,
    *,
    method: str = "youden",
    fixed_threshold: float = 0.5,
    split: str = "val",
) -> ThresholdSelection:
    """Choose a fixed or Youden threshold using validation predictions only."""

    if str(split).strip().lower() not in {"val", "validation"}:
        raise ValueError("decision thresholds may only be selected on validation data")
    method = str(method).strip().lower()
    fixed = float(fixed_threshold)
    if method not in {"fixed", "youden"}:
        raise ValueError("method must be 'fixed' or 'youden'")
    if not 0.0 <= fixed <= 1.0:
        raise ValueError("fixed_threshold must lie in [0, 1]")
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probs.shape or labels.size == 0:
        raise ValueError("validation labels/probabilities must be non-empty and aligned")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probs).all():
        raise ValueError("validation labels/probabilities are invalid")
    if method == "fixed":
        return ThresholdSelection(fixed, method, math.nan)
    if np.unique(labels).size < 2:
        return ThresholdSelection(
            fixed,
            method,
            math.nan,
            "youden_requires_both_classes",
        )

    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probs)
    objective = true_positive_rate - false_positive_rate
    finite = np.isfinite(thresholds)
    if not finite.any():
        return ThresholdSelection(fixed, method, math.nan, "no_finite_threshold")
    best_objective = float(np.max(objective[finite]))
    candidates = thresholds[finite & np.isclose(objective, best_objective)]
    # Stable tie break: select the candidate nearest the predeclared fixed
    # threshold, then the smaller value.  No test outcomes enter this choice.
    order = np.lexsort((candidates, np.abs(candidates - fixed)))
    selected = float(np.clip(candidates[order[0]], 0.0, 1.0))
    return ThresholdSelection(selected, method, best_objective)


def _model_logits(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    task: Optional[str],
    disabled_groups: Sequence[str] = (),
) -> torch.Tensor:
    forward_kwargs: Dict[str, Any] = {}
    if disabled_groups:
        forward_kwargs["disabled_groups"] = tuple(disabled_groups)
    output = (
        model(batch, task=task, **forward_kwargs)
        if task is not None
        else model(batch, **forward_kwargs)
    )
    if isinstance(output, Mapping):
        value = output.get("logits", output.get("y_logit"))
    else:
        value = output
    if not isinstance(value, torch.Tensor):
        raise TypeError("model output must be a tensor or contain logits/y_logit")
    logits = value.reshape(-1)
    if logits.numel() == 0:
        raise ValueError("model returned empty logits")
    return logits


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"bf16 autocast is unsupported for device type {device.type!r}")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("use_bf16_amp=True but this CUDA device lacks bf16 support")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_binary_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    task: Optional[str],
    pos_weight: Optional[torch.Tensor] = None,
    gradient_accumulation_steps: int = 1,
    grad_clip_norm: float = 0.0,
    use_bf16_amp: bool = False,
    disabled_groups: Sequence[str] = (),
) -> Dict[str, float]:
    """Train one epoch with accumulation, clipping and optional bf16 AMP."""

    accumulation = int(gradient_accumulation_steps)
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        n_batches = len(loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        n_batches = None
    total_loss = 0.0
    total_examples = 0
    batches = 0
    optimizer_steps = 0

    for batch_index, raw_batch in enumerate(loader):
        if not isinstance(raw_batch, Mapping):
            raise TypeError("loader must yield mapping batches")
        batch = recursive_to_device(raw_batch, device, non_blocking=device.type == "cuda")
        labels = _as_1d_float_tensor(batch.get("y"), name="batch labels").to(device)
        if n_batches is not None:
            remainder = n_batches % accumulation
            final_group_start = n_batches - remainder if remainder else n_batches
            divisor = (
                remainder
                if remainder and batch_index >= final_group_start
                else accumulation
            )
        else:
            divisor = accumulation
        with _autocast_context(device, use_bf16_amp):
            logits = _model_logits(
                model, batch, task=task, disabled_groups=disabled_groups
            )
            if logits.shape != labels.shape:
                raise ValueError(
                    f"logits shape {tuple(logits.shape)} != labels {tuple(labels.shape)}"
                )
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels, pos_weight=pos_weight
            )
        (loss / float(divisor)).backward()
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_examples += labels.numel()
        batches += 1

        is_boundary = (batch_index + 1) % accumulation == 0
        is_last = n_batches is not None and batch_index + 1 == n_batches
        if is_boundary or is_last:
            if float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip_norm)
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    if batches == 0:
        raise ValueError("training loader yielded no batches")
    if n_batches is None and batches % accumulation:
        # Unknown-length iterables cannot announce their final microbatch.  The
        # remaining gradients still need one optimizer step.  They were each
        # divided by the full accumulation count, so rescale the partial group
        # to the mean over the microbatches that actually arrived.
        partial = batches % accumulation
        correction = float(accumulation) / float(partial)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "loss": total_loss / float(total_examples),
        "n": float(total_examples),
        "optimizer_steps": float(optimizer_steps),
    }


@torch.no_grad()
def evaluate_binary(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    task: Optional[str],
    threshold: float = 0.5,
    use_bf16_amp: bool = False,
    expected_split: Optional[str] = None,
    disabled_groups: Sequence[str] = (),
) -> EvaluationResult:
    """Score a loader and retain aligned per-person predictions."""

    device = torch.device(device)
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    total_bce = 0.0
    total_examples = 0
    generated_id = 0
    for raw_batch in loader:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("loader must yield mapping batches")
        if expected_split is not None and "split" in raw_batch:
            splits = {item.strip().lower() for item in _batch_strings(raw_batch["split"])}
            if splits != {expected_split.strip().lower()}:
                raise ValueError(
                    f"expected {expected_split!r} loader, found splits {sorted(splits)}"
                )
        batch = recursive_to_device(raw_batch, device, non_blocking=device.type == "cuda")
        labels = _as_1d_float_tensor(batch.get("y"), name="batch labels").to(device)
        with _autocast_context(device, use_bf16_amp):
            logits = _model_logits(
                model, batch, task=task, disabled_groups=disabled_groups
            )
            if logits.shape != labels.shape:
                raise ValueError(
                    f"logits shape {tuple(logits.shape)} != labels {tuple(labels.shape)}"
                )
            bce = F.binary_cross_entropy_with_logits(
                logits.float(), labels, reduction="sum"
            )
        total_bce += float(bce.cpu())
        total_examples += labels.numel()
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if "person_id" in raw_batch:
            identifiers = _batch_strings(raw_batch["person_id"])
            if len(identifiers) != labels.numel():
                raise ValueError("person_id count does not match batch labels")
        else:
            identifiers = [
                f"generated_{index:09d}"
                for index in range(generated_id, generated_id + labels.numel())
            ]
            generated_id += labels.numel()
        all_ids.extend(identifiers)

    if not all_labels:
        raise ValueError("evaluation loader yielded no batches")
    y_true = np.concatenate(all_labels).astype(np.int64)
    logits_array = np.concatenate(all_logits).astype(np.float64)
    probabilities = _sigmoid_numpy(logits_array)
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = binary_classification_metrics(
        y_true, probabilities=probabilities, threshold=threshold
    )
    # This independently accumulated loss is useful for auditing mixed batch
    # sizes and should agree with log_loss up to floating-point precision.
    metrics["loss"] = total_bce / float(total_examples)
    return EvaluationResult(
        metrics=metrics,
        person_ids=all_ids,
        y_true=y_true,
        logits=logits_array,
        probabilities=probabilities,
        predictions=predictions,
        threshold=float(threshold),
    )


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a torch checkpoint atomically on the destination filesystem."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return dict(payload)


def save_prediction_csv(
    result: EvaluationResult,
    destination: str | Path,
    *,
    task: str,
    split: str = "test",
) -> Path:
    """Atomically save one aligned prediction row per person."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(result.person_ids) != result.y_true.size:
        raise ValueError("person IDs and prediction arrays are not aligned")
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "person_id",
                    "task",
                    "split",
                    "y_true",
                    "y_logit",
                    "y_probability",
                    "y_pred",
                    "threshold",
                ),
            )
            writer.writeheader()
            for person_id, label, logit, probability, prediction in zip(
                result.person_ids,
                result.y_true,
                result.logits,
                result.probabilities,
                result.predictions,
            ):
                writer.writerow(
                    {
                        "person_id": person_id,
                        "task": task,
                        "split": split,
                        "y_true": int(label),
                        "y_logit": float(logit),
                        "y_probability": float(probability),
                        "y_pred": int(prediction),
                        "threshold": float(result.threshold),
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def _checkpoint_payload(
    *,
    kind: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: BinaryTrainingConfig,
    provenance: CheckpointProvenance,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    pos_weight: Optional[float],
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "engine_format_version": ENGINE_FORMAT_VERSION,
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "task": config.task,
        "seed": int(config.seed),
        "training_config": dataclasses.asdict(config),
        **provenance.as_checkpoint_fields(),
        "selection_metric": config.selection_metric,
        "selection_direction": selection_metric_direction(config.selection_metric),
        "selection_score": float(validation_metrics[config.selection_metric]),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "pos_weight": pos_weight,
        "model_state": _unwrapped(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": [dict(item) for item in history],
    }


def fit_evaluate_binary(
    model: torch.nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    test_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    output_dir: str | Path,
    config: BinaryTrainingConfig,
    provenance: CheckpointProvenance,
    device: torch.device | str = "cpu",
    train_labels: Any | None = None,
) -> FitResult:
    """Fit, select on validation, restore best, and score test exactly once."""

    if not isinstance(config, BinaryTrainingConfig):
        raise TypeError("config must be BinaryTrainingConfig")
    if not isinstance(provenance, CheckpointProvenance):
        raise TypeError("provenance must be CheckpointProvenance")
    device = torch.device(device)
    set_deterministic_seed(config.seed, deterministic=config.deterministic)
    model.to(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best_model.pt"
    last_path = destination / "last_model.pt"
    validation_prediction_path = destination / "validation_predictions.csv"
    prediction_path = destination / "test_predictions.csv"

    pos_weight_tensor: Optional[torch.Tensor] = None
    if config.use_pos_weight:
        labels = (
            collect_train_labels(train_loader)
            if train_labels is None
            else _as_1d_float_tensor(train_labels, name="train labels")
        )
        pos_weight_tensor = compute_train_pos_weight(labels, device=device)
    pos_weight_value = (
        None if pos_weight_tensor is None else float(pos_weight_tensor.detach().cpu())
    )

    history: list[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_binary_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            task=config.task,
            pos_weight=pos_weight_tensor,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            grad_clip_norm=config.grad_clip_norm,
            use_bf16_amp=config.use_bf16_amp,
            disabled_groups=config.disabled_groups,
        )
        validation = evaluate_binary(
            model,
            validation_loader,
            device=device,
            task=config.task,
            threshold=config.fixed_threshold,
            use_bf16_amp=config.use_bf16_amp,
            expected_split="val",
            disabled_groups=config.disabled_groups,
        )
        if config.selection_metric not in validation.metrics:
            raise KeyError(
                f"selection metric {config.selection_metric!r} missing from validation"
            )
        score = float(validation.metrics[config.selection_metric])
        record: Dict[str, Any] = {
            "epoch": epoch,
            "train": dict(train_metrics),
            "validation": dict(validation.metrics),
        }
        history.append(record)
        payload = _checkpoint_payload(
            kind="last",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            provenance=provenance,
            train_metrics=train_metrics,
            validation_metrics=validation.metrics,
            pos_weight=pos_weight_value,
            history=history,
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
            epochs_without_improvement = 0
            payload["checkpoint_kind"] = "best"
            atomic_torch_save(payload, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break

    if not best_path.is_file() or best_epoch == 0 or best_score is None:
        raise RuntimeError("training did not produce a best checkpoint")

    # Mandatory safeguard: last_model remains the final optimization state,
    # while all threshold selection and test scoring use best_model.
    best_payload = _load_checkpoint(best_path, device)
    _unwrapped(model).load_state_dict(best_payload["model_state"], strict=True)

    validation_best_fixed = evaluate_binary(
        model,
        validation_loader,
        device=device,
        task=config.task,
        threshold=config.fixed_threshold,
        use_bf16_amp=config.use_bf16_amp,
        expected_split="val",
        disabled_groups=config.disabled_groups,
    )
    threshold_selection = select_validation_threshold(
        validation_best_fixed.y_true,
        validation_best_fixed.probabilities,
        method=config.threshold_method,
        fixed_threshold=config.fixed_threshold,
        split="val",
    )
    validation_best = evaluate_binary(
        model,
        validation_loader,
        device=device,
        task=config.task,
        threshold=threshold_selection.threshold,
        use_bf16_amp=config.use_bf16_amp,
        expected_split="val",
        disabled_groups=config.disabled_groups,
    )
    # Validation predictions are part of the auditable model/subset-selection
    # evidence.  Persisting participant-aligned probabilities lets the later
    # cost/Pareto stage perform paired comparisons without reopening test.
    save_prediction_csv(
        validation_best,
        validation_prediction_path,
        task=config.task,
        split="val",
    )
    test: Optional[EvaluationResult]
    saved_prediction_path: Optional[Path]
    if str(config.evaluation_phase).strip().lower() == "final_test":
        test = evaluate_binary(
            model,
            test_loader,
            device=device,
            task=config.task,
            threshold=threshold_selection.threshold,
            use_bf16_amp=config.use_bf16_amp,
            expected_split="test",
            disabled_groups=config.disabled_groups,
        )
        saved_prediction_path = save_prediction_csv(
            test, prediction_path, task=config.task, split="test"
        )
    else:
        # Never even request the first test batch during model/subset/seed
        # screening.  Remove a stale file if this output directory was reused
        # so downstream tooling cannot mistake it for the current run.
        test = None
        saved_prediction_path = None
        prediction_path.unlink(missing_ok=True)

    # Enrich the already-selected best checkpoint without changing its state.
    best_payload["selected_threshold"] = dataclasses.asdict(threshold_selection)
    best_payload["validation_metrics_selected_threshold"] = validation_best.metrics
    atomic_torch_save(best_payload, best_path)
    _atomic_json_save(
        {
            "engine_format_version": ENGINE_FORMAT_VERSION,
            "task": config.task,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_metric": config.selection_metric,
            "selection_direction": selection_metric_direction(config.selection_metric),
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "pos_weight": pos_weight_value,
            "threshold_selection": dataclasses.asdict(threshold_selection),
            "validation_metrics": validation_best.metrics,
            "validation_predictions": str(validation_prediction_path),
            "evaluation_phase": config.evaluation_phase,
            "test_evaluated": test is not None,
            "test_metrics": None if test is None else test.metrics,
            **provenance.as_checkpoint_fields(),
            "history": history,
        },
        destination / "run_summary.json",
    )

    return FitResult(
        best_epoch=best_epoch,
        best_score=float(best_score),
        epochs_completed=len(history),
        stopped_early=stopped_early,
        pos_weight=pos_weight_value,
        threshold_selection=threshold_selection,
        validation=validation_best,
        test=test,
        history=history,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        validation_predictions_csv=validation_prediction_path,
        predictions_csv=saved_prediction_path,
    )


__all__ = [
    "ENGINE_FORMAT_VERSION",
    "BinaryTrainingConfig",
    "CheckpointProvenance",
    "EvaluationResult",
    "FitResult",
    "ThresholdSelection",
    "atomic_torch_save",
    "binary_classification_metrics",
    "collect_train_labels",
    "compute_train_pos_weight",
    "evaluate_binary",
    "fit_evaluate_binary",
    "metric_is_better",
    "recursive_to_device",
    "save_prediction_csv",
    "select_validation_threshold",
    "selection_metric_direction",
    "set_deterministic_seed",
    "train_binary_epoch",
]
