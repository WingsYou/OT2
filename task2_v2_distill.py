#!/usr/bin/env python3
"""Teacher-to-cheap-modality distillation for the OT2 Task2 V2 model.

This module is deliberately a training *core*, not a command-line runner.  A
full nine-modality :class:`Task2V2Model` acts as a frozen teacher while a
second model is trained using an explicit, deployment-time subset of cheap
modalities.  The student path always disables every non-cheap modality at the
model boundary, even when those tensors are present in the training batch.

The distillation targets are representations, never raw images or raw time
series.  Expensive source tokens are predicted from the cheap-only patient
embedding, and every distillation loss is normalized only over patients for
whom at least one relevant teacher source is actually observed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:  # Package import when used as ``ot2.task2_v2_distill``.
    from .task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model
except ImportError:  # Script-style import used by the repository tests.
    from task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model


# These fields may supervise an objective, but must never become model inputs.
# Task-specific raw label concepts must already have been removed by the
# audited clinical preprocessing/schema stage.
OUTCOME_ONLY_BATCH_KEYS = frozenset(
    {
        "y",
        "label",
        "labels",
        "target",
        "targets",
        "outcome",
        "outcomes",
    }
)

LEGACY_DISTILLED_INFERENCE_MODE = "cheap_only_single_pass_missing_tokens"
RECONSTRUCTED_DISTILLED_INFERENCE_MODE = (
    "cheap_only_two_pass_latent_reconstruction"
)
LATENT_RECONSTRUCTION_TARGET_SPACE = "teacher_native_source_token_pre_fusion"
LATENT_RECONSTRUCTION_MASK_SEMANTICS = (
    "eligible iff any effective acquired source is present; synthetic "
    "source_present is true only for eligible rows; observed_source_present "
    "always records native acquisition support"
)


@dataclasses.dataclass(frozen=True)
class Task2V2DistillationConfig:
    """Objective and deployment contract for cheap-modality distillation."""

    task_name: str
    cheap_groups: tuple[str, ...]
    expensive_groups: tuple[str, ...]
    temperature: float = 2.0
    supervised_weight: float = 1.0
    logit_kd_weight: float = 1.0
    patient_embedding_weight: float = 1.0
    expensive_reconstruction_weight: float = 1.0
    reconstructor_hidden_multiplier: int = 2
    # When enabled, reconstructed expensive-source tokens are part of the
    # supervised/KD prediction path and remain in the deployment artifact.
    # False preserves the original auxiliary-only/missing-token behavior.
    deployable_latent_reconstruction: bool = False

    def __post_init__(self) -> None:
        if not str(self.task_name).strip():
            raise ValueError("task_name cannot be empty")
        _validate_modality_partition(self.cheap_groups, self.expensive_groups)
        if not torch.isfinite(torch.tensor(float(self.temperature))):
            raise ValueError("temperature must be finite and positive")
        if float(self.temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        weights = {
            "supervised_weight": self.supervised_weight,
            "logit_kd_weight": self.logit_kd_weight,
            "patient_embedding_weight": self.patient_embedding_weight,
            "expensive_reconstruction_weight": self.expensive_reconstruction_weight,
        }
        for name, value in weights.items():
            value = float(value)
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not any(float(value) > 0 for value in weights.values()):
            raise ValueError("at least one objective weight must be positive")
        if int(self.reconstructor_hidden_multiplier) <= 0:
            raise ValueError("reconstructor_hidden_multiplier must be positive")
        if not isinstance(self.deployable_latent_reconstruction, bool):
            raise ValueError("deployable_latent_reconstruction must be boolean")


def _validate_modality_partition(
    cheap_groups: Iterable[str], expensive_groups: Iterable[str]
) -> None:
    cheap = tuple(str(group) for group in cheap_groups)
    expensive = tuple(str(group) for group in expensive_groups)
    if not cheap:
        raise ValueError("cheap_groups cannot be empty")
    if not expensive:
        raise ValueError("expensive_groups cannot be empty")
    if len(set(cheap)) != len(cheap):
        raise ValueError("cheap_groups contains duplicate modalities")
    if len(set(expensive)) != len(expensive):
        raise ValueError("expensive_groups contains duplicate modalities")
    overlap = sorted(set(cheap).intersection(expensive))
    if overlap:
        raise ValueError(f"cheap_groups and expensive_groups overlap: {overlap}")
    declared = set(cheap).union(expensive)
    canonical = set(PRIMARY_MODALITY_GROUPS)
    unknown = sorted(declared.difference(canonical))
    if unknown:
        raise ValueError(f"unknown modality groups in partition: {unknown}")
    missing = [group for group in PRIMARY_MODALITY_GROUPS if group not in declared]
    if missing:
        raise ValueError(f"modality partition is incomplete; missing: {missing}")


def _zero_from(tensor: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero on ``tensor``'s device/dtype."""

    return tensor.sum() * 0.0


def masked_mean_loss(
    per_example_loss: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Average scalar per-example losses over observed patients only."""

    if per_example_loss.ndim != 1:
        raise ValueError("per_example_loss must have shape [B]")
    if observed.ndim != 1 or observed.shape[0] != per_example_loss.shape[0]:
        raise ValueError("observed must have shape [B]")
    observed = observed.to(device=per_example_loss.device).bool()
    count = int(observed.sum().item())
    if count == 0:
        return _zero_from(per_example_loss), 0
    return per_example_loss[observed].sum() / float(count), count


def masked_vector_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """MSE normalized by observed patient vectors *and* embedding width."""

    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape [B, D]")
    if observed.ndim != 1 or observed.shape[0] != prediction.shape[0]:
        raise ValueError("observed must have shape [B]")
    observed = observed.to(device=prediction.device).bool()
    if not torch.isfinite(target[observed]).all():
        raise ValueError("observed distillation targets must be finite")
    if not torch.isfinite(prediction[observed]).all():
        raise ValueError("observed distillation predictions must be finite")
    # Mask *before* arithmetic.  A NaN stored in an absent target must not
    # contaminate autograd through a later multiply-by-zero or reduction.
    expanded = observed.unsqueeze(-1)
    safe_prediction = torch.where(expanded, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(expanded, target.detach(), torch.zeros_like(target))
    squared = (safe_prediction - safe_target).square().mean(dim=-1)
    loss, vectors = masked_mean_loss(squared, observed)
    return loss, vectors, vectors * int(prediction.shape[-1])


def temperature_scaled_binary_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    observed: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Binary soft-target KD with the standard ``temperature**2`` scaling."""

    student_logits = student_logits.reshape(-1)
    teacher_logits = teacher_logits.reshape(-1)
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    if observed.ndim != 1 or observed.shape != student_logits.shape:
        raise ValueError("observed must have shape [B]")
    observed = observed.to(device=student_logits.device).bool()
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(teacher_logits[observed]).all():
        raise ValueError("observed teacher logits must be finite")
    if not torch.isfinite(student_logits[observed]).all():
        raise ValueError("observed student logits must be finite")
    safe_student = torch.where(observed, student_logits, torch.zeros_like(student_logits))
    safe_teacher = torch.where(observed, teacher_logits.detach(), torch.zeros_like(teacher_logits))
    teacher_probability = torch.sigmoid(safe_teacher / temperature)
    per_example = F.binary_cross_entropy_with_logits(
        safe_student / temperature,
        teacher_probability,
        reduction="none",
    ) * (temperature**2)
    return masked_mean_loss(per_example, observed)


def masked_supervised_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_observed: Optional[torch.Tensor] = None,
    *,
    pos_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int]:
    """Binary supervision normalized over finite, explicitly observed labels."""

    logits = logits.reshape(-1)
    labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if labels.shape != logits.shape:
        raise ValueError("labels must have shape [B] matching logits")
    finite = torch.isfinite(labels)
    if label_observed is not None:
        if label_observed.ndim != 1 or label_observed.shape != labels.shape:
            raise ValueError("label_observed must have shape [B]")
        finite = finite & label_observed.to(device=logits.device).bool()
    observed_labels = labels[finite]
    if observed_labels.numel() and not torch.all(
        (observed_labels == 0) | (observed_labels == 1)
    ):
        raise ValueError("observed labels must be binary values 0 or 1")
    if not torch.isfinite(logits[finite]).all():
        raise ValueError("logits for observed labels must be finite")
    safe_labels = torch.where(finite, labels, torch.zeros_like(labels))
    safe_logits = torch.where(finite, logits, torch.zeros_like(logits))
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    per_example = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    return masked_mean_loss(per_example, finite)


class ExpensiveLatentReconstructor(nn.Module):
    """Predict one teacher source token from the cheap patient embedding."""

    def __init__(
        self,
        d_model: int,
        hidden_multiplier: int,
        *,
        deployable: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.deployable = bool(deployable)
        hidden = int(d_model) * int(hidden_multiplier)
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        if self.deployable:
            # Reconstructed content must not masquerade as a measured native
            # encoder token.  Both parameters are source-specific because each
            # reconstructor instance belongs to exactly one expensive group.
            self.synthetic_token_indicator = nn.Parameter(torch.zeros(d_model))
            self.reliability_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("synthetic_token_indicator", None)
            self.register_parameter("reliability_logit", None)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return the raw prediction used by the teacher-token MSE target."""

        return self.network(embedding)

    def deployment_token(
        self, raw_prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tag and reliability-scale a raw reconstruction before fusion."""

        if not self.deployable:
            raise RuntimeError("legacy auxiliary reconstructor is not deployable")
        if self.synthetic_token_indicator is None or self.reliability_logit is None:
            raise RuntimeError("deployable reconstructor lacks synthetic-token parameters")
        if raw_prediction.ndim != 2 or raw_prediction.shape[-1] != self.d_model:
            raise ValueError("raw reconstructed token must have shape [B, d_model]")
        reliability = torch.sigmoid(self.reliability_logit).to(
            dtype=raw_prediction.dtype
        )
        token = (
            raw_prediction * reliability
            + self.synthetic_token_indicator.to(dtype=raw_prediction.dtype)
        )
        return token, reliability


def _label_free_model_inputs(batch: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    return {
        key: value
        for key, value in batch.items()
        if key not in OUTCOME_ONLY_BATCH_KEYS
        and not str(key).startswith("label_")
    }


def _cheap_student_inference(
    student: Task2V2Model,
    reconstructors: nn.ModuleDict,
    config: Task2V2DistillationConfig,
    batch: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the exact cheap-input deployment graph used during training.

    Native encoders for expensive groups are disabled before any source lookup.
    Reconstruction eligibility is determined solely from effective cheap-source
    masks.  It never depends on whether the unacquired expensive source happened
    to be available in the teacher cohort.
    """

    model_batch = _label_free_model_inputs(batch)
    encoded = student.encode_sources(
        model_batch, disabled_groups=config.expensive_groups
    )
    observed_tokens = encoded["source_tokens"]
    observed_present = encoded["source_present"].bool()
    source_index = {
        group: index for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
    }
    expensive_indices = [source_index[group] for group in config.expensive_groups]
    if bool(observed_present[:, expensive_indices].any().item()):
        raise RuntimeError("student exposed an expensive source despite acquisition mask")

    # Stage 1 is the cheap-only representation.  It is the only input to every
    # deployable reconstructor, avoiding any circular dependence on its own
    # predicted expensive token.
    cheap_encoded = student.fuse_encoded_sources(
        observed_tokens, observed_present
    )
    cheap_embedding = cheap_encoded["embedding"]
    cheap_indices = [source_index[group] for group in config.cheap_groups]
    reconstruction_eligible = observed_present[:, cheap_indices].any(dim=1)
    raw_reconstructed_by_group: Dict[str, torch.Tensor] = {}
    fusion_reconstructed_by_group: Dict[str, torch.Tensor] = {}
    reconstruction_reliability_by_group: Dict[str, torch.Tensor] = {}
    reconstructed_columns: list[torch.Tensor] = []
    fusion_columns: list[torch.Tensor] = []
    fusion_presence_columns: list[torch.Tensor] = []
    reconstructed_presence_columns: list[torch.Tensor] = []
    for index, group in enumerate(PRIMARY_MODALITY_GROUPS):
        if config.deployable_latent_reconstruction and group in reconstructors:
            raw_prediction = reconstructors[group](cheap_embedding)
            if raw_prediction.shape != observed_tokens[:, index].shape:
                raise ValueError(
                    f"reconstructor for {group} returned {tuple(raw_prediction.shape)}, "
                    f"expected {tuple(observed_tokens[:, index].shape)}"
                )
            if not torch.isfinite(raw_prediction).all():
                raise ValueError(f"reconstructor for {group} returned non-finite tokens")
            fusion_prediction, reliability = reconstructors[group].deployment_token(
                raw_prediction
            )
            if not torch.isfinite(fusion_prediction).all() or not torch.isfinite(
                reliability
            ).all():
                raise ValueError(
                    f"reconstructor for {group} returned a non-finite deployment token"
                )
            # Reconstruction MSE consumes the raw prediction. The learned
            # source-specific indicator and reliability gate are exclusively
            # part of the downstream fusion path.
            raw_reconstructed_by_group[group] = raw_prediction
            fusion_reconstructed_by_group[group] = fusion_prediction
            reconstruction_reliability_by_group[group] = reliability
            eligible = reconstruction_eligible
            injected = torch.where(
                eligible.unsqueeze(-1),
                fusion_prediction,
                torch.zeros_like(fusion_prediction),
            )
            fusion_columns.append(injected)
            fusion_presence_columns.append(eligible)
            reconstructed_columns.append(injected)
            reconstructed_presence_columns.append(eligible)
        else:
            fusion_columns.append(observed_tokens[:, index])
            fusion_presence_columns.append(observed_present[:, index])
            reconstructed_columns.append(torch.zeros_like(observed_tokens[:, index]))
            reconstructed_presence_columns.append(
                torch.zeros_like(observed_present[:, index])
            )
    fusion_tokens = torch.stack(fusion_columns, dim=1)
    fusion_present = torch.stack(fusion_presence_columns, dim=1)
    reconstructed_tokens = torch.stack(reconstructed_columns, dim=1)
    reconstructed_present = torch.stack(reconstructed_presence_columns, dim=1)
    output = student.predict_from_source_tokens(
        fusion_tokens,
        fusion_present,
        task=config.task_name,
    )
    return {
        **output,
        "cheap_embedding": cheap_embedding,
        "observed_source_tokens": observed_tokens,
        "observed_source_present": observed_present,
        "reconstructed_source_tokens": reconstructed_tokens,
        "reconstructed_source_present": reconstructed_present,
        # Backward-compatible name: these are the raw pre-gate predictions
        # used by the teacher-token reconstruction target.
        "reconstructed_source_tokens_by_group": raw_reconstructed_by_group,
        "raw_reconstructed_source_tokens_by_group": raw_reconstructed_by_group,
        "fusion_reconstructed_source_tokens_by_group": (
            fusion_reconstructed_by_group
        ),
        "reconstruction_reliability_by_group": (
            reconstruction_reliability_by_group
        ),
        "fusion_source_present": fusion_present,
        "reconstruction_eligible": reconstruction_eligible,
        "latent_reconstruction_deployed": bool(
            config.deployable_latent_reconstruction
        ),
        "source_presence_semantics": (
            "source_present/fusion_source_present includes eligible reconstructed "
            "slots; observed_source_present records acquired usable sources"
            if config.deployable_latent_reconstruction
            else "source_present records acquired usable cheap sources only"
        ),
    }


class Task2V2DeploymentStudent(nn.Module):
    """Teacher-free inference module for a fixed cheap acquisition contract."""

    def __init__(
        self,
        student: Task2V2Model,
        reconstructors: nn.ModuleDict,
        config: Task2V2DistillationConfig,
    ) -> None:
        super().__init__()
        # The repository supports both package imports (``ot2.*``) and direct
        # execution from ``ot2/``.  Those paths can load the same model class
        # under two Python module names, so validate the public interface rather
        # than rejecting an otherwise identical instance by class identity.
        if not isinstance(student, nn.Module) or not all(
            callable(getattr(student, method, None))
            for method in (
                "encode_sources",
                "fuse_encoded_sources",
                "predict_from_source_tokens",
            )
        ):
            raise TypeError("deployment student lacks the Task2V2Model interface")
        if set(reconstructors) != set(config.expensive_groups):
            raise ValueError(
                "deployment reconstructors must match the expensive-group contract"
            )
        self.student = student
        self.expensive_reconstructors = reconstructors
        self.distillation_config = config
        self.task_name = config.task_name
        self.cheap_groups = tuple(config.cheap_groups)
        self.expensive_groups = tuple(config.expensive_groups)

    def forward(
        self,
        batch: Mapping[str, Any],
        task: Optional[str] = None,
        *,
        disabled_groups: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if task is not None and str(task) != self.task_name:
            raise ValueError("deployment task differs from the frozen contract")
        if disabled_groups is not None and tuple(disabled_groups) != self.expensive_groups:
            raise ValueError(
                "deployment disabled_groups differs from the frozen acquisition contract"
            )
        return _cheap_student_inference(
            self.student,
            self.expensive_reconstructors,
            self.distillation_config,
            batch,
        )


class Task2V2CheapStudentDistiller(nn.Module):
    """Train a deployment-safe cheap-modality student from a full teacher."""

    def __init__(
        self,
        teacher: Task2V2Model,
        student: Task2V2Model,
        config: Task2V2DistillationConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        if isinstance(config, Mapping):
            config = Task2V2DistillationConfig(**dict(config))
        if not isinstance(config, Task2V2DistillationConfig):
            raise TypeError("config must be Task2V2DistillationConfig or a mapping")
        if not isinstance(teacher, Task2V2Model):
            raise TypeError("teacher must be a Task2V2Model")
        if not isinstance(student, Task2V2Model):
            raise TypeError("student must be a Task2V2Model")
        if teacher is student:
            raise ValueError("teacher and student must be distinct model instances")
        shared_parameters = {id(parameter) for parameter in teacher.parameters()}.intersection(
            id(parameter) for parameter in student.parameters()
        )
        if shared_parameters:
            raise ValueError("teacher and student must not share parameter objects")
        if int(teacher.config.d_model) != int(student.config.d_model):
            raise ValueError("teacher and student d_model must match")
        if config.task_name not in teacher.heads:
            raise ValueError(f"teacher has no head for task {config.task_name!r}")
        if config.task_name not in student.heads:
            raise ValueError(f"student has no head for task {config.task_name!r}")

        self.config = config
        self.teacher = teacher
        self.student = student
        self.cheap_groups = tuple(config.cheap_groups)
        self.expensive_groups = tuple(config.expensive_groups)
        self.source_index = {
            group: index for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
        }
        d_model = int(student.config.d_model)
        self.expensive_reconstructors = nn.ModuleDict(
            {
                group: ExpensiveLatentReconstructor(
                    d_model,
                    config.reconstructor_hidden_multiplier,
                    deployable=config.deployable_latent_reconstruction,
                )
                for group in self.expensive_groups
            }
        )

        # Freezing is enforced centrally so an optimizer created from
        # ``distiller.parameters()`` cannot accidentally update the teacher.
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        if not any(parameter.requires_grad for parameter in self.student.parameters()):
            raise ValueError("student must have at least one trainable parameter")

    def train(self, mode: bool = True) -> "Task2V2CheapStudentDistiller":
        """Set train mode while keeping the teacher permanently in eval mode."""

        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def model_inputs(batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a shallow label-free model-input mapping without mutation."""

        return _label_free_model_inputs(batch)

    def student_forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Exact deployment forward; expensive native encoders never run."""

        return _cheap_student_inference(
            self.student,
            self.expensive_reconstructors,
            self.config,
            batch,
        )

    def deployment_model(self) -> Task2V2DeploymentStudent:
        """Return a teacher-free module sharing the selected student weights."""

        return Task2V2DeploymentStudent(
            self.student, self.expensive_reconstructors, self.config
        )

    def deployment_metadata(self) -> Dict[str, Any]:
        """JSON-safe contract that can be stored beside a student checkpoint."""

        payload: Dict[str, Any] = {
            "format_version": "ot2-task2-v2-cheap-student-2",
            "task_name": self.config.task_name,
            "primary_modality_order": list(PRIMARY_MODALITY_GROUPS),
            "student_input_groups": list(self.cheap_groups),
            "retained_input_groups": list(self.cheap_groups),
            "acquired_source_groups": list(self.cheap_groups),
            "native_encoder_groups": list(self.cheap_groups),
            "deployment_task_heads": [self.config.task_name],
            "student_state_layout": (
                "pruned_retained_encoders_fusion_single_task_head"
                if self.config.deployable_latent_reconstruction
                else "full_task2_v2_model_state"
            ),
            "disabled_at_student_boundary": list(self.expensive_groups),
            "removed_groups": list(self.expensive_groups),
            "teacher_only_training_groups": list(self.expensive_groups),
            "teacher_required_for_deployment": False,
            "raw_expensive_reconstruction": False,
            "latent_reconstruction_targets": "teacher_source_tokens",
            "deployable_latent_reconstruction": bool(
                self.config.deployable_latent_reconstruction
            ),
            "reconstructors_required_for_deployment": bool(
                self.config.deployable_latent_reconstruction
            ),
            "reconstructor_hidden_multiplier": int(
                self.config.reconstructor_hidden_multiplier
            ),
            "reconstruction_input": "cheap_only_patient_embedding",
            "reconstruction_eligibility": "any_effective_cheap_source_present",
            "reconstructed_fusion_groups": (
                list(self.expensive_groups)
                if self.config.deployable_latent_reconstruction
                else []
            ),
            "reconstructed_source_groups": (
                list(self.expensive_groups)
                if self.config.deployable_latent_reconstruction
                else []
            ),
            "reconstructed_groups": (
                list(self.expensive_groups)
                if self.config.deployable_latent_reconstruction
                else []
            ),
            "reconstruction_target_space": LATENT_RECONSTRUCTION_TARGET_SPACE,
            "reconstruction_mask_semantics": LATENT_RECONSTRUCTION_MASK_SEMANTICS,
            "inference_mode": (
                RECONSTRUCTED_DISTILLED_INFERENCE_MODE
                if self.config.deployable_latent_reconstruction
                else LEGACY_DISTILLED_INFERENCE_MODE
            ),
            "fusion_presence_contract": (
                "observed cheap slots plus eligible reconstructed expensive slots"
                if self.config.deployable_latent_reconstruction
                else "observed cheap slots; expensive slots use learned missing tokens"
            ),
            "synthetic_token_semantics": (
                "raw_prediction * sigmoid(per_group_reliability_logit) + "
                "learned_per_group_synthetic_indicator"
                if self.config.deployable_latent_reconstruction
                else "none_auxiliary_only"
            ),
            "reconstruction_mse_target_stage": (
                "raw_prediction_before_reliability_gate_and_synthetic_indicator"
            ),
            "experiment_model_id": (
                "distilled_student_latent_reconstruction"
                if self.config.deployable_latent_reconstruction
                else "distilled_student"
            ),
            "objective": {
                "temperature": float(self.config.temperature),
                "supervised_weight": float(self.config.supervised_weight),
                "logit_kd_weight": float(self.config.logit_kd_weight),
                "patient_embedding_weight": float(
                    self.config.patient_embedding_weight
                ),
                "expensive_reconstruction_weight": float(
                    self.config.expensive_reconstruction_weight
                ),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["deployment_contract_hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        labels: Optional[torch.Tensor] = None,
        label_observed: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute supervised and observed-only distillation objectives."""

        model_batch = self.model_inputs(batch)
        self.teacher.eval()
        with torch.no_grad():
            teacher_output = self.teacher(
                model_batch,
                task=self.config.task_name,
                disabled_groups=None,
            )
        student_output = self.student_forward(model_batch)

        if labels is None:
            candidate = batch.get("y")
            if not isinstance(candidate, torch.Tensor):
                raise ValueError("labels must be passed or batch['y'] must be a tensor")
            labels = candidate
        if label_observed is None:
            candidate = batch.get("label_observed")
            if isinstance(candidate, torch.Tensor):
                label_observed = candidate

        supervised_loss, supervised_count = masked_supervised_bce(
            student_output["logits"],
            labels,
            label_observed,
            pos_weight=pos_weight,
        )

        teacher_present = teacher_output["source_present"].bool()
        expensive_indices = [
            self.source_index[group] for group in self.expensive_groups
        ]
        expensive_present = teacher_present[:, expensive_indices]
        any_expensive_present = expensive_present.any(dim=1)

        logit_kd_loss, logit_kd_count = temperature_scaled_binary_kd(
            student_output["logits"],
            teacher_output["logits"],
            any_expensive_present,
            temperature=self.config.temperature,
        )
        patient_embedding_loss, patient_vectors, patient_elements = masked_vector_mse(
            student_output["embedding"],
            teacher_output["embedding"],
            any_expensive_present,
        )

        reconstruction_losses: Dict[str, torch.Tensor] = {}
        reconstruction_counts: Dict[str, Dict[str, int]] = {}
        reconstruction_numerators: list[torch.Tensor] = []
        reconstruction_elements = 0
        for group in self.expensive_groups:
            index = self.source_index[group]
            deployed_predictions = student_output[
                "reconstructed_source_tokens_by_group"
            ]
            prediction = deployed_predictions.get(group)
            if prediction is None:
                prediction = self.expensive_reconstructors[group](
                    student_output["cheap_embedding"]
                )
            group_loss, vectors, elements = masked_vector_mse(
                prediction,
                teacher_output["source_tokens"][:, index],
                teacher_present[:, index],
            )
            reconstruction_losses[group] = group_loss
            reconstruction_counts[group] = {
                "observed_teacher_vectors": vectors,
                "observed_teacher_elements": elements,
            }
            if vectors > 0:
                reconstruction_numerators.append(group_loss * float(elements))
                reconstruction_elements += int(elements)
        if reconstruction_numerators:
            expensive_reconstruction_numerator = torch.stack(
                reconstruction_numerators
            ).sum()
            expensive_reconstruction_loss = (
                expensive_reconstruction_numerator
                / float(reconstruction_elements)
            )
        else:
            expensive_reconstruction_numerator = _zero_from(
                student_output["embedding"]
            )
            expensive_reconstruction_loss = _zero_from(student_output["embedding"])

        weighted_terms: list[torch.Tensor] = []
        if self.config.supervised_weight > 0 and supervised_count > 0:
            weighted_terms.append(supervised_loss * self.config.supervised_weight)
        if self.config.logit_kd_weight > 0 and logit_kd_count > 0:
            weighted_terms.append(logit_kd_loss * self.config.logit_kd_weight)
        if self.config.patient_embedding_weight > 0 and patient_vectors > 0:
            weighted_terms.append(
                patient_embedding_loss * self.config.patient_embedding_weight
            )
        if (
            self.config.expensive_reconstruction_weight > 0
            and reconstruction_elements > 0
        ):
            weighted_terms.append(
                expensive_reconstruction_loss
                * self.config.expensive_reconstruction_weight
            )
        if not weighted_terms:
            raise ValueError(
                "batch has no eligible supervised or observed distillation targets"
            )
        total_loss = torch.stack(weighted_terms).sum()

        return {
            "loss": total_loss,
            "component_losses": {
                "supervised_bce": supervised_loss,
                "temperature_scaled_logit_kd": logit_kd_loss,
                "patient_embedding_mse": patient_embedding_loss,
                "expensive_latent_reconstruction": expensive_reconstruction_loss,
            },
            "component_counts": {
                "supervised_patients": supervised_count,
                "logit_kd_patients_with_expensive_source": logit_kd_count,
                "patient_embedding_vectors_with_expensive_source": patient_vectors,
                "patient_embedding_elements_with_expensive_source": patient_elements,
                "patients_with_any_expensive_source": int(
                    any_expensive_present.sum().item()
                ),
                "expensive_reconstruction_elements": reconstruction_elements,
            },
            # These sufficient statistics let gradient accumulation normalize
            # each objective over the *whole* accumulation window rather than
            # giving a short/sparse microbatch the same weight as a full one.
            "component_numerators": {
                "supervised_bce": supervised_loss * float(supervised_count),
                "temperature_scaled_logit_kd": logit_kd_loss
                * float(logit_kd_count),
                "patient_embedding_mse": patient_embedding_loss
                * float(patient_elements),
                "expensive_latent_reconstruction": (
                    expensive_reconstruction_numerator
                ),
            },
            "component_denominators": {
                "supervised_bce": supervised_count,
                "temperature_scaled_logit_kd": logit_kd_count,
                "patient_embedding_mse": patient_elements,
                "expensive_latent_reconstruction": reconstruction_elements,
            },
            "reconstruction_loss_by_modality": reconstruction_losses,
            "reconstruction_counts_by_modality": reconstruction_counts,
            "student_logits": student_output["logits"],
            "student_embedding": student_output["embedding"],
            "student_source_present": student_output["source_present"],
            "student_observed_source_present": student_output[
                "observed_source_present"
            ],
            "student_reconstructed_source_present": student_output[
                "reconstructed_source_present"
            ],
            "latent_reconstruction_deployed": student_output[
                "latent_reconstruction_deployed"
            ],
            "teacher_source_present": teacher_present,
            "deployment_metadata": self.deployment_metadata(),
        }


__all__ = [
    "ExpensiveLatentReconstructor",
    "LATENT_RECONSTRUCTION_MASK_SEMANTICS",
    "LATENT_RECONSTRUCTION_TARGET_SPACE",
    "LEGACY_DISTILLED_INFERENCE_MODE",
    "OUTCOME_ONLY_BATCH_KEYS",
    "RECONSTRUCTED_DISTILLED_INFERENCE_MODE",
    "Task2V2CheapStudentDistiller",
    "Task2V2DeploymentStudent",
    "Task2V2DistillationConfig",
    "masked_mean_loss",
    "masked_supervised_bce",
    "masked_vector_mse",
    "temperature_scaled_binary_kd",
]
