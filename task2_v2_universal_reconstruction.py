#!/usr/bin/env python3
"""Universal masked-modality reconstruction for the OT2 V2 token space.

The existing :class:`Task2V2Model` maps each of the nine scientific modality
groups to one ``[D]`` source token.  This module operates at that public
boundary.  During training it hides only source tokens that were genuinely
observed, reconstructs the requested tokens from the remaining set, and sends
the observed plus explicitly synthetic tokens through the ordinary OT2 fusion
module and task head.

This is deliberately *latent* reconstruction.  It never claims to synthesize
an ECG, retinal image, or longitudinal measurement.  Native, artificially
hidden, naturally missing, and reconstructed states remain separate in every
returned result.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Optional

import torch
from torch import nn

try:
    from .task2_v2_distill import (
        masked_supervised_bce,
        masked_vector_mse,
        temperature_scaled_binary_kd,
    )
    from .task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model
except ImportError:  # Direct execution from ``ot2/``.
    from task2_v2_distill import (  # type: ignore
        masked_supervised_bce,
        masked_vector_mse,
        temperature_scaled_binary_kd,
    )
    from task2_v2_model import PRIMARY_MODALITY_GROUPS, Task2V2Model  # type: ignore


UNIVERSAL_RECONSTRUCTION_FORMAT_VERSION = (
    "ot2-universal-masked-reconstruction-v1"
)

DEFAULT_MASK_FAMILIES = (
    (
        "retinal_flio",
        "retinal_oct",
        "retinal_octa",
        "retinal_photography",
    ),
    (
        "environment",
        "wearable_activity_monitor",
        "wearable_blood_glucose",
    ),
)


@dataclasses.dataclass(frozen=True)
class UniversalMaskConfig:
    """Train-only distribution over artificial whole-modality masks."""

    no_drop_probability: float = 0.15
    single_drop_probability: float = 0.35
    family_drop_probability: float = 0.20
    random_drop_probability: float = 0.30
    families: tuple[tuple[str, ...], ...] = DEFAULT_MASK_FAMILIES

    def __post_init__(self) -> None:
        probabilities = (
            self.no_drop_probability,
            self.single_drop_probability,
            self.family_drop_probability,
            self.random_drop_probability,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("mask probabilities must lie in [0, 1]")
        if abs(sum(float(value) for value in probabilities) - 1.0) > 1e-8:
            raise ValueError("mask probabilities must sum to one")
        canonical = set(PRIMARY_MODALITY_GROUPS)
        for family_index, family in enumerate(self.families):
            normalized = tuple(str(group) for group in family)
            if not normalized or len(normalized) != len(set(normalized)):
                raise ValueError(
                    f"mask family {family_index} must be non-empty and unique"
                )
            unknown = sorted(set(normalized).difference(canonical))
            if unknown:
                raise ValueError(
                    f"mask family {family_index} contains unknown groups: {unknown}"
                )


@dataclasses.dataclass(frozen=True)
class UniversalReconstructionConfig:
    """Architecture and objective for a task-specific universal student."""

    task_name: str
    d_model: int
    num_heads: int = 4
    decoder_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.10
    temperature: float = 2.0
    supervised_weight: float = 1.0
    observed_only_supervised_weight: float = 0.5
    logit_kd_weight: float = 1.0
    patient_embedding_weight: float = 1.0
    token_reconstruction_weight: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.task_name).strip():
            raise ValueError("task_name cannot be empty")
        for name in ("d_model", "num_heads", "decoder_layers", "ffn_multiplier"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) <= 0:
            raise ValueError("temperature must be positive")
        weights = (
            self.supervised_weight,
            self.observed_only_supervised_weight,
            self.logit_kd_weight,
            self.patient_embedding_weight,
            self.token_reconstruction_weight,
        )
        if any(not math.isfinite(float(value)) for value in weights):
            raise ValueError("objective weights must be finite")
        if any(float(value) < 0 for value in weights) or not any(
            float(value) > 0 for value in weights
        ):
            raise ValueError("objective weights must be non-negative and not all zero")


class ModalityMaskSampler:
    """Sample participant-level masks without inventing reconstruction targets.

    Only genuinely present sources may be dropped.  A participant with at
    least one native source always retains at least one, and a participant with
    fewer than two sources receives the no-drop policy.
    """

    MODE_NO_DROP = 0
    MODE_SINGLE = 1
    MODE_FAMILY = 2
    MODE_RANDOM = 3
    MODE_NAMES = {
        MODE_NO_DROP: "no_drop",
        MODE_SINGLE: "single_drop",
        MODE_FAMILY: "family_drop",
        MODE_RANDOM: "random_drop",
    }

    def __init__(self, config: UniversalMaskConfig | Mapping[str, Any]) -> None:
        if isinstance(config, Mapping):
            config = UniversalMaskConfig(**dict(config))
        if not isinstance(config, UniversalMaskConfig):
            raise TypeError("config must be UniversalMaskConfig or a mapping")
        self.config = config
        source_index = {
            group: index for index, group in enumerate(PRIMARY_MODALITY_GROUPS)
        }
        self.family_indices = tuple(
            tuple(source_index[group] for group in family)
            for family in config.families
        )

    @staticmethod
    def _rand(generator: Optional[torch.Generator]) -> float:
        return float(torch.rand((), generator=generator).item())

    @staticmethod
    def _choice(
        values: Sequence[int], generator: Optional[torch.Generator]
    ) -> int:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        index = int(torch.randint(len(values), (), generator=generator).item())
        return int(values[index])

    def sample(
        self,
        native_present: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if native_present.ndim != 2 or native_present.shape[1] != len(
            PRIMARY_MODALITY_GROUPS
        ):
            raise ValueError("native_present must have shape [B, 9]")
        device = native_present.device
        present_cpu = native_present.detach().to(device="cpu").bool()
        dropped = torch.zeros_like(present_cpu)
        modes = torch.full(
            (present_cpu.shape[0],), self.MODE_NO_DROP, dtype=torch.int64
        )
        p_no = float(self.config.no_drop_probability)
        p_single = p_no + float(self.config.single_drop_probability)
        p_family = p_single + float(self.config.family_drop_probability)

        for row_index, row in enumerate(present_cpu):
            present_indices = row.nonzero(as_tuple=False).flatten().tolist()
            if len(present_indices) < 2:
                continue
            draw = self._rand(generator)
            if draw < p_no:
                continue
            if draw < p_single:
                target = self._choice(present_indices, generator)
                dropped[row_index, target] = True
                modes[row_index] = self.MODE_SINGLE
                continue
            if draw < p_family:
                eligible_families: list[tuple[int, ...]] = []
                present_set = set(present_indices)
                for family in self.family_indices:
                    targets = tuple(index for index in family if index in present_set)
                    if targets and len(targets) < len(present_indices):
                        eligible_families.append(targets)
                if eligible_families:
                    family = eligible_families[
                        self._choice(range(len(eligible_families)), generator)
                    ]
                    dropped[row_index, list(family)] = True
                    modes[row_index] = self.MODE_FAMILY
                    continue
                # A family cannot be removed without hiding every native source;
                # fall through to a valid singleton mask.
                target = self._choice(present_indices, generator)
                dropped[row_index, target] = True
                modes[row_index] = self.MODE_SINGLE
                continue

            maximum = len(present_indices) - 1
            n_drop = int(
                torch.randint(1, maximum + 1, (), generator=generator).item()
            )
            order = torch.randperm(len(present_indices), generator=generator)
            targets = [present_indices[int(index)] for index in order[:n_drop]]
            dropped[row_index, targets] = True
            modes[row_index] = self.MODE_RANDOM

        if bool((dropped & ~present_cpu).any().item()):
            raise RuntimeError("mask sampler attempted to drop a naturally absent source")
        observed = present_cpu & ~dropped
        native_any = present_cpu.any(dim=1)
        if bool((native_any & ~observed.any(dim=1)).any().item()):
            raise RuntimeError("mask sampler removed every native source")
        return {
            "native_present": present_cpu.to(device=device),
            "observed_present": observed.to(device=device),
            "artificially_dropped": dropped.to(device=device),
            "requested_reconstruction": dropped.to(device=device),
            "mask_mode": modes.to(device=device),
        }


class TargetQueryCrossModalReconstructor(nn.Module):
    """Predict all requested source tokens from the currently observed set."""

    def __init__(self, config: UniversalReconstructionConfig) -> None:
        super().__init__()
        self.config = config
        n_sources = len(PRIMARY_MODALITY_GROUPS)
        width = int(config.d_model)
        self.source_embeddings = nn.Parameter(torch.zeros(1, n_sources, width))
        self.target_queries = nn.Parameter(torch.zeros(1, n_sources, width))
        self.synthetic_indicators = nn.Parameter(torch.zeros(1, n_sources, width))
        nn.init.trunc_normal_(self.source_embeddings, std=0.02)
        nn.init.trunc_normal_(self.target_queries, std=0.02)
        nn.init.trunc_normal_(self.synthetic_indicators, std=0.02)
        self.memory_norm = nn.LayerNorm(width)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=int(config.num_heads),
            dim_feedforward=width * int(config.ffn_multiplier),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(config.decoder_layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, width)
        self.confidence_head = nn.Linear(width, 1)

    def forward(
        self,
        observed_tokens: torch.Tensor,
        observed_present: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if observed_tokens.ndim != 3:
            raise ValueError("observed_tokens must have shape [B, 9, D]")
        expected = (len(PRIMARY_MODALITY_GROUPS), int(self.config.d_model))
        if tuple(observed_tokens.shape[1:]) != expected:
            raise ValueError(
                f"observed_tokens trailing shape must be {expected}, got "
                f"{tuple(observed_tokens.shape[1:])}"
            )
        if observed_present.shape != observed_tokens.shape[:2]:
            raise ValueError("observed_present must have shape [B, 9]")
        if not torch.isfinite(observed_tokens).all():
            raise ValueError("observed_tokens must be finite")

        present = observed_present.bool()
        safe_tokens = torch.where(
            present.unsqueeze(-1), observed_tokens, torch.zeros_like(observed_tokens)
        )
        memory = self.memory_norm(safe_tokens) + self.source_embeddings.to(
            dtype=safe_tokens.dtype
        )
        raw = torch.zeros_like(observed_tokens)
        confidence = observed_tokens.new_zeros(observed_tokens.shape[:2])

        eligible_rows = present.any(dim=1)
        eligible_indices = eligible_rows.nonzero(as_tuple=False).flatten()
        if eligible_indices.numel():
            selected_memory = memory.index_select(0, eligible_indices)
            selected_present = present.index_select(0, eligible_indices)
            queries = self.target_queries.to(dtype=selected_memory.dtype).expand(
                selected_memory.shape[0], -1, -1
            )
            decoded = self.decoder(
                queries,
                selected_memory,
                memory_key_padding_mask=~selected_present,
            )
            decoded = self.output_norm(decoded)
            selected_raw = self.output_projection(decoded)
            selected_confidence = torch.sigmoid(
                self.confidence_head(decoded).squeeze(-1)
            )
            raw = raw.index_copy(0, eligible_indices, selected_raw)
            confidence = confidence.index_copy(
                0, eligible_indices, selected_confidence
            )

        return {
            "raw_reconstructed_tokens": raw,
            "reconstruction_confidence": confidence,
            "reconstruction_eligible_rows": eligible_rows,
        }

    def compose_for_fusion(
        self,
        *,
        observed_tokens: torch.Tensor,
        observed_present: torch.Tensor,
        requested_reconstruction: torch.Tensor,
        missing_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if requested_reconstruction.shape != observed_present.shape:
            raise ValueError("requested_reconstruction must have shape [B, 9]")
        requested = requested_reconstruction.to(
            device=observed_present.device
        ).bool()
        if bool((requested & observed_present.bool()).any().item()):
            raise ValueError("an observed source cannot simultaneously be reconstructed")
        result = self(observed_tokens, observed_present)
        eligible = observed_present.bool().any(dim=1, keepdim=True)
        reconstructed_present = requested & eligible
        raw = result["raw_reconstructed_tokens"]
        confidence = result["reconstruction_confidence"]
        fallback = missing_tokens.to(
            device=raw.device, dtype=raw.dtype
        ).expand(raw.shape[0], -1, -1)
        synthetic = raw + self.synthetic_indicators.to(dtype=raw.dtype)
        mixed = (
            confidence.unsqueeze(-1) * synthetic
            + (1.0 - confidence.unsqueeze(-1)) * fallback
        )
        fusion_tokens = torch.where(
            observed_present.bool().unsqueeze(-1), observed_tokens, torch.zeros_like(raw)
        )
        fusion_tokens = torch.where(
            reconstructed_present.unsqueeze(-1), mixed, fusion_tokens
        )
        fusion_present = observed_present.bool() | reconstructed_present
        return {
            **result,
            "fusion_source_tokens": fusion_tokens,
            "fusion_source_present": fusion_present,
            "reconstructed_source_present": reconstructed_present,
            "requested_reconstruction": requested,
            "synthetic_fusion_tokens": mixed,
        }


class Task2V2UniversalReconstructionStudent(nn.Module):
    """A dynamic-subset OT2 model with explicit latent reconstruction."""

    def __init__(
        self,
        base_model: Task2V2Model,
        reconstructor: TargetQueryCrossModalReconstructor,
        config: UniversalReconstructionConfig,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, Task2V2Model):
            raise TypeError("base_model must be Task2V2Model")
        if not isinstance(reconstructor, TargetQueryCrossModalReconstructor):
            raise TypeError(
                "reconstructor must be TargetQueryCrossModalReconstructor"
            )
        if reconstructor.config != config:
            raise ValueError("reconstructor and student configs differ")
        if int(base_model.config.d_model) != int(config.d_model):
            raise ValueError("base model and reconstructor d_model differ")
        if config.task_name not in base_model.heads:
            raise ValueError(f"base model has no head for {config.task_name!r}")
        self.base_model = base_model
        self.reconstructor = reconstructor
        self.config = config

    def forward_from_source_tokens(
        self,
        source_tokens: torch.Tensor,
        native_present: torch.Tensor,
        *,
        observed_source_mask: Optional[torch.Tensor] = None,
        requested_reconstruction: Optional[torch.Tensor] = None,
        inject_reconstruction: bool = True,
    ) -> Dict[str, Any]:
        if source_tokens.ndim != 3 or source_tokens.shape[1:] != (
            len(PRIMARY_MODALITY_GROUPS),
            int(self.config.d_model),
        ):
            raise ValueError("source_tokens must have shape [B, 9, d_model]")
        if native_present.shape != source_tokens.shape[:2]:
            raise ValueError("native_present must have shape [B, 9]")
        native = native_present.to(device=source_tokens.device).bool()
        if observed_source_mask is None:
            observed = native
        else:
            if observed_source_mask.shape != native.shape:
                raise ValueError("observed_source_mask must have shape [B, 9]")
            observed = native & observed_source_mask.to(
                device=source_tokens.device
            ).bool()
        requested = (
            torch.zeros_like(native)
            if requested_reconstruction is None
            else requested_reconstruction.to(device=source_tokens.device).bool()
        )
        if requested.shape != native.shape:
            raise ValueError("requested_reconstruction must have shape [B, 9]")
        requested = requested & ~observed
        observed_tokens = torch.where(
            observed.unsqueeze(-1), source_tokens, torch.zeros_like(source_tokens)
        )
        reconstructed = self.reconstructor.compose_for_fusion(
            observed_tokens=observed_tokens,
            observed_present=observed,
            requested_reconstruction=requested,
            missing_tokens=self.base_model.fusion.missing_tokens,
        )
        if inject_reconstruction:
            fusion_tokens = reconstructed["fusion_source_tokens"]
            fusion_present = reconstructed["fusion_source_present"]
            reconstructed_present = reconstructed["reconstructed_source_present"]
        else:
            fusion_tokens = observed_tokens
            fusion_present = observed
            reconstructed_present = torch.zeros_like(observed)
        prediction = self.base_model.predict_from_source_tokens(
            fusion_tokens,
            fusion_present,
            task=self.config.task_name,
        )
        return {
            **prediction,
            "native_source_present": native,
            "observed_source_present": observed,
            "requested_reconstruction": requested,
            "reconstructed_source_present": reconstructed_present,
            "fusion_source_present": fusion_present,
            "raw_reconstructed_source_tokens": reconstructed[
                "raw_reconstructed_tokens"
            ],
            "reconstruction_confidence": reconstructed[
                "reconstruction_confidence"
            ],
            "reconstruction_injected": bool(inject_reconstruction),
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        task: Optional[str] = None,
        *,
        observed_source_mask: Optional[torch.Tensor] = None,
        requested_reconstruction: Optional[torch.Tensor] = None,
        inject_reconstruction: bool = True,
        disabled_groups: Optional[Iterable[str]] = None,
        reconstruct_disabled: bool = False,
    ) -> Dict[str, Any]:
        if task is not None and str(task) != self.config.task_name:
            raise ValueError("requested task differs from universal student task")
        disabled = frozenset(str(group) for group in (disabled_groups or ()))
        unknown = disabled.difference(PRIMARY_MODALITY_GROUPS)
        if unknown:
            raise ValueError(f"unknown disabled_groups: {sorted(unknown)}")
        encoded = self.base_model.encode_sources(batch, disabled_groups=disabled)
        requested = requested_reconstruction
        if reconstruct_disabled and disabled:
            disabled_mask = torch.zeros_like(encoded["source_present"], dtype=torch.bool)
            for group in disabled:
                disabled_mask[:, PRIMARY_MODALITY_GROUPS.index(group)] = True
            requested = (
                disabled_mask
                if requested is None
                else requested.to(device=disabled_mask.device).bool() | disabled_mask
            )
        return self.forward_from_source_tokens(
            encoded["source_tokens"],
            encoded["source_present"],
            observed_source_mask=observed_source_mask,
            requested_reconstruction=requested,
            inject_reconstruction=inject_reconstruction,
        )


class Task2V2UniversalDistiller(nn.Module):
    """Frozen full teacher supervising a dynamic masked-modality student."""

    COMPONENTS = (
        "reconstructed_supervised_bce",
        "observed_only_supervised_bce",
        "temperature_scaled_logit_kd",
        "patient_embedding_mse",
        "masked_source_token_mse",
    )

    def __init__(
        self,
        teacher: Task2V2Model,
        student: Task2V2UniversalReconstructionStudent,
        mask_sampler: ModalityMaskSampler,
    ) -> None:
        super().__init__()
        if teacher is student.base_model:
            raise ValueError("teacher and student base model must be distinct")
        if int(teacher.config.d_model) != int(student.config.d_model):
            raise ValueError("teacher and student dimensions differ")
        shared = {id(parameter) for parameter in teacher.parameters()}.intersection(
            id(parameter) for parameter in student.parameters()
        )
        if shared:
            raise ValueError("teacher and student must not share parameter objects")
        self.teacher = teacher
        self.student = student
        self.mask_sampler = mask_sampler
        self.config = student.config
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "Task2V2UniversalDistiller":
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def component_weights(self) -> Dict[str, float]:
        return {
            "reconstructed_supervised_bce": float(self.config.supervised_weight),
            "observed_only_supervised_bce": float(
                self.config.observed_only_supervised_weight
            ),
            "temperature_scaled_logit_kd": float(self.config.logit_kd_weight),
            "patient_embedding_mse": float(
                self.config.patient_embedding_weight
            ),
            "masked_source_token_mse": float(
                self.config.token_reconstruction_weight
            ),
        }

    def forward_cached(
        self,
        *,
        source_tokens: torch.Tensor,
        native_present: torch.Tensor,
        teacher_embedding: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        masks = self.mask_sampler.sample(native_present, generator=generator)
        student_output = self.student.forward_from_source_tokens(
            source_tokens,
            native_present,
            observed_source_mask=masks["observed_present"],
            requested_reconstruction=masks["requested_reconstruction"],
            inject_reconstruction=True,
        )
        observed_only_output = self.student.forward_from_source_tokens(
            source_tokens,
            native_present,
            observed_source_mask=masks["observed_present"],
            requested_reconstruction=masks["requested_reconstruction"],
            inject_reconstruction=False,
        )
        supervised_loss, supervised_count = masked_supervised_bce(
            student_output["logits"], labels, pos_weight=pos_weight
        )
        observed_only_loss, observed_only_count = masked_supervised_bce(
            observed_only_output["logits"], labels, pos_weight=pos_weight
        )
        distillation_eligible = masks["observed_present"].any(dim=1)
        kd_loss, kd_count = temperature_scaled_binary_kd(
            student_output["logits"],
            teacher_logits,
            distillation_eligible,
            temperature=float(self.config.temperature),
        )
        embedding_loss, embedding_vectors, embedding_elements = masked_vector_mse(
            student_output["embedding"],
            teacher_embedding,
            distillation_eligible,
        )

        reconstruction_mask = (
            masks["artificially_dropped"] & native_present.bool()
        )
        prediction = student_output["raw_reconstructed_source_tokens"]
        safe_prediction = torch.where(
            reconstruction_mask.unsqueeze(-1), prediction, torch.zeros_like(prediction)
        )
        safe_target = torch.where(
            reconstruction_mask.unsqueeze(-1),
            source_tokens.detach(),
            torch.zeros_like(source_tokens),
        )
        reconstruction_numerator = (safe_prediction - safe_target).square().sum()
        reconstruction_vectors = int(reconstruction_mask.sum().item())
        reconstruction_elements = reconstruction_vectors * int(source_tokens.shape[-1])
        reconstruction_loss = (
            reconstruction_numerator / float(reconstruction_elements)
            if reconstruction_elements
            else prediction.sum() * 0.0
        )

        numerators = {
            "reconstructed_supervised_bce": (
                supervised_loss * float(supervised_count)
            ),
            "observed_only_supervised_bce": (
                observed_only_loss * float(observed_only_count)
            ),
            "temperature_scaled_logit_kd": kd_loss * float(kd_count),
            "patient_embedding_mse": embedding_loss * float(embedding_elements),
            "masked_source_token_mse": reconstruction_numerator,
        }
        denominators = {
            "reconstructed_supervised_bce": supervised_count,
            "observed_only_supervised_bce": observed_only_count,
            "temperature_scaled_logit_kd": kd_count,
            "patient_embedding_mse": embedding_elements,
            "masked_source_token_mse": reconstruction_elements,
        }
        weighted = [
            numerators[name] / float(denominators[name]) * weight
            for name, weight in self.component_weights.items()
            if weight > 0 and denominators[name] > 0
        ]
        if not weighted:
            raise ValueError("batch has no eligible universal reconstruction loss")
        per_modality: Dict[str, Dict[str, Any]] = {}
        squared_by_vector = (safe_prediction - safe_target).square().mean(dim=-1)
        for index, group in enumerate(PRIMARY_MODALITY_GROUPS):
            selected = reconstruction_mask[:, index]
            count = int(selected.sum().item())
            per_modality[group] = {
                "vectors": count,
                "mse": (
                    squared_by_vector[selected, index].mean()
                    if count
                    else squared_by_vector[:, index].sum() * 0.0
                ),
            }
        mode_counts = {
            name: int((masks["mask_mode"] == mode).sum().item())
            for mode, name in ModalityMaskSampler.MODE_NAMES.items()
        }
        return {
            "loss": torch.stack(weighted).sum(),
            "component_losses": {
                "reconstructed_supervised_bce": supervised_loss,
                "observed_only_supervised_bce": observed_only_loss,
                "temperature_scaled_logit_kd": kd_loss,
                "patient_embedding_mse": embedding_loss,
                "masked_source_token_mse": reconstruction_loss,
            },
            "component_numerators": numerators,
            "component_denominators": denominators,
            "reconstruction_by_modality": per_modality,
            "mask_mode_counts": mode_counts,
            "mask_state": masks,
            "student_output": student_output,
            "observed_only_output": observed_only_output,
        }


__all__ = [
    "DEFAULT_MASK_FAMILIES",
    "ModalityMaskSampler",
    "TargetQueryCrossModalReconstructor",
    "Task2V2UniversalDistiller",
    "Task2V2UniversalReconstructionStudent",
    "UNIVERSAL_RECONSTRUCTION_FORMAT_VERSION",
    "UniversalMaskConfig",
    "UniversalReconstructionConfig",
]
