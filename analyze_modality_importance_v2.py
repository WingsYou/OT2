#!/usr/bin/env python3
"""Scientifically guarded post-hoc modality importance for OT2 Model V2.

This program explains a *frozen, validation-selected* binary classifier.  It
does not train, tune a threshold, choose a checkpoint, or fit preprocessing.
The broad nine-modality screen is intended for the validation split.  A test
run is allowed only after the decision being confirmed has been written to a
hash-bound lock file, so the test labels cannot quietly become a modality
selection set.

Three complementary interventions are supported:

``ablation``
    The complete source is disabled at model level.  Its native encoder is not
    called and fusion sees the learned missing-source representation.

``permutation``
    The complete participant-level bundle (values, masks, metadata and that
    source's presence bit) is exchanged before collation.  Only source-present
    participants are permuted, without replacement and without fixed points.

``matched_permutation``
    The same whole-bundle exchange, constrained by a deterministic hierarchy
    of pre-specified, non-outcome strata.  Candidate fields are clinical site,
    sex when present, fixed age bins, and the *other* eight source-availability
    signature.  ``y`` is never inspected while constructing a permutation.

All perturbation predictions remain paired to the recipient participant and
label.  Reported confidence intervals use paired participant resampling,
stratified by the binary label; repeated permutations are also resampled in
the outer estimate.  These intervals quantify this evaluated cohort and the
specified perturbation procedure only.  They are not synthetic standard
errors and do not replace independent-seed model training.
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
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from ot2.task2_v2_config import (
        PRIMARY_MODALITIES,
        ModelV2RunConfig,
        canonical_json_hash,
        load_model_v2_config,
        validate_model_v2_lineage,
    )
    from ot2.task2_v2_dataset import (
        OT2V2Dataset,
        SOURCE_FLAG_BY_GROUP,
        task2_v2_collate,
    )
    from ot2.task2_v2_engine import (
        ENGINE_FORMAT_VERSION,
        EvaluationResult,
        binary_classification_metrics,
        recursive_to_device,
        set_deterministic_seed,
    )
    from ot2.task2_v2_model import Task2V2Model, Task2V2ModelConfig
except ModuleNotFoundError:  # Support ``python ot2/analyze_...py``.
    from task2_v2_config import (  # type: ignore
        PRIMARY_MODALITIES,
        ModelV2RunConfig,
        canonical_json_hash,
        load_model_v2_config,
        validate_model_v2_lineage,
    )
    from task2_v2_dataset import (  # type: ignore
        OT2V2Dataset,
        SOURCE_FLAG_BY_GROUP,
        task2_v2_collate,
    )
    from task2_v2_engine import (  # type: ignore
        ENGINE_FORMAT_VERSION,
        EvaluationResult,
        binary_classification_metrics,
        recursive_to_device,
        set_deterministic_seed,
    )
    from task2_v2_model import Task2V2Model, Task2V2ModelConfig  # type: ignore


IMPORTANCE_FORMAT_VERSION = "ot2-modality-importance-v2-1"
SELECTION_LOCK_VERSION = "ot2-modality-selection-lock-v2-1"
SOURCE_TOKEN_CACHE_FORMAT_VERSION = "ot2-source-token-cache-v2-1"
COHORT_ANALYSIS_VERSION = "ot2-importance-cohorts-v1"
COHORT_SCOPES = (
    "all_participants",
    "native_present_only",
    "native_present_with_remaining_input",
)
FAMILY_ABLATION_GROUPS = {
    "retinal_all": (
        "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
    ),
    "wearable_all": ("wearable_activity_monitor", "wearable_blood_glucose"),
    "retinal_wearable_all": (
        "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
        "wearable_activity_monitor", "wearable_blood_glucose",
    ),
}
FAMILY_COHORT_SCOPES = (
    "all_participants",
    "any_target_present",
    "any_target_present_with_remaining_input",
    "all_targets_present",
    "all_targets_present_with_remaining_input",
)
METRIC_NAMES = (
    "auroc",
    "auprc",
    "log_loss",
    "brier",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)
BENEFIT_METRICS = frozenset(
    {"auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity"}
)
LOSS_METRICS = frozenset({"log_loss", "brier"})

# Fixed before any outcome inspection.  Missing and out-of-range ages receive
# explicit categories; no quantile boundary is estimated from this cohort.
AGE_BIN_EDGES = (-math.inf, 40.0, 50.0, 60.0, 70.0, 80.0, math.inf)
AGE_BIN_LABELS = ("lt40", "40_49", "50_59", "60_69", "70_79", "ge80")
SEX_COLUMN_CANDIDATES = ("sex", "sex_at_birth", "gender")


@dataclasses.dataclass(frozen=True)
class PermutationPlan:
    """Recipient-index to donor-index mapping plus its label-free audit."""

    donor_indices: np.ndarray
    audit: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class SourceTokenCache:
    """Validated frozen native-encoder outputs for one ordered cohort.

    The tensors live on CPU between fusion passes.  ``full_forward_logits``
    were captured in the *same* pass that produced ``source_tokens`` and are
    used as a fail-closed equivalence oracle whenever the cache is built or
    reused.
    """

    source_tokens: torch.Tensor
    source_present: torch.Tensor
    person_ids: tuple[str, ...]
    y_true: np.ndarray
    full_forward_logits: np.ndarray
    matching_frame: pd.DataFrame
    identity: Mapping[str, Any]
    content_sha256: str
    reused: bool = False
    path: Optional[Path] = None


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be a strict boolean, got {value!r}")


def _normalised_category(value: Any) -> str:
    text = str(value).strip().casefold()
    return text if text and text not in {"nan", "none", "null"} else "__missing__"


def _fixed_age_bin(value: Any) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(age) or age < 0.0 or age > 125.0:
        return "missing"
    index = int(np.searchsorted(np.asarray(AGE_BIN_EDGES[1:-1]), age, side="right"))
    return AGE_BIN_LABELS[index]


def _presence_array(frame: pd.DataFrame, group: str) -> np.ndarray:
    if group not in PRIMARY_MODALITIES:
        raise ValueError(f"unknown primary modality {group!r}")
    flag = SOURCE_FLAG_BY_GROUP[group]
    if flag not in frame:
        raise ValueError(f"cohort is missing source-presence column {flag!r}")
    return np.asarray(
        [_strict_bool(value, flag) for value in frame[flag].tolist()], dtype=bool
    )


def _matching_frame(frame: pd.DataFrame, group: str) -> tuple[pd.DataFrame, list[str]]:
    """Construct only pre-specified non-outcome matching fields.

    This function deliberately never indexes ``frame['y']``.  Its return value
    contains no label or outcome-derived statistic, which makes that property
    straightforward to test and audit.
    """

    result = pd.DataFrame(index=np.arange(len(frame)))
    demographic_fields: list[str] = []
    if "clinical_site" in frame:
        result["clinical_site"] = frame["clinical_site"].map(_normalised_category).tolist()
        demographic_fields.append("clinical_site")

    sex_column = next((name for name in SEX_COLUMN_CANDIDATES if name in frame), None)
    if sex_column is not None:
        result["sex"] = frame[sex_column].map(_normalised_category).tolist()
        demographic_fields.append("sex")

    if "age" in frame:
        result["age_bin"] = frame["age"].map(_fixed_age_bin).tolist()
        demographic_fields.append("age_bin")

    other_flags: list[str] = []
    for other_group in PRIMARY_MODALITIES:
        if other_group == group:
            continue
        flag = SOURCE_FLAG_BY_GROUP[other_group]
        if flag not in frame:
            raise ValueError(
                "matched permutation requires every other-modality availability "
                f"column; missing {flag!r}"
            )
        other_flags.append(flag)
    signatures: list[str] = []
    for row in frame[other_flags].itertuples(index=False, name=None):
        signatures.append(
            "".join("1" if _strict_bool(value, flag) else "0" for value, flag in zip(row, other_flags))
        )
    result["other_modality_availability"] = signatures
    return result, demographic_fields


def _candidate_matching_levels(demographic_fields: Sequence[str]) -> list[list[str]]:
    """Return a deterministic fine-to-coarse relaxation hierarchy."""

    demographics = list(demographic_fields)
    candidates: list[list[str]] = [demographics + ["other_modality_availability"], demographics]
    # Preserve clinical site as long as possible because acquisition/site shift
    # is an important nuisance factor in this dataset.
    if "clinical_site" in demographics:
        for tail in (
            [name for name in ("sex", "age_bin") if name in demographics],
            [name for name in ("sex",) if name in demographics],
            [name for name in ("age_bin",) if name in demographics],
            [],
        ):
            candidates.append(["clinical_site", *tail])
    if "sex" in demographics and "age_bin" in demographics:
        candidates.append(["sex", "age_bin"])
    for name in ("sex", "age_bin"):
        if name in demographics:
            candidates.append([name])
    candidates.append([])
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for fields in candidates:
        key = tuple(fields)
        if key not in seen:
            seen.add(key)
            unique.append(fields)
    return unique


def _stratum_members(
    covariates: pd.DataFrame,
    present_indices: np.ndarray,
    fields: Sequence[str],
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    if not fields:
        return [((), present_indices.copy())]
    groups: Dict[tuple[str, ...], list[int]] = {}
    for index in present_indices.tolist():
        key = tuple(str(covariates.at[index, field]) for field in fields)
        groups.setdefault(key, []).append(index)
    return [
        (key, np.asarray(indices, dtype=np.int64))
        for key, indices in sorted(groups.items(), key=lambda item: item[0])
    ]


def _deranged_within_strata(
    n_rows: int,
    strata: Sequence[tuple[tuple[str, ...], np.ndarray]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[int, tuple[str, ...]]]:
    donors = np.arange(n_rows, dtype=np.int64)
    recipient_strata: Dict[int, tuple[str, ...]] = {}
    for key, members in strata:
        if len(members) < 2:
            raise ValueError("cannot derange a singleton permutation stratum")
        order = rng.permutation(members)
        shift = int(rng.integers(1, len(order)))
        shifted = np.roll(order, shift)
        donors[order] = shifted
        for recipient in members.tolist():
            recipient_strata[int(recipient)] = key
    return donors, recipient_strata


def build_permutation_plan(
    frame: pd.DataFrame,
    group: str,
    *,
    seed: int,
    matched: bool,
) -> PermutationPlan:
    """Build a present-only, without-replacement, fixed-point-free mapping.

    Matching selects the finest common hierarchy level for which every stratum
    has at least two participants.  Rejected levels and their singleton counts
    are recorded.  Using a common level retains a true permutation/bijection;
    ad-hoc nearest-neighbour donor reuse would not.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("cohort frame must be a non-empty DataFrame")
    if "person_id" not in frame:
        raise ValueError("cohort frame is missing person_id")
    if frame["person_id"].astype(str).duplicated().any():
        raise ValueError("cohort person_id values must be unique")
    present = _presence_array(frame, group)
    present_indices = np.flatnonzero(present).astype(np.int64)
    # Crossing a singleton with an absent participant would change source
    # availability, while returning a zero importance would falsely imply an
    # estimable null effect.  Fail explicitly instead.
    if len(present_indices) < 2:
        raise ValueError(
            f"{group} has {len(present_indices)} source-present participant(s); "
            "at least two are required for permutation"
        )

    rejected: list[Dict[str, Any]] = []
    selected_fields: list[str] = []
    tier_counts: Dict[str, int] = {}
    recipient_strata: Dict[int, tuple[str, ...]] = {}
    recipient_matching_fields: Dict[int, list[str]] = {}
    recipient_matching_levels: Dict[int, str] = {}
    rng = np.random.default_rng(int(seed))
    donors = np.arange(len(frame), dtype=np.int64)
    if matched:
        covariates, demographics = _matching_frame(frame, group)
        unresolved = present_indices.copy()
        accepted_strata: list[tuple[tuple[str, ...], np.ndarray]] = []
        levels = _candidate_matching_levels(demographics)
        for level_index, fields in enumerate(levels):
            if len(unresolved) < 2:
                break
            candidate = _stratum_members(covariates, unresolved, fields)
            sizes = [len(indices) for _, indices in candidate]
            n_singletons = sum(size == 1 for size in sizes)
            level_audit = {
                "level": int(level_index),
                "fields": list(fields),
                "n_strata": len(candidate),
                "minimum_stratum_size": min(sizes),
                "n_singleton_strata": n_singletons,
            }
            accepted = [(key, members) for key, members in candidate if len(members) >= 2]
            n_accepted = sum(len(members) for _, members in accepted)
            level_audit["n_participants_matched_at_level"] = int(n_accepted)
            rejected.append(level_audit)
            if not accepted:
                continue
            level_donors, level_keys = _deranged_within_strata(len(frame), accepted, rng)
            accepted_indices = np.concatenate([members for _, members in accepted])
            donors[accepted_indices] = level_donors[accepted_indices]
            recipient_strata.update(level_keys)
            accepted_strata.extend(accepted)
            tier_name = f"level_{level_index}:" + "+".join(fields or ["unrestricted"])
            tier_counts[tier_name] = int(n_accepted)
            for recipient in accepted_indices.tolist():
                recipient_matching_fields[int(recipient)] = list(fields)
                recipient_matching_levels[int(recipient)] = tier_name
            accepted_set = set(accepted_indices.tolist())
            unresolved = np.asarray(
                [index for index in unresolved.tolist() if index not in accepted_set],
                dtype=np.int64,
            )
        if len(unresolved):
            tier_counts["unpermutable"] = int(len(unresolved))
        selected_fields = sorted(
            {
                field
                for item in rejected
                if int(item["n_participants_matched_at_level"]) > 0
                for field in item["fields"]
            }
        )
        selected_strata = accepted_strata
        n_unpermutable = int(len(unresolved))
    else:
        selected_strata = [((), present_indices)]
        donors, recipient_strata = _deranged_within_strata(
            len(frame), selected_strata, rng
        )
        tier_counts = {"unmatched": int(len(present_indices))}
        n_unpermutable = 0
        for recipient in present_indices.tolist():
            recipient_matching_fields[int(recipient)] = []
            recipient_matching_levels[int(recipient)] = "unmatched"
    absent_indices = np.flatnonzero(~present)
    if not np.array_equal(donors[absent_indices], absent_indices):
        raise RuntimeError("internal error: absent participants were permuted")
    permuted_present = present_indices[donors[present_indices] != present_indices]
    if np.any(donors[permuted_present] == permuted_present):
        raise RuntimeError("internal error: permutable present mapping has fixed points")
    if set(donors[present_indices].tolist()) != set(present_indices.tolist()):
        raise RuntimeError("internal error: source-present donors are not a bijection")
    if not present[donors[present_indices]].all():
        raise RuntimeError("internal error: an absent donor entered the permutation")

    stratum_keys = {
        str(index): list(recipient_strata.get(index, ()))
        for index in present_indices.tolist()
    }
    audit: Dict[str, Any] = {
        "group": group,
        "matched": bool(matched),
        "seed": int(seed),
        "n_cohort": int(len(frame)),
        "n_present": int(len(present_indices)),
        "n_source_present": int(len(present_indices)),
        "n_unpermutable_present": n_unpermutable,
        "n_source_absent_unchanged": int(len(absent_indices)),
        "n_changed": int(np.sum(donors != np.arange(len(frame)))),
        "without_replacement": True,
        "fixed_points_among_permutable_present": 0,
        "matching_fields": selected_fields,
        "matching_level_relaxations": rejected,
        "relaxation_counts": tier_counts,
        "n_strata": int(len(selected_strata)),
        "minimum_stratum_size": int(min((len(indices) for _, indices in selected_strata), default=0)),
        "maximum_stratum_size": int(max((len(indices) for _, indices in selected_strata), default=0)),
        "age_bin_edges": ["-inf", 40, 50, 60, 70, 80, "inf"],
        "outcome_fields_used": [],
        "recipient_stratum_keys": stratum_keys,
        "recipient_matching_fields": {
            str(index): fields for index, fields in recipient_matching_fields.items()
        },
        "recipient_matching_levels": {
            str(index): level for index, level in recipient_matching_levels.items()
        },
    }
    return PermutationPlan(donor_indices=donors, audit=audit)


def _group_keys(sample: Mapping[str, Any], group: str) -> set[str]:
    flag = SOURCE_FLAG_BY_GROUP[group]
    prefixes = {
        "clinical_data": ("clinical_",),
        "cardiac_ecg": ("ecg_",),
        "environment": ("environment_",),
        "retinal_flio": ("retinal_flio_",),
        "retinal_oct": ("retinal_oct_",),
        "retinal_octa": ("retinal_octa_",),
        "retinal_photography": ("retinal_photography_",),
        "wearable_activity_monitor": ("activity_",),
        "wearable_blood_glucose": ("cgm_",),
    }[group]
    keys = {
        key
        for key in sample
        if key == group
        or key == flag
        or any(key.startswith(prefix) for prefix in prefixes)
    }
    # Canonical value tensors use these unprefixed names.
    unprefixed = {
        "environment": "environment",
        "wearable_activity_monitor": "activity",
        "wearable_blood_glucose": "cgm",
    }.get(group)
    if unprefixed in sample:
        keys.add(str(unprefixed))
    return keys


def _bool_scalar(value: Any, label: str) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be scalar")
        return bool(value.detach().cpu().item())
    return _strict_bool(value, label)


def _same_label(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return bool(torch.equal(torch.as_tensor(left).detach().cpu(), torch.as_tensor(right).detach().cpu()))
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return left == right


class BundlePermutationDataset(Dataset):
    """Dataset wrapper that substitutes a complete source before collation."""

    def __init__(
        self,
        base_dataset: Dataset,
        group: str,
        donor_indices: Sequence[int] | np.ndarray,
    ) -> None:
        if group not in PRIMARY_MODALITIES:
            raise ValueError(f"unknown primary modality {group!r}")
        donors = np.asarray(donor_indices, dtype=np.int64)
        self.base_dataset = base_dataset
        self.group = group
        self.present = np.zeros(len(base_dataset), dtype=bool)
        self.source_index = PRIMARY_MODALITIES.index(group)
        # Shared storage lets persistent DataLoader workers see a new repeat's
        # donor map without respawning worker processes hundreds of times.
        self._donor_indices_tensor = torch.empty(len(base_dataset), dtype=torch.int64)
        try:
            self._donor_indices_tensor.share_memory_()
            self.donor_mapping_shared = True
        except RuntimeError:
            # Some restricted containers disable torch_shm_manager.  The
            # analysis remains correct; run_importance_analysis will rebuild
            # worker loaders rather than publishing maps to persistent workers.
            self.donor_mapping_shared = False
        self._set_and_validate_donors(donors)

    def _set_and_validate_donors(self, donors: np.ndarray) -> None:
        if donors.shape != (len(self.base_dataset),):
            raise ValueError(
                f"donor_indices must have shape {(len(self.base_dataset),)}, got {donors.shape}"
            )
        if np.any(donors < 0) or np.any(donors >= len(self.base_dataset)):
            raise ValueError("donor_indices contains an out-of-range index")
        index = getattr(self.base_dataset, "index", None)
        if not isinstance(index, pd.DataFrame):
            raise TypeError("base_dataset must expose its aligned cohort as .index")
        present = _presence_array(index, self.group)
        recipients = np.arange(len(self.base_dataset))
        if np.any(donors[~present] != recipients[~present]):
            raise ValueError("source-absent recipients must map to themselves")
        if np.any(~present[donors[present]]):
            raise ValueError("every source-present recipient must have a present donor")
        self.present = present
        self._donor_indices_tensor.copy_(torch.as_tensor(donors, dtype=torch.int64))

    @property
    def donor_indices(self) -> np.ndarray:
        return self._donor_indices_tensor.detach().cpu().numpy().copy()

    def set_donor_indices(self, donor_indices: Sequence[int] | np.ndarray) -> None:
        """Atomically publish the next repeat's validated map to workers."""

        donors = np.asarray(donor_indices, dtype=np.int64)
        self._set_and_validate_donors(donors)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        recipient_index = int(index)
        donor_index = int(self._donor_indices_tensor[recipient_index].item())
        recipient_raw = self.base_dataset[recipient_index]
        if not isinstance(recipient_raw, Mapping):
            raise TypeError("base dataset samples must be mappings")
        recipient = dict(recipient_raw)
        if donor_index == recipient_index:
            return recipient
        donor_raw = self.base_dataset[donor_index]
        if not isinstance(donor_raw, Mapping):
            raise TypeError("base dataset samples must be mappings")
        donor = dict(donor_raw)

        protected = {
            key: recipient.get(key)
            for key in ("person_id", "y", "label_value", "task", "split")
            if key in recipient
        }
        recipient_keys = _group_keys(recipient, self.group)
        donor_keys = _group_keys(donor, self.group)
        if not donor_keys:
            raise ValueError(f"donor sample exposes no keys for {self.group}")
        for key in recipient_keys:
            recipient.pop(key, None)
        for key in donor_keys:
            recipient[key] = donor[key]

        # source_present contains all nine groups, so copy only the target bit.
        if "source_present" in recipient_raw:
            if "source_present" not in donor:
                raise ValueError("donor is missing source_present")
            target_presence = recipient_raw["source_present"]
            donor_presence = donor["source_present"]
            if not isinstance(target_presence, torch.Tensor) or not isinstance(donor_presence, torch.Tensor):
                raise TypeError("source_present must be a tensor before collation")
            if target_presence.shape != (len(PRIMARY_MODALITIES),) or donor_presence.shape != target_presence.shape:
                raise ValueError("unbatched source_present must have shape [9]")
            combined = target_presence.clone()
            combined[self.source_index] = donor_presence[self.source_index]
            recipient["source_present"] = combined

        flag = SOURCE_FLAG_BY_GROUP[self.group]
        if flag not in recipient or not _bool_scalar(recipient[flag], flag):
            raise ValueError("permutation donor/recipient must both be source-present")
        for key, expected in protected.items():
            if key not in recipient or not _same_label(recipient[key], expected):
                raise RuntimeError(f"protected recipient field {key!r} changed during permutation")
        return recipient


def degradation_delta(metric: str, baseline: float, perturbed: float) -> float:
    """Return a positive number when the perturbation makes performance worse."""

    name = str(metric).strip().lower()
    if name in BENEFIT_METRICS:
        return float(baseline) - float(perturbed)
    if name in LOSS_METRICS:
        return float(perturbed) - float(baseline)
    raise ValueError(f"unsupported importance metric {metric!r}")


def _bootstrap_indices_stratified(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        if len(members) == 0:
            raise ValueError("paired stratified bootstrap requires both label classes")
        sampled.append(rng.choice(members, size=len(members), replace=True))
    combined = np.concatenate(sampled)
    return combined[rng.permutation(len(combined))]


def _binary_metrics_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, np.ndarray]:
    """Exact tie-aware binary metrics for many prediction rows at once.

    Bootstrap duplicates create many tied probabilities.  Sorting once along
    the participant axis and evaluating only tie-group endpoints reproduces
    ROC trapezoids and non-interpolated average precision without thousands of
    per-repeat scikit-learn calls.
    """

    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("probability matrix is not aligned to labels")
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("matrix ranking metrics require both outcome classes")

    predicted = matrix >= float(threshold)
    sensitivity = predicted[:, positive].mean(axis=1)
    specificity = (~predicted[:, negative]).mean(axis=1)
    balanced = 0.5 * (sensitivity + specificity)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(matrix, epsilon, 1.0 - epsilon)
    labels_float = labels.astype(np.float64)[None, :]
    log_loss = -np.mean(
        labels_float * np.log(clipped)
        + (1.0 - labels_float) * np.log1p(-clipped),
        axis=1,
    )
    brier = np.mean((matrix - labels_float) ** 2, axis=1)

    order = np.argsort(-matrix, axis=1, kind="stable")
    sorted_probability = np.take_along_axis(matrix, order, axis=1)
    sorted_positive = np.take_along_axis(
        np.broadcast_to(positive, matrix.shape), order, axis=1
    ).astype(np.float64)
    cumulative_tp = np.cumsum(sorted_positive, axis=1)
    cumulative_fp = np.cumsum(1.0 - sorted_positive, axis=1)
    group_end = np.ones(matrix.shape, dtype=bool)
    if matrix.shape[1] > 1:
        group_end[:, :-1] = (
            sorted_probability[:, :-1] != sorted_probability[:, 1:]
        )

    previous_end_tp_seed = np.zeros_like(cumulative_tp)
    previous_end_fp_seed = np.zeros_like(cumulative_fp)
    if matrix.shape[1] > 1:
        previous_end_tp_seed[:, 1:] = np.where(
            group_end[:, :-1], cumulative_tp[:, :-1], 0.0
        )
        previous_end_fp_seed[:, 1:] = np.where(
            group_end[:, :-1], cumulative_fp[:, :-1], 0.0
        )
    previous_tp = np.maximum.accumulate(previous_end_tp_seed, axis=1)
    previous_fp = np.maximum.accumulate(previous_end_fp_seed, axis=1)
    fp_increment = np.where(group_end, cumulative_fp - previous_fp, 0.0)
    auroc = np.sum(
        fp_increment * (cumulative_tp + previous_tp), axis=1
    ) / (2.0 * float(n_positive * n_negative))

    positive_in_group = np.where(
        group_end, cumulative_tp - previous_tp, 0.0
    )
    positions = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)[None, :]
    precision_at_end = cumulative_tp / positions
    auprc = np.sum(precision_at_end * positive_in_group, axis=1) / float(
        n_positive
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "log_loss": log_loss,
        "brier": brier,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def paired_stratified_bootstrap_deltas(
    y_true: Any,
    baseline_probabilities: Any,
    perturbed_probabilities: Any,
    *,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    confidence_level: float = 0.95,
) -> list[Dict[str, Any]]:
    """Paired participant/repeat bootstrap intervals for degradation deltas.

    ``perturbed_probabilities`` has shape ``[repeats, participants]`` (a vector
    is interpreted as one repeat).  Each replicate samples participants within
    label class and samples the observed permutation repeats with replacement.
    Baseline and perturbed predictions always use identical participant draws.
    """

    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64).reshape(-1)
    perturbed = np.asarray(perturbed_probabilities, dtype=np.float64)
    if perturbed.ndim == 1:
        perturbed = perturbed.reshape(1, -1)
    if perturbed.ndim != 2 or perturbed.shape[1] != len(labels):
        raise ValueError("perturbed probabilities must have shape [repeats, participants]")
    if baseline.shape != labels.shape:
        raise ValueError("baseline probabilities and labels are not aligned")
    if len(labels) < 2 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("paired stratified bootstrap requires finite binary labels of both classes")
    if not np.isfinite(baseline).all() or not np.isfinite(perturbed).all():
        raise ValueError("bootstrap probabilities must be finite")
    if np.any((baseline < 0) | (baseline > 1)) or np.any((perturbed < 0) | (perturbed > 1)):
        raise ValueError("bootstrap probabilities must lie in [0, 1]")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    observed_metrics = _binary_metrics_matrix(
        labels,
        np.concatenate([baseline.reshape(1, -1), perturbed], axis=0),
        threshold=threshold,
    )
    point = {
        metric: float(
            np.mean(
                observed_metrics[metric][0] - observed_metrics[metric][1:]
                if metric in BENEFIT_METRICS
                else observed_metrics[metric][1:] - observed_metrics[metric][0]
            )
        )
        for metric in METRIC_NAMES
    }

    rng = np.random.default_rng(int(seed))
    bootstrap_values = {
        metric: np.empty(int(n_bootstrap), dtype=np.float64) for metric in METRIC_NAMES
    }
    for sample_number in range(int(n_bootstrap)):
        indices = _bootstrap_indices_stratified(labels, rng)
        repeat_indices = rng.integers(0, perturbed.shape[0], size=perturbed.shape[0])
        sampled_probabilities = np.concatenate(
            [
                baseline[indices].reshape(1, -1),
                perturbed[repeat_indices][:, indices],
            ],
            axis=0,
        )
        sampled_metrics = _binary_metrics_matrix(
            labels[indices], sampled_probabilities, threshold=threshold
        )
        for metric in METRIC_NAMES:
            baseline_value = sampled_metrics[metric][0]
            perturbed_values = sampled_metrics[metric][1:]
            deltas = (
                baseline_value - perturbed_values
                if metric in BENEFIT_METRICS
                else perturbed_values - baseline_value
            )
            bootstrap_values[metric][sample_number] = float(np.mean(deltas))

    alpha = (1.0 - float(confidence_level)) / 2.0
    records: list[Dict[str, Any]] = []
    for metric in METRIC_NAMES:
        values = bootstrap_values[metric]
        finite = values[np.isfinite(values)]
        if not math.isfinite(point[metric]) or len(finite) == 0:
            lower = upper = math.nan
        else:
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha]).tolist()
        records.append(
            {
                "metric": metric,
                "delta_definition": (
                    "baseline_minus_perturbed"
                    if metric in BENEFIT_METRICS
                    else "perturbed_minus_baseline"
                ),
                # ``delta`` is the concise machine-facing alias; both fields
                # use the same degradation-positive convention.
                "delta": point[metric],
                "point_estimate": point[metric],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": float(confidence_level),
                "n_bootstrap_requested": int(n_bootstrap),
                "n_bootstrap_finite": int(len(finite)),
                "n_permutation_repeats": int(perturbed.shape[0]),
                "bootstrap_design": "paired_participant_stratified_by_y_and_repeat_resampling",
            }
        )
    return records


def _batch_strings(value: Any) -> list[str]:
    if isinstance(value, torch.Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _person_ids_sha256(person_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for person_id in person_ids:
        encoded = str(person_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_matching_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return JSON-safe, label-free covariates needed by donor planning."""

    columns = ["person_id"]
    columns.extend(SOURCE_FLAG_BY_GROUP[group] for group in PRIMARY_MODALITIES)
    for candidate in ("clinical_site", *SEX_COLUMN_CANDIDATES, "age"):
        if candidate in frame and candidate not in columns:
            columns.append(candidate)
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(
            "source-token cache matching metadata is missing columns: "
            f"{missing}"
        )
    # pandas' JSON encoder normalizes numpy scalars, timestamps and missing
    # values without allowing arbitrary pickle objects into this audit block.
    records = json.loads(
        frame.loc[:, columns].to_json(
            orient="records", date_format="iso", date_unit="ns"
        )
    )
    return {"columns": columns, "records": records}


def _matching_frame_from_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns")
    records = payload.get("records")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(str(value) for value in columns))
        or not isinstance(records, list)
    ):
        raise ValueError("source-token cache matching metadata is malformed")
    names = [str(value) for value in columns]
    if "y" in names or "label_value" in names:
        raise ValueError("source-token cache donor metadata must be outcome-free")
    frame = pd.DataFrame(records, columns=names)
    if "person_id" not in frame:
        raise ValueError("source-token cache matching metadata lacks person_id")
    return frame


def _hash_tensor(digest: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))


def _source_token_cache_content_hash(
    *,
    identity: Mapping[str, Any],
    person_ids: Sequence[str],
    y_true: torch.Tensor,
    full_forward_logits: torch.Tensor,
    source_tokens: torch.Tensor,
    source_present: torch.Tensor,
    matching_metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            dict(identity), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    )
    digest.update(_person_ids_sha256(person_ids).encode("ascii"))
    digest.update(
        json.dumps(
            dict(matching_metadata),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    _hash_tensor(digest, "y_true", y_true)
    _hash_tensor(digest, "full_forward_logits", full_forward_logits)
    _hash_tensor(digest, "source_tokens", source_tokens)
    _hash_tensor(digest, "source_present", source_present)
    return digest.hexdigest()


def _atomic_torch_save_no_overwrite(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically publish a cache while refusing to replace prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing source-token cache: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # A concurrent writer wins rather than being silently replaced.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"source-token cache appeared concurrently; refusing overwrite: {path}"
            ) from exc
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_cache_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(
        json.dumps(dict(identity), sort_keys=True, default=str)
    )
    if not isinstance(normalized, dict):
        raise ValueError("source-token cache identity must be a mapping")
    return normalized


def _validate_cache_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_cache_identity(identity)
    required = {
        "checkpoint_sha256",
        "config_hash",
        "master_config_hash",
        "scaler_config_hash",
        "split",
        "source_order",
        "model_config_hash",
        "task",
        "encoder_batch_size",
        "amp_mode",
        "cohort_person_ids_sha256",
        "n_participants",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(
            "persistent source-token cache identity is incomplete: "
            f"{missing}"
        )
    for name in (
        "checkpoint_sha256",
        "config_hash",
        "master_config_hash",
        "scaler_config_hash",
        "model_config_hash",
        "task",
        "cohort_person_ids_sha256",
    ):
        if not str(normalized.get(name) or "").strip():
            raise ValueError(f"source-token cache identity {name} is empty")
    if normalized.get("source_order") != list(PRIMARY_MODALITIES):
        raise ValueError("source-token cache source order is not canonical")
    if str(normalized.get("split") or "").lower() not in {"val", "test"}:
        raise ValueError("source-token cache split must be val or test")
    if normalized.get("amp_mode") not in {"float32", "bfloat16_autocast"}:
        raise ValueError("source-token cache amp_mode is invalid")
    for name in ("encoder_batch_size", "n_participants"):
        value = normalized.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source-token cache identity {name} must be positive")
    return normalized


def _validate_source_token_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_frame: pd.DataFrame,
    path: Optional[Path],
    reused: bool,
) -> SourceTokenCache:
    if payload.get("schema_version") != SOURCE_TOKEN_CACHE_FORMAT_VERSION:
        raise ValueError("source-token cache has the wrong schema version")
    identity = _validate_cache_identity(
        _mapping(payload.get("identity"), "source-token cache identity")
    )
    expected = _validate_cache_identity(expected_identity)
    if identity != expected:
        observed_hash = canonical_json_hash(identity)
        expected_hash = canonical_json_hash(expected)
        raise ValueError(
            "source-token cache identity mismatch; refusing stale evidence "
            f"(observed={observed_hash}, expected={expected_hash})"
        )
    tensors = _mapping(payload.get("tensors"), "source-token cache tensors")
    source_tokens = tensors.get("source_tokens")
    source_present = tensors.get("source_present")
    y_true = tensors.get("y_true")
    full_logits = tensors.get("full_forward_logits")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (source_tokens, source_present, y_true, full_logits)
    ):
        raise ValueError("source-token cache tensor block is incomplete")
    assert isinstance(source_tokens, torch.Tensor)
    assert isinstance(source_present, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(full_logits, torch.Tensor)
    n_people = int(identity["n_participants"])
    if source_tokens.ndim != 3 or tuple(source_tokens.shape[:2]) != (
        n_people,
        len(PRIMARY_MODALITIES),
    ):
        raise ValueError("source-token cache source_tokens shape is invalid")
    if source_present.shape != source_tokens.shape[:2]:
        raise ValueError("source-token cache source_present shape is invalid")
    if y_true.shape != (n_people,) or full_logits.shape != (n_people,):
        raise ValueError("source-token cache outcome/logit shape is invalid")
    if source_tokens.device.type != "cpu" or source_present.device.type != "cpu":
        raise ValueError("source-token cache tensors must be stored on CPU")
    if not torch.isfinite(source_tokens).all() or not torch.isfinite(full_logits).all():
        raise ValueError("source-token cache contains non-finite values")
    if source_present.dtype is not torch.bool:
        raise ValueError("source-token cache source_present must be boolean")
    y_long = y_true.to(dtype=torch.int64)
    if not torch.equal(y_true, y_long) or not torch.all((y_long == 0) | (y_long == 1)):
        raise ValueError("source-token cache labels must be binary integers")
    person_ids_value = payload.get("person_ids")
    if not isinstance(person_ids_value, list):
        raise ValueError("source-token cache person_ids is malformed")
    person_ids = tuple(str(value) for value in person_ids_value)
    if len(person_ids) != n_people or len(set(person_ids)) != n_people:
        raise ValueError("source-token cache person IDs must be aligned and unique")
    current_ids = tuple(str(value) for value in expected_frame["person_id"].tolist())
    if person_ids != current_ids:
        raise ValueError("source-token cache person order differs from current cohort")
    current_y = pd.to_numeric(expected_frame["y"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(y_long.numpy(), current_y):
        raise ValueError("source-token cache labels differ from current cohort")
    matching_metadata = _mapping(
        payload.get("matching_metadata"), "source-token cache matching metadata"
    )
    current_matching = _cache_matching_payload(expected_frame)
    if canonical_json_hash(matching_metadata) != canonical_json_hash(current_matching):
        raise ValueError(
            "source-token cache matching metadata differs from current cohort"
        )
    matching_frame = _matching_frame_from_payload(matching_metadata)
    if tuple(str(value) for value in matching_frame["person_id"].tolist()) != person_ids:
        raise ValueError("source-token cache matching metadata is misaligned")
    recorded_content_hash = str(payload.get("content_sha256") or "").strip()
    calculated_content_hash = _source_token_cache_content_hash(
        identity=identity,
        person_ids=person_ids,
        y_true=y_long,
        full_forward_logits=full_logits,
        source_tokens=source_tokens,
        source_present=source_present,
        matching_metadata=matching_metadata,
    )
    if not recorded_content_hash or recorded_content_hash != calculated_content_hash:
        raise ValueError("source-token cache content hash mismatch")
    return SourceTokenCache(
        source_tokens=source_tokens.detach().cpu(),
        source_present=source_present.detach().cpu().bool(),
        person_ids=person_ids,
        y_true=y_long.detach().cpu().numpy(),
        full_forward_logits=full_logits.detach().cpu().float().numpy(),
        matching_frame=matching_frame,
        identity=identity,
        content_sha256=recorded_content_hash,
        reused=bool(reused),
        path=path,
    )


@torch.no_grad()
def build_or_load_source_token_cache(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    frame: pd.DataFrame,
    device: str | torch.device,
    task: str,
    expected_split: str,
    identity: Mapping[str, Any],
    cache_path: Optional[Path] = None,
    use_bf16_amp: bool = False,
) -> SourceTokenCache:
    """Read/encode the cohort once, or fail-closed on a stale disk cache."""

    normalized_identity = _validate_cache_identity(identity)
    target_device = torch.device(device)
    model.eval()
    if str(normalized_identity["task"]) != str(task):
        raise ValueError("source-token cache identity task differs from current run")
    if str(normalized_identity["split"]).lower() != str(expected_split).lower():
        raise ValueError("source-token cache identity split differs from current run")
    expected_amp = "bfloat16_autocast" if use_bf16_amp else "float32"
    if normalized_identity["amp_mode"] != expected_amp:
        raise ValueError("source-token cache identity autocast mode differs from current run")
    current_ids = tuple(str(value) for value in frame["person_id"].tolist())
    if int(normalized_identity["n_participants"]) != len(current_ids):
        raise ValueError("source-token cache identity participant count is stale")
    if normalized_identity["cohort_person_ids_sha256"] != _person_ids_sha256(
        current_ids
    ):
        raise ValueError("source-token cache identity participant-order hash is stale")
    path = None if cache_path is None else Path(cache_path).expanduser().resolve()
    if path is not None and path.exists():
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        cache = _validate_source_token_cache_payload(
            _mapping(loaded, "source-token cache"),
            expected_identity=normalized_identity,
            expected_frame=frame,
            path=path,
            reused=True,
        )
        equivalence = evaluate_cached_condition(
            model,
            cache,
            device=target_device,
            task=task,
            threshold=0.5,
            batch_size=int(normalized_identity["encoder_batch_size"]),
            use_bf16_amp=use_bf16_amp,
        )
        _assert_cached_baseline_exact(cache, equivalence)
        return cache

    ids: list[str] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    presences: list[torch.Tensor] = []
    for raw_batch in loader:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("source-token cache loader must yield mappings")
        splits = {
            item.strip().lower()
            for item in _batch_strings(raw_batch.get("split", []))
        }
        if splits != {str(expected_split).strip().lower()}:
            raise ValueError(
                f"expected split {expected_split!r}, observed {sorted(splits)}"
            )
        batch = recursive_to_device(
            raw_batch, target_device, non_blocking=target_device.type == "cuda"
        )
        y = torch.as_tensor(
            batch.get("y"), dtype=torch.float32, device=target_device
        ).reshape(-1)
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("source-token cache labels must be binary")
        amp = (
            torch.autocast(device_type=target_device.type, dtype=torch.bfloat16)
            if use_bf16_amp and target_device.type in {"cpu", "cuda"}
            else contextlib.nullcontext()
        )
        with amp:
            output = model(batch, task=task)
        if not isinstance(output, Mapping):
            raise TypeError("cacheable model output must be a mapping")
        batch_tokens = output.get("source_tokens")
        batch_present = output.get("source_present")
        batch_logits = output.get("logits", output.get("y_logit"))
        if not isinstance(batch_tokens, torch.Tensor) or not isinstance(
            batch_present, torch.Tensor
        ):
            raise TypeError(
                "cacheable model must expose source_tokens and source_present"
            )
        if not isinstance(batch_logits, torch.Tensor):
            raise TypeError("cacheable model must expose logits")
        batch_logits = batch_logits.reshape(-1)
        expected_shape = (len(y), len(PRIMARY_MODALITIES))
        if tuple(batch_tokens.shape[:2]) != expected_shape:
            raise ValueError("model source tokens do not follow canonical source order")
        if batch_present.shape != batch_tokens.shape[:2]:
            raise ValueError("model source presence is not aligned to source tokens")
        if batch_logits.shape != y.shape:
            raise ValueError("model logits are not aligned to source tokens")
        batch_ids = _batch_strings(raw_batch.get("person_id", []))
        if len(batch_ids) != len(y):
            raise ValueError("person IDs are missing or misaligned")
        ids.extend(batch_ids)
        labels.append(y.detach().cpu().to(dtype=torch.int64))
        logits.append(batch_logits.detach().cpu().float())
        tokens.append(batch_tokens.detach().cpu())
        presences.append(batch_present.detach().cpu().bool())
    if not ids:
        raise ValueError("source-token cache loader yielded no batches")
    tensors = {
        "source_tokens": torch.cat(tokens, dim=0).contiguous(),
        "source_present": torch.cat(presences, dim=0).contiguous(),
        "y_true": torch.cat(labels, dim=0).contiguous(),
        "full_forward_logits": torch.cat(logits, dim=0).contiguous(),
    }
    matching_metadata = _cache_matching_payload(frame)
    content_hash = _source_token_cache_content_hash(
        identity=normalized_identity,
        person_ids=ids,
        matching_metadata=matching_metadata,
        **tensors,
    )
    payload = {
        "schema_version": SOURCE_TOKEN_CACHE_FORMAT_VERSION,
        "identity": normalized_identity,
        "person_ids": ids,
        "matching_metadata": matching_metadata,
        "tensors": tensors,
        "content_sha256": content_hash,
    }
    cache = _validate_source_token_cache_payload(
        payload,
        expected_identity=normalized_identity,
        expected_frame=frame,
        path=path,
        reused=False,
    )
    equivalence = evaluate_cached_condition(
        model,
        cache,
        device=target_device,
        task=task,
        threshold=0.5,
        batch_size=int(normalized_identity["encoder_batch_size"]),
        use_bf16_amp=use_bf16_amp,
    )
    _assert_cached_baseline_exact(cache, equivalence)
    if path is not None:
        _atomic_torch_save_no_overwrite(payload, path)
    return cache


@torch.no_grad()
def evaluate_cached_condition(
    model: torch.nn.Module,
    cache: SourceTokenCache,
    *,
    device: str | torch.device,
    task: str,
    threshold: float,
    batch_size: int,
    disabled_groups: Optional[Iterable[str]] = None,
    donor_indices: Optional[Sequence[int] | np.ndarray] = None,
    permutation_group: Optional[str] = None,
    use_bf16_amp: bool = False,
    clinical_token_override: Optional[torch.Tensor] = None,
    clinical_present_override: Optional[torch.Tensor] = None,
) -> EvaluationResult:
    """Evaluate fusion/head only after an exact cached-token intervention."""

    if int(batch_size) <= 0:
        raise ValueError("cached evaluation batch_size must be positive")
    predict = getattr(model, "predict_from_source_tokens", None)
    if not callable(predict):
        raise TypeError("model does not expose predict_from_source_tokens")
    disabled = frozenset(disabled_groups or ())
    unknown = disabled.difference(PRIMARY_MODALITIES)
    if unknown:
        raise ValueError(f"unknown cached disabled groups: {sorted(unknown)}")
    if donor_indices is not None and permutation_group is None:
        raise ValueError("cached donor_indices require permutation_group")
    if permutation_group is not None and permutation_group not in PRIMARY_MODALITIES:
        raise ValueError(f"unknown cached permutation group {permutation_group!r}")
    if disabled and donor_indices is not None:
        raise ValueError("cached ablation and permutation are mutually exclusive")
    clinical_index = tuple(PRIMARY_MODALITIES).index("clinical_data")
    clinical_tokens = clinical_present = None
    if (clinical_token_override is None) != (clinical_present_override is None):
        raise ValueError("clinical token and presence overrides must be supplied together")
    if clinical_token_override is not None:
        if donor_indices is not None or permutation_group is not None:
            raise ValueError("clinical ablation overrides cannot be combined with permutation")
        if not isinstance(clinical_token_override, torch.Tensor) or not isinstance(
            clinical_present_override, torch.Tensor
        ):
            raise TypeError("clinical overrides must be tensors")
        expected = (len(cache.person_ids), cache.source_tokens.shape[-1])
        if tuple(clinical_token_override.shape) != expected:
            raise ValueError("clinical token override is not aligned to the cache")
        if tuple(clinical_present_override.shape) != (len(cache.person_ids),):
            raise ValueError("clinical presence override is not aligned to the cache")
        if clinical_present_override.dtype != torch.bool:
            raise ValueError("clinical presence override must be boolean")
        if not clinical_token_override.is_floating_point() or not torch.isfinite(
            clinical_token_override
        ).all():
            raise ValueError("clinical token override must contain finite floating-point values")
        clinical_tokens = clinical_token_override.detach().cpu()
        clinical_present = clinical_present_override.detach().cpu()
        if torch.any(clinical_present & ~cache.source_present[:, clinical_index]):
            raise ValueError("clinical ablation cannot create newly present sources")
        if torch.any(clinical_tokens[~clinical_present] != 0):
            raise ValueError("absent clinical overrides must have zero native tokens")
    donors = None
    if donor_indices is not None:
        donors = np.asarray(donor_indices, dtype=np.int64)
        if donors.shape != (len(cache.person_ids),):
            raise ValueError("cached donor mapping has the wrong shape")
        if np.any(donors < 0) or np.any(donors >= len(cache.person_ids)):
            raise ValueError("cached donor mapping contains an invalid index")

    target_device = torch.device(device)
    model.eval()
    logits_parts: list[np.ndarray] = []
    total_bce = 0.0
    n_people = len(cache.person_ids)
    for start in range(0, n_people, int(batch_size)):
        stop = min(start + int(batch_size), n_people)
        source_tokens = cache.source_tokens[start:stop].clone()
        source_present = cache.source_present[start:stop].clone()
        if clinical_tokens is not None:
            source_tokens[:, clinical_index] = clinical_tokens[start:stop].to(
                dtype=source_tokens.dtype
            )
            source_present[:, clinical_index] = clinical_present[start:stop]
        for group in disabled:
            source_present[:, PRIMARY_MODALITIES.index(group)] = False
        if donors is not None and permutation_group is not None:
            source_index = PRIMARY_MODALITIES.index(permutation_group)
            donor_tensor = torch.as_tensor(donors[start:stop], dtype=torch.int64)
            source_tokens[:, source_index] = cache.source_tokens.index_select(
                0, donor_tensor
            )[:, source_index]
            source_present[:, source_index] = cache.source_present.index_select(
                0, donor_tensor
            )[:, source_index]
        source_tokens = source_tokens.to(
            target_device, non_blocking=target_device.type == "cuda"
        )
        source_present = source_present.to(
            target_device, non_blocking=target_device.type == "cuda"
        )
        y = torch.as_tensor(
            cache.y_true[start:stop], dtype=torch.float32, device=target_device
        )
        amp = (
            torch.autocast(device_type=target_device.type, dtype=torch.bfloat16)
            if use_bf16_amp and target_device.type in {"cpu", "cuda"}
            else contextlib.nullcontext()
        )
        with amp:
            output = predict(source_tokens, source_present, task=task)
            value = (
                output.get("logits", output.get("y_logit"))
                if isinstance(output, Mapping)
                else output
            )
            if not isinstance(value, torch.Tensor):
                raise TypeError("cached model output must expose logits")
            batch_logits = value.reshape(-1)
            if batch_logits.shape != y.shape or not torch.isfinite(batch_logits).all():
                raise ValueError("cached model logits must be finite and aligned")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                batch_logits.float(), y, reduction="sum"
            )
        logits_parts.append(batch_logits.detach().cpu().float().numpy())
        total_bce += float(loss.detach().cpu())
    logits_array = np.concatenate(logits_parts).astype(np.float64)
    probabilities = _sigmoid(logits_array)
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = binary_classification_metrics(
        cache.y_true, probabilities=probabilities, threshold=threshold
    )
    metrics["loss"] = total_bce / float(n_people)
    return EvaluationResult(
        metrics=metrics,
        person_ids=list(cache.person_ids),
        y_true=cache.y_true.copy(),
        logits=logits_array,
        probabilities=probabilities,
        predictions=predictions,
        threshold=float(threshold),
    )


def _assert_cached_baseline_exact(
    cache: SourceTokenCache,
    cached_baseline: EvaluationResult,
) -> None:
    if cached_baseline.person_ids != list(cache.person_ids):
        raise RuntimeError("cached baseline person order differs from full forward")
    if not np.array_equal(cached_baseline.y_true, cache.y_true):
        raise RuntimeError("cached baseline labels differ from full forward")
    observed = np.asarray(cached_baseline.logits, dtype=np.float64)
    expected = np.asarray(cache.full_forward_logits, dtype=np.float64)
    difference = np.abs(observed - expected)
    tolerance = 1e-6 + 1e-6 * np.abs(expected)
    if not np.all(difference <= tolerance):
        worst = int(np.argmax(difference))
        raise RuntimeError(
            "cached fusion/head baseline is not equivalent to full model forward: "
            f"person_id={cache.person_ids[worst]!r}, "
            f"full_logit={expected[worst]:.10g}, "
            f"cached_logit={observed[worst]:.10g}, "
            f"abs_difference={difference[worst]:.3g}, "
            f"tolerance={tolerance[worst]:.3g}"
        )


@torch.no_grad()
def evaluate_frozen_condition(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device,
    task: str,
    threshold: float,
    expected_split: str,
    disabled_groups: Optional[Iterable[str]] = None,
    use_bf16_amp: bool = False,
) -> EvaluationResult:
    """Evaluate one frozen-model condition and retain participant pairing."""

    target_device = torch.device(device)
    model.eval()
    ids: list[str] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    total_bce = 0.0
    total_examples = 0
    for raw_batch in loader:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("evaluation loader must yield mappings")
        splits = {item.strip().lower() for item in _batch_strings(raw_batch.get("split", []))}
        if splits != {str(expected_split).strip().lower()}:
            raise ValueError(f"expected split {expected_split!r}, observed {sorted(splits)}")
        batch = recursive_to_device(
            raw_batch, target_device, non_blocking=target_device.type == "cuda"
        )
        y = torch.as_tensor(batch.get("y"), dtype=torch.float32, device=target_device).reshape(-1)
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("evaluation labels must be binary")
        amp = (
            torch.autocast(device_type=target_device.type, dtype=torch.bfloat16)
            if use_bf16_amp and target_device.type in {"cpu", "cuda"}
            else contextlib.nullcontext()
        )
        with amp:
            output = model(batch, task=task, disabled_groups=disabled_groups)
            value = output.get("logits", output.get("y_logit")) if isinstance(output, Mapping) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError("model output must be a tensor or contain logits/y_logit")
            batch_logits = value.reshape(-1)
            if batch_logits.shape != y.shape or not torch.isfinite(batch_logits).all():
                raise ValueError("model logits must be finite and aligned with labels")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                batch_logits.float(), y, reduction="sum"
            )
        batch_ids = _batch_strings(raw_batch.get("person_id", []))
        if len(batch_ids) != len(y):
            raise ValueError("person IDs are missing or misaligned")
        ids.extend(batch_ids)
        labels.append(y.detach().cpu().numpy())
        logits.append(batch_logits.float().detach().cpu().numpy())
        total_bce += float(loss.detach().cpu())
        total_examples += int(len(y))
    if not labels:
        raise ValueError("evaluation loader yielded no batches")
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation person IDs must be unique")
    y_array = np.concatenate(labels).astype(np.int64)
    logits_array = np.concatenate(logits).astype(np.float64)
    probabilities = _sigmoid(logits_array)
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = binary_classification_metrics(
        y_array, probabilities=probabilities, threshold=threshold
    )
    metrics["loss"] = total_bce / float(total_examples)
    return EvaluationResult(
        metrics=metrics,
        person_ids=ids,
        y_true=y_array,
        logits=logits_array,
        probabilities=probabilities,
        predictions=predictions,
        threshold=float(threshold),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _find_bound_hash(provenance: Mapping[str, Any], aliases: Sequence[str]) -> Optional[str]:
    for key in aliases:
        value = str(provenance.get(key) or "").strip()
        if value:
            return value
    # Training provenance nests validated schema hashes under ``lineage`` and
    # file-byte hashes under ``files``.  Search those named containers only;
    # an unconstrained recursive hash search could accidentally accept the
    # hash of a different input.
    lineage = provenance.get("lineage")
    if isinstance(lineage, Mapping):
        for key in aliases:
            value = str(lineage.get(key) or "").strip()
            if value:
                return value
    files = provenance.get("files")
    if isinstance(files, Mapping):
        file_aliases = {
            "master_config_hash": ("master_metadata", "master_metadata_json", "master_csv"),
            "master_hash": ("master_csv", "master_metadata", "master_metadata_json"),
            "scaler_config_hash": ("scaler", "scaler_json"),
            "scaler_hash": ("scaler", "scaler_json"),
        }
        for alias in aliases:
            for file_key in file_aliases.get(alias, ()):
                entry = files.get(file_key)
                if isinstance(entry, Mapping):
                    value = str(entry.get("sha256") or "").strip()
                    if value:
                        return value
    return None


def validate_importance_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_config_hash: str,
    expected_task: str,
    expected_master_hash: Optional[str] = None,
    expected_scaler_hash: Optional[str] = None,
) -> float:
    """Validate best-checkpoint lineage and return its saved val threshold."""

    checkpoint = _mapping(payload, "checkpoint")
    if checkpoint.get("engine_format_version") != ENGINE_FORMAT_VERSION:
        raise ValueError("checkpoint engine format is not OT2 V2")
    if str(checkpoint.get("checkpoint_kind") or "") != "best":
        raise ValueError("importance requires the validation-selected best checkpoint")
    if str(checkpoint.get("task") or "") != str(expected_task):
        raise ValueError("checkpoint task does not match the requested task")
    if not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError("checkpoint is missing model_state")

    config_provenance = _mapping(checkpoint.get("config_provenance"), "config provenance")
    observed_config_hash = str(config_provenance.get("config_hash") or "").strip()
    if not observed_config_hash or observed_config_hash != str(expected_config_hash):
        raise ValueError("checkpoint/config hash mismatch")
    if bool(config_provenance.get("allow_test_set_model_selection", False)):
        raise ValueError("checkpoint provenance permits forbidden test-set model selection")
    subset = _mapping(
        config_provenance.get("modality_subset"),
        "checkpoint modality-subset provenance",
    )
    disabled = subset.get("disabled_groups")
    if not isinstance(disabled, list):
        raise ValueError("checkpoint does not declare its fitted modality subset")
    if disabled:
        raise ValueError(
            "primary modality importance requires a model fitted with all nine groups"
        )

    data_provenance = _mapping(checkpoint.get("data_provenance"), "data provenance")
    scaler_provenance = _mapping(checkpoint.get("scaler_provenance"), "scaler provenance")
    if str(scaler_provenance.get("fit_split") or "") != "recommended_split=train":
        raise ValueError("checkpoint scaler provenance is not official-train-only")
    if expected_master_hash is not None:
        observed = _find_bound_hash(
            data_provenance,
            ("master_config_hash", "master_hash", "config_hash"),
        )
        if observed is None or observed != str(expected_master_hash):
            raise ValueError("checkpoint/master provenance hash mismatch")
    if expected_scaler_hash is not None:
        observed = _find_bound_hash(
            scaler_provenance,
            ("scaler_config_hash", "scaler_hash", "config_hash"),
        )
        if observed is None or observed != str(expected_scaler_hash):
            raise ValueError("checkpoint/scaler provenance hash mismatch")

    selected = _mapping(checkpoint.get("selected_threshold"), "selected threshold")
    try:
        threshold = float(selected.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint selected threshold is invalid") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("checkpoint selected threshold must lie in [0, 1]")
    if str(selected.get("method") or "").strip().lower() not in {"fixed", "youden"}:
        raise ValueError("checkpoint threshold method is not recognized")

    split_declarations = []
    for container in (selected, checkpoint):
        for key in ("selection_split", "threshold_selection_split", "source_split"):
            value = str(container.get(key) or "").strip().lower()
            if value:
                split_declarations.append(value)
    if any(value not in {"val", "validation"} for value in split_declarations):
        raise ValueError("threshold provenance is not validation-only")
    if not isinstance(checkpoint.get("validation_metrics_selected_threshold"), Mapping):
        raise ValueError("checkpoint lacks saved validation-threshold provenance")
    return threshold


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _assert_aligned(baseline: EvaluationResult, candidate: EvaluationResult) -> None:
    if candidate.person_ids != baseline.person_ids:
        raise RuntimeError("perturbed predictions are not aligned to baseline person IDs")
    if not np.array_equal(candidate.y_true, baseline.y_true):
        raise RuntimeError("perturbation changed or reordered outcome labels")


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=task2_v2_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers),
    )


def _metric_record(
    result: EvaluationResult,
    baseline: EvaluationResult,
    *,
    method: str,
    group: str,
    repeat: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"method": method, "group": group, "repeat": int(repeat)}
    for metric in METRIC_NAMES:
        row[metric] = float(result.metrics[metric])
        row[f"delta_{metric}"] = degradation_delta(
            metric, baseline.metrics[metric], result.metrics[metric]
        )
    return row


def _prediction_rows(
    baseline: EvaluationResult,
    result: EvaluationResult,
    *,
    task: str,
    split: str,
    method: str,
    group: str,
    repeat: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for index, person_id in enumerate(baseline.person_ids):
        rows.append(
            {
                "person_id": person_id,
                "task": task,
                "split": split,
                "y_true": int(baseline.y_true[index]),
                "threshold_from_validation": float(baseline.threshold),
                "method": method,
                "group": group,
                "repeat": int(repeat),
                "baseline_logit": float(baseline.logits[index]),
                "baseline_probability": float(baseline.probabilities[index]),
                "perturbed_logit": float(result.logits[index]),
                "perturbed_probability": float(result.probabilities[index]),
                "perturbed_prediction": int(result.predictions[index]),
            }
        )
    return rows


def _cohort_source_presence(
    frame: pd.DataFrame,
    baseline: EvaluationResult,
    source_cache: Optional[SourceTokenCache],
) -> tuple[np.ndarray, str]:
    """Read aligned native availability without another raw-dataset pass."""

    frame_ids = tuple(str(value) for value in frame["person_id"].tolist())
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("cohort availability requires unique participant IDs")
    if frame_ids != tuple(baseline.person_ids):
        raise ValueError("cohort availability index is not aligned to baseline person IDs")
    frame_labels = pd.to_numeric(frame["y"], errors="raise").to_numpy()
    if not np.array_equal(frame_labels, baseline.y_true):
        raise ValueError("cohort availability index labels differ from baseline")
    if source_cache is not None:
        if source_cache.person_ids != tuple(baseline.person_ids) or not np.array_equal(
            source_cache.y_true, baseline.y_true
        ):
            raise ValueError("cohort source-token cache is not aligned to baseline")
        presence = source_cache.source_present.detach().cpu().numpy()
        origin = "native_encoder_source_token_cache.source_present"
    else:
        presence = np.column_stack(
            [_presence_array(frame, group) for group in PRIMARY_MODALITIES]
        )
        origin = "aligned_dataset_index_SOURCE_FLAG_BY_GROUP"
    if presence.shape != (len(baseline.person_ids), len(PRIMARY_MODALITIES)):
        raise ValueError("cohort source availability has an invalid shape")
    if not np.isin(presence, (False, True)).all():
        raise ValueError("cohort source availability must be boolean")
    return presence.astype(bool, copy=False), origin


def _target_modalities(group: str) -> tuple[str, ...]:
    if group in FAMILY_ABLATION_GROUPS:
        return FAMILY_ABLATION_GROUPS[group]
    if group in PRIMARY_MODALITIES:
        return (group,)
    raise ValueError(f"unknown importance group {group!r}")


def _family_cohort_masks_and_counts(
    source_present: np.ndarray,
    y_true: np.ndarray,
    group: str,
) -> tuple[Dict[str, np.ndarray], list[Dict[str, Any]]]:
    """Select recipients by native family availability, never by outcomes."""

    members = FAMILY_ABLATION_GROUPS[group]
    indices = [tuple(PRIMARY_MODALITIES).index(member) for member in members]
    n_present = source_present[:, indices].sum(axis=1)
    remaining = source_present.sum(axis=1) - n_present
    any_present = n_present > 0
    all_present = n_present == len(members)
    eligible = any_present & (remaining > 0)
    all_eligible = all_present & (remaining > 0)
    masks = dict(zip(FAMILY_COHORT_SCOPES, (
        np.ones(len(y_true), dtype=bool), any_present, eligible,
        all_present, all_eligible,
    )))
    counts = []
    for scope, mask in masks.items():
        labels = y_true[mask]
        n_positive = int(np.sum(labels == 1))
        n_negative = int(np.sum(labels == 0))
        counts.append({
            "group": group,
            "cohort_scope": scope,
            "removed_modalities_json": json.dumps(list(members), separators=(",", ":")),
            "n_removed_modalities": len(members),
            "n_total": int(len(y_true)),
            "n_evaluated": int(len(labels)),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence": float(np.mean(labels)) if len(labels) else math.nan,
            "n_native_present": int(any_present.sum()),
            "n_eligible": int(eligible.sum()),
            "n_all_targets_present": int(all_present.sum()),
            "n_all_targets_eligible": int(all_eligible.sum()),
            "status": (
                "empty_cohort" if not len(labels)
                else "single_class_cohort" if not n_positive or not n_negative
                else "ok"
            ),
        })
    return masks, counts


def _cohort_masks_and_counts(
    source_present: np.ndarray,
    y_true: np.ndarray,
    group: str,
) -> tuple[Dict[str, np.ndarray], list[Dict[str, Any]]]:
    """Define outcome-independent recipient cohorts for a target source."""

    if group in FAMILY_ABLATION_GROUPS:
        return _family_cohort_masks_and_counts(source_present, y_true, group)
    target_index = tuple(PRIMARY_MODALITIES).index(group)
    native_present = source_present[:, target_index]
    remaining = source_present.sum(axis=1) - native_present.astype(np.int64)
    eligible = native_present & (remaining > 0)
    masks = {
        "all_participants": np.ones(len(y_true), dtype=bool),
        "native_present_only": native_present,
        "native_present_with_remaining_input": eligible,
    }
    counts = []
    for scope, mask in masks.items():
        labels = y_true[mask]
        n_positive = int(np.sum(labels == 1))
        n_negative = int(np.sum(labels == 0))
        status = (
            "empty_cohort" if not len(labels)
            else "single_class_cohort" if not n_positive or not n_negative
            else "ok"
        )
        counts.append(
            {
                "group": group,
                "cohort_scope": scope,
                "n_total": int(len(y_true)),
                "n_evaluated": int(len(labels)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "prevalence": float(np.mean(labels)) if len(labels) else math.nan,
                "n_native_present": int(native_present.sum()),
                "n_eligible": int(eligible.sum()),
                "status": status,
            }
        )
    return masks, counts


def _cohort_metrics_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, np.ndarray]:
    """Keep defined point metrics when a subgroup has zero or one class.

    Ranking metrics and balanced accuracy require both classes for these
    reports.  Log loss, Brier and the supported class's recall remain useful
    descriptive point estimates, but single-class cohorts receive no CI.
    """

    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("cohort probability matrix is not aligned to labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("cohort labels must be binary")
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("cohort probabilities must be finite and lie in [0, 1]")
    if set(np.unique(labels)) == {0, 1}:
        return _binary_metrics_matrix(labels, matrix, threshold=threshold)
    metrics = {
        name: np.full(matrix.shape[0], math.nan, dtype=np.float64)
        for name in METRIC_NAMES
    }
    if not len(labels):
        return metrics
    clipped = np.clip(matrix, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    labels_float = labels.astype(np.float64)[None, :]
    metrics["log_loss"] = -np.mean(
        labels_float * np.log(clipped)
        + (1.0 - labels_float) * np.log1p(-clipped), axis=1
    )
    metrics["brier"] = np.mean((matrix - labels_float) ** 2, axis=1)
    predicted = matrix >= float(threshold)
    if np.any(labels == 1):
        metrics["sensitivity"] = predicted[:, labels == 1].mean(axis=1)
    if np.any(labels == 0):
        metrics["specificity"] = (~predicted[:, labels == 0]).mean(axis=1)
    return metrics


def _cohort_condition_rows(
    baseline: EvaluationResult,
    group_results: Sequence[EvaluationResult],
    *,
    mask: np.ndarray,
    counts: Mapping[str, Any],
    method: str,
    group: str,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
    existing_ci_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Summarize paired predictions on one pre-specified recipient cohort."""

    probabilities = np.stack([result.probabilities for result in group_results])
    labels = baseline.y_true[mask]
    observed = _cohort_metrics_matrix(
        labels,
        np.concatenate([baseline.probabilities[None, mask], probabilities[:, mask]]),
        threshold=threshold,
    )
    # Reuse original all-participant metrics and intervals exactly, including
    # their seeds and floating-point implementation, for backward comparison.
    if existing_ci_records is not None:
        if counts["cohort_scope"] != "all_participants" or not np.all(mask):
            raise ValueError("legacy intervals can only be reused for all participants")
        observed = {
            name: np.asarray(
                [baseline.metrics[name], *[result.metrics[name] for result in group_results]],
                dtype=np.float64,
            )
            for name in METRIC_NAMES
        }
        intervals = [dict(record) for record in existing_ci_records]
    elif counts["status"] == "ok":
        intervals = paired_stratified_bootstrap_deltas(
            labels,
            baseline.probabilities[mask],
            probabilities[:, mask],
            threshold=threshold,
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
    else:
        intervals = [
            {
                "metric": name,
                "delta_definition": (
                    "baseline_minus_perturbed" if name in BENEFIT_METRICS
                    else "perturbed_minus_baseline"
                ),
                "delta": degradation_delta(name, observed[name][0], np.mean(observed[name][1:])),
                "point_estimate": degradation_delta(name, observed[name][0], np.mean(observed[name][1:])),
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "confidence_level": 0.95,
                "n_bootstrap_requested": int(bootstrap_samples),
                "n_bootstrap_finite": 0,
                "n_permutation_repeats": len(group_results),
                "bootstrap_design": "paired_participant_stratified_by_y_and_repeat_resampling",
            }
            for name in METRIC_NAMES
        ]
    summary_rows = []
    interval_rows = []
    for record in intervals:
        name = str(record["metric"])
        common = {
            "method": method,
            **dict(counts),
            "threshold_from_validation": float(threshold),
            "metric_status": (
                "ok" if math.isfinite(float(record["point_estimate"]))
                else "undefined_for_cohort"
            ),
            "ci_status": str(counts["status"]),
        }
        interval_rows.append({**common, **record})
        summary_rows.append(
            {
                **common,
                "metric": name,
                "baseline_metric": float(observed[name][0]),
                "mean_perturbed_metric": float(np.mean(observed[name][1:])),
                "mean_degradation_delta": float(record["point_estimate"]),
                "ci_lower": float(record["ci_lower"]),
                "ci_upper": float(record["ci_upper"]),
                "confidence_level": float(record["confidence_level"]),
                "n_repeats": int(record["n_permutation_repeats"]),
                "delta_definition": record["delta_definition"],
            }
        )
    return summary_rows, interval_rows


def run_importance_analysis(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    task: str,
    split: str,
    threshold: float,
    methods: Sequence[str],
    permutation_repeats: int,
    bootstrap_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    use_bf16_amp: bool = False,
    groups: Sequence[str] = PRIMARY_MODALITIES,
    source_token_cache_path: Optional[Path] = None,
    cache_identity: Optional[Mapping[str, Any]] = None,
    require_source_token_cache: bool = False,
) -> Dict[str, Any]:
    """Run all conditions and return auditable tables (without writing files)."""

    allowed = {"ablation", "permutation", "matched_permutation"}
    requested = tuple(dict.fromkeys(str(value).strip() for value in methods))
    unknown = sorted(set(requested) - allowed)
    if not requested or unknown:
        raise ValueError(f"invalid importance methods: {unknown or requested}")
    target_groups = tuple(dict.fromkeys(str(value).strip() for value in groups))
    unknown_groups = sorted(
        set(target_groups) - set(PRIMARY_MODALITIES) - set(FAMILY_ABLATION_GROUPS)
    )
    if not target_groups or unknown_groups:
        raise ValueError(f"invalid importance groups: {unknown_groups or target_groups}")
    family_groups = tuple(group for group in target_groups if group in FAMILY_ABLATION_GROUPS)
    if family_groups and (str(split).strip().lower() != "val" or requested != ("ablation",)):
        raise ValueError("family ablation groups require split=val and methods=ablation only")
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if int(permutation_repeats) <= 0 or int(bootstrap_samples) <= 0:
        raise ValueError("permutation_repeats and bootstrap_samples must be positive")
    frame = getattr(dataset, "index", None)
    if not isinstance(frame, pd.DataFrame) or len(frame) != len(dataset):
        raise TypeError("dataset must expose an aligned .index DataFrame")
    if set(np.unique(pd.to_numeric(frame["y"], errors="coerce"))) != {0, 1}:
        raise ValueError("importance cohort must contain both outcome classes")

    target_device = torch.device(device)
    model.to(target_device)
    base_loader = _loader(
        dataset, batch_size=batch_size, num_workers=num_workers, device=target_device
    )
    cache_capable = callable(getattr(model, "predict_from_source_tokens", None))
    if require_source_token_cache and not cache_capable:
        raise TypeError(
            "importance requires a model with predict_from_source_tokens; "
            "raw-asset repeat evaluation is forbidden"
        )
    source_cache: Optional[SourceTokenCache] = None
    if cache_capable:
        observed_source_order = tuple(
            str(value) for value in getattr(model, "source_order", ())
        )
        if observed_source_order != tuple(PRIMARY_MODALITIES):
            raise ValueError(
                "cacheable model source order differs from the canonical nine-source order"
            )
        if source_token_cache_path is not None and cache_identity is None:
            raise ValueError(
                "persistent source-token caches require checkpoint/config lineage identity"
            )
        person_ids = tuple(str(value) for value in frame["person_id"].tolist())
        derived_identity: Dict[str, Any] = {
            "task": str(task),
            "split": str(split).strip().lower(),
            "source_order": list(PRIMARY_MODALITIES),
            "encoder_batch_size": int(batch_size),
            "amp_mode": "bfloat16_autocast" if use_bf16_amp else "float32",
            "cohort_person_ids_sha256": _person_ids_sha256(person_ids),
            "n_participants": len(person_ids),
        }
        if cache_identity is None:
            # An in-memory cache cannot outlive this process, so sentinel
            # lineage is safe and makes direct library use convenient.
            identity: Dict[str, Any] = {
                "checkpoint_sha256": "unpersisted-in-memory-model",
                "config_hash": "unpersisted-in-memory-config",
                "master_config_hash": "unpersisted-in-memory-master",
                "scaler_config_hash": "unpersisted-in-memory-scaler",
                "model_config_hash": "unpersisted-in-memory-model-config",
            }
        else:
            identity = dict(cache_identity)
        for name, value in derived_identity.items():
            if name in identity and identity[name] != value:
                raise ValueError(
                    f"source-token cache identity {name} conflicts with current run"
                )
            identity[name] = value
        source_cache = build_or_load_source_token_cache(
            model,
            base_loader,
            frame=frame,
            device=target_device,
            task=task,
            expected_split=split,
            identity=identity,
            cache_path=source_token_cache_path,
            use_bf16_amp=use_bf16_amp,
        )
        baseline = evaluate_cached_condition(
            model,
            source_cache,
            device=target_device,
            task=task,
            threshold=threshold,
            batch_size=batch_size,
            use_bf16_amp=use_bf16_amp,
        )
        _assert_cached_baseline_exact(source_cache, baseline)
        plan_frame = source_cache.matching_frame
    else:
        baseline = evaluate_frozen_condition(
            model,
            base_loader,
            device=target_device,
            task=task,
            threshold=threshold,
            expected_split=split,
            use_bf16_amp=use_bf16_amp,
        )
        plan_frame = frame
    repeat_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    bootstrap_rows: list[Dict[str, Any]] = []
    permutation_audits: list[Dict[str, Any]] = []
    pair_rows: list[Dict[str, Any]] = []
    cohort_summary_rows: list[Dict[str, Any]] = []
    cohort_interval_rows: list[Dict[str, Any]] = []
    cohort_count_rows: list[Dict[str, Any]] = []
    cohort_masks: Dict[str, Dict[str, np.ndarray]] = {}
    cohort_counts: Dict[str, list[Dict[str, Any]]] = {}
    # Add exploratory cohort reporting only to validation.  Existing locked
    # test protocols did not select these scopes and must not gain them here.
    cohort_enabled = str(split).strip().lower() == "val"
    source_presence: Optional[np.ndarray] = None
    presence_origin: Optional[str] = None
    if cohort_enabled:
        source_presence, presence_origin = _cohort_source_presence(
            frame, baseline, source_cache
        )
        for group in target_groups:
            masks, counts = _cohort_masks_and_counts(
                source_presence, baseline.y_true, group
            )
            cohort_masks[group] = masks
            cohort_counts[group] = counts
            cohort_count_rows.extend(counts)

    for method in requested:
        repeats = 1 if method == "ablation" else int(permutation_repeats)
        for group in target_groups:
            group_results: list[EvaluationResult] = []
            permutation_wrapper: Optional[BundlePermutationDataset] = None
            permutation_loader: Optional[DataLoader] = None
            for repeat in range(repeats):
                condition_seed = _stable_seed(seed, method, group, repeat)
                if method == "ablation":
                    disabled: Optional[Iterable[str]] = set(_target_modalities(group))
                    plan = None
                else:
                    plan = build_permutation_plan(
                        plan_frame,
                        group,
                        seed=condition_seed,
                        matched=method == "matched_permutation",
                    )
                    disabled = None
                    audit = dict(plan.audit)
                    # The pair table already stores each recipient's stratum;
                    # omit this O(N) map from every repeat-level JSON record.
                    stratum_keys = audit.pop("recipient_stratum_keys", {})
                    recipient_fields = audit.pop("recipient_matching_fields", {})
                    recipient_levels = audit.pop("recipient_matching_levels", {})
                    audit.update({"method": method, "repeat": repeat})
                    permutation_audits.append(audit)
                    for recipient_index, donor_index in enumerate(plan.donor_indices.tolist()):
                        if recipient_index == donor_index:
                            continue
                        pair_rows.append(
                            {
                                "method": method,
                                "group": group,
                                "repeat": repeat,
                                "seed": condition_seed,
                                "recipient_person_id": str(plan_frame.iloc[recipient_index]["person_id"]),
                                "donor_person_id": str(plan_frame.iloc[donor_index]["person_id"]),
                                "matching_fields_json": json.dumps(
                                    recipient_fields.get(str(recipient_index), []),
                                    separators=(",", ":"),
                                ),
                                "matching_level": recipient_levels.get(
                                    str(recipient_index), ""
                                ),
                                "recipient_stratum_json": json.dumps(
                                    stratum_keys.get(str(recipient_index), []),
                                    separators=(",", ":"),
                                ),
                            }
                        )
                if source_cache is not None:
                    result = evaluate_cached_condition(
                        model,
                        source_cache,
                        device=target_device,
                        task=task,
                        threshold=threshold,
                        batch_size=batch_size,
                        disabled_groups=disabled,
                        donor_indices=(
                            None if plan is None else plan.donor_indices
                        ),
                        permutation_group=(None if plan is None else group),
                        use_bf16_amp=use_bf16_amp,
                    )
                else:
                    if method == "ablation":
                        loader = base_loader
                    else:
                        assert plan is not None
                        if permutation_wrapper is None:
                            permutation_wrapper = BundlePermutationDataset(
                                dataset, group, plan.donor_indices
                            )
                            permutation_loader = _loader(
                                permutation_wrapper,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                device=target_device,
                            )
                        elif permutation_wrapper.donor_mapping_shared or num_workers == 0:
                            permutation_wrapper.set_donor_indices(plan.donor_indices)
                        else:
                            permutation_wrapper = BundlePermutationDataset(
                                dataset, group, plan.donor_indices
                            )
                            permutation_loader = _loader(
                                permutation_wrapper,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                device=target_device,
                            )
                        if permutation_loader is None:
                            raise RuntimeError(
                                "internal error: permutation loader was not built"
                            )
                        loader = permutation_loader
                    result = evaluate_frozen_condition(
                        model,
                        loader,
                        device=target_device,
                        task=task,
                        threshold=threshold,
                        expected_split=split,
                        disabled_groups=disabled,
                        use_bf16_amp=use_bf16_amp,
                    )
                _assert_aligned(baseline, result)
                group_results.append(result)
                repeat_rows.append(
                    _metric_record(
                        result, baseline, method=method, group=group, repeat=repeat
                    )
                )
                condition_prediction_rows = _prediction_rows(
                    baseline,
                    result,
                    task=task,
                    split=split,
                    method=method,
                    group=group,
                    repeat=repeat,
                )
                if source_presence is not None:
                    members = _target_modalities(group)
                    source_indices = [tuple(PRIMARY_MODALITIES).index(member) for member in members]
                    n_native = source_presence[:, source_indices].sum(axis=1)
                    native = n_native > 0
                    remaining = source_presence.sum(axis=1) - n_native
                    for index, row in enumerate(condition_prediction_rows):
                        row.update(
                            {
                                "native_source_present": bool(native[index]),
                                "n_remaining_sources": int(remaining[index]),
                                "eligible_native_present_with_remaining_input": bool(
                                    native[index] and remaining[index] > 0
                                ),
                            }
                        )
                        if group in FAMILY_ABLATION_GROUPS:
                            row.update({
                                "removed_modalities_json": json.dumps(list(members), separators=(",", ":")),
                                "n_removed_modalities": len(members),
                                "n_native_targets_present": int(n_native[index]),
                                "any_target_present": bool(native[index]),
                                "all_targets_present": bool(n_native[index] == len(members)),
                                "eligible_any_target_present_with_remaining_input": bool(
                                    native[index] and remaining[index] > 0
                                ),
                                "eligible_all_targets_present_with_remaining_input": bool(
                                    n_native[index] == len(members) and remaining[index] > 0
                                ),
                            })
                prediction_rows.extend(condition_prediction_rows)

            probabilities = np.stack(
                [result.probabilities for result in group_results], axis=0
            )
            ci_records = paired_stratified_bootstrap_deltas(
                baseline.y_true,
                baseline.probabilities,
                probabilities,
                threshold=threshold,
                n_bootstrap=bootstrap_samples,
                seed=_stable_seed(seed, "bootstrap", method, group),
            )
            for record in ci_records:
                bootstrap_rows.append({"method": method, "group": group, **record})
            if cohort_enabled:
                family_rows_by_mask: Dict[bytes, tuple[list[Dict[str, Any]], list[Dict[str, Any]]]] = {}
                for counts in cohort_counts[group]:
                    scope = str(counts["cohort_scope"])
                    mask_key = cohort_masks[group][scope].tobytes()
                    if group in FAMILY_ABLATION_GROUPS and mask_key in family_rows_by_mask:
                        # Identical recipients imply the same estimand. Reuse
                        # both point estimates and bootstrap draws exactly.
                        saved_summary, saved_intervals = family_rows_by_mask[mask_key]
                        cohort_summary_rows.extend({**row, **counts} for row in saved_summary)
                        cohort_interval_rows.extend({**row, **counts} for row in saved_intervals)
                        continue
                    cohort_summary, cohort_intervals = _cohort_condition_rows(
                        baseline,
                        group_results,
                        mask=cohort_masks[group][scope],
                        counts=counts,
                        method=method,
                        group=group,
                        threshold=threshold,
                        bootstrap_samples=bootstrap_samples,
                        seed=_stable_seed(seed, "bootstrap", method, group, scope),
                        existing_ci_records=(
                            ci_records if scope == "all_participants" else None
                        ),
                    )
                    cohort_summary_rows.extend(cohort_summary)
                    cohort_interval_rows.extend(cohort_intervals)
                    if group in FAMILY_ABLATION_GROUPS:
                        family_rows_by_mask[mask_key] = (cohort_summary, cohort_intervals)

    baseline_row = {
        "method": "baseline",
        "group": "all_sources",
        "repeat": 0,
        **{metric: float(baseline.metrics[metric]) for metric in METRIC_NAMES},
        **{f"delta_{metric}": 0.0 for metric in METRIC_NAMES},
    }
    repeat_frame = pd.DataFrame([baseline_row, *repeat_rows])
    interval_frame = pd.DataFrame(bootstrap_rows)
    summary_rows: list[Dict[str, Any]] = []
    for record in bootstrap_rows:
        subset = repeat_frame[
            repeat_frame["method"].eq(record["method"])
            & repeat_frame["group"].eq(record["group"])
        ]
        metric = str(record["metric"])
        summary_rows.append(
            {
                "method": record["method"],
                "group": record["group"],
                "metric": metric,
                "baseline_metric": float(baseline.metrics[metric]),
                "mean_perturbed_metric": float(subset[metric].mean()),
                "mean_degradation_delta": float(record["point_estimate"]),
                "ci_lower": float(record["ci_lower"]),
                "ci_upper": float(record["ci_upper"]),
                "confidence_level": float(record["confidence_level"]),
                "n_repeats": int(record["n_permutation_repeats"]),
                "delta_definition": record["delta_definition"],
            }
        )
    if family_groups:
        # Keep the original single-source table layout unchanged.
        for rows in (summary_rows, bootstrap_rows, repeat_rows):
            for row in rows:
                members = _target_modalities(str(row["group"]))
                row["removed_modalities_json"] = json.dumps(list(members), separators=(",", ":"))
                row["n_removed_modalities"] = len(members)
        repeat_frame = pd.DataFrame([baseline_row, *repeat_rows])
        interval_frame = pd.DataFrame(bootstrap_rows)
    primary_scopes = {
        group: (
            "any_target_present_with_remaining_input" if group in FAMILY_ABLATION_GROUPS
            else "native_present_with_remaining_input"
        )
        for group in target_groups
    } if cohort_enabled else {}
    return {
        "baseline": baseline,
        "evaluated_groups": list(target_groups),
        "condition_definitions": {group: list(_target_modalities(group)) for group in target_groups},
        "evaluated_modalities": [
            member for member in PRIMARY_MODALITIES
            if any(member in _target_modalities(group) for group in target_groups)
        ],
        "family_ablation": {
            "enabled": bool(family_groups),
            "schema_version": "ot2-family-ablation-v1",
            "conditions": {group: list(FAMILY_ABLATION_GROUPS[group]) for group in family_groups},
            "intervention": "All named members are disabled together in one prediction; effects are not summed.",
            "interpretation": "Joint fitted-model reliance; not a formal modality interaction or redundancy estimate.",
        },
        "summary": pd.DataFrame(summary_rows),
        "repeat_metrics": repeat_frame,
        "paired_predictions": pd.DataFrame(prediction_rows),
        "bootstrap_intervals": interval_frame,
        "permutation_audits": permutation_audits,
        "permutation_pairs": pd.DataFrame(pair_rows),
        "cohort_summary": pd.DataFrame(cohort_summary_rows),
        "cohort_bootstrap_intervals": pd.DataFrame(cohort_interval_rows),
        "cohort_counts": pd.DataFrame(cohort_count_rows),
        "cohort_analysis": {
            "schema_version": COHORT_ANALYSIS_VERSION,
            "enabled": cohort_enabled,
            "validation_only": True,
            "scopes": list(dict.fromkeys(row["cohort_scope"] for row in cohort_count_rows)),
            "primary_eligible_scope": (
                next(iter(primary_scopes.values())) if len(set(primary_scopes.values())) == 1 else None
            ),
            "primary_eligible_scope_by_group": primary_scopes,
            "presence_source": presence_origin,
            "scope_definitions": {
                "all_participants": "Every participant in the evaluated split.",
                "native_present_only": "Target source was natively present before perturbation.",
                "native_present_with_remaining_input": (
                    "Target source was natively present and at least one of the other "
                    "eight sources was natively present before perturbation."
                ),
                "any_target_present": "At least one removed family member was natively present.",
                "any_target_present_with_remaining_input": (
                    "At least one removed member was natively present and at least one "
                    "source outside the entire removed family was natively present. Primary family scope."
                ),
                "all_targets_present": "Every removed family member was natively present.",
                "all_targets_present_with_remaining_input": (
                    "Every removed member and at least one source outside the entire family "
                    "were natively present. Secondary family scope."
                ),
            },
            "count_definitions": {
                "n_evaluated": "Participants in this scope; class counts and prevalence use these participants.",
                "n_native_present": "Participants with at least one target source in the entire evaluated split.",
                "n_eligible": "Participants with at least one target and at least one source outside the removed set.",
                "n_all_targets_present": "Participants with every removed family member in the entire split.",
                "n_all_targets_eligible": "Participants with every removed member and an outside source in the entire split.",
            },
            "comparison": "Full and perturbed predictions are restricted to the same participant mask.",
            "threshold": "Saved full-model validation threshold; no subgroup refitting.",
            "permutation_scope": (
                "Scopes select recipients from the original full-cohort permutation plans; "
                "donors are not reselected and subgroup results are not subgroup-only permutations."
            ),
            "undefined_cohorts": (
                "Empty cohorts have undefined metrics and CIs. Single-class cohorts retain "
                "log loss, Brier and available-class recall, but no ranking metrics, balanced "
                "accuracy or confidence estimates. CSV NaNs are blank, never zero."
            ),
            "bootstrap_seed_contract": (
                "Original all-participant intervals are reused exactly; subgroup seeds "
                "add the scope to the existing method/group seed derivation. Family scopes "
                "with identical participant masks reuse the first scope's estimates and intervals exactly."
            ),
            "counts": [
                {
                    name: (None if isinstance(value, float) and not math.isfinite(value) else value)
                    for name, value in row.items()
                }
                for row in cohort_count_rows
            ],
            "disabled_reason": (
                None if cohort_enabled else "Cohort scopes are not part of the existing locked test protocol."
            ),
        },
        "source_token_cache_audit": (
            {
                "enabled": True,
                "schema_version": SOURCE_TOKEN_CACHE_FORMAT_VERSION,
                "path": (
                    None if source_cache.path is None else str(source_cache.path)
                ),
                "reused": bool(source_cache.reused),
                "content_sha256": source_cache.content_sha256,
                "identity": dict(source_cache.identity),
                "raw_dataset_passes_this_run": 0 if source_cache.reused else 1,
                "native_encoder_passes_this_run": 0 if source_cache.reused else 1,
                "perturbation_path": "cached_source_tokens_then_fusion_and_head_only",
                "full_forward_equivalence_tolerance": "atol=1e-6, rtol=1e-6",
            }
            if source_cache is not None
            else {
                "enabled": False,
                "reason": "model_lacks_predict_from_source_tokens_library_fallback",
            }
        ),
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _expected_lineage_hashes(config: ModelV2RunConfig) -> tuple[str, str]:
    master = json.loads(config.dataset.master_metadata_json.read_text(encoding="utf-8"))
    scaler = json.loads(config.dataset.scaler_json.read_text(encoding="utf-8"))
    master_hash = str(master.get("config_hash") or "").strip()
    scaler_hash = str(scaler.get("scaler_config_hash") or "").strip()
    if not master_hash or not scaler_hash:
        raise ValueError("master/scaler files do not declare required lineage hashes")
    return master_hash, scaler_hash


def _validate_checkpoint_input_file_hashes(
    checkpoint: Mapping[str, Any], config: ModelV2RunConfig
) -> None:
    """Bind every current Dataset/schema byte stream to training provenance."""

    data = _mapping(checkpoint.get("data_provenance"), "data provenance")
    files = _mapping(data.get("files"), "data provenance files")
    expected_paths = {
        "master_csv": config.dataset.master_csv,
        "master_metadata_json": config.dataset.master_metadata_json,
        "scaler_json": config.dataset.scaler_json,
        "clinical_schema_json": config.dataset.clinical_schema_json,
        "environment_schema_json": config.dataset.environment_schema_json,
        "wearable_schema_json": config.dataset.wearable_schema_json,
        "ecg_schema_json": config.dataset.ecg_schema_json,
    }
    for name, path in expected_paths.items():
        entry = _mapping(files.get(name), f"checkpoint file provenance {name}")
        trained_hash = str(entry.get("sha256") or "").strip()
        if not trained_hash or trained_hash != _sha256_file(path):
            raise ValueError(f"checkpoint/current input file hash mismatch: {name}")


def _validate_clinical_feature_bytes(
    checkpoint: Mapping[str, Any], dataset: OT2V2Dataset
) -> None:
    """Bind the large clinical wide table that is referenced by the master."""

    data = _mapping(checkpoint.get("data_provenance"), "data provenance")
    entry = _mapping(
        data.get("clinical_features_csv"),
        "checkpoint clinical feature provenance",
    )
    trained_hash = str(entry.get("sha256") or "").strip()
    trained_path = str(entry.get("path") or "").strip()
    current_path = dataset.clinical_features_path
    if not trained_hash or trained_hash != _sha256_file(current_path):
        raise ValueError("checkpoint/current clinical wide CSV hash mismatch")
    if not trained_path:
        raise ValueError("checkpoint lacks the clinical wide CSV source path")


def _validate_resolved_model_config(
    checkpoint: Mapping[str, Any], model_config: Task2V2ModelConfig
) -> None:
    provenance = _mapping(checkpoint.get("config_provenance"), "config provenance")
    trained = _mapping(
        provenance.get("resolved_model_config"), "resolved model config provenance"
    )
    current = dataclasses.asdict(model_config)
    # JSON normalization treats tuple/list task-name representations equally.
    trained_normalized = json.loads(json.dumps(trained, sort_keys=True, default=str))
    current_normalized = json.loads(json.dumps(current, sort_keys=True, default=str))
    if trained_normalized != current_normalized:
        raise ValueError(
            "checkpoint resolved model architecture differs from current configuration"
        )


def _validate_evaluation_cohort_provenance(
    checkpoint: Mapping[str, Any], dataset: OT2V2Dataset, split: str
) -> None:
    data = _mapping(checkpoint.get("data_provenance"), "data provenance")
    counts = _mapping(data.get("split_counts"), "checkpoint split counts")
    hashes = _mapping(
        data.get("split_person_ids_sha256"), "checkpoint split person-ID hashes"
    )
    expected_count = counts.get(split)
    expected_hash = str(hashes.get(split) or "").strip()
    try:
        count_matches = not isinstance(expected_count, bool) and int(expected_count) == len(dataset)
    except (TypeError, ValueError):
        count_matches = False
    if not count_matches:
        raise ValueError("checkpoint/current evaluation split count mismatch")
    digest = hashlib.sha256()
    for person_id in sorted(str(value) for value in dataset.index["person_id"]):
        digest.update(person_id.encode("utf-8"))
        digest.update(b"\n")
    if not expected_hash or digest.hexdigest() != expected_hash:
        raise ValueError("checkpoint/current evaluation participant hash mismatch")


def _build_model_config(config: ModelV2RunConfig, dataset: OT2V2Dataset) -> Task2V2ModelConfig:
    # There must be exactly one architecture-binding implementation.  The
    # supervised CLI rejects unknown/ignored YAML keys and validates every
    # LANISTR-style section; reusing it prevents an importance run from
    # silently reconstructing a slightly different network.
    try:
        from ot2.train_task2_v2 import build_model_config
    except ModuleNotFoundError:
        from train_task2_v2 import build_model_config  # type: ignore

    return build_model_config(config, dataset.model_feature_dimensions)


def _validate_test_lock(
    path: Optional[Path],
    *,
    config_hash: str,
    checkpoint_hash: str,
) -> Mapping[str, Any]:
    if path is None:
        raise ValueError(
            "all-nine test importance can become test-set modality selection; "
            "--test-selection-lock is required for confirmatory test evaluation"
        )
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "test selection lock")
    if payload.get("schema_version") != SELECTION_LOCK_VERSION:
        raise ValueError("test selection lock has the wrong schema version")
    if str(payload.get("selection_split") or "").lower() not in {"val", "validation"}:
        raise ValueError("test selection lock must have been produced from validation")
    if str(payload.get("config_hash") or "") != config_hash:
        raise ValueError("test selection lock/config hash mismatch")
    if str(payload.get("checkpoint_sha256") or "") != checkpoint_hash:
        raise ValueError("test selection lock/checkpoint hash mismatch")
    locked_groups = payload.get("locked_groups")
    if not isinstance(locked_groups, list) or any(group not in PRIMARY_MODALITIES for group in locked_groups):
        raise ValueError("test selection lock must declare valid locked_groups")
    if not locked_groups or len(locked_groups) != len(set(locked_groups)):
        raise ValueError("test selection lock locked_groups must be non-empty and unique")
    expected_metadata_hash = str(
        payload.get("validation_importance_metadata_sha256") or ""
    ).strip()
    if not expected_metadata_hash:
        raise ValueError("test selection lock lacks validation importance provenance")
    metadata_name = str(payload.get("validation_importance_metadata_path") or "").strip()
    if not metadata_name:
        raise ValueError("test selection lock lacks validation importance metadata path")
    metadata_path = Path(metadata_name).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = (path.parent / metadata_path).resolve()
    if not metadata_path.is_file() or _sha256_file(metadata_path) != expected_metadata_hash:
        raise ValueError("test selection lock validation-metadata hash mismatch")
    validation_metadata = _mapping(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        "validation importance metadata",
    )
    if validation_metadata.get("schema_version") != IMPORTANCE_FORMAT_VERSION:
        raise ValueError("locked validation metadata has the wrong schema version")
    if str(validation_metadata.get("split") or "").lower() not in {"val", "validation"}:
        raise ValueError("locked importance metadata was not produced on validation")
    if str(validation_metadata.get("config_hash") or "") != config_hash:
        raise ValueError("locked validation metadata/config hash mismatch")
    if str(validation_metadata.get("checkpoint_sha256") or "") != checkpoint_hash:
        raise ValueError("locked validation metadata/checkpoint hash mismatch")
    validation_groups = validation_metadata.get("primary_modalities")
    if not isinstance(validation_groups, list) or not set(locked_groups).issubset(
        set(validation_groups)
    ):
        raise ValueError("locked groups were not evaluated in validation metadata")

    locked_methods = payload.get("locked_methods")
    if (
        not isinstance(locked_methods, list)
        or not locked_methods
        or len(locked_methods) != len(set(locked_methods))
        or any(
            method not in {"ablation", "permutation", "matched_permutation"}
            for method in locked_methods
        )
    ):
        raise ValueError("test selection lock must declare valid unique locked_methods")
    validation_methods = validation_metadata.get("methods")
    if not isinstance(validation_methods, list) or not set(locked_methods).issubset(
        set(validation_methods)
    ):
        raise ValueError("locked methods were not evaluated in validation metadata")
    for key in ("permutation_repeats", "bootstrap_samples"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"test selection lock {key} must be a positive integer")
        validation_value = validation_metadata.get(key)
        if value != validation_value:
            raise ValueError(
                f"test selection lock {key} differs from validation procedure"
            )
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("test selection lock seed must be an integer")
    if seed != validation_metadata.get("seed"):
        raise ValueError("test selection lock seed differs from validation procedure")
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--test-selection-lock", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--source-token-cache",
        type=Path,
        default=None,
        help=(
            "Hash-bound native-encoder cache. Defaults to "
            "OUT_DIR/source_token_cache.pt; an existing mismatched cache is "
            "rejected and never overwritten."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("ablation", "permutation", "matched_permutation"),
        default=None,
    )
    parser.add_argument(
        "--ablation-groups",
        nargs="+",
        choices=tuple(PRIMARY_MODALITIES) + tuple(FAMILY_ABLATION_GROUPS),
        default=None,
        help=(
            "Validation ablation conditions. Named families disable all their members "
            "together. When supplied, methods defaults to ablation only."
        ),
    )
    parser.add_argument("--permutation-repeats", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.ablation_groups is not None:
        if args.split != "val" or (args.methods is not None and set(args.methods) != {"ablation"}):
            raise ValueError("--ablation-groups requires split=val and methods=ablation only")
        args.methods = ["ablation"]
    try:
        from ot2.train_task2_v2 import load_supervised_config, resolve_device
    except ModuleNotFoundError:
        from train_task2_v2 import load_supervised_config, resolve_device  # type: ignore

    _, config = load_supervised_config(args.config)
    validate_model_v2_lineage(config)
    checkpoint_path = (args.checkpoint or (config.output_dir / "best_model.pt")).resolve()
    out_dir = (args.out_dir or (config.output_dir / f"importance_{args.split}" )).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint does not exist: {checkpoint_path}")
    checkpoint_hash = _sha256_file(checkpoint_path)
    master_hash, scaler_hash = _expected_lineage_hashes(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    threshold = validate_importance_checkpoint(
        checkpoint,
        expected_config_hash=config.config_hash,
        expected_task=config.dataset.task,
        expected_master_hash=master_hash,
        expected_scaler_hash=scaler_hash,
    )
    _validate_checkpoint_input_file_hashes(checkpoint, config)
    test_lock = None
    if args.split == "test":
        test_lock = _validate_test_lock(
            args.test_selection_lock,
            config_hash=config.config_hash,
            checkpoint_hash=checkpoint_hash,
        )

    caps = {f"retinal_{name}": value for name, value in config.dataset.retinal_max_assets.items()}
    dataset = OT2V2Dataset(
        config.dataset.master_csv,
        config.dataset.scaler_json,
        task=config.dataset.task,
        artifact_root=config.dataset.artifact_root,
        split=args.split,
        master_metadata_json=config.dataset.master_metadata_json,
        environment_schema_json=config.dataset.environment_schema_json,
        max_retinal_assets_per_group=caps,
        require_y=True,
    )
    _validate_clinical_feature_bytes(checkpoint, dataset)
    _validate_evaluation_cohort_provenance(checkpoint, dataset, args.split)
    resolved_model_config = _build_model_config(config, dataset)
    _validate_resolved_model_config(checkpoint, resolved_model_config)
    model = Task2V2Model(resolved_model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device_name = resolve_device(args.device)
    if test_lock is None:
        methods = tuple(args.methods or config.importance.methods)
        repeats = int(
            config.importance.permutation_repeats
            if args.permutation_repeats is None
            else args.permutation_repeats
        )
        bootstraps = int(
            config.importance.bootstrap_samples
            if args.bootstrap_samples is None
            else args.bootstrap_samples
        )
        seed = int(config.seed if args.seed is None else args.seed)
    else:
        locked_methods = tuple(str(value) for value in test_lock["locked_methods"])
        locked_repeats = int(test_lock["permutation_repeats"])
        locked_bootstraps = int(test_lock["bootstrap_samples"])
        locked_seed = int(test_lock["seed"])
        supplied = {
            "methods": tuple(args.methods) if args.methods is not None else locked_methods,
            "permutation_repeats": (
                int(args.permutation_repeats)
                if args.permutation_repeats is not None
                else locked_repeats
            ),
            "bootstrap_samples": (
                int(args.bootstrap_samples)
                if args.bootstrap_samples is not None
                else locked_bootstraps
            ),
            "seed": int(args.seed) if args.seed is not None else locked_seed,
        }
        expected = {
            "methods": locked_methods,
            "permutation_repeats": locked_repeats,
            "bootstrap_samples": locked_bootstraps,
            "seed": locked_seed,
        }
        if supplied != expected:
            raise ValueError(
                "test analysis overrides disagree with the validation selection lock"
            )
        methods = locked_methods
        repeats = locked_repeats
        bootstraps = locked_bootstraps
        seed = locked_seed
    if repeats <= 0 or bootstraps <= 0 or int(args.num_workers) < 0:
        raise ValueError("repeat/bootstrap counts must be positive and workers non-negative")
    set_deterministic_seed(seed, deterministic=True)
    cache_path = (
        Path(args.source_token_cache).expanduser().resolve()
        if args.source_token_cache is not None
        else (out_dir / "source_token_cache.pt").resolve()
    )
    cache_identity = {
        "checkpoint_sha256": checkpoint_hash,
        "config_hash": config.config_hash,
        "master_config_hash": master_hash,
        "scaler_config_hash": scaler_hash,
        "model_config_hash": canonical_json_hash(
            dataclasses.asdict(resolved_model_config)
        ),
    }
    result = run_importance_analysis(
        model,
        dataset,
        task=config.dataset.task,
        split=args.split,
        threshold=threshold,
        methods=methods,
        permutation_repeats=repeats,
        bootstrap_samples=bootstraps,
        seed=seed,
        batch_size=int(args.batch_size or config.training.batch_size),
        num_workers=int(args.num_workers),
        device=device_name,
        use_bf16_amp=bool(args.bf16),
        groups=(
            tuple(str(group) for group in test_lock["locked_groups"])
            if test_lock is not None
            else tuple(args.ablation_groups or PRIMARY_MODALITIES)
        ),
        source_token_cache_path=cache_path,
        cache_identity=cache_identity,
        require_source_token_cache=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "importance_summary.csv"
    repeat_path = out_dir / "importance_repeat_metrics.csv"
    prediction_path = out_dir / "importance_paired_predictions.csv"
    ci_path = out_dir / "importance_bootstrap_intervals.csv"
    pair_path = out_dir / "importance_permutation_pairs.csv"
    audit_path = out_dir / "importance_permutation_audits.json"
    metadata_path = out_dir / "importance_metadata.json"
    _atomic_csv(result["summary"], summary_path)
    _atomic_csv(result["repeat_metrics"], repeat_path)
    _atomic_csv(result["paired_predictions"], prediction_path)
    _atomic_csv(result["bootstrap_intervals"], ci_path)
    _atomic_csv(result["permutation_pairs"], pair_path)
    _atomic_json({"audits": result["permutation_audits"]}, audit_path)

    baseline: EvaluationResult = result["baseline"]
    output_files = [
        summary_path,
        repeat_path,
        prediction_path,
        ci_path,
        pair_path,
        audit_path,
    ]
    if result["cohort_analysis"]["enabled"]:
        for key, filename in (
            ("cohort_summary", "importance_cohort_summary.csv"),
            ("cohort_bootstrap_intervals", "importance_cohort_bootstrap_intervals.csv"),
            ("cohort_counts", "importance_cohort_counts.csv"),
        ):
            path = out_dir / filename
            _atomic_csv(result[key], path)
            output_files.append(path)
    metadata = {
        "schema_version": IMPORTANCE_FORMAT_VERSION,
        "analysis_role": (
            "validation_modality_selection" if args.split == "val" else "locked_confirmatory_test"
        ),
        "task": config.dataset.task,
        "split": args.split,
        "n_participants": len(dataset),
        "config_path": str(Path(args.config).resolve()),
        "config_hash": config.config_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_kind": "best",
        "analysis_code_sha256": _sha256_file(Path(__file__).resolve()),
        "threshold": threshold,
        "threshold_origin": "saved validation-selected best-checkpoint provenance",
        "master_config_hash": master_hash,
        "scaler_config_hash": scaler_hash,
        "methods": list(methods),
        # Legacy test-selection locks refer to evaluated single-source
        # conditions. A family result must not authorize unmeasured singles.
        "primary_modalities": [group for group in result["evaluated_groups"] if group in PRIMARY_MODALITIES],
        "ablated_modalities": result["evaluated_modalities"],
        "evaluated_groups": result["evaluated_groups"],
        "condition_definitions": result["condition_definitions"],
        "family_ablation": result["family_ablation"],
        "permutation_repeats": repeats,
        "bootstrap_samples": bootstraps,
        "bootstrap_design": "paired_participant_stratified_by_y_and_repeat_resampling",
        "seed": seed,
        "baseline_metrics": {name: float(baseline.metrics[name]) for name in METRIC_NAMES},
        "matching_contract": {
            "allowed_fields": [
                "clinical_site",
                "sex_if_available",
                "fixed_age_bins",
                "other_eight_modality_availability_signature",
            ],
            "age_bin_edges": ["-inf", 40, 50, 60, 70, 80, "inf"],
            "outcome_fields_forbidden": ["y", "label_value", "task outcome"],
            "hierarchical_relaxation": True,
        },
        "source_token_cache": {
            **dict(result["source_token_cache_audit"]),
            "file_sha256": _sha256_file(cache_path),
        },
        "test_selection_lock": dict(test_lock) if test_lock is not None else None,
        "cohort_analysis": result["cohort_analysis"],
        "output_sha256": {path.name: _sha256_file(path) for path in output_files},
        "limitations": [
            "Post-hoc importance is associational and depends on the fitted model.",
            "Ablation changes source availability and may be out of the training distribution.",
            "Permutation estimates conditional model reliance, not causal clinical utility.",
            "Bootstrap intervals do not include between-training-seed uncertainty.",
        ],
    }
    _atomic_json(metadata, metadata_path)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "task": config.dataset.task,
                "split": args.split,
                "n_participants": len(dataset),
                "threshold_from_validation": threshold,
                "methods": list(methods),
                "permutation_repeats": repeats,
                "bootstrap_samples": bootstraps,
                "source_token_cache": str(cache_path),
                "source_token_cache_reused": bool(
                    result["source_token_cache_audit"]["reused"]
                ),
                "metadata": str(metadata_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGE_BIN_EDGES",
    "BENEFIT_METRICS",
    "BundlePermutationDataset",
    "IMPORTANCE_FORMAT_VERSION",
    "LOSS_METRICS",
    "METRIC_NAMES",
    "PermutationPlan",
    "SOURCE_TOKEN_CACHE_FORMAT_VERSION",
    "SELECTION_LOCK_VERSION",
    "SourceTokenCache",
    "build_or_load_source_token_cache",
    "build_permutation_plan",
    "degradation_delta",
    "evaluate_cached_condition",
    "evaluate_frozen_condition",
    "paired_stratified_bootstrap_deltas",
    "run_importance_analysis",
    "validate_importance_checkpoint",
]
