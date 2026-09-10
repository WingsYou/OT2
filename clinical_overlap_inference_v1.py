"""Re-encode clinical inputs for frozen, coupled modality ablation.

Only the normalized clinical bank is read.  The validated native-token cache
remains unchanged and none of the eight non-clinical encoders is evaluated.
"""
from __future__ import annotations

import contextlib
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


CLINICAL_SOURCE = "clinical_data"


def _source_order(cache: Any) -> tuple[str, ...]:
    identity = getattr(cache, "identity", None)
    if not isinstance(identity, Mapping):
        raise ValueError("clinical override requires cache identity")
    order = tuple(identity.get("source_order", ()))
    if len(order) != 9 or len(set(order)) != 9 or order.count(CLINICAL_SOURCE) != 1:
        raise ValueError("clinical override requires nine unique source names")
    return order


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(result)


def materialize_clinical_bank(dataset: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the existing train-fitted scaler without loading any raw assets."""

    raw = torch.as_tensor(
        np.array(dataset.clinical_values_raw, dtype=np.float32, copy=True),
        dtype=torch.float32,
    )
    if raw.ndim != 2 or raw.shape[0] != len(dataset) or raw.shape[1] == 0:
        raise ValueError("clinical bank must be an aligned non-empty feature matrix")
    if "has_clinical" not in dataset.index:
        raise ValueError("dataset index lacks has_clinical")
    declared_array = dataset.index["has_clinical"].to_numpy()
    if declared_array.shape != (len(dataset),) or not np.isin(
        declared_array, (False, True)
    ).all():
        raise ValueError("has_clinical must contain aligned boolean availability")
    declared = torch.as_tensor(declared_array.astype(bool), dtype=torch.bool)
    mask = torch.isfinite(raw) & declared[:, None]
    values = dataset.scalers["clinical"].normalise(raw, mask, "clinical bank")
    if not isinstance(values, torch.Tensor) or values.shape != raw.shape:
        raise ValueError("clinical scaler returned an unaligned bank")
    if not torch.isfinite(values).all():
        raise ValueError("normalized clinical bank contains non-finite values")
    return values.detach().cpu().contiguous(), mask.detach().cpu().contiguous()


@torch.no_grad()
def encode_clinical_override(
    model: torch.nn.Module,
    cache: Any,
    clinical_values: torch.Tensor,
    clinical_masks: torch.Tensor,
    feature_indices: Sequence[int],
    *,
    device: str | torch.device,
    batch_size: int,
    use_bf16_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove complete clinical columns and encode their remaining information.

    An empty feature list performs the native-equivalence check's encoding.
    Returned tensors are new CPU allocations, aligned to the cache's people.
    Values and observation masks are both cleared; vector width is preserved.
    """

    size = _positive_integer(batch_size, "clinical encoder batch_size")
    order = _source_order(cache)
    if tuple(getattr(model, "source_order", ())) != order:
        raise ValueError("model/cache source order mismatch")
    encode = getattr(model, "encode_sources", None)
    if not callable(encode):
        raise TypeError("model must expose encode_sources")
    identity = cache.identity
    if "encoder_batch_size" in identity and identity["encoder_batch_size"] != size:
        raise ValueError("clinical encoder batch_size differs from native cache")
    amp_mode = "bfloat16_autocast" if use_bf16_amp else "float32"
    if "amp_mode" in identity and identity["amp_mode"] != amp_mode:
        raise ValueError("clinical encoder AMP mode differs from native cache")
    if not isinstance(clinical_values, torch.Tensor) or not isinstance(
        clinical_masks, torch.Tensor
    ):
        raise TypeError("clinical values and masks must be tensors")
    if clinical_values.ndim != 2 or clinical_masks.shape != clinical_values.shape:
        raise ValueError("clinical values/masks must be aligned rank-2 tensors")
    if clinical_masks.dtype != torch.bool:
        raise ValueError("clinical masks must be boolean")
    if not clinical_values.is_floating_point() or not torch.isfinite(clinical_values).all():
        raise ValueError("clinical values must be finite floating-point values")
    n_people, n_features = clinical_values.shape
    if n_people == 0 or n_features == 0 or n_people != len(cache.person_ids):
        raise ValueError("clinical bank does not align with cache participants")
    cache_tokens = cache.source_tokens
    cache_presence = cache.source_present
    if (
        not isinstance(cache_tokens, torch.Tensor)
        or cache_tokens.ndim != 3
        or tuple(cache_tokens.shape[:2]) != (n_people, len(order))
        or not torch.isfinite(cache_tokens).all()
    ):
        raise ValueError("cache source tokens are not finite and aligned")
    if (
        not isinstance(cache_presence, torch.Tensor)
        or cache_presence.dtype != torch.bool
        or tuple(cache_presence.shape) != (n_people, len(order))
    ):
        raise ValueError("cache source presence must be aligned and boolean")
    indices = []
    for value in feature_indices:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("clinical feature indices must be unique integers in range")
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise ValueError("clinical feature indices must be unique integers in range") from exc
        if index < 0 or index >= n_features or index in indices:
            raise ValueError("clinical feature indices must be unique integers in range")
        indices.append(int(index))
    clinical_index = order.index(CLINICAL_SOURCE)
    native_presence = cache_presence[:, clinical_index].detach().cpu()
    if not torch.equal(clinical_masks.detach().cpu().any(dim=1), native_presence):
        raise ValueError("clinical bank presence differs from native cache")

    target_device = torch.device(device)
    if use_bf16_amp and target_device.type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 clinical encoding requires CPU or CUDA")
    disabled = set(order) - {CLINICAL_SOURCE}
    tokens_parts = []
    presence_parts = []
    model.eval()
    for start in range(0, n_people, size):
        stop = min(start + size, n_people)
        values = clinical_values[start:stop].detach().clone().to(target_device)
        masks = clinical_masks[start:stop].detach().clone().to(target_device)
        if indices:
            values[:, indices] = 0.0
            masks[:, indices] = False
        batch = {
            "clinical_values": values,
            "clinical_mask": masks,
            "has_clinical": native_presence[start:stop].to(target_device),
        }
        amp = (
            torch.autocast(device_type=target_device.type, dtype=torch.bfloat16)
            if use_bf16_amp
            else contextlib.nullcontext()
        )
        with amp:
            output = encode(batch, disabled_groups=disabled)
        if not isinstance(output, Mapping):
            raise TypeError("encode_sources must return a mapping")
        encoded = output.get("source_tokens")
        present = output.get("source_present")
        if (
            not isinstance(encoded, torch.Tensor)
            or tuple(encoded.shape) != (stop - start, len(order), cache_tokens.shape[2])
            or not torch.isfinite(encoded).all()
        ):
            raise ValueError("clinical encoder output is not finite and aligned")
        if (
            not isinstance(present, torch.Tensor)
            or present.dtype != torch.bool
            or tuple(present.shape) != (stop - start, len(order))
        ):
            raise ValueError("clinical encoder presence is not aligned and boolean")
        clinical_present = present[:, clinical_index]
        expected_present = masks.any(dim=1) & batch["has_clinical"]
        if not torch.equal(clinical_present, expected_present):
            raise ValueError("clinical encoder presence disagrees with remaining input")
        others = [i for i in range(len(order)) if i != clinical_index]
        if present[:, others].any():
            raise ValueError("non-clinical source became available during clinical encoding")
        clinical_tokens = encoded[:, clinical_index]
        if torch.count_nonzero(clinical_tokens[~clinical_present]).item():
            raise ValueError("absent clinical sources must return zero native tokens")
        tokens_parts.append(clinical_tokens.detach().cpu().clone())
        presence_parts.append(clinical_present.detach().cpu().clone())
    return torch.cat(tokens_parts, dim=0), torch.cat(presence_parts, dim=0)


def assert_native_clinical_equivalent(
    cache: Any,
    tokens: torch.Tensor,
    presence: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Fail closed if the untouched clinical bank does not reproduce the cache."""

    clinical_index = _source_order(cache).index(CLINICAL_SOURCE)
    native = cache.source_tokens[:, clinical_index].detach().cpu()
    native_presence = cache.source_present[:, clinical_index].detach().cpu()
    if not isinstance(tokens, torch.Tensor) or tokens.shape != native.shape:
        raise ValueError("native clinical equivalence token shape mismatch")
    if (
        not isinstance(presence, torch.Tensor)
        or presence.dtype != torch.bool
        or not torch.equal(presence.detach().cpu(), native_presence)
    ):
        raise ValueError("native clinical equivalence presence mismatch")
    if not torch.isfinite(tokens).all() or not torch.allclose(
        tokens.detach().cpu().float(), native.float(), atol=atol, rtol=rtol
    ):
        raise ValueError("untouched clinical token differs from native cache")


__all__ = [
    "assert_native_clinical_equivalent",
    "encode_clinical_override",
    "materialize_clinical_bank",
]
