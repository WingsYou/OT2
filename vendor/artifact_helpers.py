from __future__ import annotations
import hashlib,json
from typing import Any,Mapping,Sequence
import numpy as np
import pandas as pd
PREPROCESSING_VERSION="aireadi-ot2-v2"
SCALER_FORMAT_VERSION="aireadi_ot2_train_scalers_v2"
NON_SCALING_PREFIX="time__"
def as_person_id(value: Any) -> str:
    """Normalize identifiers without converting valid leading-zero strings."""
    if isinstance(value, str):
        value = value.strip()
        return value[:-2] if value.endswith(".0") and value[:-2].isdigit() else value
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()

def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def validated_boolean_mask(value: Any, source: str) -> np.ndarray:
    """Accept only native booleans or finite numeric 0/1 mask values."""
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return array.astype(bool, copy=False)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            f"{source}: mask dtype must be bool or numeric 0/1, got {array.dtype}"
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{source}: mask contains NaN or infinity")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{source}: mask contains values other than 0/1")
    return array.astype(bool, copy=False)

def environment_feature_mask(
    feature_names: Sequence[str],
    sensor_columns: Sequence[str],
    channel_mask: np.ndarray,
    row_count: np.ndarray,
) -> np.ndarray:
    """Construct exact feature support for V2 environment artifacts."""
    names = [str(name) for name in feature_names]
    channel_mask = validated_boolean_mask(
        channel_mask, "environment_channel_observed_mask"
    )
    row_count = np.asarray(row_count)
    if channel_mask.ndim != 2 or row_count.ndim != 1:
        raise ValueError("Environment channel mask must be 2D and row_count must be 1D")
    if row_count.dtype.kind not in {"i", "u", "f"} or row_count.dtype.kind == "b":
        raise ValueError("Environment row_count must be numeric nonnegative integers")
    if not np.isfinite(row_count).all() or np.any(row_count < 0):
        raise ValueError("Environment row_count contains non-finite or negative values")
    if not np.equal(row_count, np.floor(row_count)).all():
        raise ValueError("Environment row_count contains non-integer values")
    if channel_mask.shape[0] != row_count.shape[0]:
        raise ValueError("Environment channel mask and row_count length mismatch")
    if channel_mask.shape[1] != len(sensor_columns):
        raise ValueError("Environment channel mask width does not match sensor_columns")
    rows_present = row_count > 0
    if np.any(channel_mask.any(axis=1) & ~rows_present):
        raise ValueError(
            "Environment channel mask marks an observed sensor where row_count is zero"
        )
    if len(names) != len(set(names)):
        raise ValueError("Environment feature_names contains duplicates")
    index = {name: i for i, name in enumerate(names)}
    mask = np.zeros((channel_mask.shape[0], len(names)), dtype=bool)
    supported_names: set[str] = set()
    for channel_index, column in enumerate(sensor_columns):
        safe = str(column).replace(".", "_")
        mean_name = f"env__{safe}__mean"
        std_name = f"env__{safe}__std"
        missing_name = f"env__{safe}__missing_rate"
        for name in [mean_name, std_name, missing_name]:
            if name not in index:
                raise ValueError(f"Environment artifact missing expected feature {name}")
            supported_names.add(name)
        mask[:, index[mean_name]] = channel_mask[:, channel_index]
        mask[:, index[std_name]] = channel_mask[:, channel_index]
        mask[:, index[missing_name]] = rows_present
    if "env__row_count" not in index:
        raise ValueError("Environment artifact missing env__row_count")
    supported_names.add("env__row_count")
    mask[:, index["env__row_count"]] = rows_present
    for name in names:
        if name.startswith(NON_SCALING_PREFIX):
            mask[:, index[name]] = True
            supported_names.add(name)
    unsupported = sorted(set(names) - supported_names)
    if unsupported:
        raise ValueError(f"No feature-mask semantics defined for environment features: {unsupported}")
    return mask
