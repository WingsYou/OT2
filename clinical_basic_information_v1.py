"""Resolve the agreed 23 basic-information columns against a trained schema.

This module uses only metadata. It never selects features from outcomes or
validation performance, and requires the audited source counts to match.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


BASIC_SOURCES = {
    ("participants", "age"): ("age_related", "numeric", 1),
    ("observation", "cage"): ("age_related", "numeric", 1),
    ("observation", "brthyy"): ("age_related", "numeric", 1),
    ("observation", "years_of_education"): ("education", "numeric", 1),
    ("observation", "pxed1"): ("education", "categorical_concept_presence", 8),
    ("observation", "cl_maristat"): ("marital_status", "categorical_concept_presence", 5),
    ("observation", "pxep1"): ("language_background", "categorical_concept_presence", 2),
    ("observation", "pxep2"): ("language_background", "categorical_concept_presence", 2),
    ("observation", "pxep3"): ("language_background", "categorical_concept_presence", 2),
}


def build_basic_clinical_plan(
    schema: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    order = schema.get("feature_names")
    if (not isinstance(order, list) or not order
            or any(not isinstance(x, str) or not x or x != x.strip() for x in order)
            or len(set(order)) != len(order)):
        raise ValueError("schema must contain unique non-empty feature_names")
    if schema.get("n_features") != len(order):
        raise ValueError("schema width differs from feature_names")
    if [r.get("feature_name") for r in catalog] != order:
        raise ValueError("catalog order must exactly equal trained schema order")
    counts: Counter = Counter()
    group_counts: Counter = Counter()
    kept = []
    identities = []
    known_keys = {source for _, source in BASIC_SOURCES}
    for i, row in enumerate(catalog):
        table, source = row.get("table"), row.get("source_key")
        if not isinstance(table, str) or not isinstance(source, str):
            raise ValueError("catalog table/source_key must be strings")
        parts = order[i].split("__")
        if len(parts) < 2 or parts[1] != source:
            raise ValueError("catalog feature/source identity mismatch")
        identity = (table, source)
        identities.append({"feature_name": order[i], "table": table, "source_key": source})
        if source in known_keys and identity not in BASIC_SOURCES:
            raise ValueError(f"basic source has unexpected table: {identity}")
        if identity in BASIC_SOURCES:
            group, kind, _ = BASIC_SOURCES[identity]
            if row.get("value_kind") != kind:
                raise ValueError(f"basic source has unexpected value kind: {identity}")
            counts[identity] += 1
            group_counts[group] += 1
            kept.append(i)
    expected = {key: value[2] for key, value in BASIC_SOURCES.items()}
    if dict(counts) != expected:
        raise ValueError(f"expected the audited 23 basic columns; observed counts: {dict(counts)}")
    kept_set = set(kept)
    removed = [i for i in range(len(order)) if i not in kept_set]
    if not removed:
        raise ValueError("the full clinical view must contain non-basic columns")
    plan = {
        "schema_version": "ot2-basic-clinical-plan-v1",
        "task": schema.get("task"), "feature_order": list(order),
        "original_width": len(order), "n_retained": len(kept), "n_masked": len(removed),
        "retained_indices_zero_based": kept, "masked_indices_zero_based": removed,
        "retained_feature_names": [order[i] for i in kept],
        "masked_feature_names": [order[i] for i in removed],
        "group_counts": dict(group_counts),
        "source_identities": identities,
        "selection_basis": "Fixed age, education, marital status and language source allowlist; no outcome selection.",
        "masking_contract": "Preserve width/order; zero removed normalized values AND observed masks; re-encode clinical; preserve retained missingness.",
        "other_modalities": "All eight remain unchanged, including the ECG numeric report.",
    }
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    plan["plan_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return plan
