"""Audited clinical companions for frozen-model modality ablations.

This module chooses feature *positions*, without changing the trained input
width or preprocessing. The caller must mask both values and observed flags,
then recompute the clinical encoder token. No array or model library is needed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple


PRIMARY_MODALITIES = (
    "clinical_data", "cardiac_ecg", "environment", "retinal_flio",
    "retinal_oct", "retinal_octa", "retinal_photography",
    "wearable_activity_monitor", "wearable_blood_glucose",
)
PROFILES = ("branch_only", "direct_clinical", "domain_clinical")
RETINAL_MODALITIES = (
    "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
)

# Exact source identities were verified against original observation descriptions.
# A common device prefix can contain OCT and OCTA, so prefix matching is unsafe.
RETINAL_ACQUISITION_SOURCES = {
    "retinal_flio": ("rtfl_od", "rtfl_os"),
    "retinal_oct": (
        "rtci_odd1", "rtci_odm1", "rtci_osd1", "rtci_osm1",
        "rtma_odm", "rtma_odw", "rtma_osm", "rtma_osw",
        "rtso_odo", "rtso_odp", "rtso_oso", "rtso_osp",
        "rttr_odr", "rttr_osr",
    ),
    "retinal_octa": (
        "rtci_odd2", "rtci_odm2", "rtci_osd2", "rtci_osm2",
        "rtma_odm1", "rtma_osm1", "rtso_odm", "rtso_osm",
        "rttr_odm1", "rttr_odm2", "rttr_osm1", "rttr_osm2",
    ),
    # Photography includes CFP, infrared, and autofluorescence; FAF is not FLIO.
    "retinal_photography": (
        "rtei_odfaf", "rtei_odir", "rtei_osfaf", "rtei_osir",
        "rtop_odd", "rtop_odm", "rtop_osd", "rtop_osm",
        "rtsm_odc", "rtsm_odm", "rtsm_odn", "rtsm_odt",
        "rtsm_osc", "rtsm_osm", "rtsm_osn", "rtsm_ost",
    ),
}
OCULAR_FUNCTION_NUMERIC_SOURCES = (
    "mlcsodfcl", "mlcsodlog", "mlcsodmiss", "mlcsosfcl", "mlcsoslog", "mlcsosmiss",
    "plcsodfcl", "plcsodlog", "plcsodmiss", "plcsosfcl", "plcsoslog", "plcsosmiss",
    "viaodaxi", "viaodcyl", "viaodmlog", "viaodmscore", "viaodplog", "viaodpscore", "viaodsph",
    "viaosaxi", "viaoscyl", "viaosmlog", "viaosmscore", "viaosplog", "viaospscore", "viaossph",
)
OCULAR_FUNCTION_SOURCES = OCULAR_FUNCTION_NUMERIC_SOURCES + ("viaomthd",)
ACTIVITY_SLEEP_SOURCES = ("ces7", "dmlact", "dmlhex", "dmlvex")
EXTRA_VISION_QUESTIONNAIRE_SOURCES = ("via1", "via2", "via3", "via4", "via5", "via6")


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _string_list(value: Any, label: str) -> List[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of feature names")
    result = list(value)
    if not result or any(not isinstance(x, str) or not x or x != x.strip() for x in result):
        raise ValueError(f"{label} must contain non-empty, unpadded strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate feature names")
    return result


def build_overlap_plan(schema: Mapping, catalog_rows: Sequence[Mapping]) -> Dict[str, Any]:
    """Resolve the audited source bundles against an exact ordered clinical schema.

    Every available encoding of a matched source is included. Unknown ``rt*``
    sources, incomplete catalogs, and source/feature identity mismatches fail
    closed. The allowlists do not require a fixed feature count or all source
    variables to survive vocabulary fitting. Thus a small fixture or another
    task can use this function with its complete, internally consistent schema.
    """
    if not isinstance(schema, Mapping):
        raise ValueError("schema must be a mapping")
    raw_order = schema.get("feature_names", schema.get("feature_order"))
    feature_order = _string_list(raw_order, "schema feature_names/feature_order")
    if "feature_names" in schema and "feature_order" in schema:
        if feature_order != _string_list(schema["feature_order"], "schema feature_order"):
            raise ValueError("schema feature_names and feature_order disagree")
    if "n_features" in schema and schema["n_features"] != len(feature_order):
        raise ValueError("schema n_features disagrees with feature order")
    if isinstance(catalog_rows, (str, bytes)) or not isinstance(catalog_rows, Sequence):
        raise ValueError("catalog_rows must be a complete ordered sequence")

    acquisition_sources = {s for sources in RETINAL_ACQUISITION_SOURCES.values() for s in sources}
    numeric_sources = set(OCULAR_FUNCTION_NUMERIC_SOURCES)
    known_observation = acquisition_sources | {"viaomthd"} | set(ACTIVITY_SLEEP_SOURCES)
    rows: List[Dict[str, str]] = []
    for index, row in enumerate(catalog_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"catalog row {index} must be a mapping")
        selected = {}
        for key in ("feature_name", "source_key", "table"):
            value = row.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"catalog row {index} has invalid {key}")
            selected[key] = value
        source = selected["source_key"]
        parts = selected["feature_name"].split("__")
        if len(parts) < 2 or parts[1] != source:
            raise ValueError(f"catalog feature/source identity mismatch: {selected['feature_name']}")
        if source.startswith("rt") and source not in acquisition_sources:
            raise ValueError(f"unmapped retinal acquisition source: {source}")
        expected_table = "measurement" if source in numeric_sources else "observation" if source in known_observation else None
        if expected_table is not None and selected["table"] != expected_table:
            raise ValueError(f"unexpected table for matched source {source}: {selected['table']}")
        if source in numeric_sources and row.get("value_kind", "numeric") != "numeric":
            raise ValueError(f"ocular function source {source} must be numeric")
        rows.append(selected)

    catalog_order = [row["feature_name"] for row in rows]
    if len(set(catalog_order)) != len(catalog_order):
        raise ValueError("catalog contains duplicate feature names")
    if catalog_order != feature_order:
        raise ValueError("catalog feature order must exactly equal schema; missing, extra, or reordered columns found")

    def describe(sources: Sequence[str]) -> Dict[str, Any]:
        source_set = set(sources)
        indices = [i for i, row in enumerate(rows) if row["source_key"] in source_set]
        present = sorted({rows[i]["source_key"] for i in indices})
        return {
            "indices": indices,
            "feature_names": [feature_order[i] for i in indices],
            "feature_count": len(indices),
            "source_keys": present,
            "source_key_count": len(present),
            "allowlisted_source_keys": sorted(source_set),
        }

    groups = {modality: describe(sources) for modality, sources in RETINAL_ACQUISITION_SOURCES.items()}
    groups["shared_ocular_function"] = describe(OCULAR_FUNCTION_SOURCES)
    groups["activity_sleep_questionnaire"] = describe(ACTIVITY_SLEEP_SOURCES)
    direct = {modality: list(groups[modality]["indices"]) if modality in RETINAL_MODALITIES else [] for modality in PRIMARY_MODALITIES}
    domain = {modality: list(indices) for modality, indices in direct.items()}
    for modality in RETINAL_MODALITIES:
        domain[modality] = sorted(set(domain[modality]) | set(groups["shared_ocular_function"]["indices"]))
    domain["wearable_activity_monitor"] = list(groups["activity_sleep_questionnaire"]["indices"])
    excluded_vision = describe(EXTRA_VISION_QUESTIONNAIRE_SOURCES)
    excluded_vision["masked_by_any_profile"] = False

    plan: Dict[str, Any] = {
        "schema_version": "ot2-clinical-modality-overlap-v1",
        "task": schema.get("task"),
        "clinical_feature_count": len(feature_order),
        "feature_order": feature_order,
        "source_keys_in_feature_order": [row["source_key"] for row in rows],
        "catalog_identity_sha256": _canonical_hash(rows),
        "direct_by_modality": direct,
        "domain_by_modality": domain,
        "direct_counts_by_modality": {m: len(v) for m, v in direct.items()},
        "domain_counts_by_modality": {m: len(v) for m, v in domain.items()},
        "groups": groups,
        "excluded_scope": {
            "additional_vision_questionnaire": excluded_vision,
            "labs_and_eos_are_not_removed": True,
            "all_eye_related_clinical_information_removed": False,
        },
        "profile_definitions": {
            "branch_only": "Remove target source branches without additional clinical feature masks.",
            "direct_clinical": "Also mask all encodings of the target retinal acquisition sources.",
            "domain_clinical": "Also mask shared ocular function/method sources for any retinal target and activity/sleep questionnaires for activity-monitor targets.",
        },
        "interpretation": "Domain profiles estimate joint removal of a modality and specified related clinical information. Shared ocular features cannot be assigned uniquely to one retinal modality.",
        "masking_contract": "Mask both feature values and observed flags; preserve width and order; recompute clinical token. If clinical_data itself is removed, no additional feature positions are needed.",
    }
    plan["plan_sha256"] = _canonical_hash(plan)
    return plan


def feature_indices_for_group(plan: Mapping, modalities: Sequence[str], profile: str) -> Tuple[int, ...]:
    """Return unique clinical positions in schema order for a target-source group."""
    if profile not in PROFILES:
        raise ValueError(f"unknown clinical overlap profile: {profile}")
    if isinstance(modalities, (str, bytes)) or not isinstance(modalities, Sequence):
        raise ValueError("modalities must be a sequence of modality names")
    for modality in modalities:
        if not isinstance(modality, str) or modality not in PRIMARY_MODALITIES:
            raise ValueError(f"unknown modality: {modality}")
    if profile == "branch_only" or "clinical_data" in modalities:
        return ()
    field = "direct_by_modality" if profile == "direct_clinical" else "domain_by_modality"
    if not isinstance(plan, Mapping) or not isinstance(plan.get(field), Mapping):
        raise ValueError(f"plan has no valid {field}")
    result = set()
    width = plan.get("clinical_feature_count")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("plan has invalid clinical_feature_count")
    for modality in modalities:
        positions = plan[field].get(modality)
        if not isinstance(positions, (list, tuple)):
            raise ValueError(f"plan has no positions for {modality}")
        for position in positions:
            if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < width:
                raise ValueError(f"plan has invalid clinical position: {position}")
            result.add(position)
    return tuple(sorted(result))
