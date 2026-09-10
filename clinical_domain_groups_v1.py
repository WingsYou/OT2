"""Explicit, source-based Clinical deletion rules for domain ablations.

These groups describe joint information removal, not exclusive device ownership.
All encodings of a selected source are removed, including cross-table copies.
"""
from __future__ import annotations

import hashlib
import json

try:
    from .clinical_basic_information_v1 import build_basic_clinical_plan
    from .clinical_modality_overlap_v1 import build_overlap_plan, RETINAL_MODALITIES
except ImportError:
    from clinical_basic_information_v1 import build_basic_clinical_plan
    from clinical_modality_overlap_v1 import build_overlap_plan, RETINAL_MODALITIES

SCHEMA_VERSION = "ot2-clinical-domain-groups-v1"
DEFAULT_GROUPS = (
    "clinical_data", "cardiac_ecg", "environment", "retinal_all",
    "wearable_activity_monitor", "wearable_blood_glucose", "wearable_all",
)
MEMBERS = {
    "clinical_data": ("clinical_data",), "cardiac_ecg": ("cardiac_ecg",),
    "environment": ("environment",), "retinal_all": RETINAL_MODALITIES,
    "wearable_activity_monitor": ("wearable_activity_monitor",),
    "wearable_blood_glucose": ("wearable_blood_glucose",),
    "wearable_all": ("wearable_activity_monitor", "wearable_blood_glucose"),
    "retinal_wearable_all": RETINAL_MODALITIES + ("wearable_activity_monitor", "wearable_blood_glucose"),
}
LABELS = {
    "clinical_data": "All Clinical data",
    "cardiac_ecg": "ECG + cardiovascular Clinical",
    "environment": "Environmental sensor",
    "retinal_all": "All retinal + ocular Clinical",
    "wearable_activity_monitor": "Activity monitor + activity/sleep Clinical",
    "wearable_blood_glucose": "CGM + metabolic Clinical",
    "wearable_all": "All wearable + associated Clinical",
    "retinal_wearable_all": "All retinal and wearable + associated Clinical",
}
# Exact lists, never substring matching (e.g. lbscat_a1c is hemoglobin, not HbA1c).
CARDIO_MEASUREMENTS = (
    "bp1_sysbp_vsorres", "bp1_diabp_vsorres", "bp2_sysbp_vsorres", "bp2_diabp_vsorres",
    "import_total_cholesterol", "import_hdl_cholesterol", "import_ldl_cholesterol",
    "import_triglycerides", "import_troponin_t", "import_nt_probnp",
)
CARDIO_HISTORY = (
    "mhoccur_hbp", "mhoccur_lbp", "mhoccur_clsh", "mhoccur_mi",
    "mhoccur_strk", "mhoccur_circ", "mhoccur_cvdot", "mhoccur_rvo",
)
OCULAR_HISTORY = (
    "mhoccur_amd", "mhoccur_crt", "mhoccur_glc", "mhoccur_pdr",
    "mhoccur_ded", "mhoccur_rvo",
)
ACTIVITY_SLEEP = ("dmlact", "dmlhex", "dmlvex", "ces7", "cm_slp", "cestl")
METABOLIC_MEASUREMENTS = ("import_glucose", "import_hba1c", "import_insulin", "import_c_peptide")
METABOLIC_HISTORY = ("mh_a1c", "mhterm_dm2", "mhterm_predm")
METABOLIC_OBSERVATIONS = ("mh_dm_age", "cmtrt_a1c", "cmtrt_insln", "cmtrt_glcs", "cmtrt_lfst")
SCOPE_NOTES = [
    "The baseline is the unmodified frozen teacher; this run does not retrain or reconstruct inputs.",
    "Basic Clinical (23 encoded columns: age, education, marital status, language) remains except in All Clinical data.",
    "Retinal acquisition 85 + ocular function/form 27 + vision questionnaire 24 columns are supplemented with task-available ocular history.",
    "Activity/sleep additionally removes cm_slp and the entire CESD-10 total cestl to avoid retaining the ces7 component through a composite. The total also contains non-sleep information; this is a conservative broader deletion. Other CESD items remain.",
    "mhoccur_rvo belongs to both ocular and cardiovascular groups; joint removals use a deduplicated union.",
    "No additional Clinical fields are assigned to environmental sensors. Housing and neighborhood questionnaires remain.",
    "BMI/anthropometry, pulse (acquisition origin unresolved), CRP, general chemistry/CBC, family history, diet, other diabetes questionnaires (including PAID), psychology and cognition remain unless All Clinical data is removed. These are specified subsets, not removal of every biological correlate.",
    "Metabolic removal uses the four named laboratory measures, diabetes/prediabetes/elevated-A1c history, diagnosis age, and four cmtrt treatment/management items. Lipids are assigned to the cardiovascular group in this version.",
    "eos_ds is unchanged in the baseline and domain-specific conditions; it disappears only when the complete Clinical branch is disabled. The earlier EOS/time-window question is not changed by this experiment.",
    "Sources already excluded by a task's trained schema are recorded as absent, never reintroduced.",
]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def build_domain_plan(schema, catalog):
    basic = build_basic_clinical_plan(schema, catalog)
    overlap = build_overlap_plan(schema, catalog)
    order = basic["feature_order"]
    # Validate table identities, then select every encoding of the exact source.
    allowed = {}
    for sources, tables in (
        (CARDIO_MEASUREMENTS + METABOLIC_MEASUREMENTS, {"measurement"}),
        (CARDIO_HISTORY + OCULAR_HISTORY + METABOLIC_HISTORY, {"observation", "condition_occurrence"}),
        (ACTIVITY_SLEEP + METABOLIC_OBSERVATIONS + tuple(f"via{i}" for i in range(1, 7)), {"observation"}),
    ):
        for source in sources:
            allowed.setdefault(source, set()).update(tables)
    # Catalogs use the normalized table name 'condition'.
    for tables in allowed.values():
        if "condition_occurrence" in tables:
            tables.add("condition")
    for row in catalog:
        source = row["source_key"]
        if source in allowed and row["table"] not in allowed[source]:
            raise ValueError(f"Unexpected source table: {source}: {row['table']}")

    retinal_sources = set(OCULAR_HISTORY) | {f"via{i}" for i in range(1, 7)}
    for group in (*RETINAL_MODALITIES, "shared_ocular_function"):
        retinal_sources.update(overlap["groups"][group]["allowlisted_source_keys"])
    sources = {
        "clinical_data": {row["source_key"] for row in catalog},
        "cardiac_ecg": set(CARDIO_MEASUREMENTS + CARDIO_HISTORY),
        "environment": set(), "retinal_all": retinal_sources,
        "wearable_activity_monitor": set(ACTIVITY_SLEEP),
        "wearable_blood_glucose": set(METABOLIC_MEASUREMENTS + METABOLIC_HISTORY + METABOLIC_OBSERVATIONS),
    }
    sources["wearable_all"] = sources["wearable_activity_monitor"] | sources["wearable_blood_glucose"]
    sources["retinal_wearable_all"] = sources["retinal_all"] | sources["wearable_all"]
    basic_indices = set(basic["retained_indices_zero_based"])
    groups = {}
    for group, selected in sources.items():
        indices = [i for i, row in enumerate(catalog) if row["source_key"] in selected]
        if group != "clinical_data" and basic_indices.intersection(indices):
            raise ValueError(f"Domain would remove basic Clinical: {group}")
        present = {catalog[i]["source_key"] for i in indices}
        groups[group] = {
            "label": LABELS[group], "disabled_modalities": list(MEMBERS[group]),
            "indices": indices, "feature_count": len(indices),
            "feature_names": [order[i] for i in indices],
            "source_keys": sorted(present), "source_key_count": len(present),
            "allowlisted_source_keys": sorted(selected),
            "absent_source_keys": sorted(selected - present),
        }
    plan = {
        "schema_version": SCHEMA_VERSION, "task": schema["task"],
        "feature_order": order, "original_width": len(order),
        "basic_indices": sorted(basic_indices), "n_basic_columns": len(basic_indices),
        "catalog_identity_sha256": overlap["catalog_identity_sha256"],
        "groups": groups, "scope_notes": SCOPE_NOTES,
        "feature_records": [
            {"index": i, "feature_name": row["feature_name"], "table": row["table"],
             "source_key": row["source_key"], "is_basic": i in basic_indices,
             "domain_groups": [g for g in DEFAULT_GROUPS if g != "clinical_data" and i in groups[g]["indices"]]}
            for i, row in enumerate(catalog)
        ],
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan
