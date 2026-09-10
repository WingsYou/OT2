"""Fixed 2 x 2 inference controls; no training or feature redefinition."""
from __future__ import annotations

import hashlib
import json

from clinical_basic_information_v1 import build_basic_clinical_plan

NONCLINICAL = (
    "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa",
    "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose",
)
CONDITIONS = ("A", "B", "C", "D")
CONTRASTS = {"A_vs_B": ("A", "B"), "C_vs_D": ("C", "D")}
COHORTS = ("all_participants", "basic_clinical_present", "basic_and_nonclinical_present")
LABELS = {"A": "Full Clinical + other eight modalities",
          "B": "Full Clinical only",
          "C": "Basic Clinical + other eight modalities",
          "D": "Basic Clinical only"}


def plan_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def build_abcd_plan(schema, catalog):
    basic = build_basic_clinical_plan(schema, catalog)
    plan = {**basic, "schema_version": "ot2-clinical-abcd-plan-v1",
            "basic_plan_sha256": basic["plan_sha256"],
            "other_modalities": "Native availability in A/C; all eight disabled in B/D.",
            "conditions": {}}
    plan.pop("plan_sha256")
    for name in CONDITIONS:
        kept = basic["n_retained"] if name in ("C", "D") else basic["original_width"]
        plan["conditions"][name] = {
            "condition": name, "label": LABELS[name],
            "n_clinical_columns_retained": kept,
            "masked_indices": basic["masked_indices_zero_based"] if name in ("C", "D") else [],
            "disabled_modalities": list(NONCLINICAL) if name in ("B", "D") else [],
        }
    plan["plan_sha256"] = plan_hash(plan)
    return plan
