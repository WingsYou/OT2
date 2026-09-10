from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib
import sys
import unittest

MODULE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from clinical_modality_overlap_v1 import (  # noqa: E402
    PRIMARY_MODALITIES, RETINAL_MODALITIES, build_overlap_plan,
    feature_indices_for_group,
)


def row(source, response="1", table="observation"):
    prefix = "meas" if table == "measurement" else "obsc"
    return {"feature_name": f"{prefix}__{source}__c{response}", "source_key": source, "table": table}


class ClinicalOverlapPlanTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("rtci_odm1", "1"), row("rtci_odm1", "2"),
            row("rtci_odm2"), row("rtei_odfaf"), row("rtfl_od"),
            row("viaodaxi", table="measurement"), row("viaomthd"),
            row("dmlact", "1"), row("dmlact", "2"), row("ces7"),
            row("via1"), row("import_glucose", table="measurement"), row("eos_ds"),
        ]
        self.schema = {"feature_names": [r["feature_name"] for r in self.rows], "n_features": len(self.rows)}
        self.plan = build_overlap_plan(self.schema, self.rows)

    def test_direct_mapping_keeps_all_available_response_encodings(self):
        self.assertEqual(self.plan["direct_by_modality"]["retinal_oct"], [0, 1])
        self.assertEqual(self.plan["direct_by_modality"]["retinal_octa"], [2])
        self.assertEqual(self.plan["direct_by_modality"]["retinal_photography"], [3])
        self.assertEqual(self.plan["direct_by_modality"]["retinal_flio"], [4])
        self.assertEqual(self.plan["groups"]["retinal_oct"]["source_keys"], ["rtci_odm1"])

    def test_shared_domain_and_questionnaires_have_different_semantics(self):
        self.assertEqual(feature_indices_for_group(self.plan, ["retinal_oct"], "domain_clinical"), (0, 1, 5, 6))
        self.assertEqual(feature_indices_for_group(self.plan, ["wearable_activity_monitor"], "domain_clinical"), (7, 8, 9))
        self.assertEqual(feature_indices_for_group(self.plan, ["wearable_activity_monitor"], "direct_clinical"), ())
        for modality in ("wearable_blood_glucose", "environment", "cardiac_ecg"):
            self.assertEqual(feature_indices_for_group(self.plan, [modality], "domain_clinical"), ())

    def test_family_union_is_deduplicated_in_original_schema_order(self):
        targets = ["retinal_flio", "retinal_oct", "retinal_oct", "retinal_photography", "retinal_octa"]
        self.assertEqual(feature_indices_for_group(self.plan, targets, "domain_clinical"), tuple(range(7)))
        self.assertEqual(feature_indices_for_group(self.plan, targets, "direct_clinical"), tuple(range(5)))
        self.assertEqual(feature_indices_for_group(self.plan, targets, "branch_only"), ())
        self.assertEqual(feature_indices_for_group(self.plan, ["clinical_data"] + targets, "domain_clinical"), ())

    def test_extra_vision_questionnaires_labs_and_eos_are_not_silently_added(self):
        selected = feature_indices_for_group(self.plan, [m for m in PRIMARY_MODALITIES if m != "clinical_data"], "domain_clinical")
        self.assertEqual(selected, tuple(range(10)))
        scope = self.plan["excluded_scope"]["additional_vision_questionnaire"]
        self.assertEqual(scope["indices"], [10])
        self.assertFalse(scope["masked_by_any_profile"])

    def test_hash_is_reproducible_normalized_and_tracks_schema_mapping(self):
        expected = copy.deepcopy(self.plan)
        digest = expected.pop("plan_sha256")
        raw = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(digest, build_overlap_plan(dict(reversed(list(self.schema.items()))), self.rows)["plan_sha256"])
        changed = dict(self.schema, task="synthetic_other_task")
        self.assertNotEqual(digest, build_overlap_plan(changed, self.rows)["plan_sha256"])

    def test_schema_alias_and_optional_task_do_not_require_production_dimensions(self):
        plan = build_overlap_plan({"feature_order": self.schema["feature_names"]}, self.rows)
        self.assertEqual(plan["clinical_feature_count"], 13)
        self.assertEqual(plan["direct_counts_by_modality"]["retinal_oct"], 2)

    def test_incomplete_catalog_bundle_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            build_overlap_plan(self.schema, self.rows[:1] + self.rows[2:])

    def test_reordered_and_duplicate_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            build_overlap_plan(self.schema, list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_overlap_plan({"feature_names": self.schema["feature_names"] + self.schema["feature_names"][:1]}, self.rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_overlap_plan(self.schema, self.rows + self.rows[:1])

    def test_schema_alias_conflict_and_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_overlap_plan(dict(self.schema, feature_order=list(reversed(self.schema["feature_names"]))), self.rows)
        with self.assertRaisesRegex(ValueError, "n_features"):
            build_overlap_plan(dict(self.schema, n_features=999), self.rows)

    def test_unmapped_acquisition_and_wrong_table_fail_closed(self):
        unknown = [row("rtnew_device")]
        with self.assertRaisesRegex(ValueError, "unmapped retinal"):
            build_overlap_plan({"feature_names": [unknown[0]["feature_name"]]}, unknown)
        wrong = [row("rtci_odm1", table="measurement")]
        with self.assertRaisesRegex(ValueError, "unexpected table"):
            build_overlap_plan({"feature_names": [wrong[0]["feature_name"]]}, wrong)

    def test_catalog_cannot_hide_selected_source_under_another_name(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["source_key"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_overlap_plan(self.schema, rows)

    def test_unknown_profile_and_modality_rejected_even_when_clinical_removed(self):
        with self.assertRaisesRegex(ValueError, "unknown clinical overlap profile"):
            feature_indices_for_group(self.plan, [], "new_profile")
        with self.assertRaisesRegex(ValueError, "unknown modality"):
            feature_indices_for_group(self.plan, ["clinical_data", "unknown"], "domain_clinical")
        with self.assertRaisesRegex(ValueError, "sequence"):
            feature_indices_for_group(self.plan, "retinal_oct", "domain_clinical")

    def test_corrupted_plan_positions_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["domain_by_modality"]["retinal_oct"] = [13]
        with self.assertRaisesRegex(ValueError, "invalid clinical position"):
            feature_indices_for_group(plan, ["retinal_oct"], "domain_clinical")


class ActualClusterCatalogTests(unittest.TestCase):
    def test_actual_three_task_feature_mapping(self):
        root = MODULE_ROOT / "audits/task_leakage_evidence_v1/evidence/artifacts/clinical"
        if not root.is_dir():
            self.skipTest("Local aggregate cluster evidence is not installed")
        for task, expected_width in (("moca_binary", 955), ("cvd_broad_binary", 965), ("cvd_strict_binary", 974)):
            with self.subTest(task=task):
                schema = json.loads((root / task / f"clinical_schema_{task}.json").read_text())
                with (root / task / f"clinical_feature_catalog_{task}.csv").open(newline="") as handle:
                    catalog = list(csv.DictReader(handle))
                plan = build_overlap_plan(schema, catalog)
                self.assertEqual(plan["clinical_feature_count"], expected_width)
                self.assertEqual([plan["direct_counts_by_modality"][m] for m in RETINAL_MODALITIES], [3, 27, 24, 31])
                self.assertEqual([plan["domain_counts_by_modality"][m] for m in RETINAL_MODALITIES], [30, 54, 51, 58])
                self.assertEqual(len(feature_indices_for_group(plan, RETINAL_MODALITIES, "direct_clinical")), 85)
                self.assertEqual(len(feature_indices_for_group(plan, RETINAL_MODALITIES, "domain_clinical")), 112)
                self.assertEqual(plan["groups"]["shared_ocular_function"]["feature_count"], 27)
                self.assertEqual(plan["groups"]["activity_sleep_questionnaire"]["feature_count"], 13)
                self.assertEqual(plan["excluded_scope"]["additional_vision_questionnaire"]["feature_count"], 24)


if __name__ == "__main__":
    unittest.main()
