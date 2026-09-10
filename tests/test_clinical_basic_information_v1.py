from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_basic_information_v1 import build_basic_clinical_plan


def catalog_fixture():
    rows = []
    # An explicit source fixture, independent of the implementation's allowlist.
    for source, prefix, count in [
        ("age", "person", 1), ("cage", "obse", 1), ("brthyy", "obse", 1),
        ("years_of_education", "obse", 1), ("pxed1", "obsc", 8),
        ("cl_maristat", "obsc", 5), ("pxep1", "obsc", 2),
        ("pxep2", "obsc", 2), ("pxep3", "obsc", 2),
        ("sualcage", "obse", 1), ("import_glucose", "meas", 1),
        ("eos_ds", "obse", 1),
    ]:
        for i in range(count):
            rows.append({"feature_name": f"{prefix}__{source}__c{i}", "source_key": source,
                         "table": "participants" if prefix == "person" else "measurement" if prefix == "meas" else "observation",
                         "value_kind": "categorical_concept_presence" if prefix == "obsc" else "numeric"})
    schema = {"task": "toy_binary", "n_features": len(rows),
              "feature_names": [r["feature_name"] for r in rows]}
    return schema, rows


class BasicMappingTests(unittest.TestCase):
    def test_exact_scope_excludes_drinking_age_labs_eos_and_preserves_positions(self):
        schema, rows = catalog_fixture()
        # Move a removed variable inside the retained block to detect slicing by width.
        rows.insert(2, rows.pop(24))
        schema["feature_names"] = [r["feature_name"] for r in rows]
        plan = build_basic_clinical_plan(schema, rows)
        self.assertEqual((plan["original_width"], plan["n_retained"], plan["n_masked"]), (26, 23, 3))
        self.assertEqual(plan["masked_indices_zero_based"], [2, 24, 25])
        self.assertEqual(plan["group_counts"], {"age_related": 3, "education": 9, "marital_status": 5, "language_background": 6})
        self.assertFalse(any("sualcage" in name or "glucose" in name or "eos_ds" in name for name in plan["retained_feature_names"]))
        self.assertEqual(plan, build_basic_clinical_plan(schema, rows))

    def test_reordered_missing_duplicate_or_misidentified_catalog_rejected(self):
        schema, rows = catalog_fixture()
        invalid = [rows[::-1], rows[:-1], rows + rows[:1]]
        bad = copy.deepcopy(rows)
        bad[0]["source_key"] = "cage"
        invalid.append(bad)
        for catalog in invalid:
            with self.subTest(catalog=catalog[:1]), self.assertRaises(ValueError):
                build_basic_clinical_plan(schema, catalog)

    def test_changed_basic_vocabulary_table_or_kind_requires_review(self):
        for mutation in ("remove_category", "wrong_table", "wrong_kind"):
            schema, rows = catalog_fixture()
            if mutation == "remove_category":
                rows.pop(4)
                schema["n_features"] = len(rows)
                schema["feature_names"] = [r["feature_name"] for r in rows]
            elif mutation == "wrong_table":
                rows[4]["table"] = "measurement"
            else:
                rows[4]["value_kind"] = "numeric"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_basic_clinical_plan(schema, rows)


if __name__ == "__main__":
    unittest.main()
