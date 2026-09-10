import copy
import unittest
from domain_test_fixtures_v1 import catalog_fixture
from clinical_domain_groups_v1 import build_domain_plan


class DomainMappingTests(unittest.TestCase):
    def test_cross_table_copies_and_shared_sources_are_all_removed(self):
        schema, rows = catalog_fixture()
        plan = build_domain_plan(schema, rows)
        expected = {i for i, r in enumerate(rows) if r["source_key"] == "mhoccur_rvo"}
        self.assertEqual(len(expected), 3)
        for group in ("cardiac_ecg", "retinal_all"):
            self.assertTrue(expected.issubset(plan["groups"][group]["indices"]))

    def test_union_basic_protection_and_unrelated_fields(self):
        schema, rows = catalog_fixture()
        plan = build_domain_plan(schema, rows)
        groups = plan["groups"]
        self.assertEqual(set(groups["wearable_all"]["indices"]), set(groups["wearable_activity_monitor"]["indices"]) | set(groups["wearable_blood_glucose"]["indices"]))
        for g, d in groups.items():
            if g != "clinical_data":
                self.assertFalse(set(plan["basic_indices"]) & set(d["indices"]))
                self.assertFalse({"eos_ds", "bmi_vsorres", "lbscat_a1c"} & set(d["source_keys"]))
        self.assertEqual(groups["environment"]["indices"], [])
        self.assertEqual(groups["clinical_data"]["indices"], list(range(len(rows))))
        self.assertTrue({"ces7", "cestl", "cm_slp"} <= set(groups["wearable_activity_monitor"]["source_keys"]))

    def test_wrong_order_source_identity_and_unknown_retinal_source_fail(self):
        schema, rows = catalog_fixture()
        for invalid in (rows[::-1], rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                build_domain_plan(schema, invalid)
        altered = copy.deepcopy(rows)
        altered[-1]["table"] = "observation"
        with self.assertRaises(ValueError):
            build_domain_plan(schema, altered)

    def test_absent_task_fields_are_recorded_not_reintroduced(self):
        schema, rows = catalog_fixture()
        plan = build_domain_plan(schema, rows)
        self.assertIn("mhoccur_mi", plan["groups"]["cardiac_ecg"]["absent_source_keys"])
        self.assertNotIn("mhoccur_mi", plan["groups"]["cardiac_ecg"]["source_keys"])
        self.assertEqual(plan, build_domain_plan(schema, rows))


if __name__ == "__main__":
    unittest.main()
