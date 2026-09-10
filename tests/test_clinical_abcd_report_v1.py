import csv
import json
from pathlib import Path
import tempfile
import unittest

from abcd_test_fixtures_v1 import write_result, write_csv, reseal
from summarize_clinical_abcd_v1 import write_report, load_task, TASKS
from clinical_abcd_plan_v1 import build_abcd_plan, NONCLINICAL, CONDITIONS
from test_clinical_basic_information_v1 import catalog_fixture


class ABCDReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def result(self, task=TASKS[0], split="val"):
        path = self.root / task / "seed42" / f"{split}_abcd"
        write_result(path, task, split=split)
        return path

    def test_exact_four_conditions_and_no_changes_to_full_clinical(self):
        schema, catalog = catalog_fixture()
        plan = build_abcd_plan(schema, catalog)
        self.assertEqual(tuple(plan["conditions"]), CONDITIONS)
        self.assertEqual(plan["n_retained"], 23)
        self.assertEqual(len(NONCLINICAL), 8)
        for n in CONDITIONS:
            d = plan["conditions"][n]
            self.assertEqual(d["disabled_modalities"], list(NONCLINICAL) if n in "BD" else [])
            self.assertEqual(d["masked_indices"], [23, 24, 25] if n in "CD" else [])
        self.assertEqual(plan, build_abcd_plan(schema, catalog))

    def test_complete_report_correct_four_values_effects_strata_and_only_23_retained_columns(self):
        for task in TASKS:
            self.result(task)
        self.assertTrue(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        for expected in ("Overall status: COMPLETE", "0.7000 | 0.6700 | 0.6000 | 0.5500",
                         "+0.0300 [+0.0200, +0.0400]", "+0.0500 [+0.0400, +0.0600]",
                         "basic_and_nonclinical_present", "checkpoint_sha256", "source_cache", "pxep3"):
            self.assertIn(expected, text)
        self.assertNotIn("PRIVATE_SYNTHETIC_ID", text)
        self.assertNotIn("eos_ds", text)  # This is removed, not in the retained-feature table.
        self.assertEqual(text.count("| person__age__c0 |"), 3)

    def test_partial_run_and_tampering_fail_clearly(self):
        path = self.result()
        (self.root / "RUN_STATUS.json").write_text(json.dumps({"state": "failed", "phase": "inference", "error": "GPU failure", "split": "val", "tasks": TASKS}))
        self.assertFalse(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        self.assertIn("INCOMPLETE / FAILED", text)
        self.assertIn("GPU failure", text)
        self.assertIn("+0.0300", text)
        with (path / "abcd_summary.csv").open("a") as f:
            f.write("corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_task(path)

    def test_mixed_code_and_split_rejected(self):
        for task in TASKS:
            self.result(task)
        path = self.root / TASKS[1] / "seed42/val_abcd/abcd_metadata.json"
        meta = json.loads(path.read_text())
        meta["code_sha256"] = {"wrong.py": "9" * 64}
        path.write_text(json.dumps(meta))
        self.assertFalse(write_report(self.root))
        with self.assertRaisesRegex(ValueError, "schema/split"):
            load_task(path.parent, expected_split="test")

    def test_test_report_explicitly_exploratory_and_threshold_from_val(self):
        path = self.result(split="test")
        self.assertTrue(write_report(self.root, (TASKS[0],), split="test"))
        self.assertIn("Exploratory test analysis", (self.root / "SUMMARY.md").read_text())
        meta_path = path / "abcd_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["threshold_source"] = "test_optimized"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Threshold"):
            load_task(path, expected_split="test")

    def test_incorrect_degradation_and_duplicate_rows_rejected_after_rehash(self):
        path = self.result()
        with (path / "abcd_summary.csv").open(newline="") as f:
            original = list(csv.DictReader(f))
        changed = [dict(r) for r in original]
        changed[0]["baseline_metric"] = 0.5
        write_csv(path / "abcd_summary.csv", changed)
        reseal(path)
        with self.assertRaisesRegex(ValueError, "degradation sign"):
            load_task(path)
        write_csv(path / "abcd_summary.csv", original + original[:1])
        reseal(path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_task(path)

    def test_empty_cohort_has_na_instead_of_failure(self):
        path = self.result()
        # Keep schema/paired comparisons intact, but represent an empty availability stratum.
        for filename in ("abcd_metrics.csv", "abcd_summary.csv", "abcd_bootstrap_intervals.csv", "abcd_cohort_counts.csv"):
            with (path / filename).open(newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                if r["cohort_scope"] == "basic_and_nonclinical_present":
                    r.update(n_evaluated=0, n_positive=0, n_negative=0, prevalence="", cohort_mask_sha256="9" * 64)
                    for key in ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "brier", "log_loss", "baseline_metric", "mean_perturbed_metric", "mean_degradation_delta", "point_estimate", "ci_lower", "ci_upper"):
                        if key in r:
                            r[key] = ""
            write_csv(path / filename, rows)
        reseal(path)
        self.assertTrue(write_report(self.root, (TASKS[0],)))
        self.assertIn("NA [NA, NA]", (self.root / "SUMMARY.md").read_text())


if __name__ == "__main__":
    unittest.main()
