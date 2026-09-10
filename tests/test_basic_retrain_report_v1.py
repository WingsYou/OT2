import csv
import tempfile
import unittest
from pathlib import Path

from basic_retrain_fixtures_v1 import complete_fixture, prepared_fixture, trained_fixture
from basic_retrain_contract_v1 import TASKS, read_json, write_json, sha256
from summarize_basic_retrain_v1 import write_report, load_importance
from prepare_basic_clinical_training_v1 import write_csv


class BasicRetrainReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_complete_report_includes_training_controls_strata_and_no_ids(self):
        for task in TASKS:
            complete_fixture(self.root, task)
        self.assertTrue(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        for s in ("Overall status: COMPLETE", "+0.0300 [+0.0100, +0.0500]", "nonclinical_all", "retinal_all", "wearable_all",
                  "New-model validation baseline", "No test inference", "any_target_present_with_remaining_input", "person__age"):
            self.assertIn(s, text)
        self.assertNotIn("PRIVATE_SYNTHETIC_ID", text)
        self.assertNotIn("eos_ds", text)

    def test_training_only_and_failure_leave_useful_partial_report(self):
        prepared_fixture(self.root)
        trained_fixture(self.root)
        write_json(self.root / "RUN_STATUS.json", {"state": "failed", "phase": "importance", "error": "synthetic OOM"})
        self.assertFalse(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        self.assertIn("synthetic OOM", text)
        self.assertIn("0.7000", text)
        self.assertIn("INCOMPLETE", text)

    def test_tampered_checkpoint_or_results_not_reported_as_complete(self):
        complete_fixture(self.root)
        (self.root / "models/moca_binary/seed42/best_model.pt").write_text("changed checkpoint")
        self.assertFalse(write_report(self.root, (TASKS[0],)))
        self.assertIn("hash/path mismatch", (self.root / "SUMMARY.md").read_text())

    def test_wrong_sign_rejected_even_after_output_hash_update(self):
        complete_fixture(self.root)
        folder = self.root / "importance/moca_binary/seed42/val"
        with (folder / "importance_summary.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["mean_perturbed_metric"] = "0.8"
        write_csv(folder / "importance_summary.csv", list(rows[0]), rows)
        meta = read_json(folder / "importance_metadata.json")
        meta["output_sha256"]["importance_summary.csv"] = sha256(folder / "importance_summary.csv")
        write_json(folder / "importance_metadata.json", meta)
        with self.assertRaisesRegex(ValueError, "degradation"):
            load_importance(self.root, TASKS[0])

    def test_old_threshold_or_test_split_is_rejected(self):
        complete_fixture(self.root)
        path = self.root / "importance/moca_binary/seed42/val/importance_metadata.json"
        original = read_json(path)
        for key, value in (("threshold", 0.45519860), ("split", "test")):
            changed = {**original, key: value}
            write_json(path, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_importance(self.root, TASKS[0])


if __name__ == "__main__":
    unittest.main()
