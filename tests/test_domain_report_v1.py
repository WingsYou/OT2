import json
from pathlib import Path
import tempfile
import unittest

from domain_test_fixtures_v1 import write_result
from summarize_domain_ablation_v1 import write_report, load_task, TASKS, sha256


class DomainReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def result(self, task):
        path = self.root / task / "seed42/val_domain"
        write_result(path, task)
        return path

    def test_complete_report_contains_correct_signs_strata_fields_and_no_ids(self):
        for task in TASKS:
            self.result(task)
        self.assertTrue(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        for expected in ("Overall status: COMPLETE", "+0.0300 [+0.0200, +0.0400]", "+0.0800 [+0.0700, +0.0900]",
                         "CVD strict", "any_target_information_present", "mhoccur_rvo", "checkpoint SHA256", "source cache SHA256"):
            self.assertIn(expected, text)
        self.assertNotIn("PRIVATE_SYNTHETIC_ID", text)

    def test_partial_failed_run_reports_available_results_without_success(self):
        self.result(TASKS[0])
        (self.root / "RUN_STATUS.json").write_text(json.dumps({"state": "failed", "phase": "inference", "error": "GPU failure"}))
        self.assertFalse(write_report(self.root))
        text = (self.root / "SUMMARY.md").read_text()
        self.assertIn("INCOMPLETE / FAILED", text)
        self.assertIn("GPU failure", text)
        self.assertIn("+0.0300", text)
        self.assertIn("Not available", text)

    def test_tampered_file_is_not_summarized(self):
        path = self.result(TASKS[0])
        with (path / "domain_summary.csv").open("a") as f:
            f.write("corrupt")
        self.assertFalse(write_report(self.root, (TASKS[0],)))
        self.assertIn("hash mismatch", (self.root / "SUMMARY.md").read_text())

    def test_mixed_code_is_flagged_even_when_all_outputs_complete(self):
        for task in TASKS:
            self.result(task)
        path = self.root / TASKS[1] / "seed42/val_domain/domain_metadata.json"
        meta = json.loads(path.read_text())
        meta["code_sha256"]["synthetic.py"] = "9" * 64
        path.write_text(json.dumps(meta))
        self.assertFalse(write_report(self.root))
        self.assertIn("Tasks differ in code", (self.root / "SUMMARY.md").read_text())

    def test_inconsistent_degradation_is_rejected_even_with_updated_file_hash(self):
        path = self.result(TASKS[0])
        summary = path / "domain_summary.csv"
        text = summary.read_text().replace("0.6699999999999999", "0.5")
        self.assertNotEqual(text, summary.read_text())
        summary.write_text(text)
        meta_path = path / "domain_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["output_sha256"][summary.name] = sha256(summary)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "degradation sign"):
            load_task(path, TASKS[0])


if __name__ == "__main__":
    unittest.main()
