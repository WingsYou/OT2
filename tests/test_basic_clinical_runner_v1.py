"""Verify launch ordering, failure propagation, snapshot integrity and installation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_basic_clinical_v1.sh"
TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TESTS = ("test_clinical_basic_information_v1.py", "test_analyze_basic_clinical_v1.py",
         "test_clinical_overlap_inference_v1.py", "test_basic_clinical_runner_v1.py")
MANIFEST = "configs/bundles/basic_clinical_v1.manifest"
CHECKSUMS = "configs/bundles/basic_clinical_v1.sha256"


def seal(root):
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                   and str(p.relative_to(root)) not in {MANIFEST, CHECKSUMS}) + [MANIFEST, CHECKSUMS]
    (root / MANIFEST).write_text("\n".join(names) + "\n")
    (root / CHECKSUMS).write_text("".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names if name != CHECKSUMS))


class BasicRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ot2-basic-runner-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.code, self.out = self.root / "code with spaces", self.root / "artifacts"
        self.code.mkdir()
        (self.code / "configs").mkdir()
        (self.code / "tests").mkdir()
        (self.code / "analyze_basic_clinical_v1.py").write_text("fixture")
        for name in TESTS:
            (self.code / "tests" / name).touch()
        for name in ("verify_basic_clinical_bundle_v1.py", "install_basic_clinical_v1.sh"):
            shutil.copyfile(ROOT / name, self.code / name)
        for task, short in zip(TASKS, ("moca", "cvd_broad", "cvd_strict")):
            (self.code / "configs" / f"task2_v2_{short}_full_assets_v1.yaml").touch()
            checkpoint = self.out / "model_full_assets_v1/models" / task / "seed42/best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
        seal(self.code)
        self.events_path = self.root / "calls.jsonl"
        fake = self.root / "fake-python"
        fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
source = sys.stdin.read() if args and args[0] == "-" else ""
phase = "config" if "basic_clinical-config-preflight" in source else "gpu" if "basic_clinical-cuda-preflight" in source else "test" if "unittest" in args else "check" if "--check-only" in args else "analysis"
with open(os.environ["MOCK_EVENTS"], "a") as f:
    f.write(json.dumps({"phase": phase, "args": args, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
if phase == os.environ.get("MOCK_FAIL_PHASE"):
    if not os.environ.get("MOCK_FAIL_TASK") or os.environ["MOCK_FAIL_TASK"] in " ".join(args):
        raise SystemExit(7)
if phase == "analysis":
    out = pathlib.Path(args[args.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=False)
    for name in ["basic_clinical_plan.json", "basic_clinical_conditions.json", "basic_clinical_metrics.csv", "basic_clinical_summary.csv", "basic_clinical_bootstrap_intervals.csv", "basic_clinical_paired_predictions.csv", "basic_clinical_cohort_counts.csv", "basic_clinical_metadata.json", "source_token_cache_val.pt"]:
        (out / name).write_text("fixture")
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, CODE_ROOT=str(self.code), OUT_ROOT=str(self.out),
                        PYTHON_BIN=str(fake), MOCK_EVENTS=str(self.events_path))
        for key in ("FULL_ROOT", "BASIC_ROOT", "TASKS", "MOCK_FAIL_PHASE", "MOCK_FAIL_TASK",
                    "BASIC_BATCH_SIZE", "BASIC_WORKERS", "BASIC_BOOTSTRAPS"):
            self.env.pop(key, None)

    def run_stage(self, stage="check", **env):
        return subprocess.run(["bash", str(RUNNER), stage], env=dict(self.env, **env),
                              text=True, capture_output=True, timeout=60)

    def events(self):
        return [json.loads(line) for line in self.events_path.read_text().splitlines()] if self.events_path.exists() else []

    def test_check_verifies_snapshot_then_cpu_preflight_for_all_tasks(self):
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SNAPSHOT_OK", result.stdout)
        self.assertEqual([e["phase"] for e in self.events()], ["config", "check", "check", "check"])
        self.assertTrue(all(e["cuda_visible"] == "" for e in self.events()))
        self.assertFalse((self.out / "model_full_assets_basic_clinical_v1").exists())

    def test_run_checks_tests_and_every_task_before_fresh_gpu_inference(self):
        result = self.run_stage("run")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual([e["phase"] for e in events], ["test"] * 4 + ["config", "check", "check", "check", "gpu", "analysis", "analysis", "analysis"])
        for e in events[-3:]:
            self.assertIn("--bf16", e["args"])
            self.assertNotIn("--source-token-cache", e["args"])
            self.assertNotIn("--groups", e["args"])

    def test_old_cache_is_ignored_and_preserved(self):
        path = self.out / "model_full_assets_importance_v1/moca_binary/seed42/source_token_cache_val.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old cache")
        result = self.run_stage("run", TASKS="moca_binary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--source-token-cache", self.events()[-1]["args"])
        self.assertEqual(path.read_bytes(), b"old cache")

    def test_modified_snapshot_blocks_every_stage_before_test_or_model_import(self):
        (self.code / "analyze_basic_clinical_v1.py").write_text("different version")
        for stage in ("check", "test", "run"):
            result = self.run_stage(stage)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("hash mismatch", result.stderr)
        self.assertEqual(self.events(), [])

    def test_late_preflight_failure_prevents_gpu_and_outputs(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="check", MOCK_FAIL_TASK="cvd_strict")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("gpu", [e["phase"] for e in self.events()])
        self.assertFalse((self.out / "model_full_assets_basic_clinical_v1").exists())

    def test_cpu_test_failure_stops_before_study_data_and_gpu(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="test")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()], ["test"])

    def test_failed_inference_propagates_through_tee_and_stops_next_task(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="analysis")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()].count("analysis"), 1)

    def test_existing_results_or_symlink_never_overwritten(self):
        path = self.out / "model_full_assets_basic_clinical_v1/cvd_strict_binary/seed42/val_basic_clinical"
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root / "nonexistent")
        result = self.run_stage("run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output already exists", result.stderr)
        self.assertNotIn("analysis", [e["phase"] for e in self.events()])

    def test_invalid_task_or_options_rejected(self):
        for env in ({"TASKS": ""}, {"TASKS": "moca_binary moca_binary"}, {"TASKS": "unknown"},
                    {"BASIC_BATCH_SIZE": "0"}, {"BASIC_WORKERS": "-1"}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_stage("check", **env).returncode, 0)
        self.assertEqual(self.events(), [])

    @unittest.skipUnless(shutil.which("rsync"), "rsync unavailable")
    def test_install_copies_and_backs_up_replaced_files_preserving_unrelated_files(self):
        target = self.root / "installed"
        target.mkdir()
        (target / "unrelated.txt").write_text("keep")
        old = target / "analyze_basic_clinical_v1.py"
        old.write_text("oldcode")  # Same byte length as the fixture, same-time edits.
        backups = self.root / "backups"
        result = subprocess.run(["bash", str(self.code / "install_basic_clinical_v1.sh")],
                                env=dict(self.env, CODE_ROOT=str(target), BASIC_CODE_BACKUP_ROOT=str(backups)),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(old.read_text(), "fixture")
        self.assertEqual((target / "unrelated.txt").read_text(), "keep")
        saved = list(backups.glob("*/analyze_basic_clinical_v1.py"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_text(), "oldcode")


if __name__ == "__main__":
    unittest.main()
