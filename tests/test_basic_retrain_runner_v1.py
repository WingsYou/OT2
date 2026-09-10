"""Real runner/installation with cheap, explicitly synthetic child processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basic_retrain_contract_v1 import CODE_FILES, TASKS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, CHECKSUMS = "configs/bundles/basic_retrain_v1.manifest", "configs/bundles/basic_retrain_v1.sha256"


def seal(root):
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(root)) not in (MANIFEST, CHECKSUMS)) + [MANIFEST, CHECKSUMS]
    (root / MANIFEST).write_text("\n".join(names) + "\n")
    (root / CHECKSUMS).write_text("".join(f"{hashlib.sha256((root / n).read_bytes()).hexdigest()}  {n}\n" for n in names if n != CHECKSUMS))


class BasicRetrainRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.code, self.out = self.root / "code with spaces", self.root / "experiment"
        self.code.mkdir()
        (self.code / "tests").mkdir()
        for name in (*CODE_FILES, "verify_basic_retrain_bundle_v1.py", "install_basic_retrain_v1.sh"):
            shutil.copyfile(ROOT / name, self.code / name)
        for name in ("basic_retrain_fixtures_v1.py", "test_clinical_basic_information_v1.py"):
            shutil.copyfile(ROOT / "tests" / name, self.code / "tests" / name)
        seal(self.code)
        fake = self.root / "fake-python"
        self.events_path = self.root / "events.jsonl"
        fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
source = sys.stdin.read() if args and args[0] == "-" else ""
phase = "test" if "unittest" in args else "gpu" if "basic-retrain-gpu-preflight" in source else "source_check" if "--config" in args and "--check-only" in args else "prepare" if "--config" in args else "train_check" if "--check-only" in args else "importance" if "--out-dir" in args else "train"
task = ""
for name, short in (("moca_binary", "moca"), ("cvd_broad_binary", "cvd_broad"), ("cvd_strict_binary", "cvd_strict")):
    if name in " ".join(args) or short + "_full_assets" in " ".join(args):
        task = name
        break
with open(os.environ["MOCK_EVENTS"], "a") as f:
    f.write(json.dumps({"phase": phase, "task": task, "args": args, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
root = pathlib.Path(os.environ["BASIC_RETRAIN_ROOT"])
if phase == os.environ.get("MOCK_FAIL_PHASE") and (not os.environ.get("MOCK_FAIL_TASK") or task == os.environ["MOCK_FAIL_TASK"]):
    if phase == "train":
        partial = root / "models" / task / "seed42"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "last_model.pt").write_text("incomplete synthetic training")
    print("synthetic subprocess failure")
    raise SystemExit(7)
sys.path.insert(0, str(pathlib.Path(os.environ["MOCK_CODE"]) / "tests"))
from basic_retrain_fixtures_v1 import prepared_fixture, trained_fixture, importance_fixture
if phase == "prepare":
    prepared_fixture(root, task)
elif phase == "train":
    trained_fixture(root, task)
elif phase == "importance":
    importance_fixture(root, task)
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, PYTHON_BIN=str(fake), BASIC_RETRAIN_ROOT=str(self.out),
                        BASIC_RETRAIN_BOOTSTRAPS="16", MOCK_EVENTS=str(self.events_path), MOCK_CODE=str(self.code))
        for name in ("TASKS", "MOCK_FAIL_PHASE", "MOCK_FAIL_TASK"):
            self.env.pop(name, None)

    def run_stage(self, stage, **kwargs):
        return subprocess.run([sys.executable, str(self.code / "run_basic_retrain_v1.py"), stage],
                              env=dict(self.env, **kwargs), capture_output=True, text=True, timeout=60)

    def events(self):
        return [json.loads(l) for l in self.events_path.read_text().splitlines()] if self.events_path.exists() else []

    def test_read_only_source_check(self):
        p = self.run_stage("check")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([e["phase"] for e in self.events()], ["source_check"] * 3)
        self.assertTrue(all(e["cuda"] == "" for e in self.events()))
        self.assertFalse(self.out.exists())

    def test_full_run_writes_one_report_log_and_no_test_inference(self):
        p = self.run_stage("run")
        self.assertEqual(p.returncode, 0, p.stderr)
        events = self.events()
        self.assertEqual(sum(e["phase"] == "train" for e in events), 3)
        self.assertEqual(sum(e["phase"] == "importance" for e in events), 3)
        self.assertIn("Overall status: COMPLETE", (self.out / "SUMMARY.md").read_text())
        self.assertIn("COMPLETE:", (self.out / "run.log").read_text())
        self.assertFalse(any("--split" in e["args"] for e in events))
        state = json.loads((self.out / "RUN_STATUS.json").read_text())
        self.assertEqual(state["completed_tasks"], list(TASKS))

    def test_cpu_failure_prevents_accessing_study_sources(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="test")
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()], ["test"])
        self.assertIn("FAILED", (self.out / "SUMMARY.md").read_text())

    def test_preflight_failure_prevents_any_training(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="train_check", MOCK_FAIL_TASK=TASKS[2])
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(any(e["phase"] in ("gpu", "train") for e in self.events()))

    def test_continue_reuses_completed_tasks_and_preserves_incomplete_training(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="train", MOCK_FAIL_TASK=TASKS[1])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("+0.0300", (self.out / "SUMMARY.md").read_text())
        self.events_path.unlink()
        p = self.run_stage("continue")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(any(e["phase"] in ("train", "importance") and e["task"] == TASKS[0] for e in self.events()))
        self.assertEqual(sum(e["phase"] == "train" for e in self.events()), 2)
        self.assertEqual(next((self.out / "incomplete_attempts").glob("*/training_*/last_model.pt")).read_text(), "incomplete synthetic training")
        self.assertIn("Overall status: COMPLETE", (self.out / "SUMMARY.md").read_text())

    def test_importance_failure_does_not_force_retraining(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="importance", MOCK_FAIL_TASK=TASKS[0])
        self.assertNotEqual(p.returncode, 0)
        self.events_path.unlink()
        p = self.run_stage("continue")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(any(e["phase"] == "train" and e["task"] == TASKS[0] for e in self.events()))

    def test_existing_run_and_changed_code_do_not_overwrite_results(self):
        self.out.mkdir()
        (self.out / "keep").write_text("original")
        self.assertNotEqual(self.run_stage("run").returncode, 0)
        self.assertFalse(self.events())
        (self.code / "train_basic_clinical_v1.py").write_text("changed")
        p = self.run_stage("check")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("hash mismatch", p.stderr)
        self.assertEqual((self.out / "keep").read_text(), "original")

    def test_invalid_settings_rejected(self):
        for env in ({"TASKS": ""}, {"TASKS": "moca_binary moca_binary"}, {"BASIC_RETRAIN_BOOTSTRAPS": "0"}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_stage("check", **env).returncode, 0)
        self.assertFalse(self.events())

    @unittest.skipUnless(shutil.which("rsync"), "rsync unavailable")
    def test_installer_backs_up_replaced_code_and_keeps_other_results(self):
        target, backup = self.root / "installed", self.root / "backups"
        target.mkdir()
        (target / "train_basic_clinical_v1.py").write_text("old code")
        (target / "unrelated_result").write_text("keep")
        p = subprocess.run(["bash", str(self.code / "install_basic_retrain_v1.sh")],
                           env=dict(self.env, CODE_ROOT=str(target), BASIC_RETRAIN_CODE_BACKUP_ROOT=str(backup)),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((target / "unrelated_result").read_text(), "keep")
        self.assertEqual(next(backup.glob("*/train_basic_clinical_v1.py")).read_text(), "old code")


if __name__ == "__main__":
    unittest.main()
