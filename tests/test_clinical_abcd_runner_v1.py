"""Exercise the real runner/report flow with synthetic subprocess results."""
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
TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
MANIFEST = "configs/bundles/clinical_abcd_v1.manifest"
CHECKSUMS = "configs/bundles/clinical_abcd_v1.sha256"


def seal(root):
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                   and str(p.relative_to(root)) not in {MANIFEST, CHECKSUMS}) + [MANIFEST, CHECKSUMS]
    (root / MANIFEST).write_text("\n".join(names) + "\n")
    (root / CHECKSUMS).write_text("".join(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
                                        for name in names if name != CHECKSUMS))


class ABCDRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ot2-abcd-runner-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.code, self.out = self.root / "code with spaces", self.root / "artifacts"
        self.code.mkdir()
        (self.code / "configs").mkdir()
        (self.code / "tests").mkdir()
        for name in ("run_clinical_abcd_v1.py", "summarize_clinical_abcd_v1.py",
                     "clinical_abcd_plan_v1.py", "clinical_basic_information_v1.py", "verify_clinical_abcd_bundle_v1.py",
                     "install_clinical_abcd_v1.sh"):
            shutil.copyfile(ROOT / name, self.code / name)
        shutil.copyfile(ROOT / "tests/abcd_test_fixtures_v1.py", self.code / "tests/abcd_test_fixtures_v1.py")
        shutil.copyfile(ROOT / "tests/test_clinical_basic_information_v1.py", self.code / "tests/test_clinical_basic_information_v1.py")
        (self.code / "analyze_clinical_abcd_v1.py").write_text("fixture")
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
phase = "config" if "abcd-config-preflight" in source else "gpu" if "abcd-cuda-preflight" in source else "test" if "unittest" in args else "check" if "--check-only" in args else "analysis"
with open(os.environ["MOCK_EVENTS"], "a") as f:
    f.write(json.dumps({"phase": phase, "args": args, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
if phase == os.environ.get("MOCK_FAIL_PHASE"):
    if not os.environ.get("MOCK_FAIL_TASK") or os.environ["MOCK_FAIL_TASK"] in " ".join(args):
        print("synthetic subprocess failure", flush=True)
        raise SystemExit(7)
if phase == "analysis":
    sys.path.insert(0, str(pathlib.Path(os.environ["CODE_ROOT"]) / "tests"))
    from abcd_test_fixtures_v1 import write_result
    out = pathlib.Path(args[args.index("--out-dir") + 1])
    split = args[args.index("--split") + 1]
    write_result(out, out.parts[-3], split=split)
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, CODE_ROOT=str(self.code), OUT_ROOT=str(self.out),
                        PYTHON_BIN=str(fake), MOCK_EVENTS=str(self.events_path))
        for key in ("FULL_ROOT", "ABCD_ROOT", "TASKS", "MOCK_FAIL_PHASE", "MOCK_FAIL_TASK",
                    "ABCD_BATCH_SIZE", "ABCD_WORKERS", "ABCD_BOOTSTRAPS", "ABCD_SPLIT", "ABCD_CACHE_ROOT", "ABCD_REUSE_CACHE"):
            self.env.pop(key, None)

    def run_stage(self, stage="check", **env):
        return subprocess.run([sys.executable, str(self.code / "run_clinical_abcd_v1.py"), stage],
                              env=dict(self.env, **env), text=True, capture_output=True, timeout=60)

    def events(self):
        return [json.loads(line) for line in self.events_path.read_text().splitlines()] if self.events_path.exists() else []

    @property
    def results(self):
        return self.out / "model_full_assets_clinical_abcd_v1"

    def test_check_is_read_only_and_every_task_is_preflighted_on_cpu(self):
        p = self.run_stage()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([e["phase"] for e in self.events()], ["config", "check", "check", "check"])
        self.assertTrue(all(e["cuda_visible"] == "" for e in self.events()))
        self.assertFalse(self.results.exists())

    def test_complete_run_writes_one_report_log_and_status(self):
        p = self.run_stage("run")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([e["phase"] for e in self.events()], ["test"] * 5 + ["config", "check", "check", "check", "gpu", "analysis", "analysis", "analysis"])
        self.assertEqual(json.loads((self.results / "RUN_STATUS.json").read_text())["state"], "complete")
        self.assertIn("Overall status: COMPLETE", (self.results / "SUMMARY.md").read_text())
        self.assertIn("[complete] cvd_strict_binary", (self.results / "run.log").read_text())
        self.assertNotIn("PRIVATE_SYNTHETIC_ID", (self.results / "SUMMARY.md").read_text())

    def test_cpu_test_failure_produces_failure_report_before_study_data(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="test")
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()], ["test"])
        self.assertIn("INCOMPLETE / FAILED", (self.results / "SUMMARY.md").read_text())
        self.assertIn("synthetic subprocess failure", (self.results / "run.log").read_text())

    def test_late_preflight_failure_prevents_gpu(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="check", MOCK_FAIL_TASK="cvd_strict")
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn("gpu", [e["phase"] for e in self.events()])
        self.assertIn("cvd_strict_binary", json.loads((self.results / "RUN_STATUS.json").read_text())["phase"])

    def test_second_task_failure_preserves_first_task_in_partial_report(self):
        p = self.run_stage("run", MOCK_FAIL_PHASE="analysis", MOCK_FAIL_TASK="cvd_broad")
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()].count("analysis"), 2)
        text = (self.results / "SUMMARY.md").read_text()
        self.assertIn("INCOMPLETE / FAILED", text)
        self.assertIn("+0.0300", text)
        state = json.loads((self.results / "RUN_STATUS.json").read_text())
        self.assertEqual(state["completed_tasks"], ["moca_binary"])

    def test_altered_snapshot_blocks_before_any_subprocess(self):
        (self.code / "analyze_clinical_abcd_v1.py").write_text("changed")
        p = self.run_stage("run")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("hash mismatch", p.stderr)
        self.assertFalse(self.events())

    def test_existing_output_not_overwritten(self):
        self.results.mkdir()
        (self.results / "keep").write_text("original")
        p = self.run_stage("run")
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(self.events())
        self.assertEqual((self.results / "keep").read_text(), "original")

    def test_test_split_uses_separate_paths_and_is_explicit(self):
        p = self.run_stage("run", ABCD_SPLIT="test")
        self.assertEqual(p.returncode, 0, p.stderr)
        root = self.out / "model_full_assets_clinical_abcd_v1_test"
        self.assertFalse(self.results.exists())
        self.assertIn("Exploratory test analysis", (root / "SUMMARY.md").read_text())
        for event in self.events():
            if event["phase"] in ("check", "analysis"):
                args = event["args"]
                self.assertEqual(args[args.index("--split") + 1], "test")
                self.assertIn("test_abcd", args[args.index("--out-dir") + 1])

    def test_reuses_only_existing_cache_and_can_explicitly_disable_reuse(self):
        cache = self.out / "model_full_assets_domain_ablation_v1" / TASKS[0] / "seed42/val_domain/source_token_cache_val.pt"
        cache.parent.mkdir(parents=True)
        cache.write_text("synthetic native cache")
        p = self.run_stage("check")
        self.assertEqual(p.returncode, 0, p.stderr)
        checks = [e for e in self.events() if e["phase"] == "check"]
        self.assertIn(str(cache), checks[0]["args"])
        self.assertFalse(any("--source-token-cache" in e["args"] for e in checks[1:]))
        self.events_path.unlink()
        p = self.run_stage("check", ABCD_REUSE_CACHE="0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(any("--source-token-cache" in e["args"] for e in self.events()))
        self.assertEqual(cache.read_text(), "synthetic native cache")

    def test_invalid_split_and_duplicate_tasks_fail(self):
        for env in ({"TASKS": ""}, {"TASKS": "moca_binary moca_binary"},
                    {"ABCD_SPLIT": "unknown"}, {"ABCD_WORKERS": "-1"}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_stage("check", **env).returncode, 0)
        self.assertFalse(self.events())

    @unittest.skipUnless(shutil.which("rsync"), "rsync unavailable")
    def test_install_checks_snapshot_and_backs_up_only_replaced_files(self):
        target, backups = self.root / "installed", self.root / "backups"
        target.mkdir()
        (target / "unrelated.txt").write_text("keep")
        (target / "analyze_clinical_abcd_v1.py").write_text("oldcode")
        p = subprocess.run(["bash", str(self.code / "install_clinical_abcd_v1.sh")],
                           env=dict(self.env, CODE_ROOT=str(target), ABCD_CODE_BACKUP_ROOT=str(backups)),
                           text=True, capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((target / "analyze_clinical_abcd_v1.py").read_text(), "fixture")
        self.assertEqual((target / "unrelated.txt").read_text(), "keep")
        self.assertEqual(next(backups.glob("*/analyze_clinical_abcd_v1.py")).read_text(), "oldcode")


if __name__ == "__main__":
    unittest.main()
