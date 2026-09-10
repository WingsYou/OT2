"""Exercise shell orchestration using a stdlib fake Python executable."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "run_coupled_modality_importance_v1.sh"
TASKS = ("moca_binary", "cvd_broad_binary", "cvd_strict_binary")
TEST_MODULES = (
    "test_clinical_modality_overlap_v1.py", "test_clinical_overlap_inference_v1.py",
    "test_analyze_coupled_modality_importance_v1.py", "test_analyze_modality_importance_v2.py",
    "test_family_modality_ablation_v3.py",
)


class CoupledRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ot2-coupled-runner-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "code with spaces"
        self.out = self.root / "artifacts"
        self.code.mkdir()
        (self.code / "configs").mkdir()
        (self.code / "tests").mkdir()
        (self.code / "analyze_coupled_modality_importance_v1.py").touch()
        for name in TEST_MODULES:
            (self.code / "tests" / name).touch()
        for task, short in zip(TASKS, ("moca", "cvd_broad", "cvd_strict")):
            (self.code / "configs" / f"task2_v2_{short}_full_assets_v1.yaml").touch()
            checkpoint = self.out / "model_full_assets_v1/models" / task / "seed42/best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
        self.events_path = self.root / "calls.jsonl"
        fake = self.root / "fake-python"
        fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
source = sys.stdin.read() if args and args[0] == "-" else ""
phase = "config" if "coupled-config-preflight" in source else "gpu" if "coupled-cuda-preflight" in source else "test" if "unittest" in args else "check" if "--check-only" in args else "analysis"
event = {"phase": phase, "args": args, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
with open(os.environ["MOCK_EVENTS"], "a") as f:
    f.write(json.dumps(event) + "\\n")
if phase == os.environ.get("MOCK_FAIL_PHASE"):
    if not os.environ.get("MOCK_FAIL_TASK") or os.environ["MOCK_FAIL_TASK"] in " ".join(args):
        raise SystemExit(7)
if phase == "analysis":
    out = pathlib.Path(args[args.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=False)
    for name in ["clinical_overlap_plan.json", "coupled_conditions.json", "coupled_importance_summary.csv", "coupled_importance_bootstrap_intervals.csv", "coupled_importance_paired_predictions.csv", "coupled_importance_cohort_counts.csv", "coupled_importance_metadata.json"]:
        (out / name).write_text("fixture")
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, CODE_ROOT=str(self.code), OUT_ROOT=str(self.out),
                        PYTHON_BIN=str(fake), MOCK_EVENTS=str(self.events_path))
        for key in ("FULL_ROOT", "IMPORTANCE_ROOT", "COUPLED_ROOT", "TASKS", "MOCK_FAIL_PHASE", "MOCK_FAIL_TASK"):
            self.env.pop(key, None)

    def run_stage(self, stage=None, **env):
        return subprocess.run(["bash", str(RUNNER)] + ([] if stage is None else [stage]),
                              env=dict(self.env, **env), text=True, capture_output=True, timeout=10)

    def events(self):
        return [json.loads(line) for line in self.events_path.read_text().splitlines()] if self.events_path.exists() else []

    def test_default_check_is_cpu_all_three_and_does_not_create_results(self):
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual([e["phase"] for e in events], ["config", "check", "check", "check"])
        self.assertTrue(all(e["cuda_visible"] == "" for e in events))
        self.assertFalse((self.out / "model_full_assets_coupled_importance_v1").exists())
        self.assertTrue(all("--device" not in e["args"] or e["args"][e["args"].index("--device") + 1] == "cpu" for e in events))

    def test_run_preflights_every_task_before_gpu_and_analysis(self):
        result = self.run_stage("run")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual([e["phase"] for e in events], ["config", "check", "check", "check", "gpu", "analysis", "analysis", "analysis"])
        for event in events[5:]:
            args = event["args"]
            self.assertIn("--bf16", args)
            self.assertEqual(args[args.index("--seed") + 1], "42")
            self.assertNotIn("--groups", args)
            self.assertNotIn("--source-token-cache", args)
        self.assertEqual(len(list((self.out / "model_full_assets_coupled_importance_v1/logs").iterdir())), 3)

    def test_failed_late_preflight_never_starts_gpu_or_analysis(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="check", MOCK_FAIL_TASK="cvd_strict")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()], ["config", "check", "check", "check"])
        self.assertFalse((self.out / "model_full_assets_coupled_importance_v1").exists())

    def test_existing_result_and_dangling_symlink_refused_before_python(self):
        target = self.out / "model_full_assets_coupled_importance_v1/cvd_strict_binary/seed42/val_coupled"
        target.parent.mkdir(parents=True)
        target.symlink_to(self.root / "missing-target")
        result = self.run_stage("run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual(self.events(), [])

    def test_existing_cache_passed_without_changing_its_contents(self):
        cache = self.out / "model_full_assets_importance_v1/moca_binary/seed42/source_token_cache_val.pt"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"native-cache")
        result = self.run_stage("run", TASKS="moca_binary")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.events()[-1]["args"]
        self.assertEqual(args[args.index("--source-token-cache") + 1], str(cache))
        self.assertEqual(cache.read_bytes(), b"native-cache")

    def test_gpu_failure_stops_before_result_logs_or_analysis(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="gpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("analysis", [e["phase"] for e in self.events()])
        self.assertFalse((self.out / "model_full_assets_coupled_importance_v1").exists())

    def test_analysis_failure_propagates_through_tee_and_stops_next_task(self):
        result = self.run_stage("run", MOCK_FAIL_PHASE="analysis")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([e["phase"] for e in self.events()].count("analysis"), 1)

    def test_test_stage_only_uses_five_cpu_test_modules_without_data(self):
        result = self.run_stage("test", FULL_ROOT=str(self.root / "nonexistent"))
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual([e["phase"] for e in events], ["test"] * 5)
        self.assertEqual([e["args"][e["args"].index("-p") + 1] for e in events], list(TEST_MODULES))
        self.assertTrue(all(e["cuda_visible"] == "" for e in events))

    def test_task_and_numeric_env_validation(self):
        for kwargs in ({"TASKS": ""}, {"TASKS": "moca_binary moca_binary"}, {"TASKS": "unknown"}, {"COUPLED_BOOTSTRAPS": "0"}, {"COUPLED_WORKERS": "-1"}):
            with self.subTest(kwargs=kwargs):
                result = self.run_stage("check", **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.events(), [])


if __name__ == "__main__":
    unittest.main()
