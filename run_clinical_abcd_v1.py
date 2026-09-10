#!/usr/bin/env python3
"""Cluster runner with one aggregate report, one log, and explicit failure status."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from summarize_clinical_abcd_v1 import TASKS, atomic_write, write_report
from verify_clinical_abcd_bundle_v1 import verify_bundle

TEST_MODULES = (
    "test_clinical_basic_information_v1.py", "test_clinical_abcd_report_v1.py",
    "test_clinical_abcd_runner_v1.py", "test_clinical_abcd_engine_v1.py",
    "test_clinical_overlap_inference_v1.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "test", "run", "report"), nargs="?", default="check")
    args = parser.parse_args(argv)
    code = Path(os.environ.get("CODE_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
    output = Path(os.environ.get("OUT_ROOT", "/data/pzhao3/ot2_output")).expanduser()
    full = Path(os.environ.get("FULL_ROOT", output / "model_full_assets_v1")).expanduser()
    split = os.environ.get("ABCD_SPLIT", "val")
    if split not in ("val", "test"):
        raise ValueError("ABCD_SPLIT must be val or test")
    suffix = "_test" if split == "test" else ""
    root = Path(os.environ.get("ABCD_ROOT", output / f"model_full_assets_clinical_abcd_v1{suffix}")).expanduser()
    cache_root = Path(os.environ.get("ABCD_CACHE_ROOT", output / f"model_full_assets_domain_ablation_v1{suffix}")).expanduser()
    reuse = os.environ.get("ABCD_REUSE_CACHE", "1")
    if reuse not in ("0", "1"):
        raise ValueError("ABCD_REUSE_CACHE must be 0 or 1")
    python = os.environ.get("PYTHON_BIN", sys.executable)
    tasks = tuple(os.environ.get("TASKS", " ".join(TASKS)).split())
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError("TASKS must contain unique supported task names")
    batch, workers, bootstrap = (int(os.environ.get(key, default)) for key, default in
                                  (("ABCD_BATCH_SIZE", "2"), ("ABCD_WORKERS", "2"), ("ABCD_BOOTSTRAPS", "2000")))
    if batch <= 0 or workers < 0 or bootstrap <= 0:
        raise ValueError("Invalid batch size, worker count, or bootstrap count")
    print(f"SNAPSHOT_OK: {verify_bundle(code)} files verified", flush=True)
    if args.stage == "report":
        status_path = root / "RUN_STATUS.json"
        if "TASKS" not in os.environ and status_path.exists():
            tasks = tuple(json.loads(status_path.read_text())["tasks"])
        complete = write_report(root, tasks, split=split)
        print(f"REPORT: {root / 'SUMMARY.md'}", flush=True)
        return 0 if complete else 1
    # A run root is reserved atomically; no existing results are overwritten.
    log = None
    state = {"schema_version": "ot2-clinical-abcd-run-v1", "tasks": tasks, "split": split,
             "state": "running", "phase": "initialization", "started_utc": utc(),
             "updated_utc": utc(), "completed_tasks": [], "code_root": str(code),
             "full_root": str(full), "bootstrap_samples": bootstrap, "cache_root": str(cache_root), "reuse_cache": reuse == "1"}
    if args.stage == "run":
        root.mkdir(parents=True, exist_ok=False)
        log = (root / "run.log").open("x", encoding="utf-8", buffering=1)

    def emit(message):
        print(message, flush=True)
        if log:
            log.write(message + "\n")

    def update(phase):
        state.update(phase=phase, updated_utc=utc())
        if log:
            atomic_write(root / "RUN_STATUS.json", json.dumps(state, indent=2) + "\n")
            write_report(root, tasks, split=split)

    def execute(arguments, *, cpu=False, input_text=None):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        emit("[command] " + json.dumps(arguments))
        process = subprocess.Popen(arguments, cwd=code, env=env, text=True,
                                   stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        try:
            if input_text is not None:
                process.stdin.write(input_text)
                process.stdin.close()
            for line in process.stdout:
                emit(line.rstrip("\n"))
            rc = process.wait()
            if rc:
                raise subprocess.CalledProcessError(rc, arguments)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()

    try:
        update("initialization")
        if args.stage in ("test", "run"):
            update("CPU tests")
            for name in TEST_MODULES:
                execute([python, "-m", "unittest", "discover", "-s", str(code / "tests"), "-p", name, "-v"], cpu=True)
            if args.stage == "test":
                return 0
        configs = {t: code / "configs" / f"task2_v2_{s}_full_assets_v1.yaml" for t, s in
                   zip(TASKS, ("moca", "cvd_broad", "cvd_strict"))}
        for task in tasks:
            checkpoint = full / "models" / task / "seed42/best_model.pt"
            for path in (configs[task], checkpoint):
                if not path.is_file():
                    raise FileNotFoundError(path)
        update("config preflight")
        execute([python, "-", *[x for t in tasks for x in (t, str(configs[t]))]], cpu=True, input_text='''
# abcd-config-preflight
import sys
from pathlib import Path
from train_task2_v2 import load_supervised_config
for task, path in zip(sys.argv[1::2], sys.argv[2::2]):
    _, config = load_supervised_config(Path(path))
    if config.dataset.task != task or config.seed != 42:
        raise ValueError(f"Expected {task}, seed 42 in {path}")
''')

        def command(task, check=False):
            result = [python, str(code / "analyze_clinical_abcd_v1.py"),
                      "--config", str(configs[task]), "--checkpoint", str(full / "models" / task / "seed42/best_model.pt"),
                      "--out-dir", str(root / task / "seed42" / f"{split}_abcd"), "--split", split,
                      "--seed", "42", "--batch-size", str(batch), "--num-workers", str(workers),
                      "--bootstrap-samples", str(bootstrap), "--device", "cpu" if check else "cuda"]
            cache = cache_root / task / "seed42" / f"{split}_domain" / f"source_token_cache_{split}.pt"
            if reuse == "1" and cache.is_file():
                result += ["--source-token-cache", str(cache)]
            return result + (["--check-only"] if check else ["--bf16"])

        for task in tasks:
            update(f"lineage/mapping preflight: {task}")
            execute(command(task, check=True), cpu=True)
        if args.stage == "check":
            emit("CHECK_OK: all selected tasks passed; no inference or output files created.")
            return 0
        update("GPU preflight")
        execute([python, "-"], input_text='''
# abcd-cuda-preflight
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("CUDA with bfloat16 support is required")
print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), "bf16: True")
''')
        for task in tasks:
            update(f"inference: {task}")
            execute(command(task))
            # Validate all artifacts before declaring this task complete.
            from summarize_clinical_abcd_v1 import load_task
            load_task(root / task / "seed42" / f"{split}_abcd", task, expected_split=split)
            state["completed_tasks"].append(task)
            emit(f"[complete] {task}")
            update(f"completed: {task}")
        state["state"] = "complete"
        update("complete")
        if not write_report(root, tasks, split=split):
            raise RuntimeError("Aggregate report verification failed")
        emit(f"COMPLETE: send this file for review: {root / 'SUMMARY.md'}")
        return 0
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=utc())
        emit("[failed] " + state["error"])
        if log:
            log.write(traceback.format_exc())
        raise
    finally:
        if log:
            atomic_write(root / "RUN_STATUS.json", json.dumps(state, indent=2) + "\n")
            try:
                write_report(root, tasks, split=split)
            finally:
                log.close()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
