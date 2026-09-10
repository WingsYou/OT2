#!/usr/bin/env python3
"""Prepare 23-column inputs, retrain each task, and report validation ablation."""
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

from basic_retrain_contract_v1 import (TASKS, sha256, read_json, write_json, verify_prepared, verify_training, code_hashes)
from summarize_basic_retrain_v1 import write_report, load_importance
from verify_basic_retrain_bundle_v1 import verify_bundle

TESTS = ("test_clinical_basic_information_v1.py", "test_basic_retrain_inputs_v1.py",
         "test_basic_retrain_report_v1.py", "test_basic_retrain_runner_v1.py", "test_basic_retrain_runtime_v1.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="check", choices=("check", "run", "continue", "report", "test"))
    args = parser.parse_args(argv)
    code = Path(__file__).resolve().parent
    root = Path(os.environ.get("BASIC_RETRAIN_ROOT", "/data/pzhao3/ot2_output/model_basic_clinical_retrain_v1")).expanduser().resolve()
    tasks = tuple(os.environ.get("TASKS", " ".join(TASKS)).split())
    bootstrap = int(os.environ.get("BASIC_RETRAIN_BOOTSTRAPS", "2000"))
    python = os.environ.get("PYTHON_BIN", sys.executable)
    if bootstrap <= 0 or not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError("Invalid tasks/bootstrap setting")
    print(f"SNAPSHOT_OK: {verify_bundle(code)} files verified", flush=True)
    snapshot = code_hashes()
    configs = {t: code / "configs" / f"task2_v2_{short}_full_assets_v1.yaml" for t, short in zip(TASKS, ("moca", "cvd_broad", "cvd_strict"))}
    state = {"schema_version": "ot2-basic-retrain-run-v1", "state": "running", "phase": "initialization",
             "tasks": list(tasks), "bootstrap_samples": bootstrap, "code_sha256": snapshot,
             "started_utc": utc(), "updated_utc": utc(), "completed_tasks": [], "archived_incomplete_attempts": []}
    if args.stage in ("continue", "report"):
        state = read_json(root / "RUN_STATUS.json")
        if "TASKS" not in os.environ:
            tasks = tuple(state["tasks"])
        if "BASIC_RETRAIN_BOOTSTRAPS" not in os.environ:
            bootstrap = state["bootstrap_samples"]
        if state["schema_version"] != "ot2-basic-retrain-run-v1" or state["code_sha256"] != snapshot or list(tasks) != state["tasks"] or bootstrap != state["bootstrap_samples"]:
            raise ValueError("Continue/report requires the same code, tasks and bootstrap settings")
        if args.stage == "report":
            return 0 if write_report(root, tasks) else 1
        state.update(state="running", error=None, updated_utc=utc())
    log = None
    if args.stage == "run":
        root.mkdir(parents=True, exist_ok=False)
        log = (root / "run.log").open("x", buffering=1)
    elif args.stage == "continue":
        log = (root / "run.log").open("a", buffering=1)

    def emit(text):
        print(text, flush=True)
        if log:
            log.write(text + "\n")

    def update(phase):
        state.update(phase=phase, updated_utc=utc())
        if log:
            write_json(root / "RUN_STATUS.json", state)
            write_report(root, tasks)

    def execute(command, *, cpu=False, input_text=None):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        emit("[command] " + json.dumps(command))
        proc = subprocess.Popen(command, cwd=code, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.PIPE if input_text else subprocess.DEVNULL, bufsize=1)
        try:
            if input_text:
                proc.stdin.write(input_text)
                proc.stdin.close()
            for line in proc.stdout:
                emit(line.rstrip("\n"))
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, command)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()

    def archive_incomplete(directory, task, kind):
        if directory.exists():
            destination = root / "incomplete_attempts" / task / f"{kind}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(destination)
            state["archived_incomplete_attempts"].append(str(destination))
            emit(f"[continue] Preserved incomplete {kind}: {destination}; restarting this step from the beginning.")

    try:
        update("initialization")
        if args.stage != "check":
            update("CPU tests")
            for name in TESTS:
                execute([python, "-m", "unittest", "discover", "-s", str(code / "tests"), "-p", name, "-v"], cpu=True)
            if args.stage == "test":
                return 0
        for task in tasks:
            update(f"source preflight: {task}")
            execute([python, str(code / "prepare_basic_clinical_training_v1.py"), "--config", str(configs[task]),
                     "--root", str(root), "--check-only"], cpu=True)
        if args.stage == "check":
            emit("CHECK_OK: original data and fixed 23-column projection verified; no output or inference created.")
            return 0
        for task in tasks:
            update(f"prepare inputs: {task}")
            execute([python, str(code / "prepare_basic_clinical_training_v1.py"), "--config", str(configs[task]), "--root", str(root)], cpu=True)
            verify_prepared(root / "inputs" / task)
            update(f"training input preflight: {task}")
            execute([python, str(code / "train_basic_clinical_v1.py"), "--inputs", str(root / "inputs" / task), "--check-only"], cpu=True)
        update("GPU preflight")
        execute([python, "-"], input_text='''
# basic-retrain-gpu-preflight
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("CUDA with BF16 support is required for the production training configs")
print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
''')
        for task in tasks:
            inputs, model, importance = root / "inputs" / task, root / "models" / task / "seed42", root / "importance" / task / "seed42/val"
            if (model / "TRAINED.json").exists():
                trained = verify_training(model, sha256(inputs / "PREPARED.json"))
                if trained["code_sha256"] != snapshot:
                    raise ValueError("Completed training used different code")
                emit(f"[skip verified training] {task}")
            else:
                archive_incomplete(model, task, "training")
                update(f"training: {task}")
                execute([python, str(code / "train_basic_clinical_v1.py"), "--inputs", str(inputs)])
                verify_training(model, sha256(inputs / "PREPARED.json"))
            if (importance / "importance_metadata.json").exists():
                load_importance(root, task)
                emit(f"[skip verified importance] {task}")
            else:
                archive_incomplete(importance, task, "importance")
                update(f"validation importance: {task}")
                checkpoint_hash = sha256(model / "best_model.pt")
                cache = root / "caches" / task / f"{checkpoint_hash}_val.pt"
                execute([python, str(code / "analyze_basic_retrained_v1.py"), "--inputs", str(inputs),
                         "--out-dir", str(importance), "--cache", str(cache), "--bootstrap-samples", str(bootstrap)])
                load_importance(root, task)
            if task not in state["completed_tasks"]:
                state["completed_tasks"].append(task)
            update(f"completed: {task}")
        state["state"] = "complete"
        update("complete")
        if not write_report(root, tasks):
            raise RuntimeError("Aggregate report validation failed")
        emit(f"COMPLETE: {root / 'SUMMARY.md'}")
        return 0
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_utc=utc())
        emit("[failed] " + state["error"])
        if log:
            log.write(traceback.format_exc())
        raise
    finally:
        if log:
            write_json(root / "RUN_STATUS.json", state)
            try:
                write_report(root, tasks)
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
