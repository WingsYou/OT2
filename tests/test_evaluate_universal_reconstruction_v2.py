from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pandas as pd


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from ot2.evaluate_universal_reconstruction_v2 import (
    _safe_torch_load,
    audit_availability_states,
    build_deployment_model,
    build_validation_dataset,
    inspect_deployment_checkpoint,
    isolated_validation_master_view,
    normalize_observed_groups,
    normalize_reconstruction_target_policy,
    predict_validation,
    run_raw_validation,
)
from ot2.task2_v2_model import (
    PRIMARY_MODALITY_GROUPS,
    Task2V2Model,
    Task2V2ModelConfig,
)
from ot2.task2_v2_universal_reconstruction import (
    TargetQueryCrossModalReconstructor,
    UniversalReconstructionConfig,
)
from ot2.train_universal_reconstruction_v2 import (
    DEPLOYMENT_FORMAT_VERSION,
    _runtime_code_sha256,
)


def _model_config() -> Task2V2ModelConfig:
    return Task2V2ModelConfig(
        clinical_dim=5,
        environment_dim=3,
        activity_dim=4,
        cgm_dim=2,
        ecg_waveform_dim=3,
        ecg_numeric_dim=2,
        d_model=16,
        temporal_layers=1,
        fusion_layers=1,
        num_heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        environment_patch_size=2,
        activity_patch_size=2,
        cgm_patch_size=2,
        ecg_patch_size=2,
        retinal_backbone_dim=16,
        task_names=("moca_binary",),
    )


def _universal_config() -> UniversalReconstructionConfig:
    return UniversalReconstructionConfig(
        task_name="moca_binary",
        d_model=16,
        num_heads=4,
        decoder_layers=1,
        ffn_multiplier=2,
        dropout=0.0,
    )


def _deployment_payload() -> dict:
    model_config = _model_config()
    universal_config = _universal_config()
    base = Task2V2Model(model_config)
    reconstructor = TargetQueryCrossModalReconstructor(universal_config)
    teacher_sha = "b" * 64
    config_sha = "a" * 64
    provenance = {
        "task": "moca_binary",
        "seed": 42,
        "teacher_checkpoint": {
            "path": "/not/needed/at/deployment/best_model.pt",
            "sha256": teacher_sha,
        },
        "supervised_config": {
            "path": "/portable/config.yaml",
            "sha256": config_sha,
            "config_hash": "config-hash",
        },
        "universal_config": {
            "path": "/portable/universal.yaml",
            "sha256": "c" * 64,
            "config_hash": "universal-hash",
        },
        "token_caches": {
            "train": {"path": "/portable/train.pt", "sha256": "d" * 64},
            "val": {"path": "/portable/val.pt", "sha256": "e" * 64},
        },
        "resolved_model_config": dataclasses.asdict(model_config),
        "teacher_audit": {
            "sha256": teacher_sha,
            "config_hash": "config-hash",
            "full_modality_teacher": True,
            "master_csv_sha256": "f" * 64,
            "scaler_sha256": "1" * 64,
            "clinical_features_csv_sha256": "2" * 64,
        },
        "runtime_code_sha256": _runtime_code_sha256(),
        "optimization_splits": ["train"],
        "selection_splits": ["val"],
        "test_evaluated": False,
    }
    return {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "checkpoint_kind": "universal_reconstruction_student",
        "model_role": "universal_dynamic_subset_reconstruction",
        "task": "moca_binary",
        "seed": 42,
        "base_model_state": base.state_dict(),
        "reconstructor_state": reconstructor.state_dict(),
        "resolved_model_config": dataclasses.asdict(model_config),
        "universal_reconstruction_config": dataclasses.asdict(universal_config),
        "mask_config": {},
        "native_encoder_groups": list(PRIMARY_MODALITY_GROUPS),
        "runtime_observed_groups_required": True,
        "supported_inference_modes": [
            "dynamic_subset_observed_only",
            "dynamic_subset_latent_reconstruction",
        ],
        "teacher_required": False,
        "raw_modality_reconstruction": False,
        "latent_target_space": "full_teacher_native_source_token_pre_fusion",
        "selected_threshold": {
            "threshold": 0.41,
            "method": "youden",
            "objective": 0.37,
            "fallback_reason": None,
        },
        "best_epoch": 3,
        "provenance": provenance,
        "test_evaluated": False,
    }


def _master_metadata_payload(*, n_rows: int) -> dict:
    return {
        "preprocessing_version": "aireadi-ot2-v2",
        "config_hash": "3" * 64,
        "task": "moca_binary",
        "n_rows": int(n_rows),
        "clinical_schema_hash": "4" * 64,
        "source_indexes": {
            "environment": {
                "config_hash": "5" * 64,
                "mapping_hash": "6" * 64,
                "path": "/private/environment.csv",
            },
            "wearable": {
                "config_hash": "7" * 64,
                "mapping_hash": "8" * 64,
                "path": "/private/wearable.csv",
            },
            "ecg": {
                "config_hash": "9" * 64,
                "mapping_hash": "a" * 64,
                "path": "/private/ecg.csv",
            },
            "retinal": {
                "config_hash": "b" * 64,
                "mapping_hash": "c" * 64,
                "model_view_config_hash": "d" * 64,
                "path": "/private/retinal.csv",
            },
        },
        # These fields describe the full train/val/test master and must not be
        # copied into the validation-only projection.
        "label_positive_count": "SECRET_FULL_OUTCOME_COUNT",
        "split_counts": {"test": "SECRET_TEST_COUNT"},
        "coverage_by_split": {"test": "SECRET_TEST_COVERAGE"},
        "input_bindings": {"test": "SECRET_TEST_BINDING"},
        "source_files": [{"path": "/private/test-label-source.csv"}],
    }


class UniversalRawCheckpointTests(unittest.TestCase):
    def test_checkpoint_deserialization_is_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.pt"
            path.write_bytes(b"placeholder")
            with patch(
                "ot2.evaluate_universal_reconstruction_v2.torch.load",
                return_value={"format_version": "placeholder"},
            ) as loader:
                self.assertEqual(
                    _safe_torch_load(path), {"format_version": "placeholder"}
                )
            loader.assert_called_once_with(
                path.resolve(), map_location="cpu", weights_only=True
            )

    def test_checkpoint_is_strictly_inspected_and_loaded(self) -> None:
        payload = _deployment_payload()
        contract = inspect_deployment_checkpoint(
            payload,
            expected_task="moca_binary",
            expected_seed=42,
            expected_config_hash="config-hash",
            expected_config_sha256="a" * 64,
        )
        self.assertEqual(contract.threshold, 0.41)
        model = build_deployment_model(payload, contract)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

        broken = copy.deepcopy(payload)
        key = next(iter(broken["reconstructor_state"]))
        broken["reconstructor_state"][key] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_deployment_model(broken, contract)

    def test_checkpoint_rejects_task_runtime_and_test_provenance_mismatch(self) -> None:
        base = _deployment_payload()
        mutations = (
            (lambda value: value.update(task="cvd_broad_binary"), "task differs"),
            (
                lambda value: value["provenance"].update(test_evaluated=True),
                "test isolation",
            ),
            (
                lambda value: value["provenance"]["runtime_code_sha256"].update(
                    {"task2_v2_model.py": "0" * 64}
                ),
                "runtime code bundle",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                payload = copy.deepcopy(base)
                mutation(payload)
                with self.assertRaisesRegex(ValueError, message):
                    inspect_deployment_checkpoint(
                        payload,
                        expected_task="moca_binary",
                        expected_seed=42,
                        expected_config_hash="config-hash",
                        expected_config_sha256="a" * 64,
                    )


class UniversalRawAcquisitionBoundaryTests(unittest.TestCase):
    def test_validation_master_view_excludes_invalid_test_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.csv"
            metadata = root / "master.metadata.json"
            master.write_text(
                "person_id,task,split,y,artifact\n"
                "v1,moca_binary,val,1,val.pt\n"
                "t1,moca_binary,test,SECRET_TEST_LABEL,test.pt\n"
                "r1,moca_binary,train,0,train.pt\n",
                encoding="utf-8",
            )
            metadata.write_text(
                json.dumps(_master_metadata_payload(n_rows=3)) + "\n",
                encoding="utf-8",
            )
            with isolated_validation_master_view(
                master, metadata, temporary_parent=root
            ) as (derived, derived_metadata, audit):
                frame = pd.read_csv(derived, dtype={"person_id": "string"})
                self.assertEqual(frame["person_id"].tolist(), ["v1"])
                self.assertEqual(frame["y"].tolist(), [1])
                self.assertNotIn("SECRET_TEST_LABEL", derived.read_text())
                self.assertTrue(derived_metadata.is_file())
                self.assertEqual(audit["test_rows_passed_to_dataset"], 0)
                self.assertFalse(audit["non_validation_y_materialized"])
                projected = json.loads(derived_metadata.read_text(encoding="utf-8"))
                self.assertEqual(projected["n_rows"], 1)
                self.assertEqual(projected["task"], "moca_binary")
                self.assertEqual(projected["clinical_schema_hash"], "4" * 64)
                self.assertEqual(
                    projected["source_indexes"]["retinal"][
                        "model_view_config_hash"
                    ],
                    "d" * 64,
                )
                self.assertEqual(
                    set(projected["source_indexes"]["environment"]),
                    {"config_hash", "mapping_hash"},
                )
                projected_text = derived_metadata.read_text(encoding="utf-8")
                for forbidden in (
                    "SECRET_FULL_OUTCOME_COUNT",
                    "SECRET_TEST_COUNT",
                    "SECRET_TEST_COVERAGE",
                    "SECRET_TEST_BINDING",
                    "/private/test-label-source.csv",
                    "/private/environment.csv",
                ):
                    self.assertNotIn(forbidden, projected_text)
                metadata_audit = audit["derived_metadata_policy"]
                self.assertEqual(
                    metadata_audit["outcome_aggregate_fields_copied"], []
                )
                self.assertIn(
                    "label_positive_count",
                    metadata_audit["omitted_source_metadata_fields"],
                )

    def test_validation_master_view_requires_lineage_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.csv"
            metadata = root / "master.metadata.json"
            master.write_text(
                "person_id,task,split,y\n"
                "v1,moca_binary,val,1\n",
                encoding="utf-8",
            )
            metadata.write_text(
                '{"task":"moca_binary","n_rows":1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "clinical_schema_hash"):
                with isolated_validation_master_view(
                    master, metadata, temporary_parent=root
                ):
                    self.fail("lineage-free metadata must not be accepted")

    def test_test_split_is_rejected_before_checkpoint_or_data_access(self) -> None:
        args = argparse.Namespace(
            split="test",
            reconstruction_mode="on",
            observed_groups=["clinical_data"],
            out=Path("never-write.csv"),
            metadata_out=None,
            overwrite=False,
        )
        with (
            patch(
                "ot2.evaluate_universal_reconstruction_v2.load_supervised_config"
            ) as config_loader,
            patch(
                "ot2.evaluate_universal_reconstruction_v2._safe_torch_load"
            ) as checkpoint_loader,
            patch(
                "ot2.evaluate_universal_reconstruction_v2.OT2V2Dataset"
            ) as dataset_constructor,
        ):
            with self.assertRaisesRegex(ValueError, "validation-only"):
                run_raw_validation(args)
        config_loader.assert_not_called()
        checkpoint_loader.assert_not_called()
        dataset_constructor.assert_not_called()

    def test_dataset_receives_only_observed_groups(self) -> None:
        dataset_config = SimpleNamespace(
            deterministic_retinal_selection=True,
            master_csv=Path("master.csv"),
            scaler_json=Path("scaler.json"),
            task="moca_binary",
            artifact_root=Path("artifacts"),
            master_metadata_json=Path("master.metadata.json"),
            environment_schema_json=Path("environment.json"),
            retinal_max_assets={
                "photography": 6,
                "oct": 2,
                "octa": 16,
                "flio": 4,
            },
        )
        run_config = SimpleNamespace(dataset=dataset_config)
        observed = ("cardiac_ecg", "clinical_data", "environment")
        sentinel = object()
        with patch(
            "ot2.evaluate_universal_reconstruction_v2.OT2V2Dataset",
            return_value=sentinel,
        ) as constructor:
            self.assertIs(build_validation_dataset(run_config, observed), sentinel)
        keywords = constructor.call_args.kwargs
        self.assertEqual(keywords["split"], "val")
        self.assertEqual(
            keywords["loaded_source_groups"],
            ("clinical_data", "cardiac_ecg", "environment"),
        )
        self.assertEqual(
            keywords["max_retinal_assets_per_group"]["retinal_octa"], 16
        )

    def test_default_reconstruction_targets_only_declared_disabled_sources(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def eval(self):
                return self

            def __call__(self, batch, **kwargs):
                self.calls.append(kwargs)
                n = int(batch["y"].numel())
                observed = torch.zeros(n, 9, dtype=torch.bool)
                observed[:, :2] = True
                requested = kwargs["requested_reconstruction"].bool()
                reconstructed = (
                    requested & observed.any(dim=1, keepdim=True)
                    if kwargs["inject_reconstruction"]
                    else torch.zeros_like(observed)
                )
                return {
                    "logits": torch.linspace(-0.4, 0.4, n),
                    "observed_source_present": observed,
                    "requested_reconstruction": requested,
                    "reconstructed_source_present": reconstructed,
                    "reconstruction_confidence": torch.full((n, 9), 0.6),
                }

        disabled = tuple(PRIMARY_MODALITY_GROUPS[2:])
        raw_batch = {
            "person_id": ["p1", "p2"],
            "split": ["val", "val"],
            "y": torch.tensor([0.0, 1.0]),
        }
        declared = {
            "p1": (True, True, True, False, True, False, False, True, False),
            "p2": (True, True, False, True, False, False, True, False, True),
        }
        expected_requested = torch.tensor(
            [
                [False, False, True, False, True, False, False, True, False],
                [False, False, False, True, False, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                model = FakeModel()
                result = predict_validation(
                    model,  # type: ignore[arg-type]
                    [raw_batch],
                    task="moca_binary",
                    disabled_groups=disabled,
                    declared_availability=declared,
                    reconstruction_mode=mode,
                    threshold=0.41,
                    device=torch.device("cpu"),
                    use_bf16_amp=False,
                )
                self.assertEqual(result.person_ids, ("p1", "p2"))
                self.assertEqual(model.calls[0]["disabled_groups"], disabled)
                expected = mode == "on"
                self.assertIs(model.calls[0]["reconstruct_disabled"], False)
                self.assertIs(model.calls[0]["inject_reconstruction"], expected)
                self.assertTrue(
                    torch.equal(
                        model.calls[0]["requested_reconstruction"].cpu(),
                        expected_requested,
                    )
                )
                self.assertTrue(
                    (result.requested_reconstruction == expected_requested.numpy()).all()
                )
                expected_reconstructed = (
                    expected_requested.numpy()
                    if expected
                    else torch.zeros_like(expected_requested).numpy()
                )
                self.assertTrue(
                    (
                        result.reconstructed_source_present
                        == expected_reconstructed
                    ).all()
                )
                self.assertIn("precision", result.metrics)
                self.assertIn("recall", result.metrics)
                self.assertIn("f1", result.metrics)
                self.assertEqual(
                    result.metrics_by_reconstruction_eligibility[
                        "requested_and_context_available"
                    ]["n"],
                    2.0,
                )
                self.assertEqual(
                    result.metrics_by_reconstruction_eligibility[
                        "requested_but_no_observed_context"
                    ]["n"],
                    0.0,
                )
                declared_array, availability_audit = audit_availability_states(
                    SimpleNamespace(
                        index=pd.DataFrame(
                            {
                                "person_id": ["p1", "p2"],
                                **{
                                    flag: [declared[person][index] for person in ("p1", "p2")]
                                    for index, flag in enumerate(
                                        (
                                            "has_clinical",
                                            "has_ecg",
                                            "has_environment",
                                            "has_retinal_flio",
                                            "has_retinal_oct",
                                            "has_retinal_octa",
                                            "has_retinal_photography",
                                            "has_activity",
                                            "has_cgm",
                                        )
                                    )
                                },
                            }
                        )
                    ),
                    result,
                    observed_groups=PRIMARY_MODALITY_GROUPS[:2],
                    reconstruction_mode=mode,
                    reconstruction_target_policy="declared-disabled",
                )
                self.assertEqual(declared_array.shape, (2, 9))
                self.assertEqual(
                    availability_audit[
                        "n_requested_tokens_for_naturally_absent_sources"
                    ],
                    0,
                )

    def test_exploratory_policy_explicitly_requests_naturally_absent_sources(self) -> None:
        class FakeModel:
            def eval(self):
                return self

            def __call__(self, batch, **kwargs):
                requested = kwargs["requested_reconstruction"].bool()
                observed = torch.zeros_like(requested)
                observed[:, 0] = True
                reconstructed = requested & observed.any(dim=1, keepdim=True)
                return {
                    "logits": torch.tensor([0.0]),
                    "observed_source_present": observed,
                    "requested_reconstruction": requested,
                    "reconstructed_source_present": reconstructed,
                    "reconstruction_confidence": torch.ones_like(
                        requested, dtype=torch.float32
                    ),
                }

        disabled = tuple(PRIMARY_MODALITY_GROUPS[1:])
        declared = {"p1": (True, True, False, False, False, False, False, False, False)}
        result = predict_validation(
            FakeModel(),  # type: ignore[arg-type]
            [
                {
                    "person_id": ["p1"],
                    "split": ["val"],
                    "y": torch.tensor([1.0]),
                }
            ],
            task="moca_binary",
            disabled_groups=disabled,
            declared_availability=declared,
            reconstruction_mode="on",
            reconstruction_target_policy="all-disabled-exploratory",
            threshold=0.41,
            device=torch.device("cpu"),
            use_bf16_amp=False,
        )
        self.assertFalse(bool(result.requested_reconstruction[0, 0]))
        self.assertTrue(bool(result.requested_reconstruction[0, 1:].all()))
        self.assertEqual(
            int(
                (
                    result.requested_reconstruction[0]
                    & ~torch.tensor(declared["p1"]).numpy()
                ).sum()
            ),
            7,
        )

    def test_reconstruction_target_declarations_fail_closed(self) -> None:
        raw_batch = {
            "person_id": ["p1", "p2"],
            "split": ["val", "val"],
            "y": torch.tensor([0.0, 1.0]),
        }
        common = {
            "task": "moca_binary",
            "disabled_groups": tuple(PRIMARY_MODALITY_GROUPS[1:]),
            "reconstruction_mode": "on",
            "threshold": 0.41,
            "device": torch.device("cpu"),
            "use_bf16_amp": False,
        }
        with self.assertRaisesRegex(ValueError, "absent from declared_availability"):
            predict_validation(
                SimpleNamespace(eval=lambda: None),  # type: ignore[arg-type]
                [raw_batch],
                declared_availability={"p1": (True,) * 9},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "not binary"):
            predict_validation(
                SimpleNamespace(eval=lambda: None),  # type: ignore[arg-type]
                [raw_batch],
                declared_availability={
                    "p1": (True,) * 9,
                    "p2": (True, True, 2, True, True, True, True, True, True),
                },
                **common,
            )

    def test_observed_group_parser_is_canonical_and_fail_closed(self) -> None:
        self.assertEqual(
            normalize_observed_groups(["retinal_oct", "clinical_data"]),
            ("clinical_data", "retinal_oct"),
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            normalize_observed_groups(["clinical_data", "clinical_data"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            normalize_observed_groups(["made_up_source"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalize_observed_groups([])
        self.assertEqual(
            normalize_reconstruction_target_policy("declared-disabled"),
            "declared-disabled",
        )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            normalize_reconstruction_target_policy("all-disabled")


if __name__ == "__main__":
    unittest.main()
