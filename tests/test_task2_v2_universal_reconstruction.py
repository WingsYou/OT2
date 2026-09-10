from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


OT2_DIR = Path(__file__).resolve().parents[1]
if str(OT2_DIR) not in sys.path:
    sys.path.insert(0, str(OT2_DIR))

from task2_v2_model import (  # noqa: E402
    PRIMARY_MODALITY_GROUPS,
    Task2V2Model,
    Task2V2ModelConfig,
)
from task2_v2_universal_reconstruction import (  # noqa: E402
    ModalityMaskSampler,
    TargetQueryCrossModalReconstructor,
    Task2V2UniversalDistiller,
    Task2V2UniversalReconstructionStudent,
    UniversalMaskConfig,
    UniversalReconstructionConfig,
)
from train_universal_reconstruction_v2 import (  # noqa: E402
    CachedTokenDataset,
    TOKEN_CACHE_FORMAT_VERSION,
    _cohort_hashes,
    _deterministic_deployment_masks,
    _load_universal_config,
    _validate_cache,
    _validate_teacher_validation_screen_payload,
    _validation_selection_record,
    _write_train_val_master_view,
    evaluate_validation_panel,
)


def small_model_config() -> Task2V2ModelConfig:
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
    )


def universal_config(**overrides: object) -> UniversalReconstructionConfig:
    values: dict[str, object] = {
        "task_name": "moca_binary",
        "d_model": 16,
        "num_heads": 4,
        "decoder_layers": 1,
        "ffn_multiplier": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return UniversalReconstructionConfig(**values)


class UniversalMaskTests(unittest.TestCase):
    def test_sampler_drops_only_native_sources_and_always_retains_one(self) -> None:
        sampler = ModalityMaskSampler(
            UniversalMaskConfig(
                no_drop_probability=0.0,
                single_drop_probability=0.0,
                family_drop_probability=0.0,
                random_drop_probability=1.0,
            )
        )
        native = torch.tensor(
            [
                [True] * 9,
                [True, True] + [False] * 7,
                [True] + [False] * 8,
                [False] * 9,
            ]
        )
        first = sampler.sample(
            native, generator=torch.Generator().manual_seed(991)
        )
        second = sampler.sample(
            native, generator=torch.Generator().manual_seed(991)
        )
        self.assertTrue(torch.equal(first["artificially_dropped"], second["artificially_dropped"]))
        self.assertFalse(bool((first["artificially_dropped"] & ~native).any()))
        self.assertTrue(bool(first["observed_present"][native.any(dim=1)].any(dim=1).all()))
        self.assertFalse(bool(first["artificially_dropped"][2:].any()))

    def test_invalid_probability_and_family_configs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            UniversalMaskConfig(no_drop_probability=float("nan"))
        with self.assertRaises(ValueError):
            UniversalMaskConfig(families=(("not_a_modality",),))
        with self.assertRaises(ValueError):
            universal_config(temperature=float("inf"))
        with self.assertRaises(ValueError):
            universal_config(logit_kd_weight=float("nan"))


class UniversalReconstructorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(810)
        self.base = Task2V2Model(small_model_config())
        self.config = universal_config()
        self.reconstructor = TargetQueryCrossModalReconstructor(self.config)
        self.student = Task2V2UniversalReconstructionStudent(
            self.base, self.reconstructor, self.config
        )

    def test_reconstructor_preserves_explicit_state_and_handles_empty_rows(self) -> None:
        tokens = torch.randn(3, 9, 16)
        present = torch.tensor(
            [
                [True] * 9,
                [True] + [False] * 8,
                [False] * 9,
            ]
        )
        requested = torch.zeros_like(present)
        requested[:, 4] = True
        requested[0, 4] = False  # It is still observed for row zero.
        result = self.reconstructor.compose_for_fusion(
            observed_tokens=tokens * present.unsqueeze(-1),
            observed_present=present,
            requested_reconstruction=requested,
            missing_tokens=self.base.fusion.missing_tokens,
        )
        self.assertEqual(result["fusion_source_tokens"].shape, (3, 9, 16))
        self.assertEqual(result["reconstruction_confidence"].shape, (3, 9))
        self.assertTrue(bool(result["reconstructed_source_present"][1, 4]))
        self.assertFalse(bool(result["reconstructed_source_present"][2, 4]))
        self.assertTrue(torch.equal(result["raw_reconstructed_tokens"][2], torch.zeros(9, 16)))

    def test_hidden_native_token_cannot_leak_into_prediction_or_reconstruction(self) -> None:
        self.student.eval()
        tokens = torch.randn(4, 9, 16)
        native = torch.ones(4, 9, dtype=torch.bool)
        observed = native.clone()
        observed[:, 5] = False
        requested = torch.zeros_like(native)
        requested[:, 5] = True
        altered = tokens.clone()
        altered[:, 5] = altered[:, 5] + 1_000_000.0
        with torch.no_grad():
            first = self.student.forward_from_source_tokens(
                tokens,
                native,
                observed_source_mask=observed,
                requested_reconstruction=requested,
                inject_reconstruction=True,
            )
            second = self.student.forward_from_source_tokens(
                altered,
                native,
                observed_source_mask=observed,
                requested_reconstruction=requested,
                inject_reconstruction=True,
            )
        self.assertTrue(torch.equal(first["logits"], second["logits"]))
        self.assertTrue(
            torch.equal(
                first["raw_reconstructed_source_tokens"],
                second["raw_reconstructed_source_tokens"],
            )
        )

    def test_distiller_trains_reconstructed_and_observed_only_paths(self) -> None:
        teacher = copy.deepcopy(self.base)
        student_base = copy.deepcopy(self.base)
        student = Task2V2UniversalReconstructionStudent(
            student_base,
            TargetQueryCrossModalReconstructor(self.config),
            self.config,
        )
        sampler = ModalityMaskSampler(
            UniversalMaskConfig(
                no_drop_probability=0.0,
                single_drop_probability=1.0,
                family_drop_probability=0.0,
                random_drop_probability=0.0,
            )
        )
        distiller = Task2V2UniversalDistiller(teacher, student, sampler)
        tokens = torch.randn(6, 9, 16)
        native = torch.ones(6, 9, dtype=torch.bool)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        with torch.no_grad():
            teacher_output = teacher.predict_from_source_tokens(
                tokens, native, task="moca_binary"
            )
        result = distiller.forward_cached(
            source_tokens=tokens,
            native_present=native,
            teacher_embedding=teacher_output["embedding"],
            teacher_logits=teacher_output["logits"],
            labels=labels,
            generator=torch.Generator().manual_seed(82),
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(
            set(result["component_losses"]),
            {
                "reconstructed_supervised_bce",
                "observed_only_supervised_bce",
                "temperature_scaled_logit_kd",
                "patient_embedding_mse",
                "masked_source_token_mse",
            },
        )
        self.assertEqual(
            int(result["mask_state"]["artificially_dropped"].sum()), 6
        )
        result["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        reconstructor_gradients = [
            parameter.grad
            for parameter in student.reconstructor.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(value is not None for value in reconstructor_gradients))
        self.assertIsNotNone(student.base_model.fusion.patient_token.grad)
        self.assertIsNotNone(
            student.base_model.heads["moca_binary"].network[0].weight.grad
        )


class UniversalInputContractTests(unittest.TestCase):
    def test_yaml_rejects_unknown_or_nonfinite_values(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universal.yaml"
            path.write_text(
                "\n".join(
                    [
                        "format_version: ot2-universal-masked-reconstruction-v1",
                        "reconstructor:",
                        "  unexpected: 1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_universal_config(path, task="moca_binary", d_model=16)

            path.write_text(
                "\n".join(
                    [
                        "format_version: ot2-universal-masked-reconstruction-v1",
                        "mask:",
                        "  families:",
                        "    clinical_data: [clinical_data]",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_universal_config(path, task="moca_binary", d_model=16)
            path.write_text(
                "\n".join(
                    [
                        "format_version: ot2-universal-masked-reconstruction-v1",
                        "training:",
                        "  learning_rate: .nan",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_universal_config(path, task="moca_binary", d_model=16)

    def test_cache_validation_checks_all_shapes_and_values(self) -> None:
        n = 4
        payload = {
            "format_version": TOKEN_CACHE_FORMAT_VERSION,
            "split": "train",
            "task": "moca_binary",
            "person_ids": [f"p{index}" for index in range(n)],
            "source_order": list(PRIMARY_MODALITY_GROUPS),
            "source_tokens": torch.randn(n, 9, 16),
            "source_present": torch.ones(n, 9, dtype=torch.bool),
            "teacher_embedding": torch.randn(n, 16),
            "teacher_logits": torch.randn(n),
            "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "metadata": {
                "teacher_checkpoint": {"sha256": "teacher-hash"},
                "config_hash": "config-hash",
            },
        }
        payload["cohort_hashes"] = _cohort_hashes(
            payload["person_ids"], payload["labels"]
        )
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.pt"
            torch.save(payload, valid_path)
            validated = _validate_cache(
                valid_path,
                split="train",
                task="moca_binary",
                teacher_sha256="teacher-hash",
                config_hash="config-hash",
                expected_d_model=16,
            )
            self.assertEqual(tuple(validated["source_tokens"].shape), (4, 9, 16))

            invalid = copy.deepcopy(payload)
            invalid["teacher_logits"] = torch.randn(n - 1)
            invalid_path = Path(directory) / "invalid.pt"
            torch.save(invalid, invalid_path)
            with self.assertRaises(ValueError):
                _validate_cache(
                    invalid_path,
                    split="train",
                    task="moca_binary",
                    teacher_sha256="teacher-hash",
                    config_hash="config-hash",
                    expected_d_model=16,
                )

            invalid_presence = copy.deepcopy(payload)
            invalid_presence["source_present"] = torch.ones(
                n, 9, dtype=torch.float32
            )
            invalid_presence["source_present"][0, 0] = float("nan")
            invalid_presence["source_present"][0, 1] = 2.0
            invalid_presence_path = Path(directory) / "invalid_presence.pt"
            torch.save(invalid_presence, invalid_presence_path)
            with self.assertRaises(ValueError):
                _validate_cache(
                    invalid_presence_path,
                    split="train",
                    task="moca_binary",
                    teacher_sha256="teacher-hash",
                    config_hash="config-hash",
                    expected_d_model=16,
                )

    def test_teacher_must_be_a_validation_screen_checkpoint(self) -> None:
        valid = {
            "training_config": {"evaluation_phase": "validation_screen"},
            "data_provenance": {
                "test_role": "not_iterated_during_validation_screen"
            },
        }
        audit = _validate_teacher_validation_screen_payload(
            valid, label="teacher"
        )
        self.assertTrue(audit["contract_passed"])

        for mutation in (
            {"training_config": {"evaluation_phase": "final_test"}},
            {"test_evaluated": True},
            {"test_metrics": {"auroc": 0.9}},
            {
                "data_provenance": {
                    "test_role": "final_evaluation_only_after_best_validation_checkpoint_reload"
                }
            },
        ):
            candidate = copy.deepcopy(valid)
            candidate.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _validate_teacher_validation_screen_payload(
                    candidate, label="teacher"
                )

    def test_train_validation_master_view_excludes_test_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "master.csv"
            source_metadata = root / "master.metadata.json"
            destination = root / "view.csv"
            source.write_text(
                "person_id,task,split,y\n"
                "p1,moca_binary,train,0\n"
                "p2,moca_binary,val,1\n"
                "p3,moca_binary,test,SECRET_TEST_OUTCOME\n",
                encoding="utf-8",
            )
            source_metadata.write_text(
                json.dumps(
                    {
                        "task": "moca_binary",
                        "n_rows": 3,
                        "label_positive_count": "SECRET_TEST_AGGREGATE",
                        "clinical_schema_hash": "c" * 64,
                        "source_indexes": {
                            "environment": {
                                "config_hash": "d" * 64,
                                "mapping_hash": "e" * 64,
                            },
                            "wearable": {
                                "config_hash": "f" * 64,
                            },
                            "ecg": {
                                "config_hash": "1" * 64,
                            },
                            "retinal": {
                                "config_hash": "a" * 64,
                                "model_view_config_hash": "b" * 64,
                                "test_label_positive_count": "SECRET_NESTED",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = _write_train_val_master_view(
                source, destination, source_metadata
            )
            content = destination.read_text(encoding="utf-8")
            metadata_content = destination.with_suffix(
                ".metadata.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn("p3", content)
            self.assertNotIn("SECRET_TEST_OUTCOME", content)
            self.assertNotIn("SECRET_TEST_AGGREGATE", metadata_content)
            self.assertNotIn("SECRET_NESTED", metadata_content)
            self.assertEqual(record["row_counts"]["train"], 1)
            self.assertEqual(record["row_counts"]["val"], 1)
            self.assertEqual(record["row_counts"]["test_excluded"], 1)
            self.assertFalse(record["test_rows_present_in_view"])
            derived_metadata = json.loads(metadata_content)
            self.assertEqual(derived_metadata["n_rows"], 2)
            self.assertEqual(
                derived_metadata["source_indexes"]["retinal"]["config_hash"],
                "a" * 64,
            )
            self.assertEqual(
                record["metadata_contract"]["retinal_model_view_config_hash"],
                "b" * 64,
            )
            self.assertEqual(
                set(derived_metadata["source_indexes"]["retinal"]),
                {"config_hash", "model_view_config_hash"},
            )


class UniversalValidationPanelTests(unittest.TestCase):
    def test_configured_no_drop_people_are_excluded_from_mixed_protocol(self) -> None:
        native = torch.ones(256, 9, dtype=torch.bool)
        people = [f"person-{index}" for index in range(len(native))]
        sampler = ModalityMaskSampler(
            UniversalMaskConfig(
                no_drop_probability=0.5,
                single_drop_probability=0.5,
                family_drop_probability=0.0,
                random_drop_probability=0.0,
            )
        )
        masks, audit = _deterministic_deployment_masks(
            native, people, sampler=sampler, global_seed=20260906
        )
        eligible = masks["requested_reconstruction"].any(dim=1) & masks[
            "observed_present"
        ].any(dim=1)
        modes = [row["mask_mode"] for row in audit]
        no_drop = torch.tensor(
            [mode == "no_drop" for mode in modes], dtype=torch.bool
        )
        self.assertTrue(bool(no_drop.any()))
        self.assertTrue(bool((~no_drop).any()))
        self.assertFalse(bool(eligible[no_drop].any()))
        self.assertTrue(bool(eligible[~no_drop].all()))
        self.assertTrue(
            all(
                bool(row["condition_eligible"]) == bool(eligible[index])
                for index, row in enumerate(audit)
            )
        )

    def test_selection_uses_hard_floor_and_lexicographic_rank(self) -> None:
        common = {
            "teacher_full_auroc": 0.80,
            "reconstructed_aurocs": [0.70, 0.72],
            "reconstruction_gains": [0.20, 0.10],
            "fidelity_cosines": [0.80, 0.90],
            "selection_config": {"full_teacher_auroc_tolerance": 0.02},
        }
        below = _validation_selection_record(
            student_full_auroc=0.779, **common
        )
        passing = _validation_selection_record(
            student_full_auroc=0.780, **common
        )
        self.assertFalse(below["checkpoint_eligible"])
        self.assertTrue(passing["checkpoint_eligible"])
        self.assertAlmostEqual(passing["score"], 0.71)
        for observed, expected in zip(
            passing["rank"], [0.71, 0.85, 0.15, 0.78]
        ):
            self.assertAlmostEqual(observed, expected)
        # A larger gain cannot outrank a higher absolute reconstructed AUROC;
        # it is considered only after the first two rank coordinates tie.
        higher_absolute = _validation_selection_record(
            teacher_full_auroc=0.80,
            student_full_auroc=0.79,
            reconstructed_aurocs=[0.73, 0.73],
            reconstruction_gains=[0.0, 0.0],
            fidelity_cosines=[0.80, 0.90],
            selection_config={"full_teacher_auroc_tolerance": 0.02},
        )
        self.assertGreater(tuple(higher_absolute["rank"]), tuple(passing["rank"]))

    def test_person_keyed_masks_are_order_invariant(self) -> None:
        native = torch.tensor(
            [
                [True] * 9,
                [True, True, True] + [False] * 6,
                [True, False, True, False, True, False, True, False, True],
            ]
        )
        people = ["p-a", "p-b", "p-c"]
        sampler = ModalityMaskSampler(
            UniversalMaskConfig(
                no_drop_probability=0.0,
                single_drop_probability=0.0,
                family_drop_probability=0.0,
                random_drop_probability=1.0,
            )
        )
        first, first_audit = _deterministic_deployment_masks(
            native, people, sampler=sampler, global_seed=20260906
        )
        order = torch.tensor([2, 0, 1])
        reordered_people = [people[int(index)] for index in order]
        second, second_audit = _deterministic_deployment_masks(
            native.index_select(0, order),
            reordered_people,
            sampler=sampler,
            global_seed=20260906,
        )
        inverse = torch.argsort(order)
        self.assertTrue(
            torch.equal(
                first["artificially_dropped"],
                second["artificially_dropped"].index_select(0, inverse),
            )
        )
        self.assertEqual(
            {row["person_id"]: row for row in first_audit},
            {row["person_id"]: row for row in second_audit},
        )

    def test_panel_uses_masked_only_mix_and_fixed_protocol_set(self) -> None:
        torch.manual_seed(1204)
        base = Task2V2Model(small_model_config())
        config = universal_config()
        student = Task2V2UniversalReconstructionStudent(
            base,
            TargetQueryCrossModalReconstructor(config),
            config,
        )
        student.eval()
        n = 18
        tokens = torch.randn(n, 9, 16)
        present = torch.ones(n, 9, dtype=torch.bool)
        with torch.no_grad():
            teacher = base.predict_from_source_tokens(
                tokens, present, task="moca_binary"
            )
        dataset = CachedTokenDataset(
            {
                "person_ids": [f"person-{index}" for index in range(n)],
                "source_tokens": tokens,
                "source_present": present,
                "teacher_embedding": teacher["embedding"],
                "teacher_logits": teacher["logits"],
                "labels": torch.tensor([index % 2 for index in range(n)]),
            }
        )
        sampler = ModalityMaskSampler(
            UniversalMaskConfig(
                no_drop_probability=0.0,
                single_drop_probability=1.0,
                family_drop_probability=0.0,
                random_drop_probability=0.0,
            )
        )
        kwargs = {
            "device": torch.device("cpu"),
            "use_bf16_amp": False,
            "threshold": 0.5,
            "validation_families": {
                "retinal": (
                    "retinal_flio",
                    "retinal_oct",
                    "retinal_octa",
                    "retinal_photography",
                )
            },
            "deployment_mask_sampler": sampler,
            "deployment_mask_seed": 20260906,
        }
        first = evaluate_validation_panel(
            student, dataset, batch_size=4, **kwargs
        )
        second = evaluate_validation_panel(
            student, dataset, batch_size=7, **kwargs
        )
        self.assertEqual(len(first["metrics"]), 24)
        self.assertEqual(
            first["selection_details"]["n_masked_protocol_pairs"], 11
        )
        self.assertTrue(
            bool(first["eligibility"]["deployment_mix_reconstructed"].all())
        )
        self.assertEqual(
            first["deployment_mask_audit_hash"],
            second["deployment_mask_audit_hash"],
        )
        self.assertTrue(
            np.allclose(
                first["logits"]["deployment_mix_reconstructed"],
                second["logits"]["deployment_mix_reconstructed"],
                atol=1e-6,
                rtol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
