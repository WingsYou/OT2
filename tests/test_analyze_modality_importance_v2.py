from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


OT2_DIR = Path(__file__).resolve().parents[1]
if str(OT2_DIR) not in sys.path:
    sys.path.insert(0, str(OT2_DIR))

from analyze_modality_importance_v2 import (  # noqa: E402
    AGE_BIN_EDGES,
    BundlePermutationDataset,
    IMPORTANCE_FORMAT_VERSION,
    SOURCE_TOKEN_CACHE_FORMAT_VERSION,
    SELECTION_LOCK_VERSION,
    _validate_test_lock,
    build_or_load_source_token_cache,
    build_permutation_plan,
    evaluate_cached_condition,
    evaluate_frozen_condition,
    paired_stratified_bootstrap_deltas,
    run_importance_analysis,
    validate_importance_checkpoint,
)
from task2_v2_dataset import (  # noqa: E402
    PRIMARY_SOURCE_ORDER,
    SOURCE_FLAG_BY_GROUP,
    task2_v2_collate,
)
from task2_v2_engine import ENGINE_FORMAT_VERSION  # noqa: E402
from task2_v2_model import Task2V2Model, Task2V2ModelConfig  # noqa: E402


def _asset(owner: str, asset_index: int, n_views: int) -> dict:
    value = float(10 * int(owner[-1]) + asset_index + 1)
    return {
        "views": torch.full((n_views, 3, 4, 4), value),
        "pixel_mask": torch.ones((n_views, 1, 4, 4), dtype=torch.bool),
        "view_mask": torch.ones(n_views, dtype=torch.bool),
        "metadata": {
            "owner": owner,
            "asset_index": asset_index,
            "nested": {"protocol": f"protocol-{owner}-{asset_index}"},
        },
    }


def _sample(person_index: int, *, octa_assets: int) -> dict:
    """A minimal real-contract sample accepted by ``task2_v2_collate``."""

    person_id = f"p{person_index}"
    present = {group: False for group in PRIMARY_SOURCE_ORDER}
    present["clinical_data"] = True
    present["retinal_octa"] = octa_assets > 0
    sample = {
        "person_id": person_id,
        "task": "toy_binary",
        "split": "test",
        "y": torch.tensor(float(person_index % 2)),
        "label_value": 20.0 + person_index,
        "clinical_values": torch.tensor([person_index + 0.1, person_index + 0.2]),
        "clinical_mask": torch.ones(2, dtype=torch.bool),
        "ecg_waveform": torch.zeros((3, 12)),
        "ecg_value_mask": torch.zeros((3, 12), dtype=torch.bool),
        "ecg_sample_mask": torch.zeros(3, dtype=torch.bool),
        "ecg_lead_mask": torch.zeros(12, dtype=torch.bool),
        "ecg_numeric": torch.zeros(8),
        "ecg_numeric_mask": torch.zeros(8, dtype=torch.bool),
        "environment": torch.zeros((2, 1)),
        "environment_feature_mask": torch.zeros((2, 1), dtype=torch.bool),
        "environment_slot_mask": torch.zeros(2, dtype=torch.bool),
        "environment_feature_names": ["environment_feature"],
        "environment_canonical_feature_names": ["environment_feature"],
        "environment_excluded_feature_names": [],
        "environment_feature_indices": [0],
        "activity": torch.zeros((2, 1)),
        "activity_feature_mask": torch.zeros((2, 1), dtype=torch.bool),
        "activity_observed_mask": torch.zeros((2, 1), dtype=torch.bool),
        "activity_valid_mask": torch.zeros((2, 1), dtype=torch.bool),
        "cgm": torch.zeros((2, 1)),
        "cgm_feature_mask": torch.zeros((2, 1), dtype=torch.bool),
        "cgm_observed_mask": torch.zeros((2, 1), dtype=torch.bool),
        "cgm_valid_mask": torch.zeros((2, 1), dtype=torch.bool),
        "retinal_flio": [],
        "retinal_oct": [],
        "retinal_octa": [
            _asset(person_id, asset_index, asset_index + 1)
            for asset_index in range(octa_assets)
        ],
        "retinal_photography": [],
    }
    sample["source_present"] = torch.tensor(
        [present[group] for group in PRIMARY_SOURCE_ORDER], dtype=torch.bool
    )
    for group, flag in SOURCE_FLAG_BY_GROUP.items():
        sample[flag] = torch.tensor(present[group], dtype=torch.bool)
    return sample


class _FakeDataset(Dataset):
    def __init__(self, samples: list[dict], index: pd.DataFrame | None = None) -> None:
        self.samples = samples
        self.index = index if index is not None else pd.DataFrame(
            {
                "person_id": [sample["person_id"] for sample in samples],
                **{
                    flag: [bool(sample[flag]) for sample in samples]
                    for flag in SOURCE_FLAG_BY_GROUP.values()
                },
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        # OT2V2Dataset constructs a fresh mapping on each access.  Returning a
        # deep copy makes this fake preserve the same non-mutating contract.
        return copy.deepcopy(self.samples[index])


def _cacheable_model() -> Task2V2Model:
    torch.manual_seed(881)
    return Task2V2Model(
        Task2V2ModelConfig(
            clinical_dim=2,
            environment_dim=1,
            activity_dim=1,
            cgm_dim=1,
            ecg_waveform_dim=12,
            ecg_numeric_dim=8,
            d_model=8,
            temporal_layers=1,
            fusion_layers=1,
            num_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
            environment_patch_size=1,
            activity_patch_size=1,
            cgm_patch_size=1,
            ecg_patch_size=1,
            retinal_backbone_dim=8,
            task_names=("toy_binary",),
        )
    ).eval()


def _cache_dataset() -> _FakeDataset:
    samples = [_sample(index, octa_assets=0) for index in range(4)]
    labels = [0, 0, 1, 1]
    for index, (sample, label) in enumerate(zip(samples, labels)):
        sample["split"] = "val"
        sample["y"] = torch.tensor(float(label))
        sample["clinical_values"][0] = float((-2, -1, 1, 2)[index])
    dataset = _FakeDataset(samples)
    dataset.index["y"] = labels
    dataset.index["clinical_site"] = ["A", "A", "B", "B"]
    dataset.index["age"] = [51, 52, 61, 62]
    return dataset


def _cache_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=task2_v2_collate,
    )


def _cache_identity(dataset: _FakeDataset) -> dict:
    digest = hashlib.sha256()
    for person_id in dataset.index["person_id"].astype(str):
        encoded = person_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {
        "checkpoint_sha256": "checkpoint-sha",
        "config_hash": "config-sha",
        "master_config_hash": "master-sha",
        "scaler_config_hash": "scaler-sha",
        "split": "val",
        "source_order": list(PRIMARY_SOURCE_ORDER),
        "model_config_hash": "model-config-sha",
        "task": "toy_binary",
        "encoder_batch_size": 4,
        "amp_mode": "float32",
        "cohort_person_ids_sha256": digest.hexdigest(),
        "n_participants": len(dataset),
    }


def _cohort_dataset() -> _FakeDataset:
    """Heterogeneous availability with known, non-identical subgroup metrics."""

    groups_by_person = {
        0: {"clinical_data"},
        1: {"clinical_data", "cardiac_ecg"},
        2: {"clinical_data", "cardiac_ecg"},
        3: {"clinical_data", "environment"},
        4: {"cardiac_ecg"},
        5: {"cardiac_ecg", "environment"},
        6: {"environment"},
        7: {"clinical_data"},
    }
    labels = (0, 0, 1, 1, 0, 1, 0, 1)
    logits = (-2.0, 2.0, -2.0, 2.0, -1.0, 1.0, -3.0, 3.0)
    samples = []
    # IDs are intentionally out of lexical order to expose accidental sorting
    # of availability independently from labels and predictions.
    for index in (5, 0, 7, 2, 6, 1, 4, 3):
        sample = _sample(index, octa_assets=0)
        sample["split"] = "val"
        sample["y"] = torch.tensor(float(labels[index]))
        # The controlled model below treats this scalar as a prescribed score;
        # it does not purport to model the native clinical encoder.
        sample["clinical_values"][0] = logits[index]
        for position, group in enumerate(PRIMARY_SOURCE_ORDER):
            present = group in groups_by_person[index]
            sample[SOURCE_FLAG_BY_GROUP[group]] = torch.tensor(present)
            sample["source_present"][position] = present
        samples.append(sample)
    dataset = _FakeDataset(samples)
    dataset.index["y"] = [int(sample["y"]) for sample in samples]
    return dataset


class _CohortToyModel(torch.nn.Module):
    def forward(self, batch, task=None, *, disabled_groups=None):
        logits = batch["clinical_values"][:, 0]
        if disabled_groups and "clinical_data" in set(disabled_groups):
            logits = torch.zeros_like(logits)
        return {"logits": logits}


def _audit_value(audit: object, *names: str):
    for name in names:
        if isinstance(audit, dict) and name in audit:
            return audit[name]
        if hasattr(audit, name):
            return getattr(audit, name)
    raise AssertionError(f"audit exposes none of the expected fields {names}")


class BundlePermutationTests(unittest.TestCase):
    def test_retinal_bundle_is_swapped_before_collate_without_identity_or_label(self) -> None:
        original = [_sample(0, octa_assets=1), _sample(1, octa_assets=2)]
        # Give the recipient a different non-target availability signature.
        # A buggy implementation that copies the donor's *entire*
        # source_present vector would erase this bit while swapping OCTA.
        activity_index = PRIMARY_SOURCE_ORDER.index("wearable_activity_monitor")
        original[0]["has_activity"] = torch.tensor(True)
        original[0]["source_present"][activity_index] = True
        original[0]["activity"][:] = 19.0
        original[0]["activity_feature_mask"][:] = True
        original[0]["activity_observed_mask"][:] = True
        original[0]["activity_valid_mask"][:] = True
        base = _FakeDataset(original)
        permuted = BundlePermutationDataset(
            base, group="retinal_octa", donor_indices=np.asarray([1, 0])
        )

        recipient_zero = permuted[0]
        recipient_one = permuted[1]
        octa_index = PRIMARY_SOURCE_ORDER.index("retinal_octa")

        # Person-level outcome/identity and all non-target modalities remain
        # attached to the recipient, not the donor.
        self.assertEqual(recipient_zero["person_id"], "p0")
        self.assertEqual(float(recipient_zero["y"]), 0.0)
        self.assertEqual(recipient_zero["label_value"], 20.0)
        torch.testing.assert_close(
            recipient_zero["clinical_values"], original[0]["clinical_values"]
        )
        self.assertTrue(bool(recipient_zero["has_activity"]))
        self.assertTrue(bool(recipient_zero["source_present"][activity_index]))
        torch.testing.assert_close(recipient_zero["activity"], original[0]["activity"])

        # The whole variable-length OCTA bundle and metadata come from p1.
        # Permutation is intentionally restricted to source-present people, so
        # its presence flag stays true rather than inventing collection.
        self.assertEqual(len(recipient_zero["retinal_octa"]), 2)
        self.assertEqual(
            recipient_zero["retinal_octa"][1]["metadata"]["owner"], "p1"
        )
        self.assertTrue(bool(recipient_zero["has_retinal_octa"]))
        self.assertTrue(bool(recipient_zero["source_present"][octa_index]))
        self.assertEqual(len(recipient_one["retinal_octa"]), 1)
        self.assertEqual(
            recipient_one["retinal_octa"][0]["metadata"]["owner"], "p0"
        )
        self.assertTrue(bool(recipient_one["has_retinal_octa"]))
        self.assertTrue(bool(recipient_one["source_present"][octa_index]))

        # Calling the production collator proves permutation happened at the
        # participant-object level, before variable retinal lists are padded.
        batch = task2_v2_collate([recipient_zero, recipient_one])
        self.assertEqual(tuple(batch["retinal_octa"].shape[:3]), (2, 2, 2))
        self.assertEqual(batch["retinal_octa_metadata"][0][0]["owner"], "p1")
        self.assertEqual(batch["retinal_octa_metadata"][1][0]["owner"], "p0")
        self.assertIsNone(batch["retinal_octa_metadata"][1][1])
        self.assertEqual(batch["person_id"], ["p0", "p1"])
        torch.testing.assert_close(batch["y"], torch.tensor([0.0, 1.0]))

        # Neither the wrapper nor later consumer mutation may alter the base
        # dataset's source sample.
        recipient_zero["retinal_octa"][0]["metadata"]["owner"] = "mutated"
        self.assertEqual(base[1]["retinal_octa"][0]["metadata"]["owner"], "p1")

    def test_bundle_wrapper_rejects_present_absent_crossing(self) -> None:
        base = _FakeDataset([_sample(0, octa_assets=0), _sample(1, octa_assets=1)])
        with self.assertRaisesRegex(ValueError, "absent|present"):
            BundlePermutationDataset(
                base, group="retinal_octa", donor_indices=np.asarray([1, 0])
            )

    def test_unmatched_plan_is_present_only_deranged_and_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(7)],
                "has_environment": [True, True, True, True, True, False, False],
                # Deliberately included: unmatched permutation must not use it.
                "y": [0, 0, 0, 1, 1, 0, 1],
            }
        )
        first = build_permutation_plan(
            frame, group="environment", seed=71, matched=False
        )
        second = build_permutation_plan(
            frame.assign(y=1 - frame["y"]),
            group="environment",
            seed=71,
            matched=False,
        )
        donors = np.asarray(first.donor_indices)

        np.testing.assert_array_equal(donors, second.donor_indices)
        self.assertEqual(sorted(donors[:5].tolist()), list(range(5)))
        self.assertTrue(np.all(donors[:5] != np.arange(5)))
        np.testing.assert_array_equal(donors[5:], np.asarray([5, 6]))
        self.assertEqual(
            int(_audit_value(first.audit, "n_present", "present_count")), 5
        )

    def test_single_present_participant_fails_as_unestimable_not_crossed_with_absent(self) -> None:
        frame = pd.DataFrame(
            {
                "person_id": ["p0", "p1", "p2"],
                "has_cgm": [False, True, False],
                "y": [0, 1, 0],
            }
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_permutation_plan(
                frame, group="wearable_blood_glucose", seed=4, matched=False
            )


class MatchedPermutationTests(unittest.TestCase):
    @staticmethod
    def _frame(labels: list[int]) -> pd.DataFrame:
        # p0/p1 match exactly. p2/p3 require dropping the other-modality
        # signature. p4/p5 require relaxing age but should retain site. p6 is
        # alone even after all allowed relaxation and must remain self-mapped.
        return pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(7)],
                "clinical_site": ["A", "A", "A", "A", "B", "B", "C"],
                "age": [31, 34, 52, 54, 24, 74, 45],
                "has_environment": [True] * 7,
                "has_clinical": [True, True, True, False, True, True, True],
                "has_ecg": [True, True, True, True, True, True, True],
                "has_retinal_flio": [False] * 7,
                "has_retinal_oct": [True] * 7,
                "has_retinal_octa": [True] * 7,
                "has_retinal_photography": [True] * 7,
                "has_activity": [True] * 7,
                "has_cgm": [True] * 7,
                "y": labels,
            }
        )

    def test_matched_plan_is_label_free_deterministic_and_audits_relaxation(self) -> None:
        self.assertEqual(
            tuple(AGE_BIN_EDGES),
            (-np.inf, 40.0, 50.0, 60.0, 70.0, 80.0, np.inf),
        )

        first = build_permutation_plan(
            self._frame([0, 1, 0, 1, 0, 1, 0]),
            group="environment",
            seed=913,
            matched=True,
        )
        relabeled = build_permutation_plan(
            self._frame([1, 1, 1, 0, 0, 0, 1]),
            group="environment",
            seed=913,
            matched=True,
        )
        repeated = build_permutation_plan(
            self._frame([0, 1, 0, 1, 0, 1, 0]),
            group="environment",
            seed=913,
            matched=True,
        )

        np.testing.assert_array_equal(first.donor_indices, relabeled.donor_indices)
        np.testing.assert_array_equal(first.donor_indices, repeated.donor_indices)
        donors = np.asarray(first.donor_indices)
        self.assertTrue(np.all(donors[:6] != np.arange(6)))
        # The sparse participant stays fixed under an audited local singleton
        # tier; all participants that found a valid stratum are deranged.
        self.assertEqual(int(donors[6]), 6)
        frame = self._frame([0, 1, 0, 1, 0, 1, 0])
        for recipient in np.flatnonzero(donors != np.arange(len(donors))):
            self.assertEqual(
                frame.iloc[recipient]["clinical_site"],
                frame.iloc[int(donors[recipient])]["clinical_site"],
            )

        # Audit must expose each matching/relaxation tier rather than making a
        # heavily relaxed permutation look like an exact conditional match.
        tier_counts = _audit_value(
            first.audit,
            "relaxation_counts",
            "tier_counts",
            "match_tier_counts",
            "matching_level_counts",
        )
        self.assertIsInstance(tier_counts, dict)
        self.assertGreaterEqual(len([v for v in tier_counts.values() if int(v) > 0]), 2)
        self.assertEqual(sum(int(value) for value in tier_counts.values()), 7)
        self.assertEqual(
            first.audit["age_bin_edges"],
            ["-inf", 40, 50, 60, 70, 80, "inf"],
        )

        audit_payload = str(first.audit).lower()
        self.assertNotIn("label", audit_payload)
        self.assertNotIn(" y", audit_payload)


class CheckpointValidationTests(unittest.TestCase):
    @staticmethod
    def _checkpoint() -> dict:
        return {
            "engine_format_version": ENGINE_FORMAT_VERSION,
            "checkpoint_kind": "best",
            "task": "moca_binary",
            "config_provenance": {
                "config_hash": "cfg-sha",
                "modality_subset": {
                    "enabled_groups": list(PRIMARY_SOURCE_ORDER),
                    "disabled_groups": [],
                },
            },
            "data_provenance": {
                "task": "moca_binary",
                "lineage": {"master_config_hash": "master-sha"},
                "files": {"master_csv": {"sha256": "master-sha"}},
                "test_role": "final_evaluation_only_after_best_validation_checkpoint_reload",
            },
            "scaler_provenance": {
                "sha256": "scaler-sha",
                "scaler_config_hash": "scaler-sha",
                "fit_split": "recommended_split=train",
            },
            "selected_threshold": {
                "threshold": 0.37,
                "method": "youden",
                "objective": 0.62,
                "fallback_reason": None,
            },
            "validation_metrics_selected_threshold": {"threshold": 0.37},
            "model_state": {"dummy": torch.tensor(1.0)},
        }

    def test_returns_frozen_validation_threshold_with_matching_lineage(self) -> None:
        threshold = validate_importance_checkpoint(
            self._checkpoint(),
            expected_config_hash="cfg-sha",
            expected_task="moca_binary",
            expected_master_hash="master-sha",
            expected_scaler_hash="scaler-sha",
        )
        self.assertAlmostEqual(threshold, 0.37)

    def test_rejects_last_unselected_or_mismatched_checkpoint(self) -> None:
        invalid_cases = []
        last = self._checkpoint()
        last["checkpoint_kind"] = "last"
        invalid_cases.append(last)
        missing_threshold = self._checkpoint()
        del missing_threshold["selected_threshold"]
        invalid_cases.append(missing_threshold)
        invalid_threshold = self._checkpoint()
        invalid_threshold["selected_threshold"]["threshold"] = float("nan")
        invalid_cases.append(invalid_threshold)
        subset_model = self._checkpoint()
        subset_model["config_provenance"]["modality_subset"][
            "disabled_groups"
        ] = ["retinal_flio"]
        invalid_cases.append(subset_model)

        for payload in invalid_cases:
            with self.subTest(payload=payload.get("checkpoint_kind")):
                with self.assertRaises((ValueError, KeyError)):
                    validate_importance_checkpoint(
                        payload,
                        expected_config_hash="cfg-sha",
                        expected_task="moca_binary",
                    )

        expectations = (
            {"expected_config_hash": "different", "expected_task": "moca_binary"},
            {"expected_config_hash": "cfg-sha", "expected_task": "cvd_binary"},
            {
                "expected_config_hash": "cfg-sha",
                "expected_task": "moca_binary",
                "expected_master_hash": "different",
            },
            {
                "expected_config_hash": "cfg-sha",
                "expected_task": "moca_binary",
                "expected_scaler_hash": "different",
            },
        )
        for kwargs in expectations:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, KeyError)):
                    validate_importance_checkpoint(self._checkpoint(), **kwargs)

    def test_confirmatory_test_lock_binds_validation_bytes_and_procedure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "schema_version": IMPORTANCE_FORMAT_VERSION,
                "split": "val",
                "config_hash": "cfg-sha",
                "checkpoint_sha256": "checkpoint-sha",
                "primary_modalities": list(PRIMARY_SOURCE_ORDER),
                "methods": ["ablation", "permutation"],
                "permutation_repeats": 20,
                "bootstrap_samples": 100,
                "seed": 42,
            }
            metadata_path = root / "importance_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            lock = {
                "schema_version": SELECTION_LOCK_VERSION,
                "selection_split": "val",
                "config_hash": "cfg-sha",
                "checkpoint_sha256": "checkpoint-sha",
                "validation_importance_metadata_path": metadata_path.name,
                "validation_importance_metadata_sha256": metadata_hash,
                "locked_groups": ["clinical_data"],
                "locked_methods": ["ablation"],
                "permutation_repeats": 20,
                "bootstrap_samples": 100,
                "seed": 42,
            }
            lock_path = root / "selection_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            observed = _validate_test_lock(
                lock_path,
                config_hash="cfg-sha",
                checkpoint_hash="checkpoint-sha",
            )
            self.assertEqual(observed["locked_groups"], ["clinical_data"])

            changed_procedure = dict(lock)
            changed_procedure["permutation_repeats"] = 21
            lock_path.write_text(json.dumps(changed_procedure), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from validation"):
                _validate_test_lock(
                    lock_path,
                    config_hash="cfg-sha",
                    checkpoint_hash="checkpoint-sha",
                )
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            metadata["split"] = "test"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _validate_test_lock(
                    lock_path,
                    config_hash="cfg-sha",
                    checkpoint_hash="checkpoint-sha",
                )


class PairedBootstrapTests(unittest.TestCase):
    def test_degradation_delta_direction_is_positive_for_benefit_and_loss_metrics(self) -> None:
        # Replicated balanced labels make every stratified resample contain
        # both classes. Baseline is excellent; perturbed is deliberately poor.
        y = np.tile(np.asarray([0, 1], dtype=np.int64), 30)
        baseline = np.tile(np.asarray([0.05, 0.95]), 30)
        perturbed = np.tile(np.asarray([0.70, 0.30]), 30)
        first = paired_stratified_bootstrap_deltas(
            y,
            baseline,
            perturbed,
            threshold=0.5,
            n_bootstrap=200,
            seed=31,
        )
        second = paired_stratified_bootstrap_deltas(
            y,
            baseline,
            perturbed,
            threshold=0.5,
            n_bootstrap=200,
            seed=31,
        )

        self.assertEqual(first, second)
        records = first if isinstance(first, dict) else {
            str(record["metric"]): record for record in first
        }
        for metric in ("auroc", "auprc", "balanced_accuracy", "brier"):
            self.assertIn(metric, records)
            record = records[metric]
            delta = float(record.get("point_estimate"))
            self.assertGreater(delta, 0.0, metric)
            self.assertGreater(float(record["ci_lower"]), 0.0, metric)
            self.assertGreater(float(record["ci_upper"]), 0.0, metric)

        # Specifically catches the common Brier sign bug: importance is
        # perturbed loss minus baseline loss, unlike AUROC/AUPRC.
        brier = records["brier"]
        brier_delta = float(brier.get("point_estimate"))
        expected_brier_delta = float(
            np.mean((perturbed - y) ** 2) - np.mean((baseline - y) ** 2)
        )
        self.assertAlmostEqual(brier_delta, expected_brier_delta, places=12)

    def test_bootstrap_is_paired_and_rejects_misaligned_or_single_class_inputs(self) -> None:
        y = np.asarray([0, 0, 1, 1], dtype=np.int64)
        baseline = np.asarray([0.1, 0.2, 0.8, 0.9])
        perturbed = baseline.copy()
        result = paired_stratified_bootstrap_deltas(
            y,
            baseline,
            perturbed,
            threshold=0.5,
            n_bootstrap=25,
            seed=7,
        )
        records = result if isinstance(result, dict) else {
            str(record["metric"]): record for record in result
        }
        for record in records.values():
            delta = float(record.get("point_estimate"))
            self.assertAlmostEqual(delta, 0.0, places=12)

        with self.assertRaises(ValueError):
            paired_stratified_bootstrap_deltas(
                y,
                baseline[:-1],
                perturbed,
                threshold=0.5,
                n_bootstrap=10,
                seed=7,
            )
        with self.assertRaises(ValueError):
            paired_stratified_bootstrap_deltas(
                np.zeros(4, dtype=np.int64),
                baseline,
                perturbed,
                threshold=0.5,
                n_bootstrap=10,
                seed=7,
            )


class SourceTokenCacheTests(unittest.TestCase):
    def test_cohort_uses_encoded_presence_instead_of_historical_index_flags(self) -> None:
        dataset = _cache_dataset()
        # The original index can declare a source for which the loaded inputs
        # provide no usable token.  Eligibility must use what the model saw.
        dataset.index[SOURCE_FLAG_BY_GROUP["cardiac_ecg"]] = True
        result = run_importance_analysis(
            _cacheable_model(),
            dataset,
            task="toy_binary",
            split="val",
            threshold=0.5,
            methods=("ablation",),
            groups=("cardiac_ecg",),
            permutation_repeats=1,
            bootstrap_samples=5,
            seed=119,
            batch_size=4,
            num_workers=0,
            device="cpu",
            require_source_token_cache=True,
        )
        counts = result["cohort_counts"].set_index("cohort_scope")
        self.assertEqual(counts.loc["native_present_only", "n_evaluated"], 0)
        self.assertEqual(counts.loc["native_present_only", "status"], "empty_cohort")
        self.assertFalse(result["paired_predictions"]["native_source_present"].any())

    def test_cached_baseline_ablation_and_permutation_match_raw_forward(self) -> None:
        dataset = _cache_dataset()
        model = _cacheable_model()
        cache = build_or_load_source_token_cache(
            model,
            _cache_loader(dataset),
            frame=dataset.index,
            device="cpu",
            task="toy_binary",
            expected_split="val",
            identity=_cache_identity(dataset),
        )
        self.assertEqual(
            cache.identity["source_order"], list(PRIMARY_SOURCE_ORDER)
        )
        self.assertEqual(cache.source_tokens.shape, (4, 9, 8))

        raw_baseline = evaluate_frozen_condition(
            model,
            _cache_loader(dataset),
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            expected_split="val",
        )
        cached_baseline = evaluate_cached_condition(
            model,
            cache,
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            batch_size=4,
        )
        np.testing.assert_allclose(
            cached_baseline.logits, raw_baseline.logits, rtol=1e-6, atol=1e-6
        )

        raw_ablation = evaluate_frozen_condition(
            model,
            _cache_loader(dataset),
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            expected_split="val",
            disabled_groups={"clinical_data"},
        )
        cached_ablation = evaluate_cached_condition(
            model,
            cache,
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            batch_size=4,
            disabled_groups={"clinical_data"},
        )
        np.testing.assert_allclose(
            cached_ablation.logits, raw_ablation.logits, rtol=1e-6, atol=1e-6
        )

        donors = np.asarray([1, 0, 3, 2], dtype=np.int64)
        raw_permuted = evaluate_frozen_condition(
            model,
            DataLoader(
                BundlePermutationDataset(dataset, "clinical_data", donors),
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=task2_v2_collate,
            ),
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            expected_split="val",
        )
        cached_permuted = evaluate_cached_condition(
            model,
            cache,
            device="cpu",
            task="toy_binary",
            threshold=0.5,
            batch_size=4,
            donor_indices=donors,
            permutation_group="clinical_data",
        )
        np.testing.assert_allclose(
            cached_permuted.logits, raw_permuted.logits, rtol=1e-6, atol=1e-6
        )

    def test_run_reads_each_sample_and_native_encoder_only_once(self) -> None:
        class CountingDataset(_FakeDataset):
            def __init__(self, base: _FakeDataset) -> None:
                super().__init__(base.samples, base.index.copy())
                self.access_counts = [0] * len(base)

            def __getitem__(self, index: int) -> dict:
                self.access_counts[index] += 1
                return super().__getitem__(index)

        dataset = CountingDataset(_cache_dataset())
        model = _cacheable_model()
        encoder_calls = 0

        def count_encoder(_module, _inputs, _output):
            nonlocal encoder_calls
            encoder_calls += 1

        hook = model.clinical_encoder.register_forward_hook(count_encoder)
        try:
            result = run_importance_analysis(
                model,
                dataset,
                task="toy_binary",
                split="val",
                threshold=0.5,
                methods=("ablation", "permutation", "matched_permutation"),
                groups=("clinical_data",),
                permutation_repeats=3,
                bootstrap_samples=5,
                seed=119,
                batch_size=4,
                num_workers=0,
                device="cpu",
                require_source_token_cache=True,
            )
        finally:
            hook.remove()
        self.assertEqual(dataset.access_counts, [1, 1, 1, 1])
        self.assertEqual(encoder_calls, 1)
        self.assertEqual(
            result["source_token_cache_audit"]["raw_dataset_passes_this_run"], 1
        )
        self.assertEqual(
            result["source_token_cache_audit"]["native_encoder_passes_this_run"],
            1,
        )
        self.assertEqual(len(result["cohort_summary"]), 3 * 3 * 7)
        self.assertEqual(len(result["cohort_counts"]), 3)
        counts = result["cohort_counts"].set_index("cohort_scope")
        self.assertEqual(counts.loc["native_present_only", "n_evaluated"], 4)
        self.assertEqual(
            counts.loc["native_present_with_remaining_input", "n_evaluated"], 0
        )

    def test_persistent_cache_reuses_and_rejects_tampered_hash(self) -> None:
        dataset = _cache_dataset()
        model = _cacheable_model()
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "source_token_cache.pt"
            first = build_or_load_source_token_cache(
                model,
                _cache_loader(dataset),
                frame=dataset.index,
                device="cpu",
                task="toy_binary",
                expected_split="val",
                identity=_cache_identity(dataset),
                cache_path=cache_path,
            )
            self.assertFalse(first.reused)
            self.assertTrue(cache_path.is_file())
            self.assertEqual(
                torch.load(cache_path, weights_only=False)["schema_version"],
                SOURCE_TOKEN_CACHE_FORMAT_VERSION,
            )

            class NoReadDataset(_FakeDataset):
                def __getitem__(self, index: int) -> dict:
                    raise AssertionError("a valid reusable cache must not read Dataset")

            no_read = NoReadDataset(dataset.samples, dataset.index.copy())
            reused = build_or_load_source_token_cache(
                model,
                _cache_loader(no_read),
                frame=no_read.index,
                device="cpu",
                task="toy_binary",
                expected_split="val",
                identity=_cache_identity(dataset),
                cache_path=cache_path,
            )
            self.assertTrue(reused.reused)

            def forbid_encoder(_module, _inputs):
                raise AssertionError("cohort reporting must reuse native source tokens")

            hook = model.clinical_encoder.register_forward_pre_hook(forbid_encoder)
            try:
                result = run_importance_analysis(
                    model,
                    no_read,
                    task="toy_binary",
                    split="val",
                    threshold=0.5,
                    methods=("ablation",),
                    groups=("clinical_data",),
                    permutation_repeats=1,
                    bootstrap_samples=5,
                    seed=119,
                    batch_size=4,
                    num_workers=0,
                    device="cpu",
                    source_token_cache_path=cache_path,
                    cache_identity=_cache_identity(dataset),
                    require_source_token_cache=True,
                )
            finally:
                hook.remove()
            self.assertTrue(result["source_token_cache_audit"]["reused"])
            self.assertEqual(
                result["source_token_cache_audit"]["raw_dataset_passes_this_run"], 0
            )
            self.assertEqual(
                result["source_token_cache_audit"]["native_encoder_passes_this_run"], 0
            )
            self.assertEqual(len(result["cohort_summary"]), 3 * 7)

            payload = torch.load(cache_path, weights_only=False)
            payload["content_sha256"] = "0" * 64
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                build_or_load_source_token_cache(
                    model,
                    _cache_loader(no_read),
                    frame=no_read.index,
                    device="cpu",
                    task="toy_binary",
                    expected_split="val",
                    identity=_cache_identity(dataset),
                    cache_path=cache_path,
                )


class AvailabilityCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _cohort_dataset()
        cls.result = run_importance_analysis(
            _CohortToyModel(),
            cls.dataset,
            task="toy_binary",
            split="val",
            threshold=0.5,
            methods=("ablation",),
            groups=("clinical_data", "cardiac_ecg", "environment", "retinal_flio"),
            permutation_repeats=1,
            bootstrap_samples=11,
            seed=143,
            batch_size=4,
            num_workers=0,
            device="cpu",
        )

    def test_counts_and_prediction_audit_match_actual_person_membership(self) -> None:
        counts = self.result["cohort_counts"].set_index(["group", "cohort_scope"])
        self.assertEqual(len(counts), 4 * 3)
        for table in ("cohort_summary", "cohort_bootstrap_intervals"):
            self.assertTrue(self.result[table]["threshold_from_validation"].eq(0.5).all())
        expected = {
            ("clinical_data", "all_participants"): (8, 4, 4, "ok"),
            ("clinical_data", "native_present_only"): (5, 3, 2, "ok"),
            ("clinical_data", "native_present_with_remaining_input"): (3, 2, 1, "ok"),
            ("cardiac_ecg", "native_present_only"): (4, 2, 2, "ok"),
            ("cardiac_ecg", "native_present_with_remaining_input"): (3, 2, 1, "ok"),
            ("environment", "native_present_only"): (3, 2, 1, "ok"),
            ("environment", "native_present_with_remaining_input"): (
                2, 2, 0, "single_class_cohort"
            ),
            ("retinal_flio", "native_present_only"): (0, 0, 0, "empty_cohort"),
            ("retinal_flio", "native_present_with_remaining_input"): (
                0, 0, 0, "empty_cohort"
            ),
        }
        for key, (n, positives, negatives, status) in expected.items():
            with self.subTest(group=key[0], scope=key[1]):
                row = counts.loc[key]
                self.assertEqual(row["n_total"], 8)
                self.assertEqual(row["n_evaluated"], n)
                self.assertEqual(row["n_positive"], positives)
                self.assertEqual(row["n_negative"], negatives)
                self.assertEqual(row["status"], status)
                if n:
                    self.assertAlmostEqual(row["prevalence"], positives / n)
                else:
                    self.assertTrue(pd.isna(row["prevalence"]))
        clinical = counts.loc["clinical_data"]
        self.assertTrue(clinical["n_native_present"].eq(5).all())
        self.assertTrue(clinical["n_eligible"].eq(3).all())

        predictions = self.result["paired_predictions"]
        predictions = predictions[predictions["group"].eq("clinical_data")]
        self.assertEqual(
            predictions["person_id"].tolist(), self.dataset.index["person_id"].tolist()
        )
        native = predictions.loc[predictions["native_source_present"], "person_id"]
        eligible = predictions.loc[
            predictions["eligible_native_present_with_remaining_input"], "person_id"
        ]
        self.assertEqual(set(native), {"p0", "p1", "p2", "p3", "p7"})
        self.assertEqual(set(eligible), {"p1", "p2", "p3"})
        by_person = predictions.set_index("person_id")
        self.assertEqual(by_person.loc["p0", "n_remaining_sources"], 0)
        self.assertEqual(by_person.loc["p1", "n_remaining_sources"], 1)
        self.assertEqual(by_person.loc["p5", "n_remaining_sources"], 2)

    def test_subgroup_baseline_is_recomputed_on_exact_same_people(self) -> None:
        rows = self.result["cohort_summary"]
        rows = rows[rows["group"].eq("clinical_data")].set_index(["cohort_scope", "metric"])
        probabilities = self.result["paired_predictions"]
        probabilities = probabilities[probabilities["group"].eq("clinical_data")]
        membership = {
            "native_present_only": {"p0", "p1", "p2", "p3", "p7"},
            "native_present_with_remaining_input": {"p1", "p2", "p3"},
        }
        for scope, people in membership.items():
            subset = probabilities[probabilities["person_id"].isin(people)]
            y = subset["y_true"].to_numpy()
            baseline = subset["baseline_probability"].to_numpy()
            perturbed = subset["perturbed_probability"].to_numpy()
            positive = baseline[y == 1, None]
            negative = baseline[y == 0][None, :]
            expected_auc = float(
                ((positive > negative) + 0.5 * (positive == negative)).mean()
            )
            auc_row = rows.loc[(scope, "auroc")]
            self.assertAlmostEqual(auc_row["baseline_metric"], expected_auc)
            self.assertNotAlmostEqual(
                auc_row["baseline_metric"], self.result["baseline"].metrics["auroc"]
            )
            self.assertAlmostEqual(auc_row["mean_perturbed_metric"], 0.5)
            self.assertAlmostEqual(auc_row["mean_degradation_delta"], expected_auc - 0.5)
            brier_row = rows.loc[(scope, "brier")]
            baseline_brier = float(np.mean((baseline - y) ** 2))
            perturbed_brier = float(np.mean((perturbed - y) ** 2))
            self.assertAlmostEqual(brier_row["baseline_metric"], baseline_brier)
            self.assertAlmostEqual(
                brier_row["mean_degradation_delta"], perturbed_brier - baseline_brier
            )

    def test_all_participant_tables_preserve_legacy_metrics_and_bootstrap(self) -> None:
        for legacy_key, cohort_key in (
            ("summary", "cohort_summary"),
            ("bootstrap_intervals", "cohort_bootstrap_intervals"),
        ):
            legacy = self.result[legacy_key]
            cohort = self.result[cohort_key]
            full = cohort[cohort["cohort_scope"].eq("all_participants")]
            sort_columns = ["method", "group", "metric"]
            pd.testing.assert_frame_equal(
                legacy.sort_values(sort_columns).reset_index(drop=True),
                full[legacy.columns].sort_values(sort_columns).reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )

        pairs = self.result["paired_predictions"]
        pairs = pairs[pairs["group"].eq("clinical_data")]
        # Existing experiment seeds must stay stable when additional scopes are
        # requested, preserving reproducibility of earlier whole-cohort results.
        payload = b"143|bootstrap|ablation|clinical_data"
        expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)
        expected = paired_stratified_bootstrap_deltas(
            pairs["y_true"].to_numpy(),
            pairs["baseline_probability"].to_numpy(),
            pairs["perturbed_probability"].to_numpy(),
            threshold=0.5,
            n_bootstrap=11,
            seed=expected_seed,
        )
        actual = self.result["bootstrap_intervals"]
        actual = actual[actual["group"].eq("clinical_data")]
        expected_frame = pd.DataFrame(expected)
        pd.testing.assert_frame_equal(
            actual[expected_frame.columns].reset_index(drop=True),
            expected_frame,
            check_exact=True,
        )

    def test_empty_and_single_class_cohorts_explicitly_withhold_invalid_intervals(self) -> None:
        rows = self.result["cohort_summary"]
        intervals = self.result["cohort_bootstrap_intervals"]
        empty = rows[
            rows["group"].eq("retinal_flio") & rows["cohort_scope"].eq("native_present_only")
        ]
        self.assertEqual(len(empty), 7)
        self.assertTrue(empty["ci_status"].eq("empty_cohort").all())
        self.assertTrue(empty["metric_status"].eq("undefined_for_cohort").all())
        self.assertTrue(empty[["baseline_metric", "mean_perturbed_metric", "ci_lower", "ci_upper"]].isna().all().all())

        scope = "native_present_with_remaining_input"
        single = rows[rows["group"].eq("environment") & rows["cohort_scope"].eq(scope)].set_index("metric")
        self.assertEqual(len(single), 7)
        self.assertTrue(single["ci_status"].eq("single_class_cohort").all())
        self.assertTrue(single[["ci_lower", "ci_upper"]].isna().all().all())
        self.assertEqual(single.loc["auroc", "metric_status"], "undefined_for_cohort")
        self.assertTrue(pd.isna(single.loc["auroc", "baseline_metric"]))
        self.assertEqual(single.loc["brier", "metric_status"], "ok")
        self.assertTrue(np.isfinite(single.loc["brier", "baseline_metric"]))
        self.assertTrue(pd.isna(single.loc["specificity", "baseline_metric"]))
        single_ci = intervals[
            intervals["group"].eq("environment") & intervals["cohort_scope"].eq(scope)
        ]
        self.assertTrue(single_ci["n_bootstrap_finite"].eq(0).all())
        self.assertTrue(single_ci["ci_status"].eq("single_class_cohort").all())

    def test_rejects_availability_rows_not_aligned_with_prediction_ids(self) -> None:
        dataset = _cohort_dataset()
        dataset.index = dataset.index.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)(align|person|order)"):
            run_importance_analysis(
                _CohortToyModel(),
                dataset,
                task="toy_binary",
                split="val",
                threshold=0.5,
                methods=("ablation",),
                groups=("clinical_data",),
                permutation_repeats=1,
                bootstrap_samples=3,
                seed=143,
                batch_size=4,
                num_workers=0,
                device="cpu",
            )

    def test_test_split_does_not_gain_unselected_validation_cohort_outputs(self) -> None:
        dataset = _cohort_dataset()
        for sample in dataset.samples:
            sample["split"] = "test"
        result = run_importance_analysis(
            _CohortToyModel(),
            dataset,
            task="toy_binary",
            split="test",
            threshold=0.5,
            methods=("ablation",),
            groups=("clinical_data",),
            permutation_repeats=1,
            bootstrap_samples=3,
            seed=143,
            batch_size=4,
            num_workers=0,
            device="cpu",
        )
        for key in ("cohort_summary", "cohort_bootstrap_intervals", "cohort_counts"):
            self.assertTrue(result[key].empty)
        self.assertFalse(result["cohort_analysis"]["enabled"])
        self.assertNotIn("native_source_present", result["paired_predictions"].columns)


class EndToEndImportanceTest(unittest.TestCase):
    def test_minimal_frozen_analysis_returns_all_paired_outputs(self) -> None:
        samples = [_sample(index, octa_assets=0) for index in range(4)]
        values = (-2.0, -1.0, 1.0, 2.0)
        labels = (0.0, 0.0, 1.0, 1.0)
        for sample, value, label in zip(samples, values, labels):
            sample["split"] = "val"
            sample["clinical_values"][0] = value
            sample["y"] = torch.tensor(label)
        base = _FakeDataset(samples)
        base.index["y"] = labels
        base.index["clinical_site"] = ["A"] * 4
        base.index["age"] = [55] * 4

        class ToyModel(torch.nn.Module):
            def forward(self, batch, task=None, *, disabled_groups=None):
                if disabled_groups and "clinical_data" in set(disabled_groups):
                    logits = torch.zeros(len(batch["clinical_values"]))
                else:
                    logits = batch["clinical_values"][:, 0]
                return {"logits": logits}

        result = run_importance_analysis(
            ToyModel(),
            base,
            task="toy_binary",
            split="val",
            threshold=0.5,
            methods=("ablation", "permutation", "matched_permutation"),
            groups=("clinical_data",),
            permutation_repeats=2,
            bootstrap_samples=5,
            seed=19,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )
        self.assertEqual(len(result["repeat_metrics"]), 6)  # baseline + 1 + 2 + 2
        self.assertEqual(len(result["summary"]), 3 * 7)
        self.assertEqual(len(result["paired_predictions"]), 5 * 4)
        self.assertEqual(len(result["bootstrap_intervals"]), 3 * 7)
        self.assertEqual(len(result["permutation_audits"]), 4)
        self.assertTrue(
            set(result["permutation_pairs"]["recipient_person_id"])
            <= set(base.index["person_id"])
        )


if __name__ == "__main__":
    unittest.main()
