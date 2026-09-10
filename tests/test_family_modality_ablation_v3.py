"""Family ablation regression tests; run in the cluster's OT2 environment.

These tests use tiny CPU fixtures, not participant data or trained checkpoints.
The full suite requires the same PyTorch dependencies as the existing tests.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


OT2_DIR = Path(__file__).resolve().parents[1]
for directory in (OT2_DIR, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_modality_importance_v2 import (  # noqa: E402
    FAMILY_ABLATION_GROUPS,
    _cohort_masks_and_counts,
    build_or_load_source_token_cache,
    evaluate_cached_condition,
    evaluate_frozen_condition,
    main,
    run_importance_analysis,
)
from task2_v2_dataset import PRIMARY_SOURCE_ORDER, SOURCE_FLAG_BY_GROUP  # noqa: E402
from test_analyze_modality_importance_v2 import (  # noqa: E402
    _FakeDataset,
    _asset,
    _cache_identity,
    _cache_loader,
    _cacheable_model,
    _sample,
)


EXPECTED_FAMILIES = {
    "retinal_all": (
        "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
    ),
    "wearable_all": ("wearable_activity_monitor", "wearable_blood_glucose"),
    "retinal_wearable_all": (
        "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
        "wearable_activity_monitor", "wearable_blood_glucose",
    ),
}
EXPECTED_PEOPLE = {
    "all_participants": set(range(8)),
    "any_target_present": {1, 2, 3, 4, 5, 6},
    "any_target_present_with_remaining_input": {1, 2, 3, 6},
    "all_targets_present": {3, 4, 6},
    "all_targets_present_with_remaining_input": {3, 6},
}


def _assign_presence(sample: dict, present: set[str]) -> None:
    for position, group in enumerate(PRIMARY_SOURCE_ORDER):
        sample[SOURCE_FLAG_BY_GROUP[group]] = torch.tensor(group in present)
        sample["source_present"][position] = group in present


def _family_dataset(members: tuple[str, ...], *, heterogeneous: bool) -> _FakeDataset:
    if heterogeneous:
        availability = {
            0: {"clinical_data"},
            1: {members[0], "clinical_data"},
            2: {members[-1], "cardiac_ecg"},
            3: {*members, "clinical_data"},
            4: set(members),
            5: {members[0]},
            6: {*members, "environment"},
            7: {"environment"},
        }
        order = (5, 0, 7, 2, 6, 1, 4, 3)
        labels = (0, 0, 1, 1, 0, 1, 0, 1)
        logits = (-2.0, 2.0, -2.0, 3.0, -1.0, 1.0, -3.0, 4.0)
    else:
        availability = {i: {*members, "clinical_data"} for i in range(4)}
        order = (0, 1, 2, 3)
        labels = (0, 0, 1, 1)
        logits = (-2.0, -1.0, 1.0, 2.0)
    samples = []
    for i in order:
        sample = _sample(i, octa_assets=0)
        sample["split"] = "val"
        sample["y"] = torch.tensor(float(labels[i]))
        sample["clinical_values"][0] = logits[i]
        _assign_presence(sample, availability[i])
        samples.append(sample)
    dataset = _FakeDataset(samples)
    dataset.index["y"] = [int(sample["y"]) for sample in samples]
    return dataset


class _RedundantSignals(torch.nn.Module):
    """Any surviving family member provides the same diagnostic signal."""

    def __init__(self, members: tuple[str, ...]) -> None:
        super().__init__()
        self.members = members
        self.disabled_calls: list[frozenset[str]] = []

    def forward(self, batch, task=None, *, disabled_groups=None):
        disabled = frozenset(disabled_groups or ())
        self.disabled_calls.append(disabled)
        present = batch["source_present"].clone()
        for group in disabled:
            present[:, PRIMARY_SOURCE_ORDER.index(group)] = False
        positions = [PRIMARY_SOURCE_ORDER.index(member) for member in self.members]
        has_signal = present[:, positions].any(dim=1)
        return {"logits": batch["clinical_values"][:, 0] * has_signal}


def _run(model, dataset, groups, **overrides):
    options = dict(
        task="toy_binary", split="val", threshold=0.63,
        methods=("ablation",), groups=groups, permutation_repeats=1,
        bootstrap_samples=7, seed=239, batch_size=4, num_workers=0, device="cpu",
    )
    options.update(overrides)
    return run_importance_analysis(model, dataset, **options)


class FamilyDefinitionsAndCohortsTests(unittest.TestCase):
    def test_family_definitions_match_the_requested_clinical_families(self) -> None:
        self.assertEqual(FAMILY_ABLATION_GROUPS, EXPECTED_FAMILIES)

    def test_any_all_and_remaining_membership_is_outcome_independent(self) -> None:
        for family, members in EXPECTED_FAMILIES.items():
            with self.subTest(family=family):
                dataset = _family_dataset(members, heterogeneous=True)
                presence = np.stack([sample["source_present"].numpy() for sample in dataset.samples])
                labels = dataset.index["y"].to_numpy()
                masks, rows = _cohort_masks_and_counts(presence, labels, family)
                relabeled, _ = _cohort_masks_and_counts(presence, 1 - labels, family)
                by_scope = {row["cohort_scope"]: row for row in rows}
                self.assertEqual(set(masks), set(EXPECTED_PEOPLE))
                for scope, people in EXPECTED_PEOPLE.items():
                    expected_ids = {f"p{i}" for i in people}
                    observed_ids = set(dataset.index.loc[masks[scope], "person_id"])
                    self.assertEqual(observed_ids, expected_ids)
                    np.testing.assert_array_equal(masks[scope], relabeled[scope])
                    row = by_scope[scope]
                    self.assertEqual(row["n_evaluated"], len(people))
                    self.assertEqual(row["n_positive"], int(labels[masks[scope]].sum()))
                    self.assertEqual(row["n_negative"], len(people) - row["n_positive"])
                    self.assertEqual(row["n_native_present"], 6)
                    self.assertEqual(row["n_eligible"], 4)
                    self.assertEqual(row["n_all_targets_present"], 3)
                    self.assertEqual(row["n_all_targets_eligible"], 2)
                    self.assertEqual(row["n_removed_modalities"], len(members))
                    self.assertEqual(json.loads(row["removed_modalities_json"]), list(members))

    def test_empty_and_single_class_family_counts_are_explicit(self) -> None:
        presence = np.zeros((4, len(PRIMARY_SOURCE_ORDER)), dtype=bool)
        presence[:, PRIMARY_SOURCE_ORDER.index("clinical_data")] = True
        labels = np.array([0, 0, 1, 1])
        _, rows = _cohort_masks_and_counts(presence, labels, "wearable_all")
        for row in rows[1:]:
            self.assertEqual(row["status"], "empty_cohort")
            self.assertTrue(np.isnan(row["prevalence"]))
        for member in EXPECTED_FAMILIES["wearable_all"]:
            presence[:2, PRIMARY_SOURCE_ORDER.index(member)] = True
        _, rows = _cohort_masks_and_counts(presence, labels, "wearable_all")
        for row in rows[1:]:
            self.assertEqual(row["status"], "single_class_cohort")
            self.assertEqual(row["n_positive"], 0)
            self.assertEqual(row["n_negative"], 2)


class FamilyEvaluationTests(unittest.TestCase):
    def test_joint_removal_is_direct_inference_and_is_not_the_sum_of_single_losses(self) -> None:
        for family, members in EXPECTED_FAMILIES.items():
            with self.subTest(family=family):
                dataset = _family_dataset(members, heterogeneous=False)
                model = _RedundantSignals(members)
                result = _run(model, dataset, (*members, family))
                self.assertIn(frozenset(members), model.disabled_calls)
                direct = evaluate_frozen_condition(
                    model, _cache_loader(dataset), device="cpu", task="toy_binary",
                    threshold=0.63, expected_split="val", disabled_groups=set(members),
                )
                pairs = result["paired_predictions"]
                family_pairs = pairs[pairs["group"].eq(family)]
                np.testing.assert_allclose(family_pairs["perturbed_logit"], direct.logits)
                np.testing.assert_allclose(family_pairs["perturbed_probability"], direct.probabilities)
                rows = result["summary"].query("metric == 'auroc'").set_index("group")
                single_sum = float(rows.loc[list(members), "mean_degradation_delta"].sum())
                self.assertAlmostEqual(single_sum, 0.0)
                self.assertAlmostEqual(rows.loc[family, "mean_degradation_delta"], 0.5)
                self.assertNotAlmostEqual(rows.loc[family, "mean_degradation_delta"], single_sum)

    def test_family_subgroup_pairs_recompute_baseline_on_the_same_people(self) -> None:
        family = "wearable_all"
        members = EXPECTED_FAMILIES[family]
        dataset = _family_dataset(members, heterogeneous=True)
        result = _run(_RedundantSignals(members), dataset, (family,))
        rows = result["cohort_summary"].set_index(["cohort_scope", "metric"])
        pairs = result["paired_predictions"].set_index("person_id")
        self.assertEqual(pairs.index.tolist(), dataset.index["person_id"].tolist())
        self.assertTrue(pairs["threshold_from_validation"].eq(0.63).all())
        self.assertTrue(result["cohort_summary"]["threshold_from_validation"].eq(0.63).all())
        for scope in ("any_target_present_with_remaining_input", "all_targets_present_with_remaining_input"):
            people = {f"p{i}" for i in EXPECTED_PEOPLE[scope]}
            selected = pairs.loc[pairs.index.isin(people)]
            y = selected["y_true"].to_numpy()
            baseline = selected["baseline_probability"].to_numpy()
            removed = selected["perturbed_probability"].to_numpy()
            pos = baseline[y == 1, None]
            neg = baseline[y == 0][None, :]
            expected_auc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
            self.assertAlmostEqual(rows.loc[(scope, "auroc"), "baseline_metric"], expected_auc)
            self.assertNotAlmostEqual(expected_auc, result["baseline"].metrics["auroc"])
            self.assertAlmostEqual(rows.loc[(scope, "auroc"), "mean_degradation_delta"], expected_auc - 0.5)
            brier = rows.loc[(scope, "brier")]
            self.assertAlmostEqual(brier["baseline_metric"], np.mean((baseline - y) ** 2))
            self.assertAlmostEqual(brier["mean_degradation_delta"], np.mean((removed - y) ** 2) - np.mean((baseline - y) ** 2))
            self.assertAlmostEqual(rows.loc[(scope, "sensitivity"), "mean_perturbed_metric"], 0.0)
            self.assertAlmostEqual(rows.loc[(scope, "specificity"), "mean_perturbed_metric"], 1.0)
        audit_scopes = {
            "any_target_present": "any_target_present",
            "all_targets_present": "all_targets_present",
            "eligible_any_target_present_with_remaining_input": "any_target_present_with_remaining_input",
            "eligible_all_targets_present_with_remaining_input": "all_targets_present_with_remaining_input",
        }
        for column, scope in audit_scopes.items():
            self.assertEqual(set(pairs.index[pairs[column]]), {f"p{i}" for i in EXPECTED_PEOPLE[scope]})
        self.assertEqual(pairs.loc["p4", "n_remaining_sources"], 0)
        self.assertEqual(pairs.loc["p4", "n_native_targets_present"], 2)
        self.assertEqual(pairs.loc["p1", "n_native_targets_present"], 1)
        self.assertTrue(pairs["n_removed_modalities"].eq(2).all())
        self.assertTrue(pairs["removed_modalities_json"].map(lambda value: tuple(json.loads(value)) == members).all())

    def test_family_test_and_permutation_are_rejected_before_model_or_data_access(self) -> None:
        class UntouchedDataset:
            @property
            def index(self):
                raise AssertionError("unsupported family request reached data")

        class UntouchedModel:
            def to(self, *_args, **_kwargs):
                raise AssertionError("unsupported family request reached model")

        for split, methods in (
            ("test", ("ablation",)),
            ("train", ("ablation",)),
            ("val", ("permutation",)),
            ("val", ("matched_permutation",)),
            ("val", ("ablation", "permutation")),
        ):
            with self.subTest(split=split, methods=methods):
                with self.assertRaisesRegex(ValueError, "family.*val|family.*ablation"):
                    _run(UntouchedModel(), UntouchedDataset(), ("clinical_data", "wearable_all"), split=split, methods=methods)

    def test_cli_rejects_family_test_and_permutation_before_loading_config(self) -> None:
        for arguments in (
            ["--split", "test"],
            ["--methods", "permutation"],
            ["--methods", "ablation", "matched_permutation"],
        ):
            with self.subTest(arguments=arguments):
                with patch("analyze_modality_importance_v2.load_model_v2_config", side_effect=AssertionError("invalid family CLI loaded config")):
                    with self.assertRaisesRegex(ValueError, "val|ablation"):
                        main([
                            "--config", "unused-must-not-be-read.yaml",
                            "--ablation-groups", "retinal_all", "wearable_all",
                            *arguments,
                        ])


def _real_family_dataset() -> _FakeDataset:
    """All six native sources carry valid inputs for cache/encoder equivalence."""
    members = EXPECTED_FAMILIES["retinal_wearable_all"]
    dataset = _family_dataset(members, heterogeneous=False)
    for sample in dataset.samples:
        for group in EXPECTED_FAMILIES["retinal_all"]:
            sample[group] = [_asset(sample["person_id"], 0, 1)]
        for prefix in ("activity", "cgm"):
            sample[prefix][:] = sample["clinical_values"][0]
            for suffix in ("feature_mask", "observed_mask", "valid_mask"):
                sample[f"{prefix}_{suffix}"][:] = True
    return dataset


class FamilySourceTokenCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _real_family_dataset()
        cls.model = _cacheable_model()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.cache_path = Path(cls.temporary.name) / "source_token_cache.pt"
        cls.cache = build_or_load_source_token_cache(
            cls.model, _cache_loader(cls.dataset), frame=cls.dataset.index,
            device="cpu", task="toy_binary", expected_split="val",
            identity=_cache_identity(cls.dataset), cache_path=cls.cache_path,
        )

    def test_cached_multi_source_removal_matches_native_forward_without_cache_mutation(self) -> None:
        original_tokens = self.cache.source_tokens.clone()
        original_present = self.cache.source_present.clone()
        target_positions = [PRIMARY_SOURCE_ORDER.index(m) for m in EXPECTED_FAMILIES["retinal_wearable_all"]]
        self.assertTrue(bool(self.cache.source_present[:, target_positions].all()))
        for family, members in EXPECTED_FAMILIES.items():
            with self.subTest(family=family):
                raw = evaluate_frozen_condition(
                    self.model, _cache_loader(self.dataset), device="cpu", task="toy_binary",
                    threshold=0.63, expected_split="val", disabled_groups=set(members),
                )
                cached = evaluate_cached_condition(
                    self.model, self.cache, device="cpu", task="toy_binary", threshold=0.63,
                    batch_size=4, disabled_groups=set(members),
                )
                np.testing.assert_allclose(cached.logits, raw.logits, atol=1e-6, rtol=1e-6)
                self.assertEqual(cached.person_ids, raw.person_ids)
                torch.testing.assert_close(self.cache.source_tokens, original_tokens, atol=0, rtol=0)
                torch.testing.assert_close(self.cache.source_present, original_present, atol=0, rtol=0)

    def test_all_family_reports_reuse_cache_without_raw_data_or_native_encoders(self) -> None:
        class NoReadDataset(_FakeDataset):
            def __getitem__(self, index):
                raise AssertionError("reused family cache read raw inputs")

        no_read = NoReadDataset(self.dataset.samples, self.dataset.index.copy())
        with patch.object(self.model, "encode_sources", side_effect=AssertionError("reused family cache encoded native inputs")):
            result = _run(
                self.model, no_read, tuple(EXPECTED_FAMILIES),
                source_token_cache_path=self.cache_path,
                cache_identity=_cache_identity(self.dataset), require_source_token_cache=True,
            )
        self.assertTrue(result["source_token_cache_audit"]["reused"])
        self.assertEqual(result["source_token_cache_audit"]["raw_dataset_passes_this_run"], 0)
        self.assertEqual(result["source_token_cache_audit"]["native_encoder_passes_this_run"], 0)
        self.assertEqual(len(result["cohort_counts"]), 3 * 5)
        self.assertEqual(len(result["cohort_summary"]), 3 * 5 * 7)
        self.assertEqual(len(result["paired_predictions"]), 3 * len(self.dataset))
        # Every fixture participant has every target and a clinical input, so
        # all five scopes represent exactly the same people. Their point
        # estimates and bootstrap draws must therefore agree bit for bit.
        for table in ("cohort_summary", "cohort_bootstrap_intervals"):
            for family in EXPECTED_FAMILIES:
                family_rows = result[table][result[table]["group"].eq(family)]
                full = family_rows[family_rows["cohort_scope"].eq("all_participants")]
                columns = [column for column in full.columns if column != "cohort_scope"]
                full = full[columns].sort_values("metric").reset_index(drop=True)
                for scope in EXPECTED_PEOPLE:
                    same_people = family_rows[family_rows["cohort_scope"].eq(scope)]
                    same_people = same_people[columns].sort_values("metric").reset_index(drop=True)
                    pd.testing.assert_frame_equal(full, same_people, check_exact=True)
        for family, members in EXPECTED_FAMILIES.items():
            expected = evaluate_cached_condition(
                self.model, self.cache, device="cpu", task="toy_binary", threshold=0.63,
                batch_size=4, disabled_groups=set(members),
            )
            pairs = result["paired_predictions"]
            pairs = pairs[pairs["group"].eq(family)]
            np.testing.assert_allclose(pairs["perturbed_logit"], expected.logits, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
