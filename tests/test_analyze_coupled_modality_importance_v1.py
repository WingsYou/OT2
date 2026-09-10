from __future__ import annotations

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


OT2_DIR = Path(__file__).resolve().parents[1]
if str(OT2_DIR) not in sys.path:
    sys.path.insert(0, str(OT2_DIR))

import analyze_modality_importance_v2 as am
from analyze_coupled_modality_importance_v1 import run_coupled_analysis
from clinical_modality_overlap_v1 import build_overlap_plan
from clinical_overlap_inference_v1 import encode_clinical_override


class _TransparentModel(torch.nn.Module):
    """Known arithmetic lets tests check paired predictions independently."""

    source_order = tuple(am.PRIMARY_MODALITIES)

    def __init__(self):
        super().__init__()
        self.encoder_calls = []
        self.prediction_inputs = []

    def encode_sources(self, batch, *, disabled_groups):
        expected = set(self.source_order) - {"clinical_data"}
        if set(disabled_groups) != expected:
            raise AssertionError("clinical interventions must skip all other encoders")
        if set(batch) != {"clinical_values", "clinical_mask", "has_clinical"}:
            raise AssertionError("clinical encoding must not read labels or native assets")
        values, masks = batch["clinical_values"], batch["clinical_mask"]
        self.encoder_calls.append((values.detach().clone(), masks.detach().clone()))
        clean = values * masks.to(values.dtype)
        clinical = torch.stack([
            clean.sum(dim=1) + 0.125 * masks.sum(dim=1),
            (clean * torch.tensor([1.0, 2.0, 3.0], device=values.device)).sum(dim=1),
        ], dim=1)
        present = masks.any(dim=1) & batch["has_clinical"]
        tokens = torch.zeros(len(values), 9, 2, dtype=values.dtype, device=values.device)
        availability = torch.zeros(len(values), 9, dtype=torch.bool, device=values.device)
        tokens[:, 0] = clinical * present[:, None]
        availability[:, 0] = present
        return {"source_tokens": tokens, "source_present": availability}

    def predict_from_source_tokens(self, tokens, present, *, task):
        if task != "toy_binary":
            raise AssertionError("wrong task")
        self.prediction_inputs.append((tokens.detach().clone(), present.detach().clone()))
        active = tokens * present[:, :, None]
        oct_index = self.source_order.index("retinal_oct")
        return {"logits": active[:, 0, 0] + 0.3 * active[:, 0, 1]
                + 0.7 * active[:, oct_index, 0] + 0.2 * active[:, 1, 0]}


def _fixture():
    values = torch.tensor([
        [-0.7, 0.0, 0.0], [0.0, 0.8, 0.0],
        [0.9, 0.0, 0.0], [0.0, -0.6, 0.0],
        [-0.1, 0.4, 0.5], [0.2, -0.7, 0.6],
        [0.0, 0.0, 0.0], [0.7, 0.8, -0.2],
    ])
    masks = torch.tensor([
        [True, False, False], [False, True, False],
        [True, False, False], [False, True, False],
        [True, True, True], [True, True, True],
        [False, False, False], [True, True, True],
    ])
    native = torch.zeros(8, 9, dtype=torch.bool)
    native[:, 0] = masks.any(dim=1)
    native[[2, 3, 6], 1] = True
    oct_index = tuple(am.PRIMARY_MODALITIES).index("retinal_oct")
    native[:, oct_index] = torch.tensor([True, True, False, False, True, True, True, True])
    model = _TransparentModel()
    encoded = model.encode_sources({
        "clinical_values": values, "clinical_mask": masks,
        "has_clinical": masks.any(dim=1),
    }, disabled_groups=set(model.source_order) - {"clinical_data"})
    tokens = encoded["source_tokens"]
    tokens[:, 1, 0] = native[:, 1].float() * 0.25
    tokens[:, oct_index, 0] = (
        torch.tensor([-0.1, 0.4, 0.2, 0.1, -0.2, 0.3, 0.9, -0.5])
        * native[:, oct_index]
    )
    with torch.no_grad():
        logits = model.predict_from_source_tokens(tokens, native, task="toy_binary")["logits"]
    cache = SimpleNamespace(
        source_tokens=tokens, source_present=native,
        person_ids=tuple(f"p{i}" for i in range(8)),
        y_true=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        full_forward_logits=logits.cpu().numpy().astype(np.float64),
        identity={
            "source_order": list(model.source_order), "encoder_batch_size": 2,
            "amp_mode": "float32", "split": "val", "task": "toy_binary",
        },
    )
    names = ["obsc__rtci_odm1__response_c45877994",
             "meas__viaodplog__c2005200042__u0sunknown", "person__age"]
    schema = {"feature_names": names, "n_features": 3, "task": "toy_binary"}
    catalog = [
        {"feature_name": names[0], "source_key": "rtci_odm1", "table": "observation"},
        {"feature_name": names[1], "source_key": "viaodplog", "table": "measurement", "value_kind": "numeric"},
        {"feature_name": names[2], "source_key": "age", "table": "participants"},
    ]
    plan = build_overlap_plan(schema, catalog)
    model.encoder_calls.clear()
    model.prediction_inputs.clear()
    return model, cache, values, masks, plan


def _manual_logits(cache, values, masks, *, removed_columns=(), remove_oct=False):
    cleaned_values = values.numpy().copy()
    cleaned_masks = masks.numpy().copy()
    if removed_columns:
        cleaned_values[:, list(removed_columns)] = 0.0
        cleaned_masks[:, list(removed_columns)] = False
    clean = cleaned_values * cleaned_masks
    logits = clean.sum(axis=1) + 0.125 * cleaned_masks.sum(axis=1)
    logits += 0.3 * (clean * np.array([1.0, 2.0, 3.0])).sum(axis=1)
    logits += 0.2 * cache.source_tokens[:, 1, 0].numpy()
    if not remove_oct:
        oct_index = tuple(am.PRIMARY_MODALITIES).index("retinal_oct")
        logits += 0.7 * cache.source_tokens[:, oct_index, 0].numpy()
    return logits


class CachedClinicalOverrideTests(unittest.TestCase):
    def setUp(self):
        self.model, self.cache, self.values, self.masks, _ = _fixture()
        self.options = dict(device="cpu", task="toy_binary", threshold=0.6, batch_size=2)

    def test_original_baseline_and_joint_override_match_independent_arithmetic(self):
        original = copy.deepcopy(self.cache)
        baseline = am.evaluate_cached_condition(self.model, self.cache, **self.options)
        np.testing.assert_allclose(
            baseline.logits, _manual_logits(self.cache, self.values, self.masks),
            rtol=1e-6, atol=1e-6,
        )
        override, presence = encode_clinical_override(
            self.model, self.cache, self.values, self.masks, (0, 1),
            device="cpu", batch_size=2,
        )
        self.model.prediction_inputs.clear()
        joint = am.evaluate_cached_condition(
            self.model, self.cache, **self.options,
            disabled_groups=("retinal_oct",), clinical_token_override=override,
            clinical_present_override=presence,
        )
        np.testing.assert_allclose(
            joint.logits,
            _manual_logits(self.cache, self.values, self.masks, removed_columns=(0, 1), remove_oct=True),
            rtol=1e-6, atol=1e-6,
        )
        used_tokens = torch.cat([t for t, _ in self.model.prediction_inputs])
        used_presence = torch.cat([p for _, p in self.model.prediction_inputs])
        self.assertTrue(torch.equal(used_tokens[:, 1:], original.source_tokens[:, 1:]))
        self.assertTrue(torch.equal(used_tokens[:, 0], override))
        self.assertTrue(torch.equal(used_presence[:, 0], presence))
        self.assertFalse(used_presence[:, 4].any())
        self.assertTrue(torch.equal(self.cache.source_tokens, original.source_tokens))
        self.assertTrue(torch.equal(self.cache.source_present, original.source_present))
        np.testing.assert_array_equal(joint.y_true, self.cache.y_true)
        self.assertEqual(joint.person_ids, list(self.cache.person_ids))
        self.assertEqual(joint.threshold, 0.6)

    def test_override_validation_rejects_partial_pair_gain_nonfinite_and_permutation(self):
        clinical = self.cache.source_tokens[:, 0].clone()
        present = self.cache.source_present[:, 0].clone()
        invalid = [
            {"clinical_token_override": clinical},
            {"clinical_present_override": present},
            {"clinical_token_override": clinical[:-1], "clinical_present_override": present},
            {"clinical_token_override": clinical, "clinical_present_override": present.float()},
            {"clinical_token_override": clinical, "clinical_present_override": present,
             "permutation_group": "retinal_oct"},
        ]
        gain = present.clone()
        gain[6] = True
        invalid.append({"clinical_token_override": clinical, "clinical_present_override": gain})
        nonfinite = clinical.clone()
        nonfinite[0, 0] = float("nan")
        invalid.append({"clinical_token_override": nonfinite, "clinical_present_override": present})
        nonzero_absent = clinical.clone()
        nonzero_absent[6, 0] = 0.1
        invalid.append({"clinical_token_override": nonzero_absent, "clinical_present_override": present})
        for arguments in invalid:
            with self.subTest(keys=list(arguments)), self.assertRaises((ValueError, TypeError)):
                am.evaluate_cached_condition(self.model, self.cache, **self.options, **arguments)


class CoupledAnalysisIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.cache, cls.values, cls.masks, cls.plan = _fixture()
        cls.original_cache = copy.deepcopy(cls.cache)
        cls.original_values = cls.values.clone()
        cls.original_masks = cls.masks.clone()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = run_coupled_analysis(
                cls.model, cls.cache, cls.values, cls.masks, cls.plan,
                groups=("retinal_oct", "wearable_blood_glucose"), task="toy_binary",
                threshold=0.6, batch_size=2, device="cpu", use_bf16_amp=False,
                bootstrap_samples=32, seed=42,
            )

    def test_shared_remaining_cohort_uses_broadest_cleanup_for_every_profile(self):
        predictions = self.result["predictions"]
        oct_rows = predictions[predictions.group == "retinal_oct"]
        for _, rows in oct_rows.groupby("profile"):
            self.assertEqual(
                rows.loc[rows.in_native_present_with_remaining_input, "person_id"].tolist(),
                ["p4", "p5", "p6", "p7"],
            )
            self.assertEqual(rows.loc[rows.in_native_present_only, "person_id"].tolist(),
                             ["p0", "p1", "p4", "p5", "p6", "p7"])
        summary = self.result["summary"]
        eligible = summary[(summary.group == "retinal_oct") &
                           (summary.cohort_scope == "native_present_with_remaining_input")]
        self.assertEqual(set(eligible.n_evaluated), {4})
        self.assertEqual(set(eligible.comparison), {
            "full_vs_branch", "full_vs_direct", "full_vs_domain",
            "extra_direct_clinical", "extra_domain_clinical",
        })

    def test_target_absent_participants_still_change_when_clinical_information_removed(self):
        rows = self.result["predictions"]
        p2 = rows[(rows.group == "retinal_oct") & (rows.person_id == "p2")].set_index("profile")
        self.assertFalse(p2.native_target_present.any())
        self.assertAlmostEqual(p2.loc["branch_only", "baseline_probability"],
                               p2.loc["branch_only", "perturbed_probability"], places=12)
        self.assertNotEqual(p2.loc["direct_clinical", "perturbed_probability"],
                            p2.loc["branch_only", "perturbed_probability"])
        self.assertTrue(p2.in_all_participants.all())
        self.assertFalse(p2.in_native_present_only.any())

    def test_incremental_intervals_are_direct_paired_branch_to_domain_comparisons(self):
        rows = self.result["predictions"]
        oct_rows = rows[rows.group == "retinal_oct"]
        branch = oct_rows[oct_rows.profile == "branch_only"].sort_values("person_id")
        domain = oct_rows[oct_rows.profile == "domain_clinical"].sort_values("person_id")
        expected = am.paired_stratified_bootstrap_deltas(
            branch.y.to_numpy(), branch.perturbed_probability.to_numpy(),
            domain.perturbed_probability.to_numpy(), threshold=0.6, n_bootstrap=32,
            seed=am._stable_seed(42, "coupled_bootstrap", "retinal_oct",
                                 "extra_domain_clinical", "all_participants"),
        )
        intervals = self.result["intervals"]
        observed = intervals[(intervals.group == "retinal_oct") &
                             (intervals.comparison == "extra_domain_clinical") &
                             (intervals.cohort_scope == "all_participants")].set_index("metric")
        for record in expected:
            for field in ("point_estimate", "ci_lower", "ci_upper"):
                self.assertAlmostEqual(observed.loc[record["metric"], field], record[field], places=12)

    def test_empty_mappings_reuse_predictions_and_have_exact_zero_increment(self):
        intervals = self.result["intervals"]
        cgm_extra = intervals[(intervals.group == "wearable_blood_glucose") &
                             (intervals.comparison.str.startswith("extra_")) &
                             (intervals.cohort_scope == "all_participants")]
        self.assertEqual(len(cgm_extra), 14)
        self.assertTrue((cgm_extra[["point_estimate", "ci_lower", "ci_upper"]] == 0).all().all())
        conditions = [row for row in self.result["conditions"]
                      if row["group"] == "wearable_blood_glucose"]
        self.assertEqual([row["n_clinical_features_masked"] for row in conditions], [0, 0, 0])
        self.assertEqual([row["predictions_reused_from_profile"] for row in conditions],
                         [None, "branch_only", "branch_only"])

    def test_analysis_preserves_original_cache_bank_people_and_threshold(self):
        self.assertTrue(torch.equal(self.cache.source_tokens, self.original_cache.source_tokens))
        self.assertTrue(torch.equal(self.cache.source_present, self.original_cache.source_present))
        self.assertTrue(torch.equal(self.values, self.original_values))
        self.assertTrue(torch.equal(self.masks, self.original_masks))
        self.assertTrue(self.result["clinical_inference"]["original_native_cache_unchanged"])
        self.assertEqual(self.result["clinical_inference"]["distinct_clinical_interventions_encoded"], 2)
        self.assertEqual(set(self.result["predictions"].threshold), {0.6})


if __name__ == "__main__":
    unittest.main()
