"""CPU integration checks; the synthetic model has hand-computable logits."""
import contextlib
import copy
import io
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from domain_test_fixtures_v1 import catalog_fixture
from clinical_domain_groups_v1 import build_domain_plan, DEFAULT_GROUPS
from domain_ablation_engine_v1 import run_domain_analysis
import analyze_modality_importance_v2 as am


class TransparentModel(torch.nn.Module):
    source_order = tuple(am.PRIMARY_MODALITIES)

    def __init__(self):
        super().__init__()
        self.encoder_calls = []
        self.prediction_inputs = []

    def encode_sources(self, batch, *, disabled_groups):
        if set(disabled_groups) != set(self.source_order) - {"clinical_data"}:
            raise AssertionError("Unexpected nonclinical encoder call")
        if set(batch) != {"clinical_values", "clinical_mask", "has_clinical"}:
            raise AssertionError("Unexpected access to extra fields or labels")
        values, masks = batch["clinical_values"], batch["clinical_mask"]
        self.encoder_calls.append((values.clone(), masks.clone()))
        weights = torch.arange(1, values.shape[1] + 1, dtype=values.dtype) / 100
        clinical = (values * masks * weights).sum(1) + 0.03 * masks.sum(1)
        present = masks.any(1) & batch["has_clinical"]
        tokens = torch.zeros(len(values), 9, 1)
        availability = torch.zeros(len(values), 9, dtype=torch.bool)
        tokens[:, 0, 0] = clinical * present
        availability[:, 0] = present
        return {"source_tokens": tokens, "source_present": availability}

    def predict_from_source_tokens(self, tokens, present, *, task):
        self.prediction_inputs.append((tokens.clone(), present.clone()))
        return {"logits": torch.where(present[:, 0], tokens[:, 0, 0], 0.15)
                + (tokens[:, 1:, 0] * present[:, 1:]).sum(1)}


def fixture():
    schema, rows = catalog_fixture("toy_binary")
    plan = build_domain_plan(schema, rows)
    width = len(rows)
    values = torch.arange(8 * width, dtype=torch.float32).reshape(8, width) / 130 - 0.8
    masks = torch.ones_like(values, dtype=torch.bool)
    masks[0] = False
    masks[0, 23] = True  # Only glucose Clinical; no CGM or remaining inputs.
    masks[1] = False
    values[2, 0] = 0.0  # Real observed zero must remain observed.
    model = TransparentModel()
    native = model.encode_sources({"clinical_values": values, "clinical_mask": masks,
                                   "has_clinical": masks.any(1)}, disabled_groups=set(model.source_order) - {"clinical_data"})
    tokens, present = native["source_tokens"], native["source_present"]
    present[2:, 1:] = True
    present[2, 4:7] = False  # Only one retinal source available for this person.
    tokens[2:, 1:, 0] = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 100
    tokens *= present[:, :, None]
    logits = model.predict_from_source_tokens(tokens, present, task="toy_binary")["logits"]
    cache = SimpleNamespace(source_tokens=tokens, source_present=present,
                            person_ids=tuple(f"test{i}" for i in range(8)), y_true=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                            full_forward_logits=logits.numpy().astype(np.float64),
                            identity={"source_order": list(model.source_order), "encoder_batch_size": 2,
                                      "amp_mode": "float32", "split": "val", "task": "toy_binary"})
    model.encoder_calls.clear()
    model.prediction_inputs.clear()
    return model, cache, values, masks, plan


class DomainEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.cache, cls.values, cls.masks, cls.plan = fixture()
        cls.original = copy.deepcopy(cls.cache)
        cls.original_values, cls.original_masks = cls.values.clone(), cls.masks.clone()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = run_domain_analysis(cls.model, cls.cache, cls.values, cls.masks, cls.plan,
                groups=DEFAULT_GROUPS, task="toy_binary", threshold=0.6, batch_size=2,
                device="cpu", use_bf16_amp=False, bootstrap_samples=16, seed=42)

    def test_every_joint_prediction_matches_independent_arithmetic(self):
        for group in DEFAULT_GROUPS:
            definition = self.plan["groups"][group]
            mask = self.original_masks.numpy().copy()
            mask[:, definition["indices"]] = False
            clinical = (self.original_values.numpy() * mask * (np.arange(1, mask.shape[1] + 1) / 100)).sum(1) + 0.03 * mask.sum(1)
            present = self.original.source_present.numpy().copy()
            for source in definition["disabled_modalities"]:
                present[:, self.model.source_order.index(source)] = False
            expected = np.where(mask.any(1), clinical, 0.15)
            expected += (self.original.source_tokens[:, 1:, 0].numpy() * present[:, 1:]).sum(1)
            rows = self.result["predictions"]
            rows = rows[rows.group == group]
            np.testing.assert_allclose(rows.perturbed_logit, expected, atol=1e-6, rtol=1e-6)
            self.assertEqual(rows.person_id.tolist(), list(self.cache.person_ids))

    def test_masks_values_and_unaffected_inputs_are_exactly_preserved(self):
        calls = self.model.encoder_calls
        offset = 4  # Untouched Clinical equivalence check.
        for group in DEFAULT_GROUPS:
            d = self.plan["groups"][group]
            if not d["indices"] or group == "clinical_data":
                continue
            used_values = torch.cat([v for v, _ in calls[offset:offset + 4]])
            used_masks = torch.cat([m for _, m in calls[offset:offset + 4]])
            expected_v, expected_m = self.original_values.clone(), self.original_masks.clone()
            expected_v[:, d["indices"]] = 0
            expected_m[:, d["indices"]] = False
            self.assertTrue(torch.equal(used_values, expected_v))
            self.assertTrue(torch.equal(used_masks, expected_m))
            self.assertTrue(used_masks[2, 0])
            offset += 4
        self.assertEqual(offset, len(calls))
        self.assertTrue(torch.equal(self.values, self.original_values))
        self.assertTrue(torch.equal(self.masks, self.original_masks))
        self.assertTrue(torch.equal(self.cache.source_tokens, self.original.source_tokens))
        self.assertTrue(torch.equal(self.cache.source_present, self.original.source_present))
        for j, group in enumerate(DEFAULT_GROUPS):
            used = self.model.prediction_inputs[(j + 1) * 4:(j + 2) * 4]
            tokens = torch.cat([x for x, _ in used])
            presence = torch.cat([x for _, x in used])
            disabled = {self.model.source_order.index(s) for s in self.plan["groups"][group]["disabled_modalities"]}
            for source_index in range(1, 9):
                if source_index in disabled:
                    self.assertFalse(presence[:, source_index].any())
                else:
                    self.assertTrue(torch.equal(tokens[:, source_index], self.original.source_tokens[:, source_index]))
                    self.assertTrue(torch.equal(presence[:, source_index], self.original.source_present[:, source_index]))

    def test_clinical_only_affected_people_and_family_availability_are_explicit(self):
        rows = self.result["predictions"]
        cgm = rows[rows.group == "wearable_blood_glucose"].set_index("person_id")
        self.assertFalse(cgm.loc["test0", "native_target_present"])
        self.assertTrue(cgm.loc["test0", "in_any_target_information_present"])
        self.assertFalse(cgm.loc["test0", "has_remaining_input"])
        retina = rows[rows.group == "retinal_all"].set_index("person_id")
        self.assertTrue(retina.loc["test2", "in_any_target_present"])
        self.assertFalse(retina.loc["test2", "in_all_targets_present"])
        self.assertTrue(retina.loc["test3", "in_all_targets_present"])

    def test_intervals_reproduce_direct_paired_bootstrap(self):
        rows = self.result["predictions"]
        rows = rows[rows.group == "wearable_all"]
        expected = am.paired_stratified_bootstrap_deltas(rows.y.to_numpy(), rows.baseline_probability.to_numpy(),
            rows.perturbed_probability.to_numpy(), threshold=0.6, n_bootstrap=16,
            seed=am._stable_seed(42, "domain_v1", "wearable_all", "all_participants"))
        observed = self.result["intervals"]
        observed = observed[(observed.group == "wearable_all") & (observed.cohort_scope == "all_participants")].set_index("metric")
        for row in expected:
            for key in ("point_estimate", "ci_lower", "ci_upper"):
                self.assertAlmostEqual(row[key], observed.loc[row["metric"], key], places=12)

    def test_test_split_is_rejected_before_encoding(self):
        model, cache, values, masks, plan = fixture()
        cache.identity["split"] = "test"
        with self.assertRaises(ValueError):
            run_domain_analysis(model, cache, values, masks, plan, groups=DEFAULT_GROUPS,
                task="toy_binary", threshold=0.6, batch_size=2, device="cpu",
                use_bf16_amp=False, bootstrap_samples=16, seed=42)
        self.assertFalse(model.encoder_calls)


if __name__ == "__main__":
    unittest.main()
