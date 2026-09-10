"""CPU integration checks with independently calculable synthetic predictions."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_basic_clinical_v1 import run_basic_clinical_analysis, _assert_local_runtime_modules
import analyze_modality_importance_v2 as am
from clinical_basic_information_v1 import build_basic_clinical_plan
from test_clinical_basic_information_v1 import catalog_fixture


class TransparentModel(torch.nn.Module):
    source_order = tuple(am.PRIMARY_MODALITIES)

    def __init__(self):
        super().__init__()
        self.encoder_calls, self.prediction_inputs = [], []

    def encode_sources(self, batch, *, disabled_groups):
        if set(disabled_groups) != set(self.source_order) - {"clinical_data"}:
            raise AssertionError("clinical re-encoding called non-clinical encoders")
        if set(batch) != {"clinical_values", "clinical_mask", "has_clinical"}:
            raise AssertionError("clinical re-encoding accessed labels or extra inputs")
        values, masks = batch["clinical_values"], batch["clinical_mask"]
        self.encoder_calls.append((values.clone(), masks.clone()))
        weight = torch.arange(1, values.shape[1] + 1, dtype=values.dtype) / 100
        clinical = (values * masks * weight).sum(1) + 0.03 * masks.sum(1)
        present = masks.any(1) & batch["has_clinical"]
        tokens = torch.zeros(len(values), 9, 1)
        availability = torch.zeros(len(values), 9, dtype=torch.bool)
        tokens[:, 0, 0] = clinical * present
        availability[:, 0] = present
        return {"source_tokens": tokens, "source_present": availability}

    def predict_from_source_tokens(self, tokens, present, *, task):
        assert task == "toy_binary"
        self.prediction_inputs.append((tokens.clone(), present.clone()))
        return {"logits": torch.where(present[:, 0], tokens[:, 0, 0], 0.15)
                + (tokens[:, 1:, 0] * present[:, 1:]).sum(1)}


def fixture():
    schema, rows = catalog_fixture()
    plan = build_basic_clinical_plan(schema, rows)
    values = torch.arange(8 * 26, dtype=torch.float32).reshape(8, 26) / 130 - 0.8
    masks = torch.ones_like(values, dtype=torch.bool)
    masks[0, :23] = False  # Has Clinical but no remaining basic information.
    masks[1] = False      # Has no Clinical input at all.
    values[2, 0] = 0.0   # Observed normalized zero must remain observed.
    masks[3, 1:23] = False
    model = TransparentModel()
    native = model.encode_sources({"clinical_values": values, "clinical_mask": masks,
                                   "has_clinical": masks.any(1)},
                                  disabled_groups=set(model.source_order) - {"clinical_data"})
    tokens, presence = native["source_tokens"], native["source_present"]
    tokens[2:, 1, 0] = torch.tensor([0.2, -0.4, 0.6, -0.8, 0.1, -0.3])
    presence[2:, 1] = True
    logits = model.predict_from_source_tokens(tokens, presence, task="toy_binary")["logits"]
    cache = SimpleNamespace(source_tokens=tokens, source_present=presence,
                            person_ids=tuple(f"p{i}" for i in range(8)),
                            y_true=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                            full_forward_logits=logits.numpy().astype(np.float64),
                            identity={"source_order": list(model.source_order), "encoder_batch_size": 2,
                                      "amp_mode": "float32", "split": "val", "task": "toy_binary"})
    model.encoder_calls.clear()
    model.prediction_inputs.clear()
    return model, cache, values, masks, plan


class BasicInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.cache, cls.values, cls.masks, cls.plan = fixture()
        cls.original = copy.deepcopy(cls.cache)
        cls.original_values, cls.original_masks = cls.values.clone(), cls.masks.clone()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = run_basic_clinical_analysis(
                cls.model, cls.cache, cls.values, cls.masks, cls.plan,
                task="toy_binary", threshold=0.6, batch_size=2, device="cpu",
                use_bf16_amp=False, bootstrap_samples=16, seed=42,
            )

    def test_all_three_predictions_match_independent_arithmetic(self):
        frame = self.result["predictions"]
        for condition in ("full_clinical", "basic_clinical", "no_clinical"):
            mask = self.original_masks.numpy().copy()
            if condition == "basic_clinical":
                mask[:, 23:] = False
            elif condition == "no_clinical":
                mask[:] = False
            clinical = (self.original_values.numpy() * mask * (np.arange(1, 27) / 100)).sum(1) + 0.03 * mask.sum(1)
            expected = np.where(mask.any(1), clinical, 0.15)
            expected += (self.original.source_tokens[:, 1:, 0].numpy() * self.original.source_present[:, 1:].numpy()).sum(1)
            np.testing.assert_allclose(frame[f"{condition}_logit"], expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(frame.person_id.tolist(), list(self.cache.person_ids))
        self.assertTrue((frame.threshold == 0.6).all())
        self.assertEqual(len(self.model.prediction_inputs), 12)  # 3 fresh conditions x 4 batches.

    def test_removed_values_and_masks_are_cleared_but_real_zeros_remain_observed(self):
        calls = self.model.encoder_calls
        self.assertEqual(len(calls), 8)  # Untouched equivalence plus basic re-encoding.
        basic_values = torch.cat([v for v, _ in calls[4:]])
        basic_masks = torch.cat([m for _, m in calls[4:]])
        self.assertFalse(basic_values[:, 23:].any())
        self.assertFalse(basic_masks[:, 23:].any())
        self.assertTrue(torch.equal(basic_masks[:, :23], self.original_masks[:, :23]))
        self.assertTrue(torch.equal(basic_values[:, :23], self.original_values[:, :23]))
        self.assertTrue(basic_masks[2, 0])
        for _, presence in self.model.prediction_inputs[8:]:
            self.assertFalse(presence[:, 0].any())

    def test_other_modalities_and_original_inputs_never_change(self):
        for start in (0, 4, 8):
            used = self.model.prediction_inputs[start:start + 4]
            self.assertTrue(torch.equal(torch.cat([v for v, _ in used])[:, 1:], self.original.source_tokens[:, 1:]))
            self.assertTrue(torch.equal(torch.cat([m for _, m in used])[:, 1:], self.original.source_present[:, 1:]))
        self.assertTrue(torch.equal(self.cache.source_tokens, self.original.source_tokens))
        self.assertTrue(torch.equal(self.cache.source_present, self.original.source_present))
        self.assertTrue(torch.equal(self.values, self.original_values))
        self.assertTrue(torch.equal(self.masks, self.original_masks))

    def test_missing_basic_and_no_remaining_input_are_explicit(self):
        frame = self.result["predictions"].set_index("person_id")
        self.assertFalse(frame.loc["p0", "basic_clinical_present"])
        self.assertFalse(frame.loc["p0", "has_nonclinical_input"])
        metric = self.result["metrics"]
        all_rows = metric[metric.cohort_scope == "all_participants"].set_index("condition")
        self.assertEqual(all_rows.loc["basic_clinical", "n_without_any_input"], 2)
        self.assertEqual(all_rows.loc["no_clinical", "n_without_any_input"], 2)
        self.assertEqual(all_rows.loc["full_clinical", "n_without_any_input"], 1)
        self.assertEqual(set(self.result["summary"].comparison), {"full_vs_basic", "full_vs_no_clinical", "basic_vs_no_clinical"})

    def test_no_clinical_interval_is_recomputed_from_the_paired_predictions(self):
        frame = self.result["predictions"]
        expected = am.paired_stratified_bootstrap_deltas(
            frame.y.to_numpy(), frame.full_clinical_probability.to_numpy(),
            frame.no_clinical_probability.to_numpy(), threshold=0.6, n_bootstrap=16,
            seed=am._stable_seed(42, "bootstrap", "ablation", "clinical_data"),
        )
        observed = self.result["intervals"]
        observed = observed[(observed.comparison == "full_vs_no_clinical") & (observed.cohort_scope == "all_participants")].set_index("metric")
        for record in expected:
            for key in ("point_estimate", "ci_lower", "ci_upper"):
                self.assertAlmostEqual(observed.loc[record["metric"], key], record[key], places=12)

    def test_test_split_is_rejected_before_encoding(self):
        model, cache, values, masks, plan = fixture()
        cache.identity["split"] = "test"
        with self.assertRaises(ValueError):
            run_basic_clinical_analysis(model, cache, values, masks, plan,
                task="toy_binary", threshold=0.6, batch_size=2, device="cpu",
                use_bf16_amp=False, bootstrap_samples=16, seed=42)
        self.assertFalse(model.encoder_calls)

    def test_shadowing_runtime_module_is_rejected(self):
        _assert_local_runtime_modules()
        with mock.patch.dict(sys.modules, {"other.task2_v2_model": SimpleNamespace(__file__="/tmp/wrong-model-snapshot.py")}):
            with self.assertRaisesRegex(RuntimeError, "mixed runtime code version"):
                _assert_local_runtime_modules()


if __name__ == "__main__":
    unittest.main()
