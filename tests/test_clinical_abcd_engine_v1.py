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
from abcd_inference_engine_v1 import run_abcd_analysis
from analyze_clinical_abcd_v1 import _assert_local_runtime_modules
import analyze_modality_importance_v2 as am
from clinical_abcd_plan_v1 import build_abcd_plan
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
    plan = build_abcd_plan(schema, rows)
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


def run_fixture(model, cache, values, masks, plan, split="val"):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_abcd_analysis(model, cache, values, masks, plan, task="toy_binary",
                                 split=split, threshold=0.6, batch_size=2, device="cpu",
                                 use_bf16_amp=False, bootstrap_samples=16, seed=42)


class ABCDInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.cache, cls.values, cls.masks, cls.plan = fixture()
        cls.original = copy.deepcopy(cls.cache)
        cls.original_values, cls.original_masks = cls.values.clone(), cls.masks.clone()
        cls.result = run_fixture(cls.model, cls.cache, cls.values, cls.masks, cls.plan)

    def test_four_conditions_match_independent_arithmetic(self):
        frame = self.result["predictions"]
        for condition in "ABCD":
            mask = self.original_masks.numpy().copy()
            if condition in "CD":
                mask[:, 23:] = False
            clinical = (self.original_values.numpy() * mask * (np.arange(1, 27) / 100)).sum(1) + 0.03 * mask.sum(1)
            expected = np.where(mask.any(1), clinical, 0.15)
            if condition in "AC":
                expected += (self.original.source_tokens[:, 1:, 0].numpy() * self.original.source_present[:, 1:].numpy()).sum(1)
            np.testing.assert_allclose(frame[f"{condition}_logit"], expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(len(self.model.prediction_inputs), 16)
        self.assertEqual(frame.person_id.tolist(), list(self.cache.person_ids))
        self.assertTrue((frame.threshold == 0.6).all())

    def test_b_and_d_disable_all_eight_sources_through_presence(self):
        for start in (4, 12):
            for _, present in self.model.prediction_inputs[start:start + 4]:
                self.assertFalse(present[:, 1:].any())
        for start in (0, 8):
            calls = self.model.prediction_inputs[start:start + 4]
            self.assertTrue(torch.equal(torch.cat([x for x, _ in calls])[:, 1:], self.original.source_tokens[:, 1:]))
            self.assertTrue(torch.equal(torch.cat([x for _, x in calls])[:, 1:], self.original.source_present[:, 1:]))

    def test_basic_masks_and_values_removed_and_native_inputs_unchanged(self):
        calls = self.model.encoder_calls
        self.assertEqual(len(calls), 8)
        masked_values = torch.cat([v for v, _ in calls[4:]])
        masked_masks = torch.cat([m for _, m in calls[4:]])
        self.assertFalse(masked_values[:, 23:].any())
        self.assertFalse(masked_masks[:, 23:].any())
        self.assertTrue(masked_masks[2, 0])  # Observed normalized zero remains observed.
        self.assertTrue(torch.equal(masked_values[:, :23], self.original_values[:, :23]))
        self.assertTrue(torch.equal(masked_masks[:, :23], self.original_masks[:, :23]))
        for observed, expected in ((self.cache.source_tokens, self.original.source_tokens),
                                   (self.cache.source_present, self.original.source_present),
                                   (self.values, self.original_values), (self.masks, self.original_masks)):
            self.assertTrue(torch.equal(observed, expected))

    def test_availability_and_paired_intervals(self):
        frame = self.result["predictions"]
        metrics = self.result["metrics"]
        metrics = metrics[metrics.cohort_scope == "all_participants"].set_index("condition")
        self.assertEqual([metrics.loc[n, "n_without_any_input"] for n in "ABCD"], [1, 1, 2, 2])
        counts = self.result["cohort_counts"].set_index("cohort_scope")
        self.assertEqual(counts.loc["basic_clinical_present", "n_evaluated"], 6)
        intervals = self.result["intervals"]
        for comparison, ref, deleted in (("A_vs_B", "A", "B"), ("C_vs_D", "C", "D")):
            seed = am._stable_seed(42, "abcd_v1", comparison, counts.loc["all_participants", "cohort_mask_sha256"])
            expected = am.paired_stratified_bootstrap_deltas(frame.y.to_numpy(),
                frame[f"{ref}_probability"].to_numpy(), frame[f"{deleted}_probability"].to_numpy(),
                threshold=0.6, n_bootstrap=16, seed=seed)
            observed = intervals[(intervals.comparison == comparison) & (intervals.cohort_scope == "all_participants")].set_index("metric")
            for row in expected:
                for key in ("point_estimate", "ci_lower", "ci_upper"):
                    self.assertAlmostEqual(observed.loc[row["metric"], key], row[key], places=12)

    def test_identical_participant_cohorts_reuse_one_bootstrap(self):
        model, cache, values, masks, plan = fixture()
        with mock.patch.object(am, "_cohort_condition_rows", wraps=am._cohort_condition_rows) as wrapped:
            result = run_fixture(model, cache, values, masks, plan)
        self.assertEqual(wrapped.call_count, 4)  # 2 contrasts x 2 unique masks; there are 3 named cohorts.
        ci = result["intervals"].set_index(["comparison", "cohort_scope", "metric"])
        for comparison in ("A_vs_B", "C_vs_D"):
            for metric in am.METRIC_NAMES:
                a = ci.loc[(comparison, "basic_clinical_present", metric)]
                b = ci.loc[(comparison, "basic_and_nonclinical_present", metric)]
                self.assertEqual(a.ci_lower, b.ci_lower)
                self.assertEqual(a.ci_upper, b.ci_upper)

    def test_test_split_is_explicit_and_wrong_cache_rejected(self):
        model, cache, values, masks, plan = fixture()
        cache.identity["split"] = "test"
        with self.assertRaises(ValueError):
            run_fixture(model, cache, values, masks, plan)
        self.assertFalse(model.encoder_calls)
        result = run_fixture(model, cache, values, masks, plan, split="test")
        self.assertEqual(set(result["predictions"].split), {"test"})

    def test_shadowing_runtime_module_is_rejected(self):
        _assert_local_runtime_modules()
        with mock.patch.dict(sys.modules, {"other.task2_v2_model": SimpleNamespace(__file__="/tmp/wrong-model-snapshot.py")}):
            with self.assertRaisesRegex(RuntimeError, "mixed runtime code version"):
                _assert_local_runtime_modules()

    def test_real_model_missing_sources_ignore_cached_nonclinical_values(self):
        from task2_v2_model import Task2V2Model, Task2V2ModelConfig
        from clinical_overlap_inference_v1 import encode_clinical_override
        torch.manual_seed(42)
        model = Task2V2Model(Task2V2ModelConfig(
            clinical_dim=26, environment_dim=1, activity_dim=1, cgm_dim=1,
            ecg_waveform_dim=12, ecg_numeric_dim=8, d_model=8, num_heads=2,
            temporal_layers=1, fusion_layers=1, dropout=0.0,
            retinal_backbone="small_cnn", retinal_backbone_dim=8,
            task_names=("toy_binary",))).eval()
        disabled = tuple(am.PRIMARY_MODALITIES[1:])
        tokens, present = [], []
        with torch.no_grad():
            for i in range(0, 8, 2):
                batch = {"clinical_values": self.values[i:i + 2], "clinical_mask": self.masks[i:i + 2],
                         "has_clinical": self.masks[i:i + 2].any(1)}
                source = model.encode_sources(batch, disabled_groups=disabled)
                tokens.append(source["source_tokens"])
                present.append(source["source_present"])
        cache = copy.deepcopy(self.cache)
        cache.source_tokens, cache.source_present = torch.cat(tokens), torch.cat(present)
        cache.source_tokens[:, 1:] = torch.randn(8, 8, 8) * 100
        cache.source_present[:, 1:] = True
        basic_token, basic_present = encode_clinical_override(model, cache, self.values, self.masks,
            self.plan["masked_indices_zero_based"], device="cpu", batch_size=2, use_bf16_amp=False)
        for basic in (False, True):
            override = dict(clinical_token_override=basic_token, clinical_present_override=basic_present) if basic else {}
            observed = am.evaluate_cached_condition(model, cache, task="toy_binary", threshold=0.6,
                device="cpu", batch_size=2, disabled_groups=disabled, **override)
            values, masks = self.values.clone(), self.masks.clone()
            if basic:
                values[:, 23:], masks[:, 23:] = 0.0, False
            expected = []
            with torch.no_grad():
                for i in range(0, 8, 2):
                    batch = {"clinical_values": values[i:i + 2], "clinical_mask": masks[i:i + 2],
                             "has_clinical": self.masks[i:i + 2].any(1)}
                    expected.extend(model(batch, task="toy_binary", disabled_groups=disabled)["logits"].reshape(-1).tolist())
            np.testing.assert_allclose(observed.logits, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
