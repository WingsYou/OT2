from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch


OT2_DIR = Path(__file__).resolve().parents[1]
if str(OT2_DIR) not in sys.path:
    sys.path.insert(0, str(OT2_DIR))

from clinical_overlap_inference_v1 import (
    assert_native_clinical_equivalent,
    encode_clinical_override,
    materialize_clinical_bank,
)
from task2_v2_model import Task2V2Model, Task2V2ModelConfig


class ClinicalOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(402)
        self.model = Task2V2Model(Task2V2ModelConfig(
            clinical_dim=3,
            environment_dim=1,
            activity_dim=1,
            cgm_dim=1,
            ecg_waveform_dim=12,
            ecg_numeric_dim=8,
            d_model=8,
            num_heads=2,
            temporal_layers=1,
            fusion_layers=1,
            dropout=0.0,
            retinal_backbone="small_cnn",
            retinal_backbone_dim=8,
            task_names=("toy_binary",),
        )).eval()
        self.values = torch.tensor([
            [0.0, 0.2, 0.7],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.1, 0.5],
            [0.7, 0.2, 0.0],
        ])
        self.masks = torch.tensor([
            [True, True, True],
            [True, False, False],
            [False, False, False],
            [True, True, True],
            [True, True, False],
        ])
        self.disabled = set(self.model.source_order) - {"clinical_data"}
        tokens, present = [], []
        with torch.no_grad():
            for start in range(0, len(self.values), 2):
                result = self.model.encode_sources({
                    "clinical_values": self.values[start:start + 2],
                    "clinical_mask": self.masks[start:start + 2],
                    "has_clinical": self.masks[start:start + 2].any(dim=1),
                }, disabled_groups=self.disabled)
                tokens.append(result["source_tokens"])
                present.append(result["source_present"])
        self.cache = SimpleNamespace(
            source_tokens=torch.cat(tokens),
            source_present=torch.cat(present),
            person_ids=tuple(f"p{i}" for i in range(len(self.values))),
            identity={
                "source_order": list(self.model.source_order),
                "encoder_batch_size": 2,
                "amp_mode": "float32",
            },
        )

    def encode(self, indices=(), **kwargs):
        arguments = {
            "device": "cpu", "batch_size": 2, "use_bf16_amp": False,
            **kwargs,
        }
        return encode_clinical_override(
            self.model, self.cache, self.values, self.masks, indices, **arguments
        )

    def test_untouched_bank_reproduces_native_tokens_without_other_encoders(self):
        with (
            mock.patch.object(self.model, "_ecg", side_effect=AssertionError("ECG read")),
            mock.patch.object(self.model, "_retinal", side_effect=AssertionError("image read")),
            mock.patch.object(self.model, "_temporal", side_effect=AssertionError("sequence read")),
        ):
            tokens, present = self.encode()
        assert_native_clinical_equivalent(self.cache, tokens, present)
        self.assertEqual(tokens.shape, (5, 8))
        self.assertEqual(tokens.device.type, "cpu")

    def test_removes_values_and_observation_masks_without_mutating_inputs(self):
        original_values = self.values.clone()
        original_masks = self.masks.clone()
        original_cache = copy.deepcopy(self.cache)
        observed = []

        def record_inputs(_module, arguments):
            observed.append(tuple(item.detach().clone() for item in arguments))

        hook = self.model.clinical_encoder.register_forward_pre_hook(record_inputs)
        try:
            tokens, present = self.encode((0,))
        finally:
            hook.remove()
        encoded_values = torch.cat([values for values, _ in observed])
        encoded_masks = torch.cat([masks for _, masks in observed])
        self.assertTrue(torch.equal(encoded_values[:, 0], torch.zeros(5)))
        self.assertFalse(encoded_masks[:, 0].any())
        self.assertTrue(torch.equal(encoded_values[:, 1:], self.values[:, 1:]))
        self.assertTrue(torch.equal(encoded_masks[:, 1:], self.masks[:, 1:]))
        self.assertTrue(torch.equal(present, torch.tensor([True, False, False, True, True])))
        self.assertEqual(torch.count_nonzero(tokens[1:3]).item(), 0)
        self.assertTrue(torch.equal(self.values, original_values))
        self.assertTrue(torch.equal(self.masks, original_masks))
        self.assertTrue(torch.equal(self.cache.source_tokens, original_cache.source_tokens))
        self.assertTrue(torch.equal(self.cache.source_present, original_cache.source_present))

    def test_all_clinical_columns_removed_uses_missing_source_semantics(self):
        tokens, present = self.encode((0, 1, 2))
        self.assertFalse(present.any())
        self.assertEqual(torch.count_nonzero(tokens).item(), 0)
        replaced = self.cache.source_tokens.clone()
        availability = self.cache.source_present.clone()
        replaced[:, 0] = tokens
        availability[:, 0] = present
        with torch.no_grad():
            through_cache = self.model.predict_from_source_tokens(
                replaced, availability, task="toy_binary"
            )["logits"]
            native_missing = self.model({
                "clinical_values": self.values,
                "clinical_mask": self.masks,
                "has_clinical": self.masks.any(dim=1),
            }, task="toy_binary", disabled_groups=set(self.model.source_order))["logits"]
        torch.testing.assert_close(through_cache, native_missing, rtol=1e-6, atol=1e-6)

    def test_partial_override_matches_full_forward_with_same_removed_columns(self):
        tokens, present = self.encode((0, 2))
        replaced = self.cache.source_tokens.clone()
        availability = self.cache.source_present.clone()
        replaced[:, 0] = tokens
        availability[:, 0] = present
        expected_values, expected_masks = self.values.clone(), self.masks.clone()
        expected_values[:, [0, 2]] = 0.0
        expected_masks[:, [0, 2]] = False
        with torch.no_grad():
            cached = self.model.predict_from_source_tokens(
                replaced, availability, task="toy_binary"
            )["logits"]
            native = self.model({
                "clinical_values": expected_values,
                "clinical_mask": expected_masks,
                "has_clinical": self.masks.any(dim=1),
            }, task="toy_binary", disabled_groups=self.disabled)["logits"]
        torch.testing.assert_close(cached, native, rtol=1e-6, atol=1e-6)

    def test_rejects_invalid_columns_batch_size_and_amp(self):
        for indices in ((-1,), (3,), (0, 0), (1.0,), (True,)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.encode(indices)
        for size in (0, -1, 1.5, True, 3):
            with self.subTest(batch_size=size), self.assertRaises(ValueError):
                self.encode(batch_size=size)
        with self.assertRaisesRegex(ValueError, "AMP"):
            self.encode(use_bf16_amp=True)

    def test_rejects_nonfinite_values_and_unaligned_native_presence(self):
        self.values[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.encode()
        self.values[0, 0] = 0.0
        self.cache.source_present[2, 0] = True
        with self.assertRaisesRegex(ValueError, "presence"):
            self.encode()

    def test_native_equivalence_fails_closed_on_drift(self):
        tokens, present = self.encode()
        altered = tokens.clone()
        altered[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "differs"):
            assert_native_clinical_equivalent(self.cache, altered, present)
        present[0] = False
        with self.assertRaisesRegex(ValueError, "presence"):
            assert_native_clinical_equivalent(self.cache, tokens, present)


class MaterializeClinicalBankTests(unittest.TestCase):
    def test_uses_only_existing_bank_and_scaler_preserves_measured_zero(self):
        class Scaler:
            def normalise(self, values, mask, _label):
                return torch.where(mask, torch.nan_to_num(values) - 10.0, 0.0)

        class Dataset:
            clinical_values_raw = np.array([[0.0, np.nan], [1.0, 2.0]])
            index = pd.DataFrame({"has_clinical": [True, False]})
            scalers = {"clinical": Scaler()}

            def __len__(self):
                return 2

            def __getitem__(self, _index):
                raise AssertionError("materialization must not load raw assets or labels")

        dataset = Dataset()
        original = dataset.clinical_values_raw.copy()
        values, masks = materialize_clinical_bank(dataset)
        self.assertTrue(torch.equal(masks, torch.tensor([[True, False], [False, False]])))
        self.assertTrue(torch.equal(values, torch.tensor([[-10.0, 0.0], [0.0, 0.0]])))
        np.testing.assert_equal(dataset.clinical_values_raw, original)


if __name__ == "__main__":
    unittest.main()
