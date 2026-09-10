"""Mandatory cluster CPU tests: real tiny training plus independent ablation math."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from basic_retrain_fixtures_v1 import prepared_fixture
from basic_retrain_contract_v1 import MODALITIES, GROUPS, read_json, sha256, verify_training
from train_task2_v2 import load_supervised_config, build_datasets
from task2_v2_dataset import OT2V2Dataset
from analyze_basic_retrained_v1 import analyze
import analyze_modality_importance_v2 as am


class BasicRetrainRuntimeTests(unittest.TestCase):
    def test_real_23_column_training_then_ablation_never_iterates_test(self):
        import train_basic_clinical_v1 as train_cli
        import analyze_basic_retrained_v1 as analyze_cli
        from summarize_basic_retrain_v1 import load_importance, write_report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = prepared_fixture(root)
            _, config = load_supervised_config(inputs / "train_config.yaml")
            datasets = build_datasets(config)
            old_features = read_json(root / "sources/moca_binary/clinical_schema.json")["feature_names"]
            for ds in (datasets.train, datasets.validation, datasets.test):
                self.assertEqual(ds.clinical_values_raw.shape[1], 23)
                self.assertEqual(ds.model_feature_dimensions["clinical_dim"], 23)
                self.assertEqual(tuple(ds.scalers["clinical"].feature_order), tuple(old_features[:23]))
                self.assertEqual(tuple(ds.loaded_source_groups), MODALITIES)
            self.assertFalse(datasets.train[1]["clinical_mask"].any())
            self.assertTrue(datasets.train[0]["clinical_mask"][0])
            original_getitem = OT2V2Dataset.__getitem__
            visited = []

            def guarded(ds, index):
                split = ds.index.iloc[index]["split"]
                if split == "test":
                    raise AssertionError("test participant batch was requested")
                item = original_getitem(ds, index)
                self.assertEqual(item["clinical_values"].shape, (23,))
                self.assertEqual(item["clinical_mask"].shape, (23,))
                visited.append(split)
                return item

            with mock.patch.object(OT2V2Dataset, "__getitem__", guarded), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, "argv", ["train_basic_clinical_v1.py", "--inputs", str(inputs)]):
                    train_cli.main()
                trained = verify_training(config.output_dir, sha256(inputs / "PREPARED.json"))
                self.assertFalse(trained["test_evaluated"])
                self.assertIsNone(trained["initialization_checkpoint"])
                with mock.patch.object(sys, "argv", ["analyze_basic_retrained_v1.py", "--inputs", str(inputs),
                    "--out-dir", str(root / "importance/moca_binary/seed42/val"), "--cache", str(root / "cache.pt"),
                    "--bootstrap-samples", "16"]):
                    analyze_cli.main()
            result = load_importance(root, "moca_binary")
            self.assertEqual(result["meta"]["threshold"], trained["threshold"])
            self.assertEqual(result["meta"]["clinical_width"], 23)
            self.assertTrue(write_report(root, ("moca_binary",)))
            self.assertIn("train", visited)
            self.assertIn("val", visited)
            self.assertNotIn("test", visited)
            self.assertFalse((config.output_dir / "test_predictions.csv").exists())
            checkpoint = torch.load(config.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config_provenance"]["resolved_model_config"]["clinical_dim"], 23)

    def test_all_deletions_match_independent_arithmetic_and_identical_cohorts_reuse_ci(self):
        class Model(torch.nn.Module):
            source_order = MODALITIES

            def predict_from_source_tokens(self, tokens, present, *, task):
                return {"logits": torch.where(present, tokens[:, :, 0], 0.05).sum(1)}

        model = Model().eval()
        tokens = (torch.arange(6 * 9).reshape(6, 9, 1).float() % 13 - 6) / 10
        presence = torch.ones(6, 9, dtype=torch.bool)
        labels = np.array([0, 1, 0, 1, 0, 1])
        cache = SimpleNamespace(source_tokens=tokens.clone(), source_present=presence.clone(),
            person_ids=[f"synthetic{i}" for i in range(6)], y_true=labels,
            full_forward_logits=tokens[:, :, 0].sum(1).numpy(),
            identity={"task": "moca_binary", "split": "val", "source_order": list(MODALITIES)})
        with mock.patch.object(am, "_cohort_condition_rows", wraps=am._cohort_condition_rows) as wrapped, contextlib.redirect_stdout(io.StringIO()):
            result = analyze(model, cache, task="moca_binary", threshold=0.6, batch_size=2,
                             device="cpu", bf16=False, bootstrap_samples=16)
        self.assertEqual(wrapped.call_count, len(GROUPS))
        self.assertTrue(torch.equal(cache.source_tokens, tokens))
        self.assertTrue(torch.equal(cache.source_present, presence))
        for group, members in GROUPS.items():
            expected_tokens = tokens[:, :, 0].numpy().copy()
            expected_tokens[:, [MODALITIES.index(m) for m in members]] = 0.05
            probs = 1 / (1 + np.exp(-expected_tokens.sum(1)))
            observed = result["predictions"][result["predictions"].group == group]
            np.testing.assert_allclose(observed.perturbed_probability, probs, atol=1e-6, rtol=1e-6)
            self.assertEqual(observed.person_id.tolist(), cache.person_ids)
            ci = result["intervals"][result["intervals"].group == group]
            for metric in am.METRIC_NAMES:
                rows = ci[ci.metric == metric]
                self.assertEqual(rows.ci_lower.nunique(), 1)
                self.assertEqual(rows.ci_upper.nunique(), 1)
        cache.identity["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation cache"):
            analyze(model, cache, task="moca_binary", threshold=0.6, batch_size=2,
                    device="cpu", bf16=False, bootstrap_samples=16)


if __name__ == "__main__":
    unittest.main()
