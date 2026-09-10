"""Paired A/B/C/D inference on one frozen model and one participant cohort."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import torch

import analyze_modality_importance_v2 as am
from clinical_abcd_plan_v1 import CONDITIONS, CONTRASTS, COHORTS, NONCLINICAL, plan_hash
from clinical_overlap_inference_v1 import encode_clinical_override, assert_native_clinical_equivalent


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.dtype, tuple(value.shape))).encode()
                          + value.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def run_abcd_analysis(model, cache, values, masks, plan, *, task, split, threshold,
                      batch_size, device, use_bf16_amp, bootstrap_samples, seed):
    if split not in ("val", "test") or cache.identity.get("split") != split or cache.identity.get("task") != task:
        raise ValueError("ABCD analysis requires the matching task/split cache")
    if plan.get("task") != task or values.shape[1] != len(plan["feature_order"]):
        raise ValueError("Clinical plan/task/width mismatch")
    if plan_hash({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Clinical plan hash mismatch")
    if tuple(am.PRIMARY_MODALITIES) != ("clinical_data", *NONCLINICAL):
        raise ValueError("Unexpected nine-modality source order")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    original = tuple(_tensor_hash(x) for x in (cache.source_tokens, cache.source_present, values, masks))
    opts = dict(device=device, batch_size=batch_size, use_bf16_amp=use_bf16_amp)
    untouched = encode_clinical_override(model, cache, values, masks, (), **opts)
    assert_native_clinical_equivalent(cache, *untouched)
    basic_token, basic_present = encode_clinical_override(
        model, cache, values, masks, plan["masked_indices_zero_based"], **opts)
    results = {}
    for name in CONDITIONS:
        print(f"[ABCD][{task}][{split}] {name}: {plan['conditions'][name]['label']}", flush=True)
        override = dict(clinical_token_override=basic_token, clinical_present_override=basic_present) if name in ("C", "D") else {}
        results[name] = am.evaluate_cached_condition(
            model, cache, task=task, threshold=threshold, **opts, **override,
            disabled_groups=tuple(plan["conditions"][name]["disabled_modalities"]))
    full = results["A"]
    am._assert_cached_baseline_exact(cache, full)
    for result in results.values():
        am._assert_aligned(full, result)
    native = cache.source_present.cpu().numpy()
    others, basic = native[:, 1:].any(axis=1), basic_present.cpu().numpy()
    availability = {"A": native.any(axis=1), "B": native[:, 0], "C": basic | others, "D": basic}
    cohort_masks = dict(zip(COHORTS, (np.ones(len(full.y_true), dtype=bool), basic, basic & others)))
    counts, metrics, predictions = [], [], []
    for scope, mask in cohort_masks.items():
        labels = full.y_true[mask]
        count = {"task": task, "split": split, "cohort_scope": scope, "n_total": len(full.y_true),
                 "n_evaluated": len(labels), "n_positive": int((labels == 1).sum()),
                 "n_negative": int((labels == 0).sum()), "prevalence": float(labels.mean()) if len(labels) else float("nan"),
                 "n_native_present": int(others.sum()), "n_eligible": len(labels),
                 "n_basic_present": int(basic[mask].sum()), "n_with_nonclinical_input": int(others[mask].sum()),
                 "cohort_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                 "status": "empty_cohort" if not len(labels) else "single_class_cohort" if len(np.unique(labels)) < 2 else "ok"}
        counts.append(count)
        for name in CONDITIONS:
            measured = am._cohort_metrics_matrix(labels, results[name].probabilities[mask], threshold=threshold)
            metrics.append({**count, "condition": name, "threshold": threshold,
                            "n_clinical_columns_retained": plan["conditions"][name]["n_clinical_columns_retained"],
                            "n_without_any_input": int((~availability[name] & mask).sum()),
                            **{key: float(value[0]) for key, value in measured.items()}})
    for i, person_id in enumerate(full.person_ids):
        row = {"task": task, "split": split, "person_id": person_id, "y": int(full.y_true[i]),
               "threshold": threshold, "native_clinical_present": bool(native[i, 0]),
               "basic_clinical_present": bool(basic[i]), "has_nonclinical_input": bool(others[i]),
               "n_observed_basic_columns": int(masks[i, plan["retained_indices_zero_based"]].sum()),
               **{f"in_{scope}": bool(mask[i]) for scope, mask in cohort_masks.items()}}
        for name, result in results.items():
            row.update({f"{name}_probability": float(result.probabilities[i]), f"{name}_logit": float(result.logits[i]),
                        f"{name}_prediction": int(result.predictions[i]), f"{name}_has_input": bool(availability[name][i])})
        predictions.append(row)
    summaries, intervals, memo = [], [], {}
    for comparison, (reference, perturbed) in CONTRASTS.items():
        print(f"[ABCD][{task}][{split}] paired bootstrap: {comparison}", flush=True)
        for count in counts:
            mask = cohort_masks[count["cohort_scope"]]
            key = (comparison, mask.tobytes())
            # Cohort names must not change the CI when people and predictions are identical.
            if key not in memo:
                memo[key] = am._cohort_condition_rows(
                    results[reference], [results[perturbed]], mask=mask, counts=count,
                    method="ablation", group="nonclinical_all", threshold=threshold,
                    bootstrap_samples=bootstrap_samples,
                    seed=am._stable_seed(seed, "abcd_v1", comparison, count["cohort_mask_sha256"]))
            summary, interval = memo[key]
            context = {**count, "comparison": comparison, "reference_condition": reference,
                       "perturbed_condition": perturbed, "group": "nonclinical_all"}
            summaries.extend({**row, **context} for row in summary)
            intervals.extend({**row, **context} for row in interval)
    if original != tuple(_tensor_hash(x) for x in (cache.source_tokens, cache.source_present, values, masks)):
        raise RuntimeError("Original cache or Clinical bank was mutated")
    return {"metrics": pd.DataFrame(metrics), "summary": pd.DataFrame(summaries),
            "intervals": pd.DataFrame(intervals), "predictions": pd.DataFrame(predictions),
            "cohort_counts": pd.DataFrame(counts), "conditions": [plan["conditions"][n] for n in CONDITIONS],
            "baseline_metrics": full.metrics,
            "checks": {"native_clinical_equivalence": "passed; atol=1e-6, rtol=1e-6",
                       "full_forward_equivalence": "passed; atol=1e-6, rtol=1e-6",
                       "original_cache_and_clinical_bank_unchanged": True,
                       "clinical_values_and_observed_masks_removed": True,
                       "all_four_predictions_recomputed_and_paired": True,
                       "identical_cohort_intervals_reused": True}}
