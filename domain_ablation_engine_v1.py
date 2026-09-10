"""Paired frozen-model inference for prespecified domain deletions."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import torch

try:
    from . import analyze_modality_importance_v2 as am
    from .clinical_overlap_inference_v1 import encode_clinical_override, assert_native_clinical_equivalent
except ImportError:
    import analyze_modality_importance_v2 as am
    from clinical_overlap_inference_v1 import encode_clinical_override, assert_native_clinical_equivalent


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.dtype, tuple(value.shape))).encode() + value.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def run_domain_analysis(model, cache, values, masks, plan, *, groups, task, threshold,
                        batch_size, device, use_bf16_amp, bootstrap_samples, seed):
    if cache.identity.get("split") != "val" or cache.identity.get("task") != task:
        raise ValueError("Domain analysis requires the matching validation cache")
    if plan.get("task") != task or values.shape[1] != len(plan["feature_order"]):
        raise ValueError("Clinical plan/task/width mismatch")
    if not groups or len(set(groups)) != len(groups) or any(g not in plan["groups"] for g in groups):
        raise ValueError("Unknown, empty or duplicate domain groups")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    original = tuple(_tensor_hash(x) for x in (cache.source_tokens, cache.source_present, values, masks))
    opts = dict(device=device, batch_size=batch_size, use_bf16_amp=use_bf16_amp)
    untouched = encode_clinical_override(model, cache, values, masks, (), **opts)
    assert_native_clinical_equivalent(cache, *untouched)
    eval_opts = dict(task=task, threshold=threshold, **opts)
    baseline = am.evaluate_cached_condition(model, cache, **eval_opts)
    am._assert_cached_baseline_exact(cache, baseline)
    original_presence = cache.source_present.cpu().numpy().copy()
    source_order = tuple(am.PRIMARY_MODALITIES)
    clinical_index = source_order.index("clinical_data")
    predictions, summaries, intervals, cohort_rows, conditions = [], [], [], [], []
    for group in groups:
        definition = plan["groups"][group]
        indices, members = definition["indices"], tuple(definition["disabled_modalities"])
        print(f"[domain][{task}][{group}] remove branches={','.join(members)}, Clinical columns={len(indices)}", flush=True)
        override = {}
        adjusted_presence = original_presence.copy()
        if indices and "clinical_data" not in members:
            token, present = encode_clinical_override(model, cache, values, masks, indices, **opts)
            override = dict(clinical_token_override=token, clinical_present_override=present)
            adjusted_presence[:, clinical_index] = present.cpu().numpy()
        result = am.evaluate_cached_condition(model, cache, disabled_groups=members, **eval_opts, **override)
        am._assert_aligned(baseline, result)
        target_indices = [source_order.index(m) for m in members]
        # Retain native target presence, but evaluate remaining input after cleanup.
        cohort_masks, counts = am._cohort_masks_and_counts(adjusted_presence, baseline.y_true, group)
        native_target = original_presence[:, target_indices].any(axis=1)
        clinical_affected = masks[:, indices].any(dim=1).cpu().numpy() if indices else np.zeros(len(baseline.y_true), dtype=bool)
        affected = native_target | clinical_affected
        remaining = adjusted_presence.copy()
        remaining[:, target_indices] = False
        remaining_any = remaining.any(axis=1)
        # A Clinical companion can be observed even when the raw target is absent.
        affected_labels = baseline.y_true[affected]
        extra = {**counts[0], "cohort_scope": "any_target_information_present",
                 "n_evaluated": int(affected.sum()),
                 "n_positive": int((affected_labels == 1).sum()),
                 "n_negative": int((affected_labels == 0).sum()),
                 "prevalence": float(affected_labels.mean()) if len(affected_labels) else float("nan"),
                 "status": "empty_cohort" if not len(affected_labels) else "single_class_cohort" if len(np.unique(affected_labels)) < 2 else "ok"}
        cohort_masks[extra["cohort_scope"]] = affected
        counts.append(extra)
        definition = {**definition, "group": group,
                      "n_people_with_selected_clinical_observed": int(clinical_affected.sum()),
                      "n_without_any_input": int((~remaining_any).sum())}
        conditions.append(definition)
        for i, person_id in enumerate(baseline.person_ids):
            predictions.append({
                "task": task, "split": "val", "group": group, "person_id": person_id,
                "y": int(baseline.y_true[i]), "threshold": threshold,
                "baseline_probability": float(baseline.probabilities[i]),
                "perturbed_probability": float(result.probabilities[i]),
                "baseline_logit": float(baseline.logits[i]), "perturbed_logit": float(result.logits[i]),
                "native_target_present": bool(native_target[i]),
                "selected_clinical_observed": bool(clinical_affected[i]),
                "has_remaining_input": bool(remaining_any[i]),
                **{f"in_{scope}": bool(mask[i]) for scope, mask in cohort_masks.items()},
            })
        for count in counts:
            scope = count["cohort_scope"]
            count = {**count, "n_without_any_input": int((~remaining_any & cohort_masks[scope]).sum()),
                     "n_selected_clinical_observed": int((clinical_affected & cohort_masks[scope]).sum())}
            cohort_rows.append({"task": task, **count})
            summary, interval = am._cohort_condition_rows(
                baseline, [result], mask=cohort_masks[scope], counts=count,
                method="ablation", group=group, threshold=threshold,
                bootstrap_samples=bootstrap_samples,
                seed=am._stable_seed(seed, "domain_v1", group, scope),
            )
            context = {"task": task, "split": "val", "comparison": "full_vs_domain",
                       "n_clinical_columns_removed": len(indices)}
            summaries.extend({**row, **context} for row in summary)
            intervals.extend({**row, **context} for row in interval)
        print(f"[domain][{task}][{group}] paired bootstrap complete", flush=True)
    if original != tuple(_tensor_hash(x) for x in (cache.source_tokens, cache.source_present, values, masks)):
        raise RuntimeError("Original cache or Clinical bank was mutated")
    return {
        "summary": pd.DataFrame(summaries), "intervals": pd.DataFrame(intervals),
        "predictions": pd.DataFrame(predictions), "cohort_counts": pd.DataFrame(cohort_rows),
        "conditions": conditions, "baseline_metrics": baseline.metrics,
        "checks": {"native_clinical_equivalence": "passed; atol=1e-6, rtol=1e-6",
                   "full_forward_equivalence": "passed; atol=1e-6, rtol=1e-6",
                   "original_cache_and_clinical_bank_unchanged": True,
                   "clinical_values_and_observed_masks_removed": True,
                   "paired_participant_alignment": True},
    }
