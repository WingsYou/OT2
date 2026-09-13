"""Strict supervised scalar concept readout of frozen multimodal features.

The task predictor receives predicted concepts only. Basic Clinical values are
used by explicitly separate reference models, never by ``predict_readout``.
All fitted transformations use training participants; validation chooses from
small prespecified grids. Observed-label metrics do not remove selection bias.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

REGRESSION_ALPHAS = (0.1, 1.0, 10.0, 100.0)
CLASSIFIER_CS = (0.01, 0.1, 1.0, 10.0)
VARIABLE_IDENTITY_KEYS = frozenset({"split", "cohort_person_ids_sha256", "n_participants",
                                  "encoder_batch_size", "amp_mode"})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    values = array(value).astype(np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return values


def validate_cache_pair(train: dict, val: dict) -> None:
    for cache, split in ((train, "train"), (val, "val")):
        if cache.get("schema_version") != "ot2-cbm-features-v1" or cache.get("split") != split:
            raise ValueError(f"Expected a {split} ot2-cbm-features-v1 cache; test fitting is forbidden")
        ids = cache.get("person_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Cache person_ids must be nonempty strings in a fixed order")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate person_ids in feature cache")
        n = len(ids)
        embedding = _finite_matrix(cache["embedding"], "embedding")
        if embedding.shape[0] != n or embedding.shape[1] < 1:
            raise ValueError("Embedding shape does not match participant order")
        y = array(cache["y"])
        if y.shape != (n,) or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
            raise ValueError("Training and validation each require finite binary labels of both classes")
        for key, shape in (("source_tokens", (n, 9, embedding.shape[1])),
                           ("source_present", (n, 9)), ("original_logits", (n,)),
                           ("clinical_values", (n, 23)), ("clinical_mask", (n, 23))):
            values = array(cache[key])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{key} must be finite with shape {shape}")
        for key in ("source_present", "clinical_mask"):
            if array(cache[key]).dtype != np.bool_:
                raise ValueError(f"{key} must have boolean dtype")
        identity = cache.get("identity", {})
        for key in ("checkpoint_sha256", "config_hash"):
            if not isinstance(identity.get(key), str) or not identity[key]:
                raise ValueError(f"Cache identity is missing {key}")
        if identity.get("task", cache.get("task")) != cache.get("task"):
            raise ValueError("Cache task and identity disagree")
        if "content_sha256" in cache:
            from cache_cbm_features_v1 import cache_content_sha256
            if cache["content_sha256"] != cache_content_sha256(cache):
                raise ValueError("Feature cache content SHA256 mismatch")
    if set(train["person_ids"]) & set(val["person_ids"]):
        raise ValueError("Training and validation person_ids overlap")
    if train["task"] != val["task"] or array(train["embedding"]).shape[1] != array(val["embedding"]).shape[1]:
        raise ValueError("Training and validation task/embedding dimensions differ")
    invariant = lambda cache: {k: v for k, v in cache["identity"].items() if k not in VARIABLE_IDENTITY_KEYS}
    if invariant(train) != invariant(val):
        raise ValueError("Training and validation caches have different source identity/provenance")


def validate_schema(schema: dict) -> list[dict]:
    if schema.get("schema_version") != "ot2-cbm-concepts-v1":
        raise ValueError("Expected ot2-cbm-concepts-v1 concept schema")
    concepts = schema.get("concepts", [])
    names = [c.get("name") for c in concepts]
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Concept names must be unique nonempty strings")
    if schema.get("concept_names", names) != names:
        raise ValueError("Schema concept_names and concepts ordering disagree")
    for c in concepts:
        if c.get("type") not in ("continuous", "categorical"):
            raise ValueError(f"Unsupported concept type for {c['name']}")
        if c["type"] == "categorical":
            cats = c.get("categories", [])
            if len(cats) < 2 or len({_category(x) for x in cats}) != len(cats):
                raise ValueError(f"Invalid categories for {c['name']}")
    return concepts


def _category(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Nonfinite category")
        return format(float(value), ".15g")
    value = str(value).strip()
    try:
        number = float(value)
        return format(number, ".15g") if np.isfinite(number) else value
    except ValueError:
        return value


def ordered_labels(frame: pd.DataFrame, ids: list[str], concepts: list[dict]) -> pd.DataFrame:
    expected = ["person_id"] + [c["name"] for c in concepts]
    if list(frame.columns) != expected:
        raise ValueError(f"Label columns/order must be exactly {expected}")
    frame = frame.copy()
    if frame.person_id.isna().any():
        raise ValueError("Missing person_id in concept labels")
    frame["person_id"] = frame.person_id.astype(str)
    if frame.person_id.duplicated().any():
        raise ValueError("Duplicate person_id in concept labels")
    missing = set(ids) - set(frame.person_id)
    if missing:
        raise ValueError(f"Concept label table missing {len(missing)} cached participants; rebuild audit")
    return frame.set_index("person_id").loc[ids].copy()


def _targets(frame: pd.DataFrame, c: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = frame[c["name"]]
    observed = raw.notna().to_numpy() & raw.astype(str).str.strip().ne("").to_numpy()
    if c["type"] == "continuous":
        values = pd.to_numeric(raw.where(observed), errors="coerce").to_numpy(dtype=float)
        if (observed & ~np.isfinite(values)).any():
            raise ValueError(f"Invalid/nonfinite observed continuous label: {c['name']}")
    else:
        lookup = {_category(v): i for i, v in enumerate(c["categories"])}
        values = np.full(len(raw), -1, dtype=int)
        for i in np.flatnonzero(observed):
            key = _category(raw.iloc[i])
            if key not in lookup:
                raise ValueError(f"Unknown category {key!r} for {c['name']}; allowed {c['categories']}")
            values[i] = lookup[key]
    return values, observed


def _fit_concepts(x_train, x_val, train_labels, val_labels, concepts, seed):
    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train)
    val = scaler.transform(x_val)
    heads, selections = [], []
    for c in concepts:
        y, present = _targets(train_labels, c)
        yv, present_v = _targets(val_labels, c)
        if present.sum() < 2:
            raise ValueError(f"Concept {c['name']}: fewer than 2 observed training labels; audit/drop explicitly")
        if np.unique(y[present]).size < 2:
            raise ValueError(f"Concept {c['name']}: constant training target; cannot fit a semantic predictor")
        if c["type"] == "categorical" and set(y[present]) != set(range(len(c["categories"]))):
            raise ValueError(f"Concept {c['name']}: some schema categories absent in train; review/merge/drop explicitly")
        mean, std = (float(y[present].mean()), float(y[present].std())) if c["type"] == "continuous" else (0., 1.)
        grid = REGRESSION_ALPHAS if c["type"] == "continuous" else CLASSIFIER_CS
        best = None
        for strength in grid:
            if c["type"] == "continuous":
                model = Ridge(alpha=strength).fit(train[present], (y[present] - mean) / std)
                score = float(mean_squared_error((yv[present_v] - mean) / std, model.predict(val[present_v]))) if present_v.any() else None
            else:
                model = LogisticRegression(C=strength, max_iter=2000, random_state=seed).fit(train[present], y[present])
                score = float(log_loss(yv[present_v], model.predict_proba(val[present_v]), labels=np.arange(len(c["categories"])))) if present_v.any() else None
            row = {"concept": c["name"], "regularization": strength, "validation_concept_loss": score,
                   "selection_metric": "standardized_mse" if c["type"] == "continuous" else "log_loss",
                   "n_train_observed": int(present.sum()), "n_val_observed": int(present_v.sum())}
            selections.append(row)
            # If val has no annotations, retain predeclared alpha/C=1, rather
            # than making a task-label-guided choice for this concept.
            objective = score if score is not None else abs(np.log10(strength))
            if best is None or objective < best[0]:
                best = (objective, model, strength)
        heads.append({"name": c["name"], "type": c["type"], "categories": c.get("categories", []),
                      "model": best[1], "regularization": best[2], "target_mean": mean, "target_std": std,
                      "n_train_observed": int(present.sum()), "n_val_observed": int(present_v.sum()),
                      "status": "fitted" if present_v.any() else "fitted_without_validation_annotations"})
    return scaler, heads, selections


def _concept_predictions(scaler, heads, embeddings):
    x = scaler.transform(embeddings)
    pieces, names = [], []
    for head in heads:
        if head["type"] == "continuous":
            pieces.append((head["model"].predict(x) * head["target_std"] + head["target_mean"])[:, None])
            names.append(head["name"])
        else:
            pieces.append(head["model"].predict_proba(x))
            names.extend(f"{head['name']}__p_{_category(cat)}" for cat in head["categories"])
    return np.concatenate(pieces, axis=1), names


def _fit_task(x, xv, y, yv, seed):
    scaler = StandardScaler().fit(x)
    z, zv = scaler.transform(x), scaler.transform(xv)
    best, candidates = None, []
    for strength in CLASSIFIER_CS:
        model = LogisticRegression(C=strength, max_iter=2000, random_state=seed).fit(z, y)
        probability = model.predict_proba(zv)[:, 1]
        score = float(log_loss(yv, probability, labels=[0, 1]))
        candidates.append({"C": strength, "validation_log_loss": score,
                           "validation_auroc": float(roc_auc_score(yv, probability))})
        if best is None or score < best[0]:
            best = score, model, strength, probability
    return {"scaler": scaler, "model": best[1], "C": best[2], "threshold": choose_threshold(yv, best[3]),
            "selection_metric": "validation_log_loss", "candidates": candidates,
            "calibration": "none; logistic head trained by log loss"}


def choose_threshold(y, probability) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    usable = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    best = np.max((tpr-fpr)[usable])
    candidates = thresholds[usable & np.isclose(tpr-fpr, best)]
    return float(candidates[np.lexsort((candidates, np.abs(candidates-0.5)))[0]])


def binary_metrics(y, probability, threshold=0.5) -> dict:
    y, probability = array(y).astype(int), array(probability).astype(float)
    if y.ndim != 1 or probability.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Binary metric labels/probabilities must be aligned")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probabilities")
    pred = probability >= threshold
    sensitivity = float(pred[y == 1].mean()) if np.any(y == 1) else np.nan
    specificity = float((~pred[y == 0]).mean()) if np.any(y == 0) else np.nan
    both = np.unique(y).size == 2
    return {"n": len(y), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, probability)) if both else np.nan,
            "auprc": float(average_precision_score(y, probability)) if both else np.nan,
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(np.mean((y-probability)**2)), "sensitivity": sensitivity,
            "specificity": specificity, "balanced_accuracy": (sensitivity+specificity)/2,
            "threshold": float(threshold)}


def basic_reference_inputs(cache: dict) -> np.ndarray:
    values, mask = array(cache["clinical_values"]), array(cache["clinical_mask"])
    return np.concatenate([np.where(mask, values, 0), mask.astype(float)], axis=1)


def true_concept_inputs(labels_df, schema, person_ids):
    """Observed concepts only, fixed-schema encoding; no imputation."""
    concepts = validate_schema(schema)
    labels = ordered_labels(labels_df, person_ids, concepts)
    complete = np.ones(len(person_ids), dtype=bool)
    targets = []
    for concept in concepts:
        values, observed = _targets(labels, concept)
        complete &= observed
        targets.append((concept, values))
    pieces = []
    for concept, values in targets:
        if concept["type"] == "continuous":
            pieces.append(values[complete, None])
        else:
            pieces.append(np.eye(len(concept["categories"]))[values[complete]])
    return np.concatenate(pieces, axis=1), complete


def _fit_true_reference(train_cache, val_cache, labels, schema, seed):
    x, train_mask = true_concept_inputs(labels, schema, train_cache["person_ids"])
    xv, val_mask = true_concept_inputs(labels, schema, val_cache["person_ids"])
    y = array(train_cache["y"])[train_mask]
    yv = array(val_cache["y"])[val_mask]
    minimum_n = max(20, 2*x.shape[1])
    result = {"status": "unavailable", "cohort": "complete_concept_labels",
              "n_train_complete": int(train_mask.sum()), "n_val_complete": int(val_mask.sum()),
              "train_class_counts": np.bincount(y.astype(int), minlength=2).tolist(),
              "val_class_counts": np.bincount(yv.astype(int), minlength=2).tolist(),
              "minimum_train_n": minimum_n, "minimum_train_per_class": 5,
              "requires_additional_measured_concepts_at_inference": True,
              "not_a_guaranteed_performance_upper_bound": True}
    if len(y) < minimum_n or min(result["train_class_counts"]) < 5:
        result["reason"] = "insufficient_complete_case_training_sample_or_classes"
    elif len(yv) < 2 or np.unique(yv).size < 2:
        result["reason"] = "complete_case_validation_requires_both_classes_for_selection"
    else:
        result.update(status="fitted", reason=None, task_head=_fit_task(x, xv, y, yv, seed))
    return result


def fit_readout(train_cache: dict, val_cache: dict, labels_df: pd.DataFrame,
                schema: dict, seed: int = 42) -> dict:
    validate_cache_pair(train_cache, val_cache)
    concepts = validate_schema(schema)
    train_labels = ordered_labels(labels_df, train_cache["person_ids"], concepts)
    val_labels = ordered_labels(labels_df, val_cache["person_ids"], concepts)
    x, xv = _finite_matrix(train_cache["embedding"], "train embedding"), _finite_matrix(val_cache["embedding"], "val embedding")
    scaler, heads, selection = _fit_concepts(x, xv, train_labels, val_labels, concepts, seed)
    c, names = _concept_predictions(scaler, heads, x)
    cv, _ = _concept_predictions(scaler, heads, xv)
    y, yv = array(train_cache["y"]).astype(int), array(val_cache["y"]).astype(int)
    task = _fit_task(c, cv, y, yv, seed)
    b, bv = basic_reference_inputs(train_cache), basic_reference_inputs(val_cache)
    basic_scaler, basic_heads, basic_selection = _fit_concepts(b, bv, train_labels, val_labels, concepts, seed)
    pca_dimension = min(c.shape[1], x.shape[1], len(x)-1)
    pca = PCA(n_components=pca_dimension, svd_solver="full").fit(scaler.transform(x))
    pca_task = _fit_task(pca.transform(scaler.transform(x)), pca.transform(scaler.transform(xv)), y, yv, seed)
    return {"schema_version": "ot2-strict-cbm-readout-v1", "task": train_cache["task"], "seed": seed,
            "concept_schema": schema, "concept_schema_sha256": canonical_hash(schema),
            "source_identity": {k: v for k, v in train_cache["identity"].items() if k not in VARIABLE_IDENTITY_KEYS},
            "embedding_dim": x.shape[1], "embedding_scaler": scaler, "concept_models": heads,
            "concept_feature_names": names, "task_head": task, "threshold": task["threshold"],
            "concept_selection": selection, "strict_bottleneck": True, "direct_basic_path": False,
            "latent_bypass": False, "task_probability_distillation": False,
            "basic_references": {"concept_scaler": basic_scaler, "concept_models": basic_heads,
                                 "concept_selection": basic_selection, "task_head": _fit_task(b, bv, y, yv, seed)},
            "pca_reference": {"status": "fitted", "type": "unsupervised_train_only_pca_compression",
                              "scaler": scaler, "pca": pca, "dimension": pca_dimension, "task_head": pca_task},
            "true_concept_reference": _fit_true_reference(train_cache, val_cache, labels_df, schema, seed),
            "cohorts": {"train_ids_sha256": canonical_hash(train_cache["person_ids"]),
                        "val_ids_sha256": canonical_hash(val_cache["person_ids"]),
                        "n_train": len(y), "n_val": len(yv)},
            "limitations": ["Concept errors evaluated only where concept labels are observed; missing-label masking does not correct selection bias.",
                            "Validation selects concept regularization, task regularization and threshold; validation metrics are exploratory.",
                            "Frozen representations were task-trained on the training cohort; these are not fully out-of-fold features.",
                            "Soft concept predictions may encode information beyond their names; no causal interpretation is established."]}


def predict_readout(artifact: dict, embeddings: Any) -> dict:
    if artifact.get("schema_version") != "ot2-strict-cbm-readout-v1" or artifact.get("direct_basic_path") is not False or artifact.get("latent_bypass") is not False:
        raise ValueError("Artifact is not a strict CBM readout")
    x = _finite_matrix(embeddings, "prediction embeddings")
    if x.shape[1] != artifact["embedding_dim"]:
        raise ValueError("Prediction embedding dimension differs from trained readout")
    concepts, names = _concept_predictions(artifact["embedding_scaler"], artifact["concept_models"], x)
    if names != artifact["concept_feature_names"]:
        raise ValueError("Artifact concept order is inconsistent")
    task = artifact["task_head"]
    logits = task["model"].decision_function(task["scaler"].transform(concepts))
    return {"concepts": concepts, "logits": logits, "probabilities": expit(logits)}


def concept_metrics(concept_predictions, labels_df, schema, person_ids, *, model="strict_cbm", split="val") -> pd.DataFrame:
    concepts = validate_schema(schema)
    labels = ordered_labels(labels_df, person_ids, concepts)
    predictions = _finite_matrix(concept_predictions, "concept predictions")
    width = sum(1 if c["type"] == "continuous" else len(c["categories"]) for c in concepts)
    if predictions.shape != (len(person_ids), width):
        raise ValueError("Concept prediction matrix shape/order differs from schema")
    rows, offset = [], 0
    for c in concepts:
        y, present = _targets(labels, c)
        columns = 1 if c["type"] == "continuous" else len(c["categories"])
        p = predictions[present, offset:offset+columns]
        row = {"model": model, "split": split, "concept": c["name"], "type": c["type"],
               "unit": c.get("unit", ""), "n_total": len(labels), "n_observed": int(present.sum()),
               "evaluation_population": "observed_concept_labels_only", "status": "evaluated" if present.any() else "no_observed_labels"}
        if present.any():
            truth = y[present]
            if c["type"] == "continuous":
                row.update(mae=float(mean_absolute_error(truth, p[:, 0])), rmse=float(np.sqrt(mean_squared_error(truth, p[:, 0]))),
                           r2=float(r2_score(truth, p[:, 0])) if len(truth) >= 2 and np.var(truth) > 0 else np.nan,
                           predicted_min=float(p[:, 0].min()), predicted_max=float(p[:, 0].max()),
                           observed_min=float(truth.min()), observed_max=float(truth.max()))
            else:
                if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1):
                    raise ValueError("Categorical concept predictions must be normalized probabilities")
                row.update(log_loss=float(log_loss(truth, p, labels=np.arange(columns))), accuracy=float(np.mean(p.argmax(1) == truth)),
                           brier=float(np.mean(np.sum((p-np.eye(columns)[truth])**2, axis=1))),
                           observed_class_counts=json.dumps(np.bincount(truth, minlength=columns).tolist()))
                if columns == 2 and np.unique(truth).size == 2:
                    row.update(auroc=float(roc_auc_score(truth, p[:, 1])), auprc=float(average_precision_score(truth, p[:, 1])))
        rows.append(row)
        offset += columns
    return pd.DataFrame(rows)


def predict_basic_references(artifact: dict, cache: dict) -> dict:
    ref = artifact["basic_references"]
    b = basic_reference_inputs(cache)
    concepts, _ = _concept_predictions(ref["concept_scaler"], ref["concept_models"], b)
    task = ref["task_head"]
    logits = task["model"].decision_function(task["scaler"].transform(b))
    return {"concepts": concepts, "logits": logits, "probabilities": expit(logits), "threshold": task["threshold"]}


def predict_pca_reference(artifact: dict, embeddings) -> dict:
    ref = artifact["pca_reference"]
    z = ref["pca"].transform(ref["scaler"].transform(_finite_matrix(embeddings, "PCA embeddings")))
    task = ref["task_head"]
    logits = task["model"].decision_function(task["scaler"].transform(z))
    return {"logits": logits, "probabilities": expit(logits), "threshold": task["threshold"]}


def predict_true_reference(artifact: dict, labels_df, person_ids) -> dict:
    ref = artifact["true_concept_reference"]
    if ref["status"] != "fitted":
        raise ValueError(f"True-concept reference unavailable: {ref['reason']}")
    x, mask = true_concept_inputs(labels_df, artifact["concept_schema"], person_ids)
    task = ref["task_head"]
    logits = task["model"].decision_function(task["scaler"].transform(x)) if len(x) else np.empty(0)
    return {"logits": logits, "probabilities": expit(logits), "threshold": task["threshold"], "complete_case_mask": mask}


def predict_training_mean_concepts(labels_df: pd.DataFrame, schema: dict,
                                   train_ids: list[str], n: int) -> np.ndarray:
    concepts = validate_schema(schema)
    labels = ordered_labels(labels_df, train_ids, concepts)
    values = []
    for c in concepts:
        y, observed = _targets(labels, c)
        if not observed.any():
            raise ValueError(f"No training observations for {c['name']}")
        if c["type"] == "continuous":
            values.append(float(y[observed].mean()))
        else:
            values.extend((np.bincount(y[observed], minlength=len(c["categories"]))/observed.sum()).tolist())
    return np.repeat(np.asarray(values)[None, :], n, axis=0)


def explain_readout(artifact: dict, embeddings: Any) -> dict:
    prediction = predict_readout(artifact, embeddings)
    task = artifact["task_head"]
    centered = task["scaler"].transform(prediction["concepts"])
    contributions = centered * task["model"].coef_[0]
    intercept = float(task["model"].intercept_[0])
    if not np.allclose(intercept + contributions.sum(1), prediction["logits"], atol=1e-10, rtol=1e-10):
        raise ValueError("Concept contributions failed exact logit reconstruction")
    return {"names": artifact["concept_feature_names"], "contributions": contributions,
            "intercept": intercept, "logits": prediction["logits"],
            "reference_values": task["scaler"].mean_,
            "raw_concept_coefficients": task["model"].coef_[0] / task["scaler"].scale_}
