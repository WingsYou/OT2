"""Joint concept/readout training on frozen embeddings; labels are supervision only."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cbm_readout_v1 import _targets, array, ordered_labels, validate_schema

DEFAULTS = {"seed": 42, "concept_epochs": 400, "head_epochs": 300, "joint_epochs": 300,
    "patience": 40, "learning_rate": 0.001, "weight_decay": 0.0001,
    "hidden_width": 32, "lambdas": [1.0, 10.0], "concept_loss_tolerance": 0.10}


def prepare(train, val, labels, schema):
    # This readout uses embeddings/labels only, for either Full or Basic Clinical.
    # The older readout's cache contract requires exactly 23 clinical columns.
    concepts = validate_schema(schema)
    frames = [ordered_labels(labels, c["person_ids"], concepts) for c in (train, val)]
    x = array(train["embedding"]).astype(np.float32)
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-8, std, 1.)
    target_mean, target_std, targets, masks, layout = [], [], [[], []], [[], []], []
    offset = 0
    for c in concepts:
        pairs = [_targets(frame, c) for frame in frames]
        y, present = pairs[0]
        if present.sum() < 2 or len(np.unique(y[present])) < 2 or not pairs[1][1].any():
            raise ValueError(f"Insufficient train/validation labels for {c['name']}; review definition, do not auto-drop")
        width = 1 if c["type"] == "continuous" else len(c["categories"])
        if c["type"] == "categorical" and set(y[present]) != set(range(width)):
            raise ValueError(f"A training category is absent for {c['name']}")
        mu, sd = (float(y[present].mean()), float(y[present].std())) if width == 1 else (0., 1.)
        target_mean.append(mu); target_std.append(sd)
        layout.append({"name": c["name"], "start": offset, "stop": offset+width, "type": c["type"]})
        offset += width
        for split, (values, mask) in enumerate(pairs):
            standardized = (values-mu)/sd if width == 1 else values
            targets[split].append(np.where(mask, standardized, 0).astype(np.float32))
            masks[split].append(mask)
    spec = {"input_dim": x.shape[1], "width": offset, "layout": layout,
        "x_mean": mean.tolist(), "x_std": std.tolist(),
        "target_mean": target_mean, "target_std": target_std, "concepts": concepts}
    data = []
    for k, cache in enumerate((train, val)):
        data.append({"x": torch.tensor((array(cache["embedding"])-mean)/std, dtype=torch.float32).detach(),
            "y": torch.tensor(array(cache["y"]), dtype=torch.float32),
            "targets": torch.tensor(np.stack(targets[k], axis=1)),
            "masks": torch.tensor(np.stack(masks[k], axis=1), dtype=torch.bool)})
    return spec, data


class JointCBM(nn.Module):
    def __init__(self, spec, hidden_width=32):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.concept_layer = nn.Linear(spec["input_dim"], spec["width"])
        self.task_head = nn.Sequential(nn.Linear(spec["width"], hidden_width), nn.ReLU(), nn.Linear(hidden_width, 1))
        # The head receives train-true-standardized numerical values and unscaled
        # probability blocks. No fitted scale based on compressed predictions.

    def bottleneck(self, x):
        raw = self.concept_layer(x)
        blocks = []
        for j, c in enumerate(self.spec["layout"]):
            z = raw[:, c["start"]:c["stop"]]
            if c["type"] == "categorical":
                z = torch.softmax(z, dim=1)
            elif c["name"] == "cesd10_total":
                z = (30*torch.sigmoid(z)-self.spec["target_mean"][j])/self.spec["target_std"][j]
            blocks.append(z)
        return raw, torch.cat(blocks, dim=1)

    def classify(self, concepts):
        return self.task_head(concepts).squeeze(-1)

    def forward(self, x):
        raw, concepts = self.bottleneck(x)
        return {"raw": raw, "concepts": concepts, "logits": self.classify(concepts)}


def losses(model, output, data):
    per_concept = []
    for j, c in enumerate(model.spec["layout"]):
        mask = data["masks"][:, j]
        if not mask.any():
            raise ValueError(f"No supervised values for {c['name']}")
        sl = slice(c["start"], c["stop"])
        if c["type"] == "continuous":
            loss = F.mse_loss(output["concepts"][mask, sl].squeeze(1), data["targets"][mask, j])
        else:
            loss = F.cross_entropy(output["raw"][mask, sl], data["targets"][mask, j].long())
        per_concept.append(loss)
    return torch.stack(per_concept), F.binary_cross_entropy_with_logits(output["logits"], data["y"])


def quality_gate(current, reference, tolerance):
    current, reference = np.asarray(current), np.asarray(reference)
    return bool(np.isfinite(current).all() and np.all(current <= reference*(1+tolerance)+1e-8))


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_phase(model, train, val, *, phase, config, strength=0., reference=None):
    if phase not in ("concept", "head", "joint"):
        raise ValueError(phase)
    model.concept_layer.requires_grad_(phase != "head")
    model.task_head.requires_grad_(phase != "concept")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config[{"concept": "concept_epochs", "head": "head_epochs", "joint": "joint_epochs"}[phase]]
    best, best_score, best_epoch, stalled, history = _state(model), float("inf"), 0, 0, []
    # The sequential starting model is a fallback, not a successful joint update.
    for epoch in range(epochs+1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            c_loss, y_loss = losses(model, model(train["x"]), train)
            objective = c_loss.mean() if phase == "concept" else y_loss if phase == "head" else y_loss+strength*c_loss.mean()
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite training loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            c_loss, y_loss = losses(model, model(val["x"]), val)
        per = c_loss.cpu().numpy()
        eligible = phase != "joint" or quality_gate(per, reference, config["concept_loss_tolerance"])
        score = float(c_loss.mean()) if phase == "concept" else float(y_loss)
        history.append({"phase": phase, "lambda": strength, "epoch": epoch,
            "val_task_log_loss": float(y_loss), "val_mean_concept_loss": float(c_loss.mean()),
            "per_concept_loss": per.tolist(), "eligible": eligible})
        if eligible and score < best_score-1e-7:
            best, best_score, best_epoch, stalled = _state(model), score, epoch, 0
        else:
            stalled += 1
        if epoch and (epoch % 50 == 0 or stalled >= config["patience"]):
            print(f"  {phase} lambda={strength:g} epoch={epoch} task_loss={float(y_loss):.4f} concept_loss={float(c_loss.mean()):.4f} gate={eligible}", flush=True)
        if stalled >= config["patience"]:
            break
    if not np.isfinite(best_score):
        raise ValueError("No eligible checkpoint, including initial model")
    model.load_state_dict(best)
    model.requires_grad_(False)
    model.eval()
    return {"best_epoch": best_epoch, "epochs_run": epoch,
        "selected_score": best_score, "status": "no_accepted_joint_update" if phase == "joint" and best_epoch == 0 else "fitted"}, history


def fit_experiment(train, val, labels, schema, *, config=None, device="cpu"):
    cfg = dict(DEFAULTS if config is None else config)
    if set(cfg) != set(DEFAULTS) or cfg["hidden_width"] < 1 or cfg["patience"] < 1:
        raise ValueError("Invalid training contract")
    if any(cfg[k] < 1 for k in ("concept_epochs", "head_epochs", "joint_epochs")) or not 0 <= cfg["concept_loss_tolerance"] <= 1:
        raise ValueError("Invalid epoch budget or concept tolerance")
    if len(set(cfg["lambdas"])) != len(cfg["lambdas"]) or not cfg["lambdas"] or any(v <= 0 for v in cfg["lambdas"]):
        raise ValueError("Positive unique concept-loss weights required")
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    spec, data = prepare(train, val, labels, schema)
    data = [{k: v.to(device) for k, v in split.items()} for split in data]
    model = JointCBM(spec, cfg["hidden_width"]).to(device)
    concept_info, history = fit_phase(model, *data, phase="concept", config=cfg)
    head_info, log = fit_phase(model, *data, phase="head", config=cfg)
    history += log
    with torch.no_grad():
        reference = losses(model, model(data[1]["x"]), data[1])[0].cpu().numpy()
    common = {"schema_version": "ot2-cbm-joint-v2", "spec": spec, "config": cfg,
        "task": train["task"], "concept_schema": schema, "strict_bottleneck": True,
        "backbone_frozen": True, "direct_basic_path": False, "latent_bypass": False,
        "concept_warmup": concept_info, "head_warmup": head_info,
        "reference_val_concept_losses": reference.tolist()}
    artifacts = {"sequential": {**common, "state_dict": _state(model), "arm": "sequential",
        "selection": head_info, "lambda": None}}
    initial = _state(model)
    for strength in cfg["lambdas"]:
        model.load_state_dict(initial)
        selection, log = fit_phase(model, *data, phase="joint", config=cfg, strength=strength, reference=reference)
        arm = "joint_lambda_"+format(strength, "g")
        artifacts[arm] = {**common, "state_dict": _state(model), "arm": arm, "lambda": strength, "selection": selection}
        history += log
    return artifacts, history


def restore(artifact, device="cpu"):
    if artifact.get("schema_version") != "ot2-cbm-joint-v2" or not artifact.get("strict_bottleneck") or artifact.get("latent_bypass") or artifact.get("direct_basic_path"):
        raise ValueError("Unsupported joint artifact / bypass")
    model = JointCBM(artifact["spec"], artifact["config"]["hidden_width"]).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def predict(model, embedding):
    spec = model.spec
    x = (array(embedding).astype(np.float32)-np.asarray(spec["x_mean"], np.float32))/np.asarray(spec["x_std"], np.float32)
    device = next(model.parameters()).device
    output = model(torch.tensor(x, dtype=torch.float32, device=device))
    result = {k: v.cpu().numpy() for k, v in output.items()}
    result["probabilities"] = torch.sigmoid(output["logits"]).cpu().numpy()
    result["clinical_concepts"] = result["concepts"].astype(np.float64)
    for j, c in enumerate(spec["layout"]):
        if c["type"] == "continuous":
            result["clinical_concepts"][:, c["start"]] = result["concepts"][:, c["start"]]*spec["target_std"][j]+spec["target_mean"][j]
        else:
            block = result["clinical_concepts"][:, c["start"]:c["stop"]]
            block /= block.sum(axis=1, keepdims=True)
    return result


@torch.no_grad()
def classify(model, concepts):
    return torch.sigmoid(model.classify(torch.tensor(concepts, dtype=torch.float32,
        device=next(model.parameters()).device))).cpu().numpy()


def measured_inputs(model, labels, ids):
    frame = ordered_labels(labels, ids, model.spec["concepts"])
    values = np.zeros((len(ids), model.spec["width"]), dtype=np.float32)
    observed = {}
    for j, c in enumerate(model.spec["concepts"]):
        target, mask = _targets(frame, c)
        layout = model.spec["layout"][j]
        observed[c["name"]] = mask
        if c["type"] == "continuous":
            values[mask, layout["start"]] = (target[mask]-model.spec["target_mean"][j])/model.spec["target_std"][j]
        else:
            values[np.flatnonzero(mask), layout["start"]+target[mask].astype(int)] = 1.
    return values, observed


def intervention_groups(schema):
    from cbm_semantics_v1 import concept_groups
    groups = concept_groups(schema)
    if "cesd10_total" in schema["concept_names"]:
        # Do not replace a component while holding its overlapping total fixed.
        groups.pop("restless_sleep", None)
        groups.pop("cesd10_total", None)
        groups.pop("activity_and_sleep", None)
        groups["cesd10_and_sleep"] = ("restless_sleep", "cesd10_total")
    return groups


def replace_group(model, predicted, replacement, observed, members):
    members = set(members)
    available = {c["name"] for c in model.spec["layout"]}
    if not members or not members.issubset(available):
        raise ValueError("Unknown/empty concept intervention")
    overlap = {"cesd10_total", "restless_sleep"}
    if overlap.issubset(available) and members & overlap and not overlap.issubset(members):
        raise ValueError("CES-D total and its sleep item must be changed together")
    mask = np.logical_and.reduce([observed[m] for m in members])
    result = np.asarray(predicted).copy()
    for c in model.spec["layout"]:
        if c["name"] in members:
            result[mask, c["start"]:c["stop"]] = replacement[mask, c["start"]:c["stop"]]
    return result, mask
