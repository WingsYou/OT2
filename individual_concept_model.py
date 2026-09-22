"""Joint CBM with individually supervised, heterogeneous concept items."""
import copy
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def numeric(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except ValueError:
        return np.nan


@lru_cache(maxsize=100000)
def date_days(value):
    # Local CSVs use M/D/YY H:MM; also accept ISO exports. A clinical name
    # containing "date" is not enough: eos_dsvst=1/2 remains categorical.
    try:
        if len(value) >= 10 and value[4] == '-' and value[7] == '-':
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        elif value.count('/') == 2:
            date, _, time = value.partition(' ')
            year = date.rsplit('/', 1)[1]
            if len(year) not in (2, 4): return np.nan
            fmt = '%m/%d/%y' if len(year) == 2 else '%m/%d/%Y'
            if time: fmt += ' %H:%M:%S' if time.count(':') == 2 else ' %H:%M'
            dt = datetime.strptime(value, fmt)
        else: return np.nan
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() / 86400
    except ValueError:
        return np.nan


def encode(values, item):
    values = np.asarray(values, str)
    if item['type'] == 'categorical':
        lookup = {v: i for i, v in enumerate(item['categories'])}
        y = np.array([lookup.get(v, -1) for v in values], dtype=np.float32)
        mask = y >= 0
    else:
        parse = date_days if item['encoding'] == 'date_days' else numeric
        y = np.array([parse(v) for v in values], dtype=np.float64)
        mask = np.isfinite(y)
        y = (y - item['mean']) / item['std']
    return np.where(mask, y, 0).astype(np.float32), mask


def prepare(train, validation, labels, schema):
    frame = labels.set_index('person_id')
    frames = [frame.reindex(list(map(str, c['person_ids']))).fillna('') for c in (train, validation)]
    x = np.asarray(train['embedding'], dtype=np.float32)
    mean, std = x.mean(0), x.std(0); std[std < 1e-8] = 1
    items, targets, masks, offset = [], [[], []], [[], []], 0
    for definition in schema['concepts']:
        item = copy.deepcopy(definition)
        values = frames[0][item['name']].to_numpy(str)
        observed = values[values != '']
        numbers = np.array([numeric(v) for v in observed])
        dates = np.array([date_days(v) for v in observed])
        categorical = bool(set(item['field_types']) & {'radio', 'checkbox', 'dropdown', 'yesno', 'truefalse'})
        item.update(mean=0., std=1., categories=[], encoding='raw_category')
        if len(observed) and not categorical and (np.isfinite(numbers).all() or np.isfinite(dates).all()):
            use_dates = not np.isfinite(numbers).all()
            y = dates if use_dates else numbers
            item.update(type='continuous', encoding='date_days' if use_dates else 'number',
                        mean=float(y.mean()), std=float(y.std()) if y.std() > 1e-8 else 1.)
            levels = len(np.unique(y)); width = 1
        else:
            categories = sorted(set(observed))
            item.update(type='categorical', categories=categories)
            levels = len(categories); width = max(1, levels)
        item.update(start=offset, stop=offset + width, n_train_observed=len(observed),
                    learnable=levels > 1, status='learned' if levels > 1 else 'constant' if levels else 'no_training_label')
        offset += width
        for k, f in enumerate(frames):
            y, present = encode(f[item['name']].to_numpy(str), item)
            targets[k].append(y); masks[k].append(present)
        items.append(item)
    if not any(c['learnable'] for c in items): raise ValueError('No varying concept labels in the training cohort')
    spec = {'items': items, 'width': offset, 'input_dim': x.shape[1], 'x_mean': mean.tolist(), 'x_std': std.tolist()}
    data = []
    for k, cache in enumerate((train, validation)):
        data.append({'x': torch.tensor((np.asarray(cache['embedding']) - mean) / std, dtype=torch.float32),
                     'y': torch.as_tensor(cache['y'], dtype=torch.float32),
                     'targets': torch.tensor(np.stack(targets[k], axis=1)),
                     'mask': torch.tensor(np.stack(masks[k], axis=1))})
    return spec, data


class IndividualCBM(nn.Module):
    def __init__(self, spec, hidden=32):
        super().__init__(); self.spec = spec
        self.concept_layer = nn.Linear(spec['input_dim'], spec['width'])
        self.task_head = nn.Sequential(nn.Linear(spec['width'], hidden), nn.ReLU(), nn.Linear(hidden, 1))
        active = np.zeros(spec['width'], np.float32); constant = active.copy()
        numeric_columns, numeric_items, categories = [], [], {}
        for j, c in enumerate(spec['items']):
            sl = slice(c['start'], c['stop'])
            if c['learnable']:
                active[sl] = 1
                if c['type'] == 'continuous': numeric_columns.append(c['start']); numeric_items.append(j)
                else: categories.setdefault(c['stop'] - c['start'], []).append(j)
            elif c['type'] == 'categorical' and c['categories']:
                constant[c['start']] = 1
        self.register_buffer('active', torch.tensor(active), persistent=False)
        self.register_buffer('constant', torch.tensor(constant), persistent=False)
        self.register_buffer('num_columns', torch.tensor(numeric_columns, dtype=torch.long), persistent=False)
        self.register_buffer('num_items', torch.tensor(numeric_items, dtype=torch.long), persistent=False)
        self.widths = sorted(categories)
        for width, js in categories.items():
            columns = [i for j in js for i in range(spec['items'][j]['start'], spec['items'][j]['stop'])]
            self.register_buffer(f'cat_columns_{width}', torch.tensor(columns), persistent=False)
            self.register_buffer(f'cat_items_{width}', torch.tensor(js), persistent=False)

    def bottleneck(self, x):
        raw = self.concept_layer(x)
        z = raw * self.active + self.constant
        for width in self.widths:
            cols = getattr(self, f'cat_columns_{width}')
            probs = raw[:, cols].reshape(len(x), -1, width).softmax(-1).flatten(1)
            z = z.index_copy(1, cols, probs)
        return raw, z

    def forward(self, x):
        raw, z = self.bottleneck(x)
        return raw, z, self.task_head(z).squeeze(-1)

    def losses(self, data):
        raw, z, logits = self(data['x'])
        losses = raw.new_full((len(self.spec['items']),), float('nan'))
        if self.num_items.numel():
            js = self.num_items; mask = data['mask'][:, js]
            per = ((z[:, self.num_columns] - data['targets'][:, js]).square() * mask).sum(0) / mask.sum(0).clamp_min(1)
            losses = losses.index_copy(0, js, torch.where(mask.any(0), per, float('nan')))
        for width in self.widths:
            js = getattr(self, f'cat_items_{width}'); cols = getattr(self, f'cat_columns_{width}')
            log_probs = raw[:, cols].reshape(len(raw), -1, width).log_softmax(-1)
            ce = -log_probs.gather(2, data['targets'][:, js].long().unsqueeze(-1)).squeeze(-1)
            mask = data['mask'][:, js]
            per = (ce * mask).sum(0) / mask.sum(0).clamp_min(1)
            losses = losses.index_copy(0, js, torch.where(mask.any(0), per, float('nan')))
        return losses, F.binary_cross_entropy_with_logits(logits, data['y'])


def state(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(train, val, labels, schema, cfg, device, masked_train=None):
    torch.manual_seed(cfg['seed'])
    spec, data = prepare(train, val, labels, schema)
    if masked_train is not None:
        masked = torch.tensor((np.asarray(masked_train) - np.asarray(spec['x_mean'])) / np.asarray(spec['x_std']), dtype=torch.float32)
        data[0] = {k: torch.cat((v, masked if k == 'x' else v)) for k, v in data[0].items()}
    data = [{k: v.to(device) for k, v in d.items()} for d in data]
    model = IndividualCBM(spec).to(device); history = []; artifacts = {}; reference = None
    phases = (('concept', cfg['concept_epochs']), ('head', cfg['head_epochs']), ('joint', cfg['joint_epochs']))
    selections = {}
    for phase, epochs in phases:
        model.concept_layer.requires_grad_(phase != 'head'); model.task_head.requires_grad_(phase != 'concept')
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.0001)
        best, best_score, best_epoch, stale = state(model), float('inf'), 0, 0
        for epoch in range(epochs + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                c, y = model.losses(data[0])
                objective = c.nanmean() if phase == 'concept' else y if phase == 'head' else y + c.nanmean()
                objective.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            with torch.no_grad(): c, y = model.losses(data[1])
            per = c.cpu().numpy(); mean_loss = float(c.nanmean())
            known = np.isfinite(per)
            eligible = phase != 'joint' or bool(np.all(per[known] <= reference[known] * 1.1 + 1e-8))
            score = mean_loss if phase == 'concept' else float(y)
            if not np.isfinite(score): raise ValueError('No finite validation objective; inspect concept label coverage')
            history.append({'phase': phase, 'epoch': epoch, 'val_task_loss': float(y),
                            'val_concept_loss': mean_loss, 'eligible': eligible})
            if eligible and score < best_score - 1e-7:
                best, best_score, best_epoch, stale = state(model), score, epoch, 0
            else: stale += 1
            if epoch % 25 == 0 or stale >= cfg['concept_patience']:
                print(f'[CBM] {phase} epoch={epoch} val_task={float(y):.4f} val_concept={mean_loss:.4f} gate={eligible}', flush=True)
            if stale >= cfg['concept_patience']: break
        model.load_state_dict(best)
        selections[phase] = {'epoch': best_epoch, 'score': best_score,
                             'status': 'sequential_fallback' if phase == 'joint' and best_epoch == 0 else 'fitted'}
        if phase == 'head':
            artifacts['sequential'] = state(model)
            with torch.no_grad(): reference = model.losses(data[1])[0].cpu().numpy()
    artifacts['joint'] = state(model)
    return {'spec': spec, 'states': artifacts, 'selection': selections, 'backbone_frozen': True,
            'training_views': ['full', 'masked'] if masked_train is not None else ['basic'],
            'lambda': 1., 'concept_loss_tolerance': .1}, history


def restore(artifact, device, arm='joint'):
    model = IndividualCBM(artifact['spec']).to(device)
    model.load_state_dict(artifact['states'][arm]); model.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def predict(model, embedding):
    x = (np.asarray(embedding) - np.asarray(model.spec['x_mean'])) / np.asarray(model.spec['x_std'])
    _, z, logits = model(torch.tensor(x, dtype=torch.float32, device=next(model.parameters()).device))
    return z, logits.sigmoid().cpu().numpy()
