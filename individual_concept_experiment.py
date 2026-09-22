"""Cached CBM fitting, single-item restoration, and validation/test summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import (TASK_LABELS, METRICS, METRIC_LABELS, FOCAL, Bootstrap, effect_rows,
                    threshold, read_json, write_json, save_torch)
from full_clinical_cbm import feature_caches, load
from individual_concept_model import fit, restore, predict, encode


def metric_batch(boot, probabilities, cut):
    """Paired bootstrap for several interventions in one GPU operation (K,S,3)."""
    probabilities = np.asarray(probabilities)
    order = np.argsort(probabilities, axis=1, kind='stable')
    low, high = [], []
    for p, idx in zip(probabilities, order):
        s = p[idx]; start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        end = np.r_[start[1:] - 1, len(s) - 1]; repeats = end - start + 1
        low.append(np.repeat(start, repeats)); high.append(np.repeat(end, repeats))
    device = boot.device
    order = torch.tensor(order, device=device)
    weights = boot.w[:, order].permute(1, 0, 2)
    y = torch.tensor(boot.y, device=device)[order]
    negative = (weights * (1 - y[:, None])).cumsum(2)
    before = torch.cat((negative[:, :, :1] * 0, negative[:, :, :-1]), dim=2)
    lo = torch.tensor(np.array(low), device=device)[:, None].expand(-1, len(boot.w), -1)
    hi = torch.tensor(np.array(high), device=device)[:, None].expand_as(lo)
    mid = (before.gather(2, lo).double() + negative.gather(2, hi).double()) * .5
    auc = (weights * y[:, None] * mid).sum(2) / (boot.pos * boot.neg)
    pred = torch.tensor(probabilities >= cut, device=device)
    truth = torch.tensor(boot.y, device=device)
    tp = ((pred & (truth == 1)).double() @ boot.fw.T)
    fp = ((pred & (truth == 0)).double() @ boot.fw.T)
    ba = .5 * (tp / boot.pos + (boot.neg - fp) / boot.neg)
    f1 = 2 * tp / (boot.pos + tp + fp).clamp_min(1)
    return torch.stack((auc, ba, f1), dim=2).cpu().numpy()


@torch.no_grad()
def restoration_batch(model, missing, full, groups):
    """Use the additive first head layer; no K-fold copies of concept vectors."""
    first, _, last = model.task_head
    base = first(missing)
    values = []
    for columns in groups:
        if len(columns) == model.spec['width']:
            values.append(model.task_head(full).squeeze(-1).sigmoid())
            continue
        delta = (full[:, columns] - missing[:, columns]) @ first.weight[:, columns].T
        values.append(last(torch.relu(base + delta)).squeeze(-1).sigmoid())
    return torch.stack(values).cpu().numpy()


def concept_metrics(task, split, model, z, labels, ids):
    frame = labels.set_index('person_id').reindex(list(map(str, ids))).fillna('')
    z = z.cpu().numpy(); rows = []
    for c in model.spec['items']:
        values = frame[c['name']].to_numpy(str); y, mask = encode(values, c)
        row = {k: c[k] for k in ('name', 'concept_id', 'concept_name', 'source_field', 'type', 'status', 'unit', 'encoding')}
        row.update(task=task, split=split, n_recorded=int((values != '').sum()), n_supported=int(mask.sum()),
                   n_unsupported=int(((values != '') & ~mask).sum()), mae=np.nan, r2=np.nan,
                   accuracy=np.nan, balanced_accuracy=np.nan)
        if c['encoding'] == 'date_days': row['unit'] = 'days'
        if mask.any() and c['status'] != 'no_training_label':
            target = y[mask]; p = z[mask, c['start']:c['stop']]
            if c['type'] == 'continuous':
                target = target * c['std'] + c['mean']; p = p[:, 0] * c['std'] + c['mean']
                denom = ((target - target.mean()) ** 2).sum()
                row.update(mae=float(np.abs(target - p).mean()), r2=float(1 - ((target - p) ** 2).sum() / denom) if denom else np.nan)
            else:
                p = p.argmax(1)
                row.update(accuracy=float((p == target).mean()),
                           balanced_accuracy=float(np.mean([(p[target == k] == k).mean() for k in np.unique(target)])))
        rows.append(row)
    return rows


def run_task(task, cfg, device):
    mode = cfg['individual_mode']; output = Path(cfg['individual_root'])
    root = output / mode / task; root.mkdir(parents=True, exist_ok=True)
    if (root / 'DONE.json').exists(): print('[reuse] ' + mode + '/' + task, flush=True); return
    source = Path(cfg['output_root'])
    if mode == 'mixed':
        cache_root = Path(cfg.get('full_cbm_root', source / 'full_clinical_cbm')) / task
        cache_root.mkdir(parents=True, exist_ok=True)
        paths, masked_paths = feature_caches(task, cfg, device, cache_root, include_masked_train=True)
    else:
        paths = {s: source / task / 'basic' / f'features_{s}.pt' for s in ('train', 'val', 'test')}
    label_dir = output / 'labels' / task
    labels = pd.read_csv(label_dir / 'labels.csv', dtype=str, keep_default_na=False)
    schema = read_json(label_dir / 'schema.json')
    # These small readouts use FP32 so incremental head calculations do not
    # introduce TF32 rounding differences between restoration conditions.
    torch.set_float32_matmul_precision('highest')
    train, val = load(paths['train']), load(paths['val'])
    validation = {**val, 'embedding': load(masked_paths['val'])['embedding']} if mode == 'mixed' else val
    if (root / 'model.pt').exists(): artifact = load(root / 'model.pt')
    else:
        print(f'[fit] {task}/{mode}: {len(schema["concepts"])} individual items', flush=True)
        artifact, log = fit(train, validation, labels, schema, cfg, device,
                            load(masked_paths['train'])['embedding'] if mode == 'mixed' else None)
        save_torch(root / 'model.pt', artifact); pd.DataFrame(log).to_csv(root / 'training.csv', index=False)
    model = restore(artifact, device)
    _, val_p = predict(model, validation['embedding']); cut = threshold(val['y'].numpy(), val_p)
    original_val = load(masked_paths['val']) if mode == 'mixed' else val
    original_cut = threshold(val['y'].numpy(), original_val['original_logits'].sigmoid().numpy())
    items = model.spec['items']; columns = {c['name']: list(range(c['start'], c['stop'])) for c in items}
    write_json(root / 'selection.json', {'mode': mode, 'threshold': cut, 'threshold_source': 'masked validation' if mode == 'mixed' else 'basic validation',
                'fit': artifact['selection'], 'inventory_ids': schema['inventory_ids'], 'source_pairs': schema['source_pairs'],
                'individual_items': len(items), 'learned_items': sum(c['learnable'] for c in items),
                'constant_items': sum(c['status'] == 'constant' for c in items),
                'no_training_label': sum(c['status'] == 'no_training_label' for c in items),
                'bottleneck_channels': model.spec['width'], 'restoration': 'same-participant unperturbed predictions'})
    # Raw-category labels and train-fitted encodings remain in local result files.
    write_json(root / 'concept_definitions.json', items)
    rows, effects, scores, ranking = [], [], [], {}
    for split in ('val', 'test'):
        cache = val if split == 'val' else load(paths[split]); y = cache['y'].numpy()
        full_z, full_p = predict(model, cache['embedding'])
        if mode == 'mixed':
            masked = load(masked_paths[split])
            removed = {'nonbasic_clinical': {**masked, 'eligible': torch.ones(len(y), dtype=torch.bool)}}
            eval_cache = masked
        else:
            removed = load(source / task / 'basic' / f'removals_{split}.pt'); eval_cache = cache
        eval_z, eval_p = predict(model, eval_cache['embedding'])
        boot = Bootstrap(y, cfg['bootstrap_samples'], cfg['seed'], device)
        original_p = eval_cache['original_logits'].sigmoid().numpy()
        for name, p, threshold_value in (('original_classifier', original_p, original_cut),
                ('cbm_deployment', eval_p, cut), ('cbm_unperturbed_reference', full_p, cut)):
            rows += effect_rows(task, split, 'classification', boot.metrics(p, threshold_value), model=name, n=len(y), threshold=threshold_value)
        scores += concept_metrics(task, split, model, eval_z, labels, cache['person_ids'])
        pd.DataFrame({'person_id': cache['person_ids'], 'y': y, 'original': original_p,
                      'cbm': eval_p, 'unperturbed_cbm': full_p}).to_csv(root / f'predictions_{split}.csv', index=False)
        np.savez_compressed(root / f'concept_predictions_{split}.npz', person_ids=np.array(cache['person_ids']),
                            unperturbed=full_z.cpu().numpy(), deployment=eval_z.cpu().numpy())
        restoration_arrays = {}
        for source_name, masked in removed.items():
            eligible = np.asarray(masked['eligible'], bool)
            if len(np.unique(y[eligible])) < 2:
                print(f'[skip] {source_name}/{split}: eligible cohort lacks both classes', flush=True); continue
            sub = Bootstrap(y[eligible], cfg['bootstrap_samples'], cfg['seed'], device)
            missing_z, missing_p = predict(model, masked['embedding'])
            baseline = sub.metrics(missing_p[eligible], cut); reference = sub.metrics(full_p[eligible], cut)
            names = list(columns) if split == 'val' else ranking.get(source_name, [])
            interventions = [(name, columns[name]) for name in names]
            interventions += [('ALL_ITEMS', list(range(model.spec['width'])))]
            source_effects = []
            for start in range(0, len(interventions), 16):
                block = interventions[start:start + 16]
                predictions = restoration_batch(model, missing_z, full_z, [cols for _, cols in block])
                draws = metric_batch(sub, predictions[:, eligible], cut)
                for (name, _), p, restored in zip(block, predictions, draws):
                    part = effect_rows(task, split, 'recovery', restored - baseline, removed_input=source_name,
                                       concept=name, n=int(eligible.sum()), threshold=cut)
                    for j, row in enumerate(part):
                        row.update(full_metric=float(reference[0, j]), masked_metric=float(baseline[0, j]),
                                   restored_metric=float(restored[0, j]), initial_drop=float(reference[0, j] - baseline[0, j]))
                    source_effects += part; restoration_arrays[source_name + '__' + name] = p
                if start % 128 == 0: print(f'[restore] {task}/{split}/{source_name}: {min(start+16,len(interventions))}/{len(interventions)}', flush=True)
            if split == 'val':
                candidates = [r for r in source_effects if r['metric'] == 'auroc' and r['concept'] != 'ALL_ITEMS'
                              and next(c['learnable'] for c in items if c['name'] == r['concept'])]
                ranking[source_name] = [r['concept'] for r in sorted(candidates, key=lambda r: (-r['value'], r['concept']))[:3]]
            chosen = ranking.get(source_name, [])
            if chosen:
                selected_columns = [i for name in chosen for i in columns[name]]
                p = restoration_batch(model, missing_z, full_z, [selected_columns])[0]
                restored = sub.metrics(p[eligible], cut)
                part = effect_rows(task, split, 'recovery', restored - baseline, removed_input=source_name,
                                   concept='TOP3_JOINT', n=int(eligible.sum()), threshold=cut)
                for j, row in enumerate(part):
                    row.update(full_metric=float(reference[0, j]), masked_metric=float(baseline[0, j]),
                               restored_metric=float(restored[0, j]), initial_drop=float(reference[0, j] - baseline[0, j]))
                source_effects += part; restoration_arrays[source_name + '__TOP3_JOINT'] = p
            effects += source_effects
        if split == 'val': write_json(root / 'VAL_TOP3.json', ranking)
        np.savez_compressed(root / f'restored_predictions_{split}.npz', **restoration_arrays)
    for name, records in (('classification', rows), ('restoration', effects), ('concept_metrics', scores)):
        pd.DataFrame(records).to_csv(root / (name + '.csv'), index=False)
    write_json(root / 'DONE.json', {'task': task, 'mode': mode, 'splits': ['val', 'test']})


def make_report(cfg):
    from report import markdown, number, interval
    output = Path(cfg['individual_root']); tables = output / 'tables'; tables.mkdir(exist_ok=True)
    for mode in cfg['individual_modes']:
        completed = [t for t in cfg['tasks'] if (output / mode / t / 'DONE.json').exists()]
        if not completed: continue
        frames = {name: pd.concat([pd.read_csv(output / mode / t / (name + '.csv')) for t in completed], ignore_index=True)
                  for name in ('classification', 'restoration', 'concept_metrics')}
        for split in ('val', 'test'):
            for name, frame in frames.items(): frame[frame.split == split].to_csv(tables / f'{mode}_{name}_{split}.csv', index=False)
            lines = [f'# Individual-concept CBM ({mode}) — {split}', '',
                f'Completed tasks: {len(completed)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}.', '',
                'Concepts follow TASK_SPECIFIC_CONCEPT_SUBSETS.xlsx, with only the approved task-specific outcome exclusions. '
                'Administrative fields, dates, identifiers and workflow items remain. No form pooling or clinical concept-group aggregation is used.', '',
                ('One frozen Full Clinical backbone; one CBM trained on equal-weight full/masked views. Selection and deployment use masked inputs.' if mode == 'mixed' else
                 'The trained Basic Clinical backbone is frozen; a new CBM is fitted on its cached embeddings.'), '',
                'Concept warmup → classifier warmup → joint training (lambda=1), AdamW; a 10% per-concept validation-loss constraint retains the sequential fallback when needed. '
                'The classification head receives predicted concepts only. Encodings are fitted on training participants.', '', '## Concept coverage', '']
            coverage = []
            for task in completed:
                s = read_json(output / mode / task / 'selection.json')
                coverage.append([TASK_LABELS[task], *[s[k] for k in ('inventory_ids', 'source_pairs', 'individual_items', 'learned_items', 'constant_items', 'no_training_label', 'bottleneck_channels')], s['fit']['joint']['status']])
            lines += [markdown(['Task','IDs','ID/source pairs','Items incl. contexts','Learned','Constant','No train label','Channels','Joint selection'], coverage), '',
                      'Constants stay fixed; items without training labels stay inactive. Neither can act as an unconstrained latent channel. '
                      'Categorical items use a probability block, so channel count is not concept count. '
                      'Detailed definitions, raw-label coverage, conflicts and per-split supported/unseen labels are saved in the per-task files.', '', '## Classification', '']
            classification = frames['classification']; restoration = frames['restoration']
            for metric in METRICS:
                rows = []
                for task in completed:
                    f = classification[(classification.task == task) & (classification.split == split) & (classification.metric == metric)]
                    rows.append([TASK_LABELS[task], *[number(f[f.model == arm].value.iloc[0]) for arm in
                                 ('original_classifier', 'cbm_deployment', 'cbm_unperturbed_reference')]])
                lines += ['**' + METRIC_LABELS[metric] + '**', '', markdown(['Task','Original, deployment input','CBM, deployment input','Same CBM, unperturbed reference'], rows), '']
            lines += ['## Single-concept restoration', '',
                      'All individual items are screened on validation. For each removed input, the three highest validation AUROC recoveries '
                      'are fixed before test evaluation; their joint restoration and all-item restoration are also evaluated. Negative effects are retained.', '']
            if mode == 'basic':
                lines += ['This summary shows each task’s previously specified focal modality removal. '
                          'The CSV files include every single/family removal.', '']
            for task in completed:
                ranking = read_json(output / mode / task / 'VAL_TOP3.json')
                definitions = {c['name']: c for c in read_json(output / mode / task / 'concept_definitions.json')}
                rows = []
                for source_name, top in ranking.items():
                    if mode == 'basic' and source_name != FOCAL[task]: continue
                    for rank, concept in enumerate([*top, 'TOP3_JOINT', 'ALL_ITEMS'], 1):
                        f = restoration[(restoration.task == task) & (restoration.split == split) &
                                        (restoration.removed_input == source_name) & (restoration.concept == concept)]
                        if f.empty: continue
                        d = definitions.get(concept)
                        label = (d['concept_name'] + ' — ' + d['source_field'] + ' (ID ' + d['concept_id'] + ')') if d else concept
                        rows.append([source_name, rank if concept in top else '—', label,
                                     *[interval(f[f.metric == m].iloc[0]) for m in METRICS]])
                lines += ['### ' + TASK_LABELS[task], '', markdown(['Removed input','Validation rank','Restored item',
                    *[METRIC_LABELS[m] + ' recovery [95% CI]' for m in METRICS]], rows), '']
            lines += ['Restoration substitutes the same participant’s unperturbed **predicted** channel(s); it is a diagnostic probe, '
                'not measured-concept injection or autonomous imputation. All-item restoration reproduces the unperturbed decision by construction. '
                'Paired intervals use the same outcome-stratified participant draws, without multiple-comparison adjustment; '
                'validation ranks are exploratory and earlier test inspection/training-seed uncertainty is not included.', '',
                'All retained items are data-dictionary entries, not necessarily clinically meaningful states. Occurrence-only Procedure records '
                'are labeled as recorded occurrences, never invented sensation scores; absence is missing, not a negative label. '
                'Dates (local M/D/YY H:MM or ISO format) use days since the Unix epoch, numeric targets are standardized, and raw categorical values (including recorded refusal codes) '
                'are retained. Unsupported test categories are counted separately, not converted into a known class.']
            (tables / f'SUMMARY_INDIVIDUAL_{mode}_{split}.md').write_text('\n'.join(lines) + '\n')
    print('[reports] ' + str(tables), flush=True)
