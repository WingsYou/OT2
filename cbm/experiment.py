"""Fit the shared Mixed CBM on a frozen Full Clinical backbone."""
from pathlib import Path
import pandas as pd
import torch
from common import read_json, write_json, save_torch, backbone_dir
from train import load
from cbm.model import fit


def run_cbm(task, cfg, device):
    task_root = Path(cfg['output_root'])/task
    root = task_root/'mixed_cbm'; root.mkdir(parents=True, exist_ok=True)
    if (root/'DONE.json').exists(): print(f'[reuse] {task}/mixed_cbm', flush=True); return
    source = backbone_dir(task,cfg)/'full'
    train = load(source/'features_train.pt'); val = load(source/'features_val.pt')
    labels = pd.read_csv(Path(cfg['output_root'])/'labels'/task/'labels.csv', dtype=str, keep_default_na=False)
    schema = read_json(Path(cfg['output_root'])/'labels'/task/'schema.json')
    validation = {**val, 'embedding':val['masked_embedding']}
    torch.set_float32_matmul_precision('highest')
    artifact, history = fit(train, validation, labels, schema, cfg, device,
                            masked_train=train['masked_embedding'])
    save_torch(root/'model.pt', artifact)
    pd.DataFrame(history).to_csv(root/'training.csv', index=False)
    write_json(root/'concept_definitions.json', artifact['spec']['items'])
    write_json(root/'DONE.json', dict(task=task, mode='mixed', threshold=artifact['threshold'],
        selection=artifact['selection'], inventory_ids=schema['inventory_ids'],
        learned_items=sum(c['learnable'] for c in artifact['spec']['items']),
        input_backbone='full', backbone_frozen=True))


def concept_id_groups(items):
    groups = {}
    for c in items:
        cid = str(c['concept_id'])
        g = groups.setdefault(cid, dict(concept_id=cid, concept_name=c['concept_name'],
            source_fields=[], columns=[], learnable=False))
        if c['source_field'] not in g['source_fields']: g['source_fields'].append(c['source_field'])
        g['columns'].extend(range(c['start'], c['stop']))
        g['learnable'] |= c['learnable']
    return groups


@torch.no_grad()
def restoration_batch(model, missing, full, columns):
    """Restore a whole ID efficiently through the additive first head layer."""
    first, _, last = model.task_head; base = first(missing)
    values = []
    for cols in columns:
        if len(cols) == model.spec['width']:
            values.append(model.task_head(full).squeeze(-1).sigmoid())
            continue
        delta = (full[:,cols]-missing[:,cols]) @ first.weight[:,cols].T
        values.append(last(torch.relu(base+delta)).squeeze(-1).sigmoid())
    return torch.stack(values).cpu().numpy()
