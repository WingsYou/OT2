"""Frozen-model analyses with exploratory within-split Concept-ID rankings."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import REMOVALS, read_json, write_json
from train import load
from stats import PairedF1, f1
from cbm.model import restore, predict
from cbm.experiment import concept_id_groups, restoration_batch


def run_analysis(task, cfg, device):
    root = Path(cfg['output_root'])/task
    out = root/'analysis'; out.mkdir(exist_ok=True)
    if (out/'DONE.json').exists() and read_json(out/'DONE.json').get('restoration_version') == 2:
        add_balanced_accuracy(task, cfg, device)
        print(f'[reuse analysis] {task}', flush=True); return
    artifacts = {mode:load(root/(mode+'_cbm')/'model.pt') for mode in ('basic','mixed')}
    models = {mode:restore(a, device) for mode,a in artifacts.items()}
    groups = {mode:concept_id_groups(a['spec']['items']) for mode,a in artifacts.items()}
    cuts = {k:read_json(root/k/'TRAINING_DONE.json')['threshold'] for k in ('full','basic')}
    cuts.update({m+'_cbm':a['threshold'] for m,a in artifacts.items()})
    torch.set_float32_matmul_precision('highest')
    classification, comparisons, ablations, recoveries, cohorts = [], [], [], [], []
    selection = dict(metric='positive_class_f1', selection_split='val', top_modalities=[], concepts={})
    for split in ('val','test'):
        basic = load(root/'basic'/f'features_{split}.pt'); full = load(root/'full'/f'features_{split}.pt')
        if basic['person_ids'] != full['person_ids'] or not torch.equal(basic['y'], full['y']):
            raise ValueError(f'{task}/{split}: Full and Basic predictions must use the same ordered participants')
        y = basic['y'].numpy(); n = len(y)
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        removed = load(root/'basic'/f'removals_{split}.pt')
        bz, bp = predict(models['basic'], basic['embedding'])
        fz, fp = predict(models['mixed'], full['embedding'])
        mz, mp = predict(models['mixed'], full['masked_embedding'])
        conditions = {
            'full':(full['original_logits'].sigmoid().numpy(), cuts['full']),
            'full_basic_inference':(full['masked_logits'].sigmoid().numpy(), cuts['full']),
            'basic':(basic['original_logits'].sigmoid().numpy(), cuts['basic']),
            'basic_cbm':(bp, cuts['basic_cbm']),
            'mixed_full':(fp, cuts['mixed_cbm']),
            'mixed_basic':(mp, cuts['mixed_cbm'])}
        cohorts.append(dict(task=task, split=split, n=n, positive=int(y.sum()), negative=int(n-y.sum()),
                            prevalence=float(y.mean()), all_positive_f1=float(2*y.sum()/(n+y.sum()))))
        predictions = dict(person_id=basic['person_ids'], y=y)
        for name,(p,cut) in conditions.items():
            classification.append(dict(task=task, split=split, model=name, threshold=cut,
                                       n=n, **boot.performance(p, cut)))
            predictions[name] = p; predictions[name+'__decision'] = (p>=cut).astype(int)
        definitions = [
            ('full_input_loss','full','full_basic_inference','downstream'),
            ('basic_vs_full','basic','full','downstream'),
            ('retraining_gain','basic','full_basic_inference','downstream'),
            ('basic_cbm_gain','basic_cbm','basic','cbm'),
            ('mixed_cbm_gain','mixed_basic','full_basic_inference','cbm'),
            ('mixed_input_loss','mixed_full','mixed_basic','cbm')]
        for label,a,b,family in definitions:
            pa,ca=conditions[a]; pb,cb=conditions[b]
            comparisons.append(dict(task=task, split=split, comparison=label, family=family,
                minuend=a, subtrahend=b, n=n, **boot.effects(pa,pb,ca,cb)[0]))
        original = conditions['basic'][0]; cut = cuts['basic']
        effects = boot.effects(original, np.stack([removed[k]['probability'] for k in REMOVALS]), cut, cut)
        if split=='val':
            ranking = sorted(zip(REMOVALS,effects), key=lambda pair:(-pair[1]['value'],pair[0]))
            selection['top_modalities'] = [name for name,_ in ranking[:3]]
            write_json(out/'SELECTION.json', selection)
        for name,effect in zip(REMOVALS,effects):
            p = removed[name]['probability']
            rank = selection['top_modalities'].index(name)+1 if name in selection['top_modalities'] else None
            ablations.append(dict(task=task, split=split, removed_input=name, family='ablation', n=n,
                n_available=int(removed[name]['available'].sum()), val_rank=rank,
                full_f1=float(f1(y,original,cut)), removed_f1=float(f1(y,p,cut)), **effect))
            predictions['without__'+name] = p
        pd.DataFrame(predictions).to_csv(out/f'predictions_{split}.csv',index=False)
        for mode in ('basic','mixed'):
            model = models[mode]; unit_groups = groups[mode]; cut = cuts[mode+'_cbm']
            full_z, full_p = (bz,bp) if mode=='basic' else (fz,fp)
            targets = selection['top_modalities'] if mode=='basic' else ['nonbasic_clinical']
            for source in targets:
                missing_z, missing_p = predict(model,removed[source]['embedding']) if mode=='basic' else (mz,mp)
                key = mode+'__'+source
                names = list(unit_groups)  # Include constant/unobserved IDs in the supplementary CSV.
                restored = []
                for start in range(0,len(names),32):
                    block = names[start:start+32]
                    restored.append(restoration_batch(model,missing_z,full_z,[unit_groups[cid]['columns'] for cid in block]))
                restored = np.concatenate(restored)
                delta = f1(y,restored,cut)-f1(y,missing_p,cut)
                order = sorted((i for i in range(len(names)) if unit_groups[names[i]]['learnable']),
                               key=lambda i:(-float(delta[i]),names[i]))
                ranks = {names[i]:rank+1 for rank,i in enumerate(order)}
                if split=='val':
                    selection['concepts'][key] = [names[i] for i in order[:10]]
                    write_json(out/'SELECTION.json',selection)
                chosen = selection['concepts'][key]
                # A strict bottleneck's all-channel restoration is exactly the
                # same head applied to the full-input concept vector.
                names.append('ALL_CONCEPTS')
                all_restored = restoration_batch(model,missing_z,full_z,[list(range(model.spec['width']))])
                restored = np.concatenate((restored,all_restored))
                rows = boot.effects(restored,missing_p,cut,cut)
                values = boot.intervals(boot.draws(restored,cut))
                masked_stat = boot.performance(missing_p,cut)
                for cid,effect,perf in zip(names,rows,values):
                    g = unit_groups.get(cid,dict(concept_name='All CBM concepts restored together',
                        source_fields=list(dict.fromkeys(c['source_field'] for c in model.spec['items'])),
                        columns=list(range(model.spec['width'])),learnable=False))
                    recoveries.append(dict(task=task,split=split,mode=mode,removed_input=source,
                        concept_id=cid,concept_name=g['concept_name'],source_fields=' | '.join(g['source_fields']),
                        channels=len(g['columns']),val_rank=chosen.index(cid)+1 if cid in chosen else None,
                        rank=ranks.get(cid),ranking_split=split,learnable=g['learnable'],
                        intervention='all_concepts' if cid=='ALL_CONCEPTS' else 'individual',
                        family=mode+('_all_restoration' if cid=='ALL_CONCEPTS' else '_restoration'),n=n,threshold=cut,
                        full_f1=float(f1(y,full_p,cut)),masked_f1=masked_stat['value'],
                        restored_f1=perf['value'],restored_ci_lower=perf['ci_lower'],restored_ci_upper=perf['ci_upper'],**effect))
                table = pd.DataFrame(restored.T,columns=names)
                table.insert(0,'masked_probability',missing_p);table.insert(0,'y',y);table.insert(0,'person_id',basic['person_ids'])
                table.to_csv(out/f'restoration_{key}_{split}.csv',index=False)
                print(f'[restore {task}/{split}/{mode}] {source}: {len(unit_groups)} IDs + all-concepts reference; ranked on {split}',flush=True)
        # Modality removals stay validation-selected; concept rankings are split-specific.
        del boot
    for name,rows in [('classification',classification),('comparisons',comparisons),('ablation',ablations),
                       ('restoration',recoveries),('cohorts',cohorts)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    write_json(out/'DONE.json',dict(task=task,seed=cfg['seed'],bootstrap_samples=cfg['bootstrap_samples'],restoration_version=2))
    add_balanced_accuracy(task, cfg, device)


def add_balanced_accuracy(task, cfg, device):
    """Upgrade completed F1 results from saved predictions, without refitting.

    Keep existing F1 estimates, thresholds, and SELECTION.json unchanged.
    """
    out = Path(cfg['output_root'])/task/'analysis'
    done = read_json(out/'DONE.json')
    if done.get('reported_metrics') == ['f1','balanced_accuracy']: return
    frames = {name:pd.read_csv(out/(name+'.csv'),dtype={'concept_id':str},float_precision='round_trip')
              for name in ('classification','comparisons','ablation','restoration')}
    def assign(frame, index, values):
        for key,value in values.items(): frame.loc[index,'ba_'+key] = value
    for split in ('val','test'):
        pred = pd.read_csv(out/f'predictions_{split}.csv',dtype={'person_id':str},float_precision='round_trip')
        boot = PairedF1(pred.y.to_numpy(),cfg['bootstrap_samples'],cfg['seed'],device)
        def performance(p,cut): return boot.performance(p,cut,metric='balanced_accuracy')
        def effects(a,b,ca,cb): return boot.effects(a,b,ca,cb,metric='balanced_accuracy')
        c = frames['classification']; subset = c[c.split==split]
        cuts = dict(zip(subset.model,subset.threshold))
        for i,row in subset.iterrows():
            assign(c,i,performance(pred[row.model+'__decision'].to_numpy(),.5))
        e = frames['comparisons']
        for i,row in e[e.split==split].iterrows():
            assign(e,i,effects(pred[row.minuend+'__decision'].to_numpy(),pred[row.subtrahend+'__decision'].to_numpy(),.5,.5)[0])
        a = frames['ablation']
        for i,row in a[a.split==split].iterrows():
            removed = pred['without__'+row.removed_input].to_numpy(np.float32)
            assign(a,i,effects(pred.basic__decision.to_numpy(),removed,.5,cuts['basic'])[0])
            a.loc[i,'full_ba'] = performance(pred.basic__decision.to_numpy(),.5)['value']
            a.loc[i,'removed_ba'] = performance(removed,cuts['basic'])['value']
        r = frames['restoration']
        for (mode,source),part in r[r.split==split].groupby(['mode','removed_input'],sort=False):
            saved = pd.read_csv(out/f'restoration_{mode}__{source}_{split}.csv',dtype={'person_id':str},float_precision='round_trip')
            if saved.person_id.tolist()!=pred.person_id.tolist() or not np.array_equal(saved.y,pred.y):
                raise ValueError('Restoration predictions must match the classification cohort')
            cut = float(part.threshold.iloc[0]); ids = part.concept_id.tolist()
            # CSV contains shortest float32 decimals; restore the original dtype
            # before comparing with the saved threshold (including exact ties).
            restored = saved[ids].to_numpy(np.float32).T
            masked = saved.masked_probability.to_numpy(np.float32)
            rows = effects(restored,masked,cut,cut)
            perfs = boot.intervals(boot.draws(restored,cut,metric='balanced_accuracy'))
            full_name = 'basic_cbm' if mode=='basic' else 'mixed_full'
            full_ba = performance(pred[full_name+'__decision'].to_numpy(),.5)['value']
            masked_ba = performance(masked,cut)['value']
            for index,effect,perf in zip(part.index,rows,perfs):
                assign(r,index,effect)
                r.loc[index,'full_ba']=full_ba; r.loc[index,'masked_ba']=masked_ba
                r.loc[index,'restored_ba']=perf['value']
                r.loc[index,'restored_ba_ci_lower']=perf['ci_lower'];r.loc[index,'restored_ba_ci_upper']=perf['ci_upper']
        print(f'[balanced accuracy] {task}/{split}: saved predictions; no training',flush=True)
    for name,frame in frames.items(): frame.to_csv(out/(name+'.csv'),index=False)
    done['reported_metrics']=['f1','balanced_accuracy']; write_json(out/'DONE.json',done)
