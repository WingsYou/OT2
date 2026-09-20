"""Diagnostic concept restoration in fitted mixed-training CBMs; no fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import (GROUPS,JOINT_GROUP,JOINT_MEMBERS,TASK_LABELS,METRICS,METRIC_LABELS,
                    Bootstrap,effect_rows,read_json,write_json,save_csv)
from full_clinical_cbm import load,restoration_predictions
from cbm_joint_v2 import restore,predict


def run_task(task,cfg,device):
    root=Path(cfg['restoration_root'])/task;root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():print('[reuse mixed restoration] '+task,flush=True);return
    mixed=Path(cfg['mixed_cbm_root'])/task
    selected=read_json(mixed/'selection.json');arm=selected['primary_arm']
    artifact=load(mixed/'models.pt')[arm];model=restore(artifact,device=device)
    cut=selected['thresholds']['mixed_cbm']
    groups={**GROUPS,JOINT_GROUP:JOINT_MEMBERS,'All 12 concepts':artifact['concept_schema']['concept_names']}
    write_json(root/'selection.json',{'source_model':str(mixed/'models.pt'),'primary_arm':arm,
        'fit_status':selected['fit_status'][arm],'threshold':cut,'threshold_source':'saved masked validation Youden',
        'restoration_reference':'same-participant full-input predictions from the SAME mixed-training CBM',
        'cohort':'entire labeled split','ranking_split':'val'})
    classification=[];degradation=[];effects=[]
    for split in ('val','test'):
        cache=load(Path(cfg['output_root'])/task/'full'/f'features_{split}.pt')
        masked=load(Path(cfg['full_cbm_root'])/task/f'basic_inference_{split}.pt')
        y=cache['y'].numpy();n=len(y);affected=int(masked['affected'].sum())
        full=predict(model,cache['embedding']);missing=predict(model,masked['embedding'])
        boot=Bootstrap(y,cfg['bootstrap_samples'],cfg['seed'],device)
        full_draws=boot.metrics(full['probabilities'],cut);missing_draws=boot.metrics(missing['probabilities'],cut)
        for name,draws in (('full',full_draws),('masked',missing_draws)):
            classification+=effect_rows(task,split,'classification',draws,model=name,n=n,threshold=cut)
        degradation+=effect_rows(task,split,'degradation',full_draws-missing_draws,n=n,threshold=cut)
        saved={'person_id':cache['person_ids'],'y':y,'full':full['probabilities'],'masked':missing['probabilities']}
        arrays={'person_ids':np.asarray(cache['person_ids']),'full_concepts':full['concepts'],'masked_concepts':missing['concepts']}
        for group,p in restoration_predictions(model,missing['concepts'],full['concepts'],groups):
            restored=boot.metrics(p,cut);remaining=full_draws-restored
            rows=effect_rows(task,split,'recovery',restored-missing_draws,concept_group=group,n=n,n_affected=affected,threshold=cut)
            for j,row in enumerate(rows):
                lo,hi=np.quantile(remaining[1:,j],[.025,.975])
                row.update(full_metric=float(full_draws[0,j]),masked_metric=float(missing_draws[0,j]),
                    restored_metric=float(restored[0,j]),initial_drop=float(full_draws[0,j]-missing_draws[0,j]),
                    remaining_drop=float(remaining[0,j]),remaining_ci_lower=float(lo),remaining_ci_upper=float(hi))
            effects+=rows;saved['restored__'+group]=p
        pd.DataFrame(saved).to_csv(root/f'predictions_{split}.csv',index=False)
        np.savez_compressed(root/f'concept_predictions_{split}.npz',**arrays)
        if split=='val':
            ranked=sorted((r for r in effects if r['split']=='val' and r['metric']=='auroc' and r['concept_group'] in GROUPS),
                          key=lambda r:(-r['value'],r['concept_group']))
            write_json(root/'VAL_TOP3.json',{'groups':[r['concept_group'] for r in ranked[:3]],'ranking_split':'val'})
        print(f'[mixed restoration] {task}/{split}: n={n}; fixed threshold={cut:.4f}',flush=True)
    for name,rows in (('classification',classification),('degradation',degradation),('restoration',effects)):
        save_csv(root/(name+'.csv'),pd.DataFrame(rows))
    write_json(root/'DONE.json',{'task':task,'splits':['val','test'],'training_performed':False})


def make_report(cfg):
    from report import number,interval,markdown
    root=Path(cfg['restoration_root']);tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    def collect(filename):
        frames=[pd.read_csv(root/t/filename) for t in cfg['tasks'] if (root/t/'DONE.json').exists()]
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    classification=collect('classification.csv');degradation=collect('degradation.csv');restoration=collect('restoration.csv')
    for split in ('val','test'):
        frames={name:f[f.split==split] if not f.empty else f for name,f in
                (('classification',classification),('degradation',degradation),('restoration',restoration))}
        for name,f in frames.items():f.to_csv(tables/f'{name}_{split}.csv',index=False)
        c,d,r=[frames[k] for k in ('classification','degradation','restoration')]
        lines=[f'# Mixed-training CBM: concept restoration — {split}','',
            f'Completed tasks: {sum((root/t/"DONE.json").exists() for t in cfg["tasks"])}/{len(cfg["tasks"])}. Seed {cfg["seed"]}.','',
            'This analysis reuses the completed full + masked training CBM without retraining. '
            'The masked input retains Basic23 and all eight nonclinical modalities. '
            'Selected channels are replaced by the same participant’s full-input predictions from this SAME mixed-training CBM. '
            'All other concept channels remain at their masked predictions. No measured concepts or old-model concept predictions are injected.','',
            'The existing masked-validation threshold is kept for full, masked and restored predictions on both splits. '
            'All comparisons use the entire labeled split; no concept-label completeness filter is applied. '
            'This diagnostic intervention needs no measured test concept labels.','',
            '## 1. Full versus masked input in the same mixed-training CBM','',
            'Drop = full-input minus masked-input performance. Negative means masked performance is higher.','']
        for metric in METRICS:
            rows=[]
            for task in cfg['tasks']:
                q=c[(c.task==task)&(c.metric==metric)] if not c.empty else c
                harm=d[(d.task==task)&(d.metric==metric)] if not d.empty else d
                if q.empty:rows.append([TASK_LABELS[task],'—','—','—']);continue
                rows.append([TASK_LABELS[task],number(q[q.model=='full'].value.iloc[0]),
                             number(q[q.model=='masked'].value.iloc[0]),interval(harm.iloc[0])])
            lines+=['**'+METRIC_LABELS[metric]+'**','',markdown(['Task','Full input','Masked input','Drop [95% CI]'],rows),'']
        lines+=['## 2. Concept-group restoration','',
            'Recovery = restored minus masked performance. Individual-group top-three ranks are selected on validation and held fixed on test. '
            'Six-concept and all-12 restoration are separate follow-ups, excluded from that ranking.','']
        top=[]
        for task in cfg['tasks']:
            q=r[r.task==task] if not r.empty else r
            if q.empty:continue
            ranks=read_json(root/task/'VAL_TOP3.json')['groups'];selected=read_json(root/task/'selection.json')['fit_status']
            label='sequential fallback' if selected['status']=='no_accepted_joint_update' else f'joint epoch {selected["best_epoch"]}'
            rows=[]
            for group,part in q.groupby('concept_group',sort=False):
                row=[group,str(ranks.index(group)+1) if group in ranks else '—']
                for metric in METRICS:
                    item=part[part.metric==metric].iloc[0]
                    row.extend((number(item.restored_metric),interval(item)))
                rows.append(row)
                if group in ranks:
                    part=part.copy();part['val_rank']=ranks.index(group)+1;top.append(part)
            headers=['Restored concepts','Val rank']
            for metric in METRICS:headers.extend((METRIC_LABELS[metric]+' restored',METRIC_LABELS[metric]+' gain [95% CI]'))
            lines+=['### '+TASK_LABELS[task],'',f'N={int(q.n.iloc[0])}; observed nonbasic clinical={int(q.n_affected.iloc[0])}; reused checkpoint: {label}.','',markdown(headers,rows),'']
        (pd.concat(top,ignore_index=True) if top else pd.DataFrame()).to_csv(tables/f'top3_{split}.csv',index=False)
        lines+=['All-12 restoration reproduces the full-input prediction by construction. If masked performance is already higher, '
                'all-12 restoration has a negative gain. Small or negative recovery is not automatically a failure: interpret it alongside the initial full–masked gap. '
                'Group effects overlap and cannot be added. Six cardiometabolic concepts means BMI, systolic/diastolic BP, HbA1c, HDL and LDL.','',
                'Full-input concepts are an unavailable reference in a genuinely masked deployment. This analysis diagnoses residual sensitivity '
                'of concept channels; it does not itself implement deployable reconstruction or measure accuracy against true concept labels.','',
                f'Pointwise 95% intervals use {cfg["bootstrap_samples"]} paired outcome-stratified bootstrap draws. '
                'They exclude training-seed and previous selection uncertainty. The test split has been inspected previously. '
                'CSV details retain full, masked and restored metrics, initial drops and remaining gaps.']
        reference=Path(__file__).resolve().parent/'report_update/CONCEPT_REFERENCE.md'
        if reference.exists():lines+=['',reference.read_text()]
        (tables/f'SUMMARY_MIXED_RESTORATION_{split}.md').write_text('\n'.join(lines)+'\n')
    print('MIXED_RESTORATION_TABLES: '+str(tables),flush=True)
