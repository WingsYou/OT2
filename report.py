"""Four English tables, separately for validation and test, from completed outputs."""
from pathlib import Path
import pandas as pd
from common import (TASK_LABELS,INPUT_LABELS,REMOVALS,METRICS,METRIC_LABELS,FOCAL,JOINT_GROUP,read_json)

MODELS={'full':'Full clinical model','basic_inference':'Full model: basic clinical at inference',
        'basic':'Basic clinical model','cbm_joint_lambda_1':'Basic clinical model + CBM'}

def number(x,delta=False):return '—' if pd.isna(x) else format(float(x),'+.4f' if delta else '.4f')
def interval(row):return f"{number(row.value,True)} [{number(row.ci_lower,True)}, {number(row.ci_upper,True)}]"
def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])
def collect(root,cfg,filename,variants):
    frames=[]
    for task in cfg['tasks']:
        for variant in variants:
            p=root/task/variant/filename
            if p.exists():frames.append(pd.read_csv(p))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
def wide_metrics(frame,keys):
    if frame.empty:return pd.DataFrame(columns=[*keys,*METRICS])
    pivot=frame.pivot(index=keys,columns='metric',values=['value','ci_lower','ci_upper']).reset_index()
    pivot.columns=[a if not b else b if a=='value' else b+'_'+a for a,b in pivot.columns]
    return pivot

def make_report(cfg):
    root=Path(cfg['output_root']);tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    classification=collect(root,cfg,'classification.csv',('full','basic','cbm'))
    ablation=collect(root,cfg,'ablation.csv',('basic',))
    restoration=collect(root,cfg,'restoration.csv',('cbm',))
    def split_rows(frame,split):return frame[frame.split==split].copy() if not frame.empty else frame
    completed=sum((root/t/v/'DONE.json').exists() for t in cfg['tasks'] for v in ('full','basic','cbm'))
    for split in ('val','test'):
        c=split_rows(classification,split);a=split_rows(ablation,split);r=split_rows(restoration,split)
        c1=c[c.model.isin(('full','basic_inference','basic'))] if not c.empty else c
        c3=c[c.model.isin(('basic','cbm_joint_lambda_1'))] if not c.empty else c
        for index,frame in ((1,c1),(3,c3)):
            wide_metrics(frame,['task','model','n','threshold']).to_csv(tables/f'TABLE_{index}_{split}.csv',index=False)
        wide_metrics(a,['task','removed_input','n','threshold']).to_csv(tables/f'TABLE_2_{split}.csv',index=False)
        top=[];joint=[]
        for task in cfg['tasks']:
            p=root/task/'cbm/VAL_TOP3.json'
            if p.exists() and not r.empty:
                ranks=read_json(p)
                for rank,group in enumerate(ranks['groups'],1):
                    part=r[(r.task==task)&(r.removed_input==ranks['source'])&(r.concept_group==group)].copy()
                    part['rank']=rank;top.append(part)
            if not r.empty:
                sources=[FOCAL[task]]+(['retinal_oct'] if task=='lung_history_binary' else [])
                joint.append(r[(r.task==task)&r.removed_input.isin(sources)&(r.concept_group==JOINT_GROUP)])
        top=pd.concat(top,ignore_index=True) if top else pd.DataFrame()
        joint=pd.concat(joint,ignore_index=True) if joint else pd.DataFrame()
        top_keys=['task','removed_input','rank','concept_group','n','threshold']
        top_table=wide_metrics(top,top_keys)
        if not top.empty:
            harm=top[[*top_keys,'metric','initial_harm']].pivot(index=top_keys,columns='metric',values='initial_harm').add_prefix('initial_drop_').reset_index()
            top_table=top_table.merge(harm,on=top_keys)
        top_table.to_csv(tables/f'TABLE_4_{split}.csv',index=False)
        # Keep initial harm and all paired confidence intervals in the detail files.
        joint.to_csv(tables/f'TABLE_4_joint_{split}.csv',index=False)
        if not c.empty:c.to_csv(tables/f'classification_detail_{split}.csv',index=False)
        if not r.empty:r.to_csv(tables/f'restoration_detail_{split}.csv',index=False)
        lines=[f'# Six-task study — {"validation" if split=="val" else "test"}','',
            f'Completed model/analysis units: {completed}/{3*len(cfg["tasks"])}. Seed {cfg["seed"]}.',
            'Basic clinical means the agreed 23 clinical columns plus all eight nonclinical modalities. '
            'Full-model basic inference masks nonbasic clinical values and their observed masks, retaining the eight modalities.',
            'Models and thresholds are selected on validation only. All perturbed predictions use the parent full-input threshold. '
            'F1 is the positive-class F1; Balanced accuracy is the mean of sensitivity and specificity.',
            f'Intervals use {cfg["bootstrap_samples"]} paired participant bootstrap draws, stratified by outcome; '
            'pointwise 95% intervals exclude training-seed and previous study-selection uncertainty.',
            'The test split has been inspected in earlier parts of this project; these results are held-out evaluations, not a claim of a previously untouched confirmatory cohort.','',
            '## Table 1. Clinical input strategy — absolute performance','']
        def performance_panel(frame,arms):
            result=[]
            for metric in METRICS:
                rows=[]
                for task in cfg['tasks']:
                    values=[]
                    for arm in arms:
                        q=frame[(frame.task==task)&(frame.model==arm)&(frame.metric==metric)] if not frame.empty else frame
                        values.append(number(q.value.iloc[0]) if len(q) else '—')
                    rows.append([TASK_LABELS[task],*values])
                result += ['**'+METRIC_LABELS[metric]+'**','',markdown(['Task',*[MODELS[x] for x in arms]],rows),'']
            return result
        lines+=performance_panel(c1,('full','basic_inference','basic'))
        lines+=['## Table 2. Input removal from the Basic clinical model','',
            'Delta = full-input performance minus removed-input performance. Positive values indicate degradation. '
            'Each task uses its entire labeled split for every removal; no model refitting or threshold retuning.','']
        for metric in METRICS:
            rows=[]
            for source in REMOVALS:
                values=[]
                for task in cfg['tasks']:
                    q=a[(a.task==task)&(a.removed_input==source)&(a.metric==metric)] if not a.empty else a
                    values.append(number(q.value.iloc[0],True) if len(q) else '—')
                rows.append([INPUT_LABELS[source],*values])
            lines+=['**'+METRIC_LABELS[metric]+' degradation**','',markdown(['Removed input',*[TASK_LABELS[t] for t in cfg['tasks']]],rows),'']
        lines+=['## Table 3. Adding the concept bottleneck — absolute performance','',
            'CBM: frozen Basic-model fusion → original12 predicted concepts → 32-unit nonlinear classifier. '
            'Primary arm is joint lambda=1, with the original concept-quality selection constraint and sequential fallback. '
            'There is no direct Basic clinical or fusion bypass.','']
        lines+=performance_panel(c3,('basic','cbm_joint_lambda_1'))
        selections=[]
        for task in cfg['tasks']:
            path=root/task/'cbm/selection.json'
            if path.exists():
                selected=read_json(path)['fit_status']['joint_lambda_1']
                status='sequential fallback' if selected['status']=='no_accepted_joint_update' else f"joint epoch {selected['best_epoch']}"
                selections.append(TASK_LABELS[task]+': '+status)
        if selections:lines+=['Selected CBM checkpoints: '+'; '.join(selections)+'.','']
        lines+=['## Table 4. Top three restored predicted concept groups','',
            'The ranking is determined by validation AUROC recovery and held fixed for test. '
            'Recovery = restored-channel performance minus source-removed performance. '
            'Only people with at least one member of the removed source originally present are included. '
            'Unlike the earlier complete12 analysis, no measured-concept completeness filter is applied to either split.',
            'Channels are replaced with the same person’s full-input PREDICTED concepts. '
            'This is a diagnostic model intervention, not measured-concept addition or a deployable missing-modality reconstruction.','']
        if not top.empty:
            context=[]
            for (task,source),part in top.groupby(['task','removed_input'],sort=False):
                values=[number(part[part.metric==metric].initial_harm.iloc[0],True) for metric in METRICS]
                context.append([TASK_LABELS[task],INPUT_LABELS[source],int(part.n.iloc[0]),*values])
            lines+=['**Initial source-removal degradation in the CBM, before channel restoration**','',
                markdown(['Task','Removed input','N','AUROC drop','Balanced accuracy drop','F1 drop'],context),'']
        def recovery_panel(frame,ranked):
            rows=[]
            if not frame.empty:
                keys=['task','removed_input']+(['rank'] if ranked else [])+['concept_group']
                for key,part in frame.groupby(keys,sort=False):
                    values=[interval(part[part.metric==metric].iloc[0]) for metric in METRICS]
                    key=list(key);row=[TASK_LABELS[key[0]],INPUT_LABELS[key[1]],*key[2:],*values];rows.append(row)
            return markdown(['Task','Removed input']+(['Val rank'] if ranked else [])+['Restored concepts',
                'AUROC recovery [95% CI]','Balanced accuracy recovery [95% CI]','F1 recovery [95% CI]'],rows)
        lines += [recovery_panel(top,True),'','**Joint-six follow-up (separate from the seven-group ranking)**','',
            'BMI, systolic BP, diastolic BP, HbA1c, HDL and LDL are restored simultaneously. '
            'The additional Lung/OCT comparison is fixed from the earlier validation finding.',
            recovery_panel(joint,False),'',
            'Inspect initial removal harm in restoration_detail before interpreting recovery. Group effects overlap and are not additive.',
            'Full clinical inputs for the three history tasks reuse the existing MoCA clinical feature view, '
            'with each history target’s own observation/condition features removed. Previously excluded MoCA fields are not reintroduced.',
            'Missing cells mean the required unit has not produced results. CSV files retain unrounded values, sample counts, thresholds and confidence intervals.']
        (tables/f'SUMMARY_{split}.md').write_text('\n'.join(lines)+'\n')
    print('TABLES: '+str(tables),flush=True)
