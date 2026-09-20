"""One CBM trained on paired full/masked inputs, with masked-only evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import (TASK_LABELS,METRICS,METRIC_LABELS,Bootstrap,effect_rows,threshold,
                    read_json,write_json,save_torch,save_csv)
from full_clinical_cbm import load,feature_caches
from cbm_joint_v2 import DEFAULTS,fit_experiment,restore,predict
from cbm_readout_v1 import _targets

ARMS={'original_masked':'Original Full classifier, masked',
      'full_only_cbm':'Full-only-trained CBM, masked',
      'mixed_cbm':'Full + masked training CBM, masked'}


def concept_scores(task,split,arm,model,prediction,labels,ids):
    frame=labels.set_index('person_id').reindex(ids);rows=[]
    for c,block in zip(model.spec['concepts'],model.spec['layout']):
        truth,present=_targets(frame,c);n=int(present.sum())
        if not n:continue
        z=prediction['clinical_concepts'][present,block['start']:block['stop']]
        y=truth[present]
        row=dict(task=task,split=split,model=arm,concept=c['name'],n=n,unit=c.get('unit',''),
                 mae=np.nan,r2=np.nan,balanced_accuracy=np.nan,accuracy=np.nan)
        if c['type']=='continuous':
            p=z[:,0];denom=((y-y.mean())**2).sum()
            row.update(mae=float(np.abs(p-y).mean()),r2=float(1-((p-y)**2).sum()/denom) if denom>0 else np.nan)
        else:
            p=z.argmax(1)
            row.update(accuracy=float((p==y).mean()),
                       balanced_accuracy=float(np.mean([(p[y==k]==k).mean() for k in np.unique(y)])))
        rows.append(row)
    return rows


def run_task(task,cfg,device):
    source=Path(cfg['output_root']);root=Path(cfg['mixed_cbm_root'])/task
    root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():print('[reuse mixed CBM] '+task,flush=True);return
    previous=Path(cfg['full_cbm_root'])/task;previous.mkdir(parents=True,exist_ok=True)
    paths,masked_paths=feature_caches(task,cfg,device,previous,include_masked_train=True)
    entry=read_json(source/'inputs.json')[task];label_dir=Path(entry['concept_dir'])
    schema=read_json(label_dir/'concept_schema.json')
    labels=pd.read_csv(label_dir/'concept_labels.csv',dtype={'person_id':str})
    write_json(root/'concept_schema.json',schema)
    train,val=load(paths['train']),load(paths['val'])
    val_masked=load(masked_paths['val'])
    # Same participant order and labels; validation never uses full-input embeddings.
    validation={**val,'embedding':val_masked['embedding']}
    if (root/'models.pt').exists():artifacts=load(root/'models.pt')
    else:
        options={**DEFAULTS,'seed':cfg['seed'],'concept_epochs':cfg['concept_epochs'],
                 'head_epochs':cfg['head_epochs'],'joint_epochs':cfg['joint_epochs'],
                 'patience':cfg['concept_patience'],'lambdas':[1.]}
        print(f'[mixed training] {task}: 50% full + 50% masked; selection on masked validation',flush=True)
        artifacts,history=fit_experiment(train,validation,labels,schema,config=options,device=device,
                                         masked_train_embedding=load(masked_paths['train'])['embedding'])
        save_torch(root/'models.pt',artifacts);pd.DataFrame(history).to_csv(root/'training.csv',index=False)
    del train,validation
    models={'mixed_cbm':restore(artifacts['joint_lambda_1'],device=device),
            'mixed_sequential':restore(artifacts['sequential'],device=device)}
    if (previous/'models.pt').exists():
        models['full_only_cbm']=restore(load(previous/'models.pt')['joint_lambda_1'],device=device)
    cuts={name:threshold(val['y'].numpy(),predict(m,val_masked['embedding'])['probabilities']) for name,m in models.items()}
    cuts['original_masked']=threshold(val['y'].numpy(),val_masked['original_logits'].sigmoid().numpy())
    write_json(root/'selection.json',{'bank':'original12','training_views':['full','nonbasic_clinical_masked'],
        'training_view_weights':[0.5,0.5],'backbone_frozen':True,'fusion_frozen':True,
        'selection_input':'masked','inference_input':'masked','threshold_source':'masked validation Youden',
        'thresholds':cuts,'primary_arm':'joint_lambda_1',
        'fit_status':{k:a['selection'] for k,a in artifacts.items()}})
    rows=[];gains=[];scores=[]
    for split in ('val','test'):
        cache=val if split=='val' else load(paths[split])
        masked=val_masked if split=='val' else load(masked_paths[split])
        y=cache['y'].numpy();n=len(y);boot=Bootstrap(y,cfg['bootstrap_samples'],cfg['seed'],device)
        probabilities={'original_masked':masked['original_logits'].sigmoid().numpy()}
        arrays={'person_ids':np.asarray(cache['person_ids'])}
        for arm,m in models.items():
            prediction=predict(m,masked['embedding']);probabilities[arm]=prediction['probabilities']
            arrays[arm]=prediction['clinical_concepts']
            if arm!='mixed_sequential':scores+=concept_scores(task,split,arm,m,prediction,labels,cache['person_ids'])
        draws={arm:boot.metrics(p,cuts[arm]) for arm,p in probabilities.items()}
        for arm,value in draws.items():
            rows+=effect_rows(task,split,'classification',value,model=arm,n=n,threshold=cuts[arm])
        for baseline in ('original_masked','full_only_cbm'):
            if baseline in draws:
                gains+=effect_rows(task,split,'gain',draws['mixed_cbm']-draws[baseline],baseline=baseline,n=n)
        pd.DataFrame({'person_id':cache['person_ids'],'y':y,**probabilities}).to_csv(root/f'predictions_{split}.csv',index=False)
        np.savez_compressed(root/f'masked_concepts_{split}.npz',**arrays)
        print(f'[mixed CBM] {task}/{split}: masked-only evaluation complete',flush=True)
    for filename,data in (('classification',rows),('gains',gains),('concept_metrics',scores)):
        save_csv(root/(filename+'.csv'),pd.DataFrame(data))
    write_json(root/'DONE.json',{'task':task,'splits':['val','test'],'inference_input':'masked'})


def make_report(cfg):
    from report import number,interval,markdown
    root=Path(cfg['mixed_cbm_root']);tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    def collect(filename):
        frames=[pd.read_csv(root/t/filename) for t in cfg['tasks'] if (root/t/'DONE.json').exists()]
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    classification=collect('classification.csv');gains=collect('gains.csv');concepts=collect('concept_metrics.csv')
    for split in ('val','test'):
        frames={name:f[f.split==split] if not f.empty else f for name,f in
                (('classification',classification),('gains',gains),('concept_metrics',concepts))}
        for name,frame in frames.items():frame.to_csv(tables/f'{name}_{split}.csv',index=False)
        c,g,s=[frames[k] for k in ('classification','gains','concept_metrics')]
        lines=[f'# Full + masked CBM training — {split}','',
            f'Completed tasks: {sum((root/t/"DONE.json").exists() for t in cfg["tasks"])}/{len(cfg["tasks"])}. Seed {cfg["seed"]}.','',
            'One shared original12 concept predictor and classifier are trained with equal-weight full and masked views of each training participant. '
            'The original Full Clinical encoders and fusion stay frozen. Both views use the same observed concept labels and task labels. '
            'Every validation/test prediction uses Basic23 + eight nonclinical modalities; all other clinical values and observation indicators are masked. '
            'No full-input concepts or measured concepts are substituted at inference.','',
            'Checkpoint selection, concept-alignment constraint and thresholds use masked validation only. '
            'The full-only-trained CBM comparator retains its original checkpoint; its threshold and the original classifier threshold '
            'are reselected on masked validation for a matched inference setting. Consequently their balanced accuracy/F1 may differ from earlier reports.','',
            '## 1. Masked-input classification','']
        for metric in METRICS:
            rows=[]
            for task in cfg['tasks']:
                values=[]
                for arm in ARMS:
                    part=c[(c.task==task)&(c.model==arm)&(c.metric==metric)] if not c.empty else c
                    values.append(number(part.value.iloc[0]) if len(part) else '—')
                rows.append([TASK_LABELS[task],*values])
            lines+=['**'+METRIC_LABELS[metric]+'**','',markdown(['Task',*ARMS.values()],rows),'']
        lines+=['## 2. Paired improvement under masked inputs','',
                'Positive values favor mixed training. Every comparison uses the same participants.','']
        rows=[]
        if not g.empty:
            for (task,baseline),part in g.groupby(['task','baseline'],sort=False):
                rows.append([TASK_LABELS[task],ARMS[baseline],*[interval(part[part.metric==m].iloc[0]) for m in METRICS]])
        lines+=[markdown(['Task','Comparator',*[METRIC_LABELS[m]+' gain [95% CI]' for m in METRICS]],rows),'',
                '## 3. Masked-input concept prediction','',
                'Errors are evaluated against available measured labels, never full-input predictions. '
                'Continuous concepts report MAE in their clinical unit and R²; categorical concepts report balanced accuracy. '
                'Missing measured labels are omitted per concept. Test concept labels may be absent from the existing label files.','']
        if s.empty:lines+=['No measured concept labels available in the supplied files for this split.','']
        else:
            for task,part in s.groupby('task',sort=False):
                rows=[[r.concept,ARMS[r.model],r.unit,int(r.n),number(r.mae),number(r.r2),number(r.balanced_accuracy)]
                      for r in part.itertuples()]
                lines+=['### '+TASK_LABELS[task],'',markdown(['Concept','Model','Unit','N','MAE ↓','R² ↑','Balanced accuracy ↑'],rows),'']
        selections=[]
        for task in cfg['tasks']:
            path=root/task/'selection.json'
            if path.exists():
                selected=read_json(path)['fit_status']['joint_lambda_1']
                label='sequential fallback' if selected['status']=='no_accepted_joint_update' else f'joint epoch {selected["best_epoch"]}'
                selections.append(TASK_LABELS[task]+': '+label)
        lines+=['Selected mixed CBM checkpoints: '+'; '.join(selections)+'.','',
            f'Intervals use {cfg["bootstrap_samples"]} paired outcome-stratified participant bootstrap draws; '
            'they exclude training-seed and prior test-inspection uncertainty. This first experiment changes CBM training exposure '
            'and selects checkpoints for masked deployment. It does not isolate the benefit of CBM versus a non-CBM model trained with the same masking. '
            'It does not evaluate complete-input deployment or cross-institution transfer.']
        (tables/f'SUMMARY_MIXED_CBM_{split}.md').write_text('\n'.join(lines)+'\n')
    print('MIXED_CBM_TABLES: '+str(tables),flush=True)
