"""Frozen Full Clinical CBM and predicted-channel restoration after clinical masking."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import (BASIC_KEYS,GROUPS,JOINT_GROUP,JOINT_MEMBERS,TASK_LABELS,METRICS,METRIC_LABELS,
                    Bootstrap,effect_rows,threshold,read_json,write_json,save_torch,save_csv)
from cbm_joint_v2 import DEFAULTS,fit_experiment,restore,predict,classify

ARMS={'full':'Full Clinical', 'full_basic_inference':'Full Clinical: Basic-only clinical inference',
      'basic':'Retrained Basic Clinical', 'basic_cbm':'Basic Clinical + CBM',
      'full_cbm':'Full Clinical + CBM', 'full_cbm_basic_inference':'Full Clinical + CBM: Basic-only clinical inference'}


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


@torch.no_grad()
def masked_clinical_features(model,cache,basic,task,device,batch_size):
    """Keep eight source tokens fixed; recompute only clinical encoding and fusion."""
    from train import amp
    model.eval();embeddings=[];logits=[]
    keep=torch.zeros(cache['clinical_values'].shape[1],dtype=torch.bool);keep[basic]=True
    affected=cache['clinical_mask'][:,~keep].any(1)
    for start in range(0,len(cache['y']),batch_size):
        sl=slice(start,start+batch_size)
        values=cache['clinical_values'][sl].to(device).clone()
        mask=cache['clinical_mask'][sl].to(device).clone()
        values[:,~keep.to(device)]=0;mask[:,~keep.to(device)]=False
        tokens=cache['source_tokens'][sl].to(device).clone()
        present=cache['source_present'][sl].to(device).clone()
        with amp(device):
            token,valid=model.clinical_encoder(values,mask)
            present[:,0]=valid&present[:,0];tokens[:,0]=token*present[:,0,None]
            out=model.predict_from_source_tokens(tokens,present,task=task)
        embeddings.append(out['embedding'].float().cpu())
        logits.append(out['logits'].reshape(-1).float().cpu())
    return {'embedding':torch.cat(embeddings),'original_logits':torch.cat(logits),'affected':affected}


def feature_caches(task,cfg,device,root):
    from train import CachedModel,CachedDataset,clinical_inputs,loader,export
    source=Path(cfg['output_root']);full=source/task/'full'
    original={s:full/f'features_{s}.pt' for s in ('train','val','test')}
    train_path=original['train'] if original['train'].exists() else root/'features_train.pt'
    mask_paths={s:root/f'basic_inference_{s}.pt' for s in ('val','test')}
    if not train_path.exists() or any(not p.exists() for p in mask_paths.values()):
        checkpoint=load(full/'best.pt')
        # Every CNN parameter comes from this checkpoint; no pretrained download.
        mc={**checkpoint['model_config'],'retinal_weights':None,'retinal_weights_path':None}
        model=CachedModel(mc).to(device);model.load_state_dict(checkpoint['model'])
        model.eval().requires_grad_(False)
        names=read_json(full/'input_columns.json')['clinical_features']
        basic=[i for i,name in enumerate(names) if name.split('__')[1] in BASIC_KEYS]
        if not train_path.exists():
            print(f'[features] {task}: exporting Full Clinical training embeddings once',flush=True)
            entry=read_json(source/'inputs.json')[task]
            master,values,mask,_,_,_=clinical_inputs(task,cfg,entry)
            ds=CachedDataset(source,master,values,mask,task,'train')
            export(model,loader(ds,cfg),task,'train',device,root,read_json(full/'TRAINING_DONE.json'))
        for split,path in mask_paths.items():
            if not path.exists():
                print(f'[features] {task}/{split}: cached tokens → masked clinical → fusion',flush=True)
                result=masked_clinical_features(model,load(original[split]),basic,task,device,cfg['batch_size'])
                save_torch(path,result)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    return {'train':train_path,'val':original['val'],'test':original['test']},mask_paths


def restoration_predictions(model,missing,full,groups):
    """Replace complete categorical probability blocks as well as numeric channels."""
    for group,members in groups.items():
        indices=[i for block in model.spec['layout'] if block['name'] in members
                 for i in range(block['start'],block['stop'])]
        z=missing.copy();z[:,indices]=full[:,indices]
        yield group,classify(model,z)


def run_task(task,cfg,device):
    root=Path(cfg['full_cbm_root'])/task;root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():print('[reuse Full CBM] '+task,flush=True);return
    paths,mask_paths=feature_caches(task,cfg,device,root)
    source=Path(cfg['output_root']);entry=read_json(source/'inputs.json')[task]
    label_dir=Path(entry['concept_dir']);schema=read_json(label_dir/'concept_schema.json')
    labels=pd.read_csv(label_dir/'concept_labels.csv',dtype={'person_id':str})
    write_json(root/'concept_schema.json',schema)
    train,val=load(paths['train']),load(paths['val'])
    if (root/'models.pt').exists():artifacts=load(root/'models.pt')
    else:
        options={**DEFAULTS,'seed':cfg['seed'],'concept_epochs':cfg['concept_epochs'],
                 'head_epochs':cfg['head_epochs'],'joint_epochs':cfg['joint_epochs'],
                 'patience':cfg['concept_patience'],'lambdas':[1.]}
        artifacts,history=fit_experiment(train,val,labels,schema,config=options,device=device)
        save_torch(root/'models.pt',artifacts);pd.DataFrame(history).to_csv(root/'training.csv',index=False)
    del train
    models={arm:restore(a,device=device) for arm,a in artifacts.items()}
    thresholds={arm:threshold(val['y'].numpy(),predict(m,val['embedding'])['probabilities']) for arm,m in models.items()}
    primary='joint_lambda_1';model=models[primary];cut=thresholds[primary]
    groups={**GROUPS,JOINT_GROUP:JOINT_MEMBERS,'All 12 concepts':schema['concept_names']}
    write_json(root/'selection.json',{'backbone':'full','bank':'original12','primary_arm':primary,
        'thresholds':thresholds,'threshold_source':'full-input validation Youden',
        'fit_status':{k:a['selection'] for k,a in artifacts.items()},
        'restoration_reference':'same-participant full-input predicted concepts','cohort':'entire labeled split'})
    rows=[];degradation=[];effects=[]
    original_cut=read_json(source/task/'full/TRAINING_DONE.json')['threshold']
    for split in ('val','test'):
        cache=val if split=='val' else load(paths[split]);masked=load(mask_paths[split])
        y=cache['y'].numpy();n=len(y);affected=int(masked['affected'].sum())
        boot=Bootstrap(y,cfg['bootstrap_samples'],cfg['seed'],device)
        saved={'person_id':cache['person_ids'],'y':y,'nonbasic_observed':masked['affected'].numpy()}
        for name,c in (('full',cache),('full_basic_inference',masked)):
            p=c['original_logits'].sigmoid().numpy();saved[name]=p
            rows+=effect_rows(task,split,'classification',boot.metrics(p,original_cut),model=name,n=n,threshold=original_cut)
        for arm,m in models.items():
            full=predict(m,cache['embedding']);missing=predict(m,masked['embedding'])
            full_draws=boot.metrics(full['probabilities'],thresholds[arm])
            missing_draws=boot.metrics(missing['probabilities'],thresholds[arm])
            name='full_cbm' if arm==primary else 'full_cbm_sequential'
            for suffix,draws,pred in (('',full_draws,full),('_basic_inference',missing_draws,missing)):
                rows+=effect_rows(task,split,'classification',draws,model=name+suffix,n=n,threshold=thresholds[arm])
                saved[name+suffix]=pred['probabilities']
            degradation+=effect_rows(task,split,'degradation',full_draws-missing_draws,model=name,n=n,threshold=thresholds[arm])
            if arm!=primary:continue
            arrays={'person_ids':np.asarray(cache['person_ids']),'full_concepts':full['concepts'],'masked_concepts':missing['concepts']}
            for group,p in restoration_predictions(m,missing['concepts'],full['concepts'],groups):
                restored=boot.metrics(p,cut);recovery=restored-missing_draws
                remaining=full_draws-restored
                part=effect_rows(task,split,'recovery',recovery,concept_group=group,n=n,n_affected=affected,threshold=cut)
                for j,row in enumerate(part):
                    lo,hi=np.quantile(remaining[1:,j],[.025,.975])
                    row.update(full_metric=float(full_draws[0,j]),masked_metric=float(missing_draws[0,j]),
                        restored_metric=float(restored[0,j]),initial_drop=float(full_draws[0,j]-missing_draws[0,j]),
                        remaining_drop=float(remaining[0,j]),remaining_ci_lower=float(lo),remaining_ci_upper=float(hi))
                effects+=part;arrays[group]=p;saved['restored__'+group]=p
            np.savez_compressed(root/f'concept_predictions_{split}.npz',**arrays)
        pd.DataFrame(saved).to_csv(root/f'predictions_{split}.csv',index=False)
        if split=='val':
            ranked=sorted((r for r in effects if r['split']=='val' and r['metric']=='auroc' and r['concept_group'] in GROUPS),
                          key=lambda r:(-r['value'],r['concept_group']))
            write_json(root/'VAL_TOP3.json',{'groups':[r['concept_group'] for r in ranked[:3]],'ranking_split':'val'})
        print(f'[Full CBM] {task}/{split} complete; n={n}, observed nonbasic clinical={affected}',flush=True)
    for file,records in (('classification',rows),('degradation',degradation),('restoration',effects)):
        save_csv(root/(file+'.csv'),pd.DataFrame(records))
    write_json(root/'DONE.json',{'task':task,'splits':['val','test'],'bank':'original12'})


def make_report(cfg):
    from report import collect,number,interval,markdown
    root=Path(cfg['full_cbm_root']);tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    def read_all(filename):
        frames=[pd.read_csv(root/t/filename) for t in cfg['tasks'] if (root/t/filename).exists()]
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    classification=read_all('classification.csv');restoration=read_all('restoration.csv')
    degradation=read_all('degradation.csv')
    reference=collect(Path(cfg['output_root']),cfg,'classification.csv',('basic','cbm'))
    if not reference.empty:
        reference=reference[reference.model.isin(('basic','cbm_joint_lambda_1'))].copy()
        reference['model']=reference.model.replace({'cbm_joint_lambda_1':'basic_cbm'})
        classification=pd.concat([classification,reference],ignore_index=True)
    done=sum((root/t/'DONE.json').exists() for t in cfg['tasks'])
    for split in ('val','test'):
        c=classification[classification.split==split] if not classification.empty else classification
        r=restoration[restoration.split==split] if not restoration.empty else restoration
        d=degradation[degradation.split==split] if not degradation.empty else degradation
        for name,frame in (('classification',c),('restoration',r),('degradation',d)):
            frame.to_csv(tables/f'{name}_{split}.csv',index=False)
        lines=[f'# Full Clinical + CBM — {split}','',f'Completed tasks: {done}/{len(cfg["tasks"])}. Seed {cfg["seed"]}.','',
            'The existing Full Clinical encoder/fusion is frozen. The original12 CBM uses joint lambda=1 with the existing concept-alignment constraint and sequential fallback. '
            'Removal masks all nonbasic clinical values and their observation indicators; Basic23 and all eight nonclinical modalities remain. '
            'All comparisons use the entire labeled split. Thresholds are selected on full-input validation predictions and remain fixed after masking/restoration.','',
            '## 1. Classification performance','']
        for metric in METRICS:
            rows=[]
            for task in cfg['tasks']:
                values=[]
                for arm in ARMS:
                    q=c[(c.task==task)&(c.model==arm)&(c.metric==metric)] if not c.empty else c
                    values.append(number(q.value.iloc[0]) if len(q) else '—')
                rows.append([TASK_LABELS[task],*values])
            lines+=['**'+METRIC_LABELS[metric]+'**','',markdown(['Task',*ARMS.values()],rows),'']
        lines+=['## 2. Nonbasic clinical removal: degradation in Full Clinical + CBM','',
                'Delta = full-input minus masked-input performance. Positive means degradation.','']
        rows=[]
        for task in cfg['tasks']:
            q=d[(d.task==task)&(d.model=='full_cbm')] if not d.empty else d
            if not q.empty:rows.append([TASK_LABELS[task],*[interval(q[q.metric==m].iloc[0]) for m in METRICS]])
        lines+=[markdown(['Task',*[METRIC_LABELS[m]+' drop [95% CI]' for m in METRICS]],rows),'',
                '## 3. Predicted concept-channel restoration','',
                'Recovery = restored minus masked performance. Restore each group separately from the same participant’s full-input predicted concepts. '
                'Val ranks are based on the seven individual groups and held fixed on test; joint-six and all-12 are separate comparisons.','']
        top=[]
        for task in cfg['tasks']:
            q=r[r.task==task] if not r.empty else r
            if q.empty:continue
            ranks=read_json(root/task/'VAL_TOP3.json')['groups'];rows=[]
            for group,part in q.groupby('concept_group',sort=False):
                rank=str(ranks.index(group)+1) if group in ranks else '—'
                rows.append([group,rank,*[interval(part[part.metric==m].iloc[0]) for m in METRICS]])
                if group in ranks:
                    part=part.copy();part['val_rank']=ranks.index(group)+1;top.append(part)
            selected=read_json(root/task/'selection.json')['fit_status']['joint_lambda_1']
            label='sequential fallback' if selected['status']=='no_accepted_joint_update' else f'joint epoch {selected["best_epoch"]}'
            lines+=['### '+TASK_LABELS[task],'',f'N={int(q.n.iloc[0])}; observed nonbasic clinical={int(q.n_affected.iloc[0])}; CBM: {label}.','',
                    markdown(['Restored concepts','Val top-3 rank',*[METRIC_LABELS[m]+' recovery [95% CI]' for m in METRICS]],rows),'']
        (pd.concat(top,ignore_index=True) if top else pd.DataFrame()).to_csv(tables/f'top3_{split}.csv',index=False)
        lines += ['All-12 restoration reproduces the full-input CBM decision and serves as a reference for complete channel recovery. '
                  'The six cardiometabolic concepts are BMI, systolic/diastolic BP, HbA1c, HDL and LDL. Group effects are not additive.','',
                  'Full Clinical inputs include variables used as concept labels. This experiment measures access to that structured information and its removal; '
                  'it does not demonstrate recovering it from independent nonclinical modalities. Restoration uses an unavailable full-input reference at deployment, '
                  'not measured-concept injection or a learned missing-modality replacement.','',
                  f'Pointwise 95% intervals use {cfg["bootstrap_samples"]} paired participant bootstrap samples stratified by outcome. '
                  'They exclude training-seed and selection uncertainty. The test split has been inspected previously. '
                  'Detailed CSVs include full, masked and restored performance, and the remaining gap to the full-input CBM.']
        reference_path=Path(__file__).resolve().parent/'report_update/CONCEPT_REFERENCE.md'
        if reference_path.exists():lines+=['','## Concept definitions','',reference_path.read_text()]
        (tables/f'SUMMARY_FULL_CBM_{split}.md').write_text('\n'.join(lines)+'\n')
    print('FULL_CBM_TABLES: '+str(tables),flush=True)
