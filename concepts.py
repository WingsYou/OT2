"""Fit the existing original12 CBM, then evaluate frozen predicted-channel recovery."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import (GROUPS,JOINT_GROUP,JOINT_MEMBERS,FOCAL,REMOVALS,Bootstrap,effect_rows,
                    threshold,read_json,write_json,save_torch,save_csv)
from cbm_joint_v2 import DEFAULTS,fit_experiment,restore,predict,classify

def run_cbm(task,cfg,device):
    task_root=Path(cfg['output_root'])/task;root=task_root/'cbm';root.mkdir(exist_ok=True)
    if (root/'DONE.json').exists():print('[reuse CBM] '+task,flush=True);return
    basic=task_root/'basic';entry=read_json(Path(cfg['output_root'])/'inputs.json')[task]
    load=lambda split:torch.load(basic/('features_'+split+'.pt'),map_location='cpu',weights_only=False)
    train,val=load('train'),load('val')
    label_dir=Path(entry['concept_dir']);schema=read_json(label_dir/'concept_schema.json')
    labels=pd.read_csv(label_dir/'concept_labels.csv',dtype={'person_id':str})
    if len(schema['concept_names'])!=12:raise ValueError('This run uses the audited original12 concept bank')
    write_json(root/'concept_schema.json',schema)
    path=root/'models.pt'
    if path.exists():artifacts=torch.load(path,map_location='cpu',weights_only=False)
    else:
        options={**DEFAULTS,'seed':cfg['seed'],'concept_epochs':cfg['concept_epochs'],
            'head_epochs':cfg['head_epochs'],'joint_epochs':cfg['joint_epochs'],
            'patience':cfg['concept_patience'],'lambdas':[1.]}
        artifacts,history=fit_experiment(train,val,labels,schema,config=options,device=device)
        save_torch(path,artifacts);pd.DataFrame(history).to_csv(root/'training.csv',index=False)
    models={arm:restore(artifact,device=device) for arm,artifact in artifacts.items()}
    thresholds={arm:threshold(val['y'].numpy(),predict(model,val['embedding'])['probabilities']) for arm,model in models.items()}
    write_json(root/'selection.json',{'bank':'original12','primary_arm':'joint_lambda_1',
        'threshold_source':'validation_youden','thresholds':thresholds,
        'fit_status':{a:v['selection'] for a,v in artifacts.items()},'focal_source':FOCAL[task],
        'restoration_scope':'any_target_present','rank_selection_split':'val'})
    classification=[];effects=[];model=models['joint_lambda_1'];cut=thresholds['joint_lambda_1']
    groups={**GROUPS,JOINT_GROUP:JOINT_MEMBERS}
    columns={name:[i for block in model.spec['layout'] if block['name'] in members for i in range(block['start'],block['stop'])]
             for name,members in groups.items()}
    for split in ('val','test'):
        cache=val if split=='val' else load(split);y=cache['y'].numpy()
        boot=Bootstrap(y,cfg['bootstrap_samples'],cfg['seed'],device)
        all_pred={'person_id':cache['person_ids'],'y':y}
        for arm,m in models.items():
            full=predict(m,cache['embedding']);p=full['probabilities']
            classification+=effect_rows(task,split,'classification',boot.metrics(p,thresholds[arm]),
                model='cbm_'+arm,n=len(y),threshold=thresholds[arm])
            all_pred[arm]=p
        pd.DataFrame(all_pred).to_csv(root/('predictions_'+split+'.csv'),index=False)
        full=predict(model,cache['embedding']);removed=torch.load(basic/('removals_'+split+'.pt'),map_location='cpu',weights_only=False)
        restored_arrays={}
        for source,result in removed.items():
            eligible=result['eligible'];n=int(eligible.sum());sub_y=y[eligible]
            if len(np.unique(sub_y))<2:
                print(f'[skip recovery] {task}/{split}/{source}: eligible cohort lacks two classes',flush=True);continue
            sub=Bootstrap(sub_y,cfg['bootstrap_samples'],cfg['seed'],device)
            missing=predict(model,result['embedding'])
            missing_metrics=sub.metrics(missing['probabilities'][eligible],cut)
            full_metrics=sub.metrics(full['probabilities'][eligible],cut)
            harm=full_metrics-missing_metrics
            for group,indices in columns.items():
                z=missing['concepts'].copy();z[:,indices]=full['concepts'][:,indices]
                probability=classify(model,z);restored_arrays[source+'__'+group]=probability
                recovery=sub.metrics(probability[eligible],cut)-missing_metrics
                rows=effect_rows(task,split,'recovery',recovery,removed_input=source,concept_group=group,
                    group_type='joint_followup' if group==JOINT_GROUP else 'individual_group',n=n,threshold=cut)
                for j,row in enumerate(rows):
                    lo,hi=np.quantile(harm[1:,j],[.025,.975]);row.update(initial_harm=float(harm[0,j]),
                        harm_ci_lower=float(lo),harm_ci_upper=float(hi))
                effects+=rows
        np.savez_compressed(root/('restored_predictions_'+split+'.npz'),**restored_arrays)
        if split=='val':
            candidates=[r for r in effects if r['split']=='val' and r['removed_input']==FOCAL[task]
                        and r['metric']=='auroc' and r['group_type']=='individual_group']
            top=sorted(candidates,key=lambda r:(-r['value'],r['concept_group']))[:3]
            write_json(root/'VAL_TOP3.json',{'task':task,'source':FOCAL[task],
                'groups':[r['concept_group'] for r in top],'ranking_metric':'AUROC recovery',
                'ranking_split':'val','test_ranked':False})
        print(f'[CBM evaluation] {task}/{split} complete',flush=True)
    save_csv(root/'classification.csv',pd.DataFrame(classification))
    save_csv(root/'restoration.csv',pd.DataFrame(effects))
    write_json(root/'DONE.json',{'task':task,'splits':['val','test'],'concept_bank':'original12'})
