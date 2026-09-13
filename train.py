"""Fresh multimodal fitting; fixed ResNet features and trainable original pooling."""
import math
import random
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from common import (MODALITIES,REMOVALS,Bootstrap,effect_rows,threshold,read_json,write_json,save_torch,save_csv,sha)
from data import CachedDataset,clinical_inputs,collate,cache_config
from task2_v2_model import Task2V2Model
from artifact_helpers import canonical_json_hash

class CachedModel(Task2V2Model):
    def _retinal(self,batch,group):
        features=batch[group+'_features'];valid=batch[group+'_view_mask']
        encoder=self.retinal_encoder
        tokens=encoder.view_projection(features)*valid.unsqueeze(-1)
        assets=encoder.view_pool(tokens,valid);present=valid.any(-1)
        value=encoder.group_projections[group](encoder.asset_pools[group](assets,present))
        return value,present.any(-1)

def move(batch,device):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
def amp(device):return torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')
def loader(dataset,cfg,shuffle=False):
    return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=cfg['num_workers'],
        collate_fn=collate,pin_memory=True,persistent_workers=bool(cfg['num_workers']),
        **({'multiprocessing_context':'spawn'} if cfg['num_workers'] else {}))

@torch.no_grad()
def predict_loader(model,dl,task,device):
    model.eval();prob=[];truth=[]
    for batch in dl:
        b=move(batch,device)
        with amp(device):logits=model(b,task=task)['logits'].reshape(-1)
        prob.append(logits.float().sigmoid().cpu().numpy());truth.append(b['y'].cpu().numpy())
    return np.concatenate(truth),np.concatenate(prob)

def fit(model,train_dl,val_dl,task,cfg,root,device):
    if (root/'TRAINING_DONE.json').exists():
        model.load_state_dict(torch.load(root/'best.pt',map_location=device,weights_only=False)['model'])
        return read_json(root/'TRAINING_DONE.json')
    y=train_dl.dataset.master.iloc[train_dl.dataset.indices].y.to_numpy(float)
    weight=torch.tensor((len(y)-y.sum())/y.sum(),device=device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start=0;best=-math.inf;bad=0;best_epoch=0;history=[]
    if (root/'last.pt').exists():
        last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer'])
        start=last['epoch'];best=last['best'];bad=last['bad'];best_epoch=last['best_epoch'];history=last['history']
        torch.set_rng_state(last['rng']);np.random.set_state(last['numpy_rng']);random.setstate(last['python_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state(last['cuda_rng'],device)
        print(f'[resume training] {task} epoch {start}',flush=True)
    total_steps=cfg['max_epochs']*len(train_dl);warmup=max(1,int(.1*total_steps))
    for epoch in range(start,cfg['max_epochs']):
        if bad>=cfg['patience']:break
        model.train();loss_sum=torch.zeros((),device=device)
        for step,batch in enumerate(train_dl):
            number=epoch*len(train_dl)+step+1
            factor=number/warmup if number<=warmup else .5*(1+math.cos(math.pi*(number-warmup)/max(1,total_steps-warmup)))
            for group in opt.param_groups:group['lr']=cfg['learning_rate']*factor
            b=move(batch,device);opt.zero_grad(set_to_none=True)
            with amp(device):
                logits=model(b,task=task)['logits'].reshape(-1)
                loss=F.binary_cross_entropy_with_logits(logits,b['y'],pos_weight=weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=loss.detach()
        truth,prob=predict_loader(model,val_dl,task,device);score=float(roc_auc_score(truth,prob))
        record={'epoch':epoch+1,'train_loss':float(loss_sum)/len(train_dl),'val_auroc':score};history.append(record)
        if score>best+1e-8:
            best=score;best_epoch=epoch+1;bad=0
            save_torch(root/'best.pt',{'model':model.state_dict(),'model_config':asdict(model.config),
                'epoch':best_epoch,'val_auroc':score,'threshold':threshold(truth,prob)})
        else:bad+=1
        save_torch(root/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,
            'best':best,'bad':bad,'best_epoch':best_epoch,'history':history,
            'rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
        pd.DataFrame(history).to_csv(root/'training.csv',index=False)
        print(f'[epoch {task} {root.name}] {epoch+1}/{cfg["max_epochs"]} loss={record["train_loss"]:.4f} val_AUROC={score:.4f} best={best:.4f}',flush=True)
    best_state=torch.load(root/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_state['model'])
    result={'task':task,'best_epoch':best_epoch,'epochs_completed':history[-1]['epoch'],
            'validation_auroc':best,'threshold':best_state['threshold'],'threshold_source':'validation_youden',
            'checkpoint_sha256':sha(root/'best.pt'),'model_config':asdict(model.config),
            'config_hash':canonical_json_hash({'model':asdict(model.config),'settings':cfg})}
    write_json(root/'TRAINING_DONE.json',result);return result

@torch.no_grad()
def export(model,dl,task,split,device,root,info,basic_indices=None):
    path=root/('features_'+split+'.pt')
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    model.eval();collected={k:[] for k in ('embedding','source_tokens','source_present','original_logits','clinical_values','clinical_mask','y')}
    ids=[];masked_logits=[]
    for batch in dl:
        b=move(batch,device);ids.extend(batch['person_id'])
        with amp(device):
            sources=model.encode_sources(b);out=model.predict_from_source_tokens(**sources,task=task)
            if basic_indices is not None:
                values=torch.zeros_like(b['clinical_values']);mask=torch.zeros_like(b['clinical_mask'])
                values[:,basic_indices]=b['clinical_values'][:,basic_indices];mask[:,basic_indices]=b['clinical_mask'][:,basic_indices]
                token,present=model.clinical_encoder(values,mask);tokens=sources['source_tokens'].clone();availability=sources['source_present'].clone()
                present=present&availability[:,0];tokens[:,0]=token*present.unsqueeze(-1);availability[:,0]=present
                masked_logits.append(model.predict_from_source_tokens(tokens,availability,task=task)['logits'].reshape(-1).float().cpu())
        for key in ('embedding','source_tokens'):collected[key].append(out[key].float().cpu())
        collected['source_present'].append(out['source_present'].cpu())
        collected['original_logits'].append(out['logits'].reshape(-1).float().cpu())
        for key in ('clinical_values','clinical_mask'):collected[key].append(b[key].cpu())
        collected['y'].append(b['y'].long().cpu())
    result={k:torch.cat(v) for k,v in collected.items()}
    result.update(schema_version='ot2-cbm-features-v1',person_ids=ids,split=split,task=task,
        identity={'task':task,'checkpoint_sha256':info['checkpoint_sha256'],'config_hash':info['config_hash']})
    if masked_logits:result['basic_inference_logits']=torch.cat(masked_logits)
    save_torch(path,result);return result

@torch.no_grad()
def source_outputs(model,cache,task,device):
    tokens=cache['source_tokens'].to(device);presence=cache['source_present'].to(device)
    results={}
    for name,members in REMOVALS.items():
        indices=[MODALITIES.index(g) for g in members];z=tokens.clone();m=presence.clone();z[:,indices]=0;m[:,indices]=False
        with amp(device):out=model.predict_from_source_tokens(z,m,task=task)
        results[name]={'embedding':out['embedding'].float().cpu(),'probability':out['logits'].reshape(-1).float().sigmoid().cpu().numpy(),
                       'eligible':presence[:,indices].any(1).cpu().numpy()}
    return results

def run_model(task,variant,cfg,device):
    root=Path(cfg['output_root'])/task/variant;root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():print(f'[reuse] {task}/{variant}',flush=True);return
    entry=read_json(Path(cfg['output_root'])/'inputs.json')[task]
    master,values,mask,basic,names,excluded=clinical_inputs(task,cfg,entry)
    if variant=='basic':values=values[:,basic];mask=mask[:,basic];names=[names[i] for i in basic]
    write_json(root/'input_columns.json',{'clinical_features':names,'target_columns_removed':excluded,
        'clinical_source':entry['clinical_csv'],'full_clinical_source_task':entry['full_clinical_source_task']})
    datasets={split:CachedDataset(cfg['output_root'],master,values,mask,task,split) for split in ('train','val','test')}
    dims=read_json(Path(cfg['output_root'])/'cache/dimensions.json');dims['clinical_dim']=len(names)
    mc=replace(cache_config(cfg,dims),task_names=(task,))
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    model=CachedModel(mc).to(device)
    train_dl=loader(datasets['train'],cfg,shuffle=True);val_dl=loader(datasets['val'],cfg)
    info=fit(model,train_dl,val_dl,task,cfg,root,device)
    del train_dl,val_dl
    rows=[];ablation=[]
    for split in (('train','val','test') if variant=='basic' else ('val','test')):
        cache=export(model,loader(datasets[split],cfg),task,split,device,root,info,
            basic_indices=basic if variant=='full' else None)
        if split=='train':continue
        y=cache['y'].numpy();p=cache['original_logits'].sigmoid().numpy();cut=info['threshold']
        boot=Bootstrap(y,cfg['bootstrap_samples'],cfg['seed'],device);base=boot.metrics(p,cut)
        rows+=effect_rows(task,split,'classification',base,model=variant,n=len(y),threshold=cut)
        pred={'person_id':cache['person_ids'],'y':y,'probability':p,'prediction':(p>=cut).astype(int)}
        if variant=='full':
            pm=cache['basic_inference_logits'].sigmoid().numpy()
            rows+=effect_rows(task,split,'classification',boot.metrics(pm,cut),model='basic_inference',n=len(y),threshold=cut)
            pred['basic_inference_probability']=pm
        else:
            removed=source_outputs(model,cache,task,device);save_torch(root/('removals_'+split+'.pt'),removed)
            for source,result in removed.items():
                estimate=boot.metrics(result['probability'],cut)
                ablation+=effect_rows(task,split,'degradation',base-estimate,removed_input=source,n=len(y),threshold=cut)
                pred['without__'+source]=result['probability']
        pd.DataFrame(pred).to_csv(root/('predictions_'+split+'.csv'),index=False)
    save_csv(root/'classification.csv',pd.DataFrame(rows))
    if ablation:save_csv(root/'ablation.csv',pd.DataFrame(ablation))
    write_json(root/'DONE.json',{'task':task,'variant':variant,'threshold':info['threshold'],'splits':['val','test']})
