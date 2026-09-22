"""Read existing processed artifacts; cache frozen retinal features once."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from common import (MODALITIES,FULL_CONFIGS,BASIC_KEYS,HISTORY_TARGETS,original_config,
                    concept_dir,read_json,write_json,save_torch)
from task2_v2_dataset import (OT2V2Dataset,FrozenScalerGroup,SOURCE_FLAG_BY_GROUP,
                            RETINAL_GROUPS,_pad_temporal)
from task2_v2_model import Task2V2ModelConfig,build_retinal_backbone
from artifact_helpers import as_person_id

PATH_COLUMNS=('environment_path','wearable_path','ecg_path','activity_path','cgm_path')

def people_csv(path):return pd.read_csv(path,dtype={'person_id':str},keep_default_na=False)
def absolute(value,root):
    if not value:return ''
    p=Path(value);return str(p if p.is_absolute() else Path(root)/p)

def prepare(cfg):
    out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    full_presence=[]
    for task in FULL_CONFIGS:
        source=people_csv(original_config(task,cfg)['dataset']['master_csv'])[['person_id','has_clinical']]
        source['person_id']=source.person_id.map(as_person_id)
        source['has_clinical']=source.has_clinical.astype(str).str.lower().isin(('true','1'))
        full_presence.append(source)
    full_presence=pd.concat(full_presence).groupby('person_id').has_clinical.any().to_dict()
    write_json(out/'full_clinical_presence.json',full_presence)
    entries={};frames=[];reference=None
    for task in cfg['tasks']:
        base=original_config(task,cfg);ds=base['dataset'];frame=people_csv(ds['master_csv'])
        frame['person_id']=frame.person_id.map(as_person_id)
        if frame.person_id.duplicated().any():raise ValueError('Duplicate participants in '+ds['master_csv'])
        native_scalers=read_json(ds['scaler_json'])['scalers']
        signature={k:{'feature_order':v['feature_order'],'features':v['features']} for k,v in native_scalers.items() if k!='clinical'}
        if reference is not None and signature!=reference:raise ValueError('Nonclinical scaler statistics differ across source tasks; cannot share this cache')
        reference=signature
        for col in PATH_COLUMNS:frame[col]=frame.get(col,pd.Series('',index=frame.index)).map(lambda s:absolute(s,ds['artifact_root']))
        frame['retinal_asset_paths_json']=frame.retinal_asset_paths_json.map(lambda s:json.dumps(sorted(absolute(p,ds['artifact_root']) for p in json.loads(s))))
        clinical_base=base if task in FULL_CONFIGS else original_config('moca_binary',cfg)
        cds=clinical_base['dataset'];cm=people_csv(cds['master_csv'])
        cpath=absolute(cm.clinical_features_path.iloc[0],cds['artifact_root'])
        entries[task]={'master':ds['master_csv'],'clinical_csv':cpath,'scaler':cds['scaler_json'],
                       'clinical_schema':cds['clinical_schema_json'],
                       'concept_dir':str(concept_dir(task,cfg)),'full_clinical_source_task':cds['task']}
        frames.append(frame)
    joined=pd.concat(frames,ignore_index=True).fillna('')
    nonclinical_flags=[SOURCE_FLAG_BY_GROUP[g] for g in MODALITIES[1:]]
    keys=['split',*PATH_COLUMNS,'retinal_asset_paths_json',*nonclinical_flags]
    for col in nonclinical_flags:joined[col]=joined[col].astype(str).str.lower().replace({'1':'true','0':'false'})
    if (joined.groupby('person_id')[keys].nunique()>1).any().any():
        raise ValueError('A participant has conflicting official splits or modality artifacts across existing masters')
    joined=joined.drop_duplicates('person_id').sort_values('person_id').reset_index(drop=True)
    base=original_config('moca_binary',cfg);ds=base['dataset']
    columns=['person_id',*keys,*[c for c in ('cgm_config_hash','activity_config_hash','wearable_config_hash') if c in joined]]
    index=joined[columns].copy();index['task']='moca_binary';index['has_clinical']=False
    index['clinical_features_path']=entries['moca_binary']['clinical_csv'] if 'moca_binary' in entries else absolute(people_csv(ds['master_csv']).clinical_features_path.iloc[0],ds['artifact_root'])
    index['task_config_hash']=read_json(ds['scaler_json'])['source_config_hashes']['clinical']['task_config_hash']
    cache=out/'cache';cache.mkdir(exist_ok=True);index.to_csv(cache/'index.csv',index=False)
    meta=read_json(ds['master_metadata_json']);meta['n_rows']=len(index)
    write_json(cache/'index.metadata.json',meta)
    write_json(out/'inputs.json',entries)
    # Only bookkeeping is written. No label rebuilding, scaler fitting or raw preprocessing.
    return entries

def raw_dataset(cfg):
    base=original_config('moca_binary',cfg);ds=base['dataset']
    return OT2V2Dataset(Path(cfg['output_root'])/'cache/index.csv',ds['scaler_json'],
        task='moca_binary',artifact_root=ds['artifact_root'],environment_schema_json=ds['environment_schema_json'],
        max_retinal_assets_per_group={f'retinal_{k}':v for k,v in ds['retinal']['max_assets'].items()},
        loaded_source_groups=MODALITIES[1:],require_y=False)

def cache_config(cfg,dimensions):
    weights=cfg.get('retinal_weights_path')
    return Task2V2ModelConfig(**dimensions,retinal_backbone='resnet18',retinal_backbone_dim=512,
        retinal_weights=None if weights else 'IMAGENET1K_V1',retinal_weights_path=weights,
        retinal_backbone_frozen=True,retinal_input_normalization='imagenet1k_v1')

def keep_list(samples):return samples

@torch.no_grad()
def build_cache(cfg,shard,count,device):
    root=Path(cfg['output_root'])/'cache';ds=raw_dataset(cfg)
    dimensions=ds.model_feature_dimensions
    dimensions={k:dimensions[k] for k in ('clinical_dim','environment_dim','activity_dim','cgm_dim','ecg_waveform_dim','ecg_numeric_dim')}
    if shard==0:write_json(root/'dimensions.json',dimensions)
    indices=[i for i in range(shard,len(ds),count) if not (root/'people'/(str(ds.index.person_id.iloc[i])+'.pt')).exists()]
    cnn=build_retinal_backbone(cache_config(cfg,dimensions)).to(device).eval()
    loader=DataLoader(Subset(ds,indices),batch_size=cfg['cache_people_per_batch'],shuffle=False,
        num_workers=cfg['num_workers'],collate_fn=keep_list,pin_memory=True,
        **({'multiprocessing_context':'spawn'} if cfg['num_workers'] else {}))
    completed=0
    for samples in loader:
        images=[];masks=[];records=[]
        for sample in samples:
            for group in RETINAL_GROUPS:
                for asset in sample[group]:
                    valid=asset['view_mask'].nonzero().flatten();asset['features']=torch.zeros(len(asset['views']),512)
                    if len(valid):
                        images.append(asset['views'][valid]);masks.append(asset['pixel_mask'][valid]);records.append((asset,valid))
        if images:
            pixels=torch.cat(images);pixel_masks=torch.cat(masks);pixels=pixels*pixel_masks;features=[]
            for start in range(0,len(pixels),cfg['retinal_batch_size']):
                stop=start+cfg['retinal_batch_size']
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    features.append(cnn(pixels[start:stop].to(device),pixel_masks[start:stop].to(device)).float().cpu())
            encoded=torch.cat(features);offset=0
            for asset,valid in records:
                asset['features'][valid]=encoded[offset:offset+len(valid)];offset+=len(valid)
        for sample in samples:
            for group in RETINAL_GROUPS:
                sample[group]=[{'features':a['features'],'view_mask':a['view_mask']} for a in sample[group]]
            for key in ('y','label_value','clinical_values','clinical_mask'):sample.pop(key,None)
            save_torch(root/'people'/(sample['person_id']+'.pt'),sample)
        completed+=len(samples)
        if completed%40<len(samples) or completed==len(indices):print(f'[cache {shard}] {completed}/{len(indices)} participants',flush=True)
    print(f'[cache {shard}] complete',flush=True)

def clinical_inputs(task,cfg,entry):
    master=people_csv(entry['master']);ids=master.person_id.map(as_person_id).tolist()
    scaler=FrozenScalerGroup.from_payload('clinical',read_json(entry['scaler'])['scalers']['clinical'])
    table=pd.read_csv(entry['clinical_csv'],dtype={'person_id':str})
    table['person_id']=table.person_id.map(as_person_id);table=table.set_index('person_id')
    names=list(scaler.feature_order);raw=torch.as_tensor(table.loc[ids,names].to_numpy(dtype=np.float32))
    if task in FULL_CONFIGS:
        acquired=master.has_clinical.astype(str).str.lower().isin(('true','1')).to_numpy()
    else:
        presence=read_json(Path(cfg.get('source_root',cfg['output_root']))/'full_clinical_presence.json')
        acquired=np.asarray([presence[pid] for pid in ids])
    mask=torch.isfinite(raw)&torch.as_tensor(acquired).unsqueeze(1)
    values=scaler.normalise(raw,mask,'clinical')
    keep=list(range(len(names)));excluded=[]
    if task in HISTORY_TARGETS:
        source,cid=HISTORY_TARGETS[task]
        excluded=[i for i,n in enumerate(names) if n.split('__')[1]==source or (n.startswith('cond__') and n.endswith('__c'+str(cid)))]
        keep=[i for i in keep if i not in excluded]
    retained=[names[i] for i in keep]
    basic=[i for i,n in enumerate(retained) if n.split('__')[1] in BASIC_KEYS]
    if len(basic)!=23:raise ValueError(f'{task}: expected the agreed Basic23, found {len(basic)}')
    master=master.copy();master['person_id']=ids
    return master,values[:,keep],mask[:,keep],basic,retained,[names[i] for i in excluded]

class CachedDataset(Dataset):
    def __init__(self,root,master,values,mask,task,split):
        self.root=Path(root)/'cache/people';self.task=task;self.split=split
        self.indices=np.flatnonzero(master.split.to_numpy()==split);self.master=master.reset_index(drop=True)
        self.values=values;self.mask=mask
    def __len__(self):return len(self.indices)
    @lru_cache(maxsize=64)
    def native(self,pid):return torch.load(self.root/(pid+'.pt'),map_location='cpu',weights_only=False,mmap=True)
    def __getitem__(self,i):
        j=int(self.indices[i]);row=self.master.iloc[j];sample=dict(self.native(row.person_id))
        sample.update(task=self.task,split=self.split,y=torch.tensor(float(row.y)),
            clinical_values=self.values[j],clinical_mask=self.mask[j])
        sample['source_present']=sample['source_present'].clone();sample['source_present'][0]=self.mask[j].any()
        sample['has_clinical']=sample['source_present'][0]
        return sample

def collate(samples):
    batch={'person_id':[s['person_id'] for s in samples]}
    for key in ('clinical_values','clinical_mask','ecg_lead_mask','ecg_numeric','ecg_numeric_mask','source_present','y'):
        batch[key]=torch.stack([s[key] for s in samples])
    for flag in SOURCE_FLAG_BY_GROUP.values():batch[flag]=torch.stack([s[flag] for s in samples])
    for value,masks in (('ecg_waveform',('ecg_value_mask','ecg_sample_mask')),
        ('environment',('environment_feature_mask','environment_slot_mask')),
        ('activity',('activity_feature_mask','activity_observed_mask','activity_valid_mask')),
        ('cgm',('cgm_feature_mask','cgm_observed_mask','cgm_valid_mask'))):
        batch.update(_pad_temporal(samples,value,masks))
    for group in RETINAL_GROUPS:
        na=max(1,max(len(s[group]) for s in samples));nv=max(1,max((len(a['features']) for s in samples for a in s[group]),default=0))
        features=torch.zeros(len(samples),na,nv,512);valid=torch.zeros(len(samples),na,nv,dtype=torch.bool)
        for b,sample in enumerate(samples):
            for a,asset in enumerate(sample[group]):
                n=len(asset['features']);features[b,a,:n]=asset['features'];valid[b,a,:n]=asset['view_mask']
        batch[group+'_features']=features;batch[group+'_view_mask']=valid
    return batch
