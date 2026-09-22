"""Study definitions, paths, and small output helpers."""
from pathlib import Path
import json
import sys
import torch
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from task2_v2_model import PRIMARY_MODALITY_GROUPS as MODALITIES

TASK_LABELS={'moca_binary':'MoCA < 26','cvd_broad_binary':'CVD broad','cvd_strict_binary':'CVD strict',
 'osteoporosis_history_binary':'Osteoporosis history','kidney_history_binary':'Kidney-problem history',
 'lung_history_binary':'Lung-problem history'}
FULL_CONFIGS={'moca_binary':'moca','cvd_broad_binary':'cvd_broad','cvd_strict_binary':'cvd_strict'}
BASIC_KEYS={'age','cage','brthyy','years_of_education','pxed1','cl_maristat','pxep1','pxep2','pxep3'}
HISTORY_TARGETS={'osteoporosis_history_binary':('mhoccur_oa',80502),
 'kidney_history_binary':('mhoccur_rnl',2005200017),'lung_history_binary':('mhoccur_plm',4186898)}
REMOVALS={**{x:(x,) for x in MODALITIES},'retinal_all':MODALITIES[3:7],
 'wearable_all':MODALITIES[7:]}
INPUT_LABELS=dict(zip(MODALITIES,('Basic clinical information','Electrocardiography (ECG)',
 'Environmental measurements','Retinal FLIO','Retinal OCT','Retinal OCTA','Retinal photography',
 'Wearable activity monitoring','Wearable blood glucose monitoring')))
INPUT_LABELS.update(retinal_all='All retinal modalities',wearable_all='All wearable modalities')
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def save_torch(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
def save_csv(path,frame):
    path=Path(path);temp=path.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(path)
def original_config(task,cfg):
    if task in FULL_CONFIGS:
        data=yaml.safe_load((HERE/'configs'/(FULL_CONFIGS[task]+'_full.yaml')).read_text())
        old='/data/pzhao3/ot2_output';new=str(cfg['artifact_root'])
        data=json.loads(json.dumps(data).replace(old,new))
        return data
    return yaml.safe_load((Path(cfg['history_root'])/'inputs'/task/'train_config.yaml').read_text())
def concept_dir(task,cfg):
    root=cfg['concept_root'] if task in FULL_CONFIGS else cfg['history_concept_root']
    return Path(root)/task/'labels'
