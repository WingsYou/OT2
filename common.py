"""Study definitions and shared, GPU-batched paired bootstrap metrics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_curve

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
 'wearable_all':MODALITIES[7:],'nonclinical_all':MODALITIES[1:]}
INPUT_LABELS=dict(zip(MODALITIES,('Basic clinical information','Electrocardiography (ECG)',
 'Environmental measurements','Retinal FLIO','Retinal OCT','Retinal OCTA','Retinal photography',
 'Wearable activity monitoring','Wearable blood glucose monitoring')))
INPUT_LABELS.update(retinal_all='All retinal modalities',wearable_all='All wearable modalities',
 nonclinical_all='All eight nonclinical modalities')
GROUPS={'BMI':('bmi',),'Blood pressure':('systolic_bp','diastolic_bp'),'HbA1c':('hba1c',),
 'Lipids':('hdl_cholesterol','ldl_cholesterol'),
 'Visual function':('photopic_logmar_right','photopic_logmar_left','photopic_log_contrast_right','photopic_log_contrast_left'),
 'Activity':('self_reported_activity',),'Restless sleep':('restless_sleep',)}
JOINT_GROUP='Six cardiometabolic concepts'
JOINT_MEMBERS=('bmi','systolic_bp','diastolic_bp','hba1c','hdl_cholesterol','ldl_cholesterol')
FOCAL={'moca_binary':'retinal_all','cvd_broad_binary':'cardiac_ecg',
 'cvd_strict_binary':'wearable_activity_monitor','osteoporosis_history_binary':'cardiac_ecg',
 'kidney_history_binary':'wearable_all','lung_history_binary':'retinal_all'}
METRICS=('auroc','balanced_accuracy','f1')
METRIC_LABELS={'auroc':'AUROC','balanced_accuracy':'Balanced accuracy','f1':'F1'}

def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def save_torch(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
def save_csv(path,frame):
    path=Path(path);temp=path.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def settings(path):
    cfg=yaml.safe_load(Path(path).read_text());cfg['config_path']=str(Path(path).resolve())
    return cfg
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
def threshold(y,p):
    fpr,tpr,t=roc_curve(y,p);valid=np.isfinite(t)
    return float(t[valid][np.argmax((tpr-fpr)[valid])])

class Bootstrap:
    """Same stratified participant draws for all paired comparisons in a cohort."""
    def __init__(self,y,samples=2000,seed=42,device='cuda'):
        self.y=np.asarray(y,dtype=int);n=len(self.y);rng=np.random.default_rng(seed)
        w=np.ones((samples+1,n),dtype=np.int64)
        for label in (0,1):
            idx=np.flatnonzero(self.y==label)
            if not len(idx):raise ValueError('Evaluation cohort needs both outcome classes')
            w[1:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=samples)
        self.w=torch.as_tensor(w,device=device);self.fw=self.w.double();self.device=device
        self.pos=self.fw@torch.as_tensor(self.y,dtype=torch.float64,device=device)
        self.neg=self.fw.sum(1)-self.pos
    def metrics(self,p,cut):
        p=np.asarray(p,dtype=float).reshape(-1)
        order=np.argsort(p,kind='stable');s=p[order];ys=self.y[order]
        start=np.flatnonzero(np.r_[True,s[1:]!=s[:-1]])
        end=np.r_[start[1:]-1,len(s)-1];repeats=end-start+1
        lo=np.repeat(start,repeats);hi=np.repeat(end,repeats)
        sw=self.w[:,torch.as_tensor(order,device=self.device)]
        sy=torch.as_tensor(ys,device=self.device)
        cneg=torch.cumsum(sw*(1-sy),dim=1)
        before=torch.cat((cneg[:,:1]*0,cneg[:,:-1]),dim=1)
        mid=(before[:,lo].double()+cneg[:,hi].double())*.5
        auc=(sw*sy*mid).sum(1)/(self.pos*self.neg)
        pred=p>=cut
        tp=self.fw@torch.as_tensor(pred&(self.y==1),dtype=torch.float64,device=self.device)
        fp=self.fw@torch.as_tensor(pred&(self.y==0),dtype=torch.float64,device=self.device)
        fn=self.pos-tp;tn=self.neg-fp
        ba=.5*(tp/self.pos+tn/self.neg)
        f1=2*tp/(2*tp+fp+fn).clamp_min(1)
        return torch.stack((auc,ba,f1),dim=1).cpu().numpy()

def effect_rows(task,split,kind,draws,**extra):
    rows=[]
    for j,metric in enumerate(METRICS):
        values=draws[:,j];lo,hi=np.quantile(values[1:],[.025,.975])
        rows.append(dict(task=task,split=split,kind=kind,metric=metric,value=float(values[0]),
                         ci_lower=float(lo),ci_upper=float(hi),**extra))
    return rows
