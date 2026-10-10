"""A GPU worker; the parent assigns one visible device per job."""
import argparse
import os
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--task',required=True)
    p.add_argument('--variant',choices=('full','basic'));p.add_argument('--analyze',action='store_true')
    p.add_argument('--concept-diagnostics',action='store_true')
    p.add_argument('--cpu',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(cfg['cpu_threads'])
    import torch
    torch.set_num_threads(cfg['cpu_threads']);torch.set_float32_matmul_precision('high')
    device=torch.device('cpu' if a.cpu else 'cuda:0')
    if device.type=='cuda':torch.backends.cudnn.benchmark=True
    if a.concept_diagnostics:
        from cbm.diagnostics import run_diagnostics
        run_diagnostics(a.task,cfg,device)
    elif a.analyze:
        from analysis import run_analysis
        run_analysis(a.task,cfg,device)
    else:
        from train import run_model
        if not cfg.get('backbone_root'):
            run_model(a.task,a.variant,cfg,device)
        if device.type=='cuda':torch.cuda.empty_cache()
        from cbm.experiment import run_cbm
        if a.variant=='full':
            run_cbm(a.task,cfg,device)


if __name__=='__main__':main()
