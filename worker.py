"""One GPU worker. The parent assigns its visible GPU before Python starts."""
import argparse
import os
import yaml

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--task')
    p.add_argument('--variant',choices=('full','basic'));p.add_argument('--cache-shard',type=int)
    p.add_argument('--cache-shards',type=int);p.add_argument('--cpu',action='store_true')
    p.add_argument('--full-cbm',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(cfg['cpu_threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(cfg['cpu_threads']);torch.set_float32_matmul_precision('high')
    device=torch.device('cpu' if a.cpu else 'cuda:0')
    if device.type=='cuda':torch.backends.cudnn.benchmark=True
    if a.full_cbm:
        from full_clinical_cbm import run_task
        run_task(a.task,cfg,device)
    elif a.cache_shard is not None:
        from data import build_cache
        build_cache(cfg,a.cache_shard,a.cache_shards,device)
    else:
        from train import run_model
        run_model(a.task,a.variant,cfg,device)
        if a.variant=='basic':
            torch.cuda.empty_cache()
            from concepts import run_cbm
            run_cbm(a.task,cfg,device)

if __name__=='__main__':main()
