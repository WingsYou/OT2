#!/usr/bin/env python3
"""Extend a completed six-task run with Full Clinical CBMs; reuse all preprocessing and models."""
import argparse
import os
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True,help='Completed six-task directory containing runtime.yaml and task/full/best.pt')
    p.add_argument('--output-root',type=Path,help='Defaults to <source-root>/full_clinical_cbm')
    p.add_argument('--devices',nargs='+',default=None)
    p.add_argument('--tasks',nargs='+',help='Optional subset of the six tasks')
    p.add_argument('--bootstrap-samples',type=int)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    source=a.source_root.expanduser().resolve()
    cfg=yaml.safe_load((source/'runtime.yaml').read_text())
    cfg['output_root']=str(source)
    root=(a.output_root or source/'full_clinical_cbm').expanduser().resolve()
    if a.report_only and (root/'runtime.yaml').exists():
        cfg=yaml.safe_load((root/'runtime.yaml').read_text())
        cfg['output_root']=str(source)
    cfg['full_cbm_root']=str(root)
    if a.devices:cfg['devices']=a.devices
    if a.tasks:cfg['tasks']=a.tasks
    if a.bootstrap_samples is not None:cfg['bootstrap_samples']=a.bootstrap_samples
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]=str(cfg['cpu_threads'])
    from full_clinical_cbm import make_report
    if a.report_only:
        make_report(cfg);return
    from run import pool
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    runtime=root/'runtime.yaml';runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
    jobs=[(task,['--task',task,'--full-cbm']) for task in cfg['tasks']]
    pool(jobs,cfg['devices'],runtime,root,after=lambda:make_report(cfg))
    make_report(cfg)
    print('FULL_CLINICAL_CBM_COMPLETE: '+str(root/'tables'),flush=True)


if __name__=='__main__':main()
