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
    p.add_argument('--mixed-training',action='store_true',help='Train one CBM on full + masked views; select/evaluate on masked inputs only')
    p.add_argument('--full-cbm-root',type=Path,help='Previous Full Clinical CBM directory for cache/model reuse')
    a=p.parse_args()
    source=a.source_root.expanduser().resolve()
    cfg=yaml.safe_load((source/'runtime.yaml').read_text())
    cfg['output_root']=str(source)
    folder='mixed_clinical_cbm' if a.mixed_training else 'full_clinical_cbm'
    root=(a.output_root or source/folder).expanduser().resolve()
    if a.report_only and (root/'runtime.yaml').exists():
        cfg=yaml.safe_load((root/'runtime.yaml').read_text())
        cfg['output_root']=str(source)
    if a.mixed_training:
        cfg['mixed_cbm_root']=str(root)
        cfg['full_cbm_root']=str((a.full_cbm_root or Path(cfg.get('full_cbm_root',source/'full_clinical_cbm'))).expanduser().resolve())
    else:cfg['full_cbm_root']=str(root)
    if a.devices:cfg['devices']=a.devices
    if a.tasks:cfg['tasks']=a.tasks
    if a.bootstrap_samples is not None:cfg['bootstrap_samples']=a.bootstrap_samples
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]=str(cfg['cpu_threads'])
    if a.mixed_training:
        from mixed_clinical_cbm import make_report
    else:
        from full_clinical_cbm import make_report
    if a.report_only:
        make_report(cfg);return
    from run import pool
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    runtime=root/'runtime.yaml';runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
    jobs=[(task,['--task',task,'--mixed-cbm' if a.mixed_training else '--full-cbm']) for task in cfg['tasks']]
    pool(jobs,cfg['devices'],runtime,root,after=lambda:make_report(cfg))
    make_report(cfg)
    print(('MIXED_CBM_COMPLETE: ' if a.mixed_training else 'FULL_CLINICAL_CBM_COMPLETE: ')+str(root/'tables'),flush=True)


if __name__=='__main__':main()
