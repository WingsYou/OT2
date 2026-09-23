#!/usr/bin/env python3
"""One-command, four-GPU fitting and val/test tables; no raw preprocessing."""
import argparse
import fcntl
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import yaml

HERE=Path(__file__).resolve().parent

def pool(jobs,devices,config,root,after=None):
    waiting=list(jobs);free=list(map(str,devices));running={};failed=[];selector=selectors.DefaultSelector()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    def launch(job,gpu):
        name,args=job;env=os.environ.copy();cpu=gpu=='cpu'
        if not cpu:env['CUDA_VISIBLE_DEVICES']=visible.split(',')[int(gpu)] if visible else gpu
        command=[sys.executable,'-u',str(HERE/'worker.py'),'--config',str(config),*args]+(['--cpu'] if cpu else [])
        log=(root/'logs'/(name+'.log')).open('ab',buffering=0)
        proc=subprocess.Popen(command,cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              bufsize=0,start_new_session=True)
        running[name]={'proc':proc,'gpu':gpu,'log':log,'buffer':b'','eof':False}
        selector.register(proc.stdout,selectors.EVENT_READ,name)
        print(f'[start] {name} on GPU {gpu} (PID {proc.pid})',flush=True)
    try:
        last=time.monotonic()
        while waiting or running:
            while waiting and free:launch(waiting.pop(0),free.pop(0))
            events=selector.select(1) if selector.get_map() else []
            if not events:time.sleep(.05)
            for key,_ in events:
                name=key.data;item=running[name];chunk=os.read(key.fd,65536)
                if not chunk:selector.unregister(key.fileobj);item['eof']=True;continue
                item['log'].write(chunk);item['buffer']+=chunk
                while b'\n' in item['buffer']:
                    line,item['buffer']=item['buffer'].split(b'\n',1)
                    print('['+name+'] '+line.decode(errors='replace'),flush=True)
            for name,item in list(running.items()):
                code=item['proc'].poll()
                if code is None or not item['eof']:continue
                if item['buffer']:print(item['buffer'].decode(errors='replace'),flush=True)
                if code:failed.append(name)
                print(f'[finished] {name}, exit={code}',flush=True)
                item['log'].close();item['proc'].stdout.close();free.append(item['gpu']);del running[name]
                if after:after()
            if time.monotonic()-last>30:
                print('[running] '+', '.join(running)+f'; queued={len(waiting)}',flush=True);last=time.monotonic()
        if failed:raise RuntimeError('Failed units: '+', '.join(failed)+'. Fix the reported error and rerun the same command; completed units are reused.')
    finally:
        for item in running.values():
            proc=item['proc']
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            item['log'].close();proc.stdout.close()
        selector.close()

def main():
    p=argparse.ArgumentParser(description='F1/BA six-task study from existing processed inputs/cache.')
    p.add_argument('--config',type=Path,default=HERE/'study.yaml')
    p.add_argument('--devices',nargs='+');p.add_argument('--source-root',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--stage',choices=('all','train','analyze','report'),default='all')
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    for key in ('source_root','output_root','devices'):
        if getattr(a,key) is not None:cfg[key]=getattr(a,key)
    for key in ('source_root','output_root','clinical_dir'):
        cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    cfg['devices']=list(map(str,cfg['devices']))
    root=Path(cfg['output_root']);source=Path(cfg['source_root'])
    if root==source:raise ValueError('Use a new output directory for the F1 rerun; source_root is read-only.')
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=str(cfg['cpu_threads'])
    from common import read_json,write_json
    from report import make_report
    runtime=root/'runtime.yaml'
    if runtime.exists():
        previous=yaml.safe_load(runtime.read_text())
        # Preserve a resumed experiment; allow only scheduling resources to change.
        resources=('devices','num_workers','cpu_threads')
        if {k:v for k,v in previous.items() if k not in resources}!={k:v for k,v in cfg.items() if k not in resources}:
            raise ValueError('Settings differ from this output. Use its runtime.yaml or choose a new output-root.')
    if a.stage=='report':make_report(cfg);return
    with (root/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A process is already using this output directory') from None
        runtime.write_text(yaml.safe_dump(cfg,sort_keys=False))
        status=dict(status='running',stage=a.stage,tasks=cfg['tasks']);write_json(root/'STATUS.json',status)
        try:
            if a.stage in ('all','train'):
                # These are existing normalized tensors and frozen ResNet features,
                # not fitted representations from the previous experiment.
                read_json(source/'inputs.json');read_json(source/'cache/dimensions.json')
                if not (root/'labels/DONE.json').exists():
                    from cbm.labels import extract,FOUR_OBSERVATION_FORMS,SURVEY_ADMIN_IDS
                    extract(cfg['clinical_dir'],root/'labels',observation_forms=FOUR_OBSERVATION_FORMS,excluded_ids=SURVEY_ADMIN_IDS)
                    write_json(root/'labels/DONE.json',dict(clinical_dir=cfg['clinical_dir']))
                jobs=[(task+'_'+variant,['--task',task,'--variant',variant]) for task in cfg['tasks'] for variant in ('full','basic')]
                pool(jobs,cfg['devices'],runtime,root)
            if a.stage in ('all','analyze'):
                jobs=[(task+'_analysis',['--task',task,'--analyze']) for task in cfg['tasks']]
                pool(jobs,cfg['devices'],runtime,root,after=lambda:make_report(cfg))
            status['status']='complete';write_json(root/'STATUS.json',status);make_report(cfg)
        except BaseException as exc:
            status.update(status='failed',error=str(exc));write_json(root/'STATUS.json',status);raise
    print('F1_STUDY_COMPLETE: '+str(root/'tables'),flush=True)

if __name__=='__main__':main()
