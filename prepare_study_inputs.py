"""Bind audited history labels and original12 supervision to processed inputs.

No tensor preprocessing, scaling, model fitting or test-concept supervision here.
History full-clinical views retain the study's existing MoCA-view definition;
data.clinical_inputs removes the current history endpoint before model input.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'data_preprocessing_cluster'))
from build_history_task_labels_v1 import build as build_history
from build_cbm_concepts_v1 import build_labels
from history_tasks_v1 import task_hash, SPEC_PATH
from preparation_helpers import canonical_hash
from common import original_config, FULL_CONFIGS, HISTORY_TARGETS, read_json, write_json
from data import people_csv, absolute, PATH_COLUMNS
from artifact_helpers import as_person_id


def bind_history(study, labels_dir, destination):
    frames = []
    for task in FULL_CONFIGS:
        ds = original_config(task, study)['dataset']
        frame = people_csv(ds['master_csv'])
        frame['person_id'] = frame.person_id.map(as_person_id)
        # Whitelist input metadata; never carry a parent endpoint into a new task.
        flags = [c for c in frame if c.startswith('has_')]
        columns = ['person_id','split',*flags,*PATH_COLUMNS,'retinal_asset_paths_json',
                   *[c for c in ('cgm_config_hash','activity_config_hash','wearable_config_hash') if c in frame]]
        for col in PATH_COLUMNS:
            frame[col] = frame.get(col, pd.Series('', index=frame.index)).map(lambda x:absolute(x, ds['artifact_root']))
        frame['retinal_asset_paths_json'] = frame.retinal_asset_paths_json.map(
            lambda x:json.dumps(sorted(absolute(p,ds['artifact_root']) for p in json.loads(x))))
        for col in flags:
            frame[col] = frame[col].astype(str).str.lower().isin(('1','true'))
        frames.append(frame[columns])
    joined = pd.concat(frames, ignore_index=True).fillna('')
    compare = [c for c in joined if c not in ('person_id','has_clinical')]
    if (joined.groupby('person_id')[compare].nunique()>1).any().any():
        raise ValueError('Parent masters disagree on split or nonclinical inputs')
    presence = joined.groupby('person_id').has_clinical.any()
    union = joined.drop_duplicates('person_id').set_index('person_id')
    union['has_clinical'] = presence
    parent = original_config('moca_binary', study)['dataset']
    clinical_path = absolute(people_csv(parent['master_csv']).clinical_features_path.iloc[0], parent['artifact_root'])
    for task in (t for t in study['tasks'] if t in HISTORY_TARGETS):
        labels = people_csv(Path(labels_dir)/(task+'_labels.csv'))
        labels['person_id'] = labels.person_id.map(as_person_id)
        if labels.person_id.duplicated().any(): raise ValueError('Duplicate history label IDs')
        missing = set(labels.person_id)-set(union.index)
        if missing: raise ValueError(f'{task}: {len(missing)} labeled participants absent from parent input union')
        master = union.loc[labels.person_id].reset_index()
        if master.split.tolist()!=labels.split.tolist(): raise ValueError('History split mismatch')
        for col in ('task','task_config_hash','y','label_value','label_datetime'):
            master[col] = labels[col].to_numpy()
        master['clinical_features_path'] = clinical_path
        folder = Path(destination)/'inputs'/task; folder.mkdir(parents=True)
        schema = copy.deepcopy(read_json(parent['clinical_schema_json']))
        derivation = {'parent_task':'moca_binary','parent_schema':parent['clinical_schema_json'],
                      'clinical_target_exclusion':'Applied by data.clinical_inputs before full/basic model input',
                      'scalers':'Existing fitted numbers copied; not refitted'}
        schema.update(task=task, task_config=str(SPEC_PATH), task_config_hash=task_hash(task),
                      preprocessing_config={'task':task,'task_config_hash':task_hash(task),
                                            'feature_names':schema['feature_names'],'derivation':derivation})
        schema['config_hash'] = canonical_hash(schema['preprocessing_config'])
        scaler = copy.deepcopy(read_json(parent['scaler_json']))
        scaler['source_config_hashes']['clinical'].update(task=task,task_config_hash=task_hash(task),
            config_hash=schema['config_hash'],schema_hash=canonical_hash(schema))
        scaler['history_derivation'] = derivation
        scaler.pop('scaler_config_hash',None); scaler['scaler_config_hash'] = canonical_hash(scaler)
        meta = copy.deepcopy(read_json(parent['master_metadata_json']))
        meta.update(task=task,n_rows=len(master),split_counts=master.split.value_counts().to_dict(),
                    clinical_schema_hash=canonical_hash(schema))
        meta['preprocessing_config'] = {'task':task,'task_config_hash':task_hash(task),
            'clinical_schema_hash':canonical_hash(schema),'derivation':derivation,
            'source_index_config_hashes':meta['preprocessing_config']['source_index_config_hashes']}
        meta['config_hash'] = canonical_hash(meta['preprocessing_config'])
        ds = copy.deepcopy(parent); ds['task'] = task
        for key,name in (('master_csv','master.csv'),('master_metadata_json','master.metadata.json'),
                         ('scaler_json','scalers.json'),('clinical_schema_json','clinical_schema.json')):
            ds[key] = str(folder/name)
        master.to_csv(folder/'master.csv',index=False)
        for name,payload in (('clinical_schema.json',schema),('scalers.json',scaler),('master.metadata.json',meta)):
            write_json(folder/name,payload)
        (folder/'train_config.yaml').write_text(yaml.safe_dump({'dataset':ds},sort_keys=False))


def prepare(paths, study_path, target):
    study = yaml.safe_load(Path(study_path).read_text())
    root = Path(paths['prepared_root']).resolve()
    work = root/'supplemental_inputs'
    work.mkdir(parents=True, exist_ok=False)
    clinical = Path(paths['clinical_dir'])
    build_history(paths['participants'],clinical/'observation.csv',clinical/'condition_occurrence.csv',work/'history_labels')
    bind_history(study,work/'history_labels',work/'history')
    study['history_root'] = str(work/'history')
    cohorts = []
    for task in study['tasks']:
        frame = people_csv(original_config(task,study)['dataset']['master_csv'])
        frame['person_id'] = frame.person_id.map(as_person_id)
        cohorts.append(frame[['person_id','split']])
    cohort = pd.concat(cohorts).drop_duplicates()
    if cohort.person_id.duplicated().any(): raise ValueError('Conflicting official splits')
    cohort = cohort[cohort.split.isin(('train','val'))].sort_values('person_id')
    cohort.to_csv(work/'concept_cohort_train_val.csv',index=False)
    print('Extract original12 concepts once for the union of train/val participants',flush=True)
    shared = work/'concepts_shared'
    build_labels(clinical/'measurement.csv',clinical/'observation.csv',cohort.person_id.tolist(),shared,
                 split_by_person=dict(zip(cohort.person_id,cohort.split)))
    for task in study['tasks']:
        shutil.copytree(shared,work/'concepts'/task/'labels')
    study['concept_root'] = study['history_concept_root'] = str(work/'concepts')
    study['output_root'] = paths['new_output_root']
    Path(target).write_text(yaml.safe_dump(study,sort_keys=False))
    print('Prepared study config: '+str(target),flush=True)
