"""Prepare replacement CVD labels and clinical views; reuse all nonclinical caches.

Only endpoint-dependent clinical tables/scalers are rebuilt. The original
participant splits, normalized signals and frozen retinal features are reused.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
PREPROCESSING = HERE / 'data_preprocessing_cluster'
SPEC = PREPROCESSING / 'configs/cardiovascular_tasks_v2.json'
TASKS = ('cardiovascular_conditions_binary', 'major_cardiovascular_events_binary')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')


def people(path, sep=','):
    frame = pd.read_csv(path, sep=sep, dtype={'person_id': str}, keep_default_na=False)
    frame['person_id'] = frame.person_id.str.replace(r'\.0$', '', regex=True)
    return frame


def run_builder(script, args, log):
    with Path(log).open('w') as output:
        result = subprocess.run([sys.executable, '-u', str(PREPROCESSING / script), *map(str, args)],
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed; details: {log}')


def prepare(cfg):
    """Return a new input registry whose cache points to the existing source."""
    from common import BASIC_KEYS
    # The existing scaler's clinical-only function does not read other modalities.
    sys.path.insert(0, str(PREPROCESSING))
    from fit_train_scalers_v2 import fit_clinical, load_fit_population

    source = Path(cfg['source_root'])
    destination = Path(cfg.get('prepared_inputs_root') or Path(cfg['output_root']) / 'prepared_inputs')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'DONE.json').exists():
        print('[inputs] reuse ' + str(destination), flush=True)
        return destination
    clinical = Path(cfg['clinical_dir'])
    participants_path = Path(cfg.get('participants_tsv') or clinical.parent / 'participants.tsv')
    participants = people(participants_path, '\t')
    registry = read_json(source / 'inputs.json')
    reference = registry['cvd_broad_binary']
    reference_schema = read_json(reference['clinical_schema'])
    reference_scaler = read_json(reference['scaler'])
    options = reference_schema['preprocessing_config']
    population = load_fit_population(participants_path)
    tasks = [task for task in cfg['tasks'] if task in TASKS]
    labels_dir = destination / 'labels'
    labels_dir.mkdir(exist_ok=True)
    print('[inputs] rebuilding the two cardiovascular labels and their clinical views', flush=True)
    run_builder('build_downstream_labels.py',
                ['--participants', participants_path, '--clinical_dir', clinical,
                 '--task_config', SPEC, '--tasks', *tasks, '--out_dir', labels_dir],
                destination / 'label_build.log')

    index = people(source / 'cache/index.csv').set_index('person_id')
    entries = {}
    counts = []
    expected_basic = {name for name in reference_scaler['scalers']['clinical']['feature_order']
                      if name.split('__')[1] in BASIC_KEYS}
    for task in tasks:
        labels = people(labels_dir / f'labels_{task}.csv')
        absent = set(labels.person_id) - set(index.index)
        uncached = [pid for pid in labels.person_id if not (source / 'cache/people' / (pid + '.pt')).exists()]
        if absent or uncached:
            raise ValueError(f'{task}: {len(absent)} eligible patients lack a processed input index; '
                             f'{len(uncached)} lack a cached participant file. Complete the existing '
                             'source cache for these patients before training; no patients were dropped.')
        folder = destination / task
        folder.mkdir(exist_ok=True)
        args = ['--clinical_dir', clinical, '--participants', participants_path,
                '--task', task, '--task_config', SPEC, '--out_dir', folder,
                '--min_train_persons', options.get('min_train_persons', 20)]
        for key in ('include_condition', 'include_procedure'):
            if options.get(key):
                args.append('--' + key)
        run_builder('build_clinical_tabular_features_v2.py', args, folder / 'clinical_build.log')
        wide_path = folder / f'clinical_features_{task}_wide.csv'
        schema_path = folder / f'clinical_schema_{task}.json'
        schema = read_json(schema_path)
        basic = {name for name in schema['feature_names'] if name.split('__')[1] in BASIC_KEYS}
        if len(basic) != 23 or basic != expected_basic:
            raise ValueError(f'{task}: rebuilt Basic Clinical columns differ from the existing Basic23')
        clinical_scaler, source_info, _ = fit_clinical(wide_path, str(schema_path), population, None)
        scaler = copy.deepcopy(reference_scaler)
        scaler['scalers']['clinical'] = clinical_scaler
        scaler['source_config_hashes']['clinical'] = source_info
        # Legacy provenance fields describe the old clinical view, so replace
        # them with the actual derivation. Nonclinical fitted numbers are intact.
        scaler.pop('scaler_config_hash', None)
        scaler['clinical_replacement'] = {'task': task, 'participants': str(participants_path),
                                         'nonclinical_scalers_copied_from': reference['scaler']}
        scaler_path = folder / 'train_scalers.json'
        write_json(scaler_path, scaler)
        master = index.loc[labels.person_id].reset_index()
        if master.split.tolist() != labels.recommended_split.tolist():
            raise ValueError(f'{task}: existing cache splits differ from participants.tsv')
        for key in ('task', 'task_config_hash', 'y', 'label_value', 'label_datetime'):
            if key in labels:
                master[key] = labels[key].to_numpy()
        master['clinical_features_path'] = str(wide_path)
        master['has_clinical'] = master.person_id.isin(schema['raw_clinical_person_ids'])
        master_path = folder / 'master.csv'
        master.to_csv(master_path, index=False)
        entries[task] = {'master': str(master_path), 'clinical_csv': str(wide_path),
                         'scaler': str(scaler_path), 'clinical_schema': str(schema_path),
                         'task_specific_clinical': True, 'full_clinical_source_task': task}
        for split in ('train', 'val', 'test'):
            y = master.loc[master.split == split, 'y'].astype(int)
            cohort_n = int((participants.recommended_split == split).sum())
            counts.append({'task': task, 'split': split, 'cohort_n': cohort_n, 'n': len(y),
                           'positive': int(y.sum()), 'negative': int((y == 0).sum()),
                           'excluded': cohort_n - len(y), 'prevalence': float(y.mean())})
        print(f'[inputs] {task}: {len(master)} patients; {len(schema["feature_names"])} clinical '
              'features; Basic23 retained; existing nonclinical cache reused', flush=True)
    cache = destination / 'cache'
    if not cache.exists():
        cache.symlink_to((source / 'cache').resolve(), target_is_directory=True)
    # Keep other already-prepared task registrations available for mixed runs.
    write_json(destination / 'inputs.json', {**registry, **entries})
    pd.DataFrame(counts).to_csv(destination / 'LABEL_COUNTS.csv', index=False)
    write_json(destination / 'DONE.json', {'tasks': tasks, 'source_root': str(source),
                                          'participants': str(participants_path)})
    return destination


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(yaml.safe_load(args.config.read_text())))
