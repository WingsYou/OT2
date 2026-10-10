#!/usr/bin/env python3
"""Prepare the released six-task study from a separately downloaded AI-READI dataset.

The original preprocessing builders are reused unchanged. The shared cache
includes every participant, including people without a valid legacy endpoint.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PREP = HERE / 'data_preprocessing_cluster'
# Historical preprocessing namespaces are intermediate files, not study endpoints.
PREPROCESSING_TASKS = ('moca_binary', 'cvd_broad_binary', 'cvd_strict_binary')
CURRENT_TASKS = ('moca_binary', 'cardiovascular_conditions_binary',
                 'major_cardiovascular_events_binary', 'osteoporosis_history_binary',
                 'kidney_history_binary', 'lung_history_binary')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_config(path):
    """Repository-relative defaults also work when invoked from another directory."""
    cfg = yaml.safe_load(Path(path).read_text())

    def absolute(value):
        value = Path(str(value)).expanduser()
        return str((value if value.is_absolute() else HERE / value).resolve())

    for key in ('dataset_root', 'prepared_root', 'source_root'):
        cfg[key] = absolute(cfg[key])
    for key, default in (('participants', '{dataset_root}/participants.tsv'),
                         ('clinical_dir', '{dataset_root}/clinical_data')):
        cfg[key] = absolute(cfg.get(key, default).format(dataset_root=cfg['dataset_root']))
    cfg['manifests'] = {key: absolute(value.format(dataset_root=cfg['dataset_root']))
                        if value else None for key, value in cfg['manifests'].items()}
    if cfg.get('retinal_weights_path'):
        cfg['retinal_weights_path'] = absolute(cfg['retinal_weights_path'])
    for key, value in dict(workers=8, num_workers=2, cpu_threads=2,
                           cache_people_per_batch=4, retinal_batch_size=128).items():
        cfg.setdefault(key, value)
    return cfg


def command(script, *args):
    return [sys.executable, '-u', str(PREP / script), *map(str, args)]


def raw_commands(cfg):
    """Use the supplied Cluster preprocessing definitions and train-only scalers."""
    root = Path(cfg['prepared_root']); people = cfg['participants']
    dataset = cfg['dataset_root']; clinical = cfg['clinical_dir']; n = cfg['workers']
    m = {k: v or '<SET_' + k + '_MANIFEST>' for k, v in cfg['manifests'].items()}
    yield command('build_downstream_labels.py', '--participants', people,
                  '--clinical_dir', clinical, '--out_dir', root/'labels')
    yield command('build_retinal_asset_index_v2.py', '--participants', people,
                  *[x for k in ('retinal_photography','retinal_oct','retinal_octa','retinal_flio')
                    for x in ('--'+k, m[k])], '--out_dir', root/'retinal_index')
    yield command('preprocess_retinal_assets_v2.py', '--asset-index', root/'retinal_index/retinal_asset_index.csv',
                  '--dataset-root', dataset, '--out-dir', root/'retinal', '--num-workers', n)
    yield command('build_retinal_model_view_sidecars_v2.py', '--retinal-index', root/'retinal/retinal_preprocessed_index.csv',
                  '--out-dir', root/'retinal_model_views', '--num-workers', n,
                  '--max-photography-assets', 6, '--max-oct-assets', 2, '--max-octa-assets', 16, '--max-flio-assets', 4)
    yield command('build_ecg_waveform_features_v2.py', '--manifest', m['cardiac_ecg'],
                  '--dataset-root', dataset, '--out-dir', root/'ecg', '--num-workers', n)
    yield command('build_environment_5min_sequences.py', '--participants', people,
                  '--manifest', m['environment'], '--dataset_root', dataset,
                  '--out_dir', root/'environment_v4', '--num_workers', n)
    yield command('build_wearable_5min_sequences.py', '--participants', people,
                  '--glucose_manifest', m['wearable_blood_glucose'], '--activity_manifest', m['wearable_activity_monitor'],
                  '--dataset_root', dataset, '--out_dir', root/'wearable_v5', '--num_workers', n)
    for task in PREPROCESSING_TASKS:
        folder = root/'clinical'/task
        wide = folder/f'clinical_features_{task}_wide.csv'; schema = folder/f'clinical_schema_{task}.json'
        yield command('build_clinical_tabular_features_v2.py', '--participants', people,
                      '--clinical_dir', clinical, '--task', task, '--out_dir', folder,
                      '--include_condition', '--include_procedure')
        yield command('fit_train_scalers_v2.py', '--participants', people,
                      '--clinical-wide', wide, '--clinical-schema', schema,
                      '--ecg-numeric-wide', root/'ecg/ecg_numeric_features_wide.csv',
                      '--ecg-channel-stats', root/'ecg/ecg_channel_stats.csv', '--ecg-schema', root/'ecg/ecg_schema.json',
                      '--environment-index', root/'environment_v4/environment_5min_index.csv',
                      '--environment-artifact-dir', root/'environment_v4', '--environment-schema', root/'environment_v4/environment_5min_schema.json',
                      '--wearable-index', root/'wearable_v5/wearable_5min_index.csv',
                      '--wearable-artifact-dir', root/'wearable_v5', '--wearable-schema', root/'wearable_v5/wearable_5min_schema.json',
                      '--output', root/f'scalers/train_scalers_{task}.json')
        yield command('build_multimodal_index_v2.py', '--participants', people,
                      '--labels', root/f'labels/labels_{task}.csv', '--clinical_features', wide, '--clinical_schema', schema,
                      '--retinal_index', root/'retinal_model_views/retinal_model_view_index.csv',
                      '--ecg_index', root/'ecg/ecg_index.csv', '--environment_index', root/'environment_v4/environment_5min_index.csv',
                      '--wearable_index', root/'wearable_v5/wearable_5min_index.csv', '--artifact_root', root,
                      '--out_csv', root/f'model_full_assets_v1/inputs/master_{task}.csv',
                      '--metadata_json', root/f'model_full_assets_v1/inputs/master_{task}.metadata.json')


def execute(cmd, dry=False):
    print(shlex.join(list(map(str, cmd))), flush=True)
    if not dry:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PREP)
        subprocess.run(cmd, check=True, cwd=PREP, env=env)


def run_raw(cfg, dry_run=False):
    root = Path(cfg['prepared_root'])
    marker = root / 'completed_commands.json'
    done = json.loads(marker.read_text()) if marker.exists() else []
    if not dry_run:
        missing = [key for key, value in cfg['manifests'].items() if not value]
        if missing:
            raise ValueError('Set manifest paths in the preparation config: ' + ', '.join(missing))
        (root / 'scalers').mkdir(parents=True, exist_ok=True)
        (root / 'model_full_assets_v1/inputs').mkdir(parents=True, exist_ok=True)
    for command in raw_commands(cfg):
        if command in done:
            print('[preprocessing] reuse ' + Path(command[2]).name, flush=True)
            continue
        execute(command, dry_run)
        if not dry_run:
            done.append(command)
            write_json(marker, done)


def make_cache_index(cfg):
    """Join processed modality artifacts to all official participants, without y."""
    sys.path.insert(0, str(PREP))
    from build_multimodal_index_v2 import _single_artifact_view, _retinal_view
    from preprocessing_v2 import as_person_id, canonical_json_hash

    prepared, source = Path(cfg['prepared_root']), Path(cfg['source_root'])
    people = pd.read_csv(cfg['participants'], sep='\t', dtype={'person_id': str})
    people['person_id'] = people.person_id.map(as_person_id)
    frame = people[['person_id', 'recommended_split']].rename(columns={'recommended_split': 'split'})
    participant_ids = set(frame.person_id)
    metadata = {}
    for name, relative, candidates in (
        ('environment', 'environment_v4/environment_5min_index.csv', ('artifact_path', 'out_path')),
        ('wearable', 'wearable_v5/wearable_5min_index.csv', ('out_path', 'artifact_path')),
        ('ecg', 'ecg/ecg_index.csv', ('artifact_path', 'out_path')),
    ):
        view, metadata[name] = _single_artifact_view(
            prepared / relative, name=name, path_candidates=candidates,
            artifact_root=None, allow_missing=False, participant_ids=participant_ids)
        frame = frame.merge(view, on='person_id', how='left', validate='one_to_one')
        frame[name + '_path'] = frame[name + '_path'].fillna('')
        frame['has_' + name] = frame[name + '_artifact_exists'].eq(True)
    frame['has_environment'] &= frame.environment_observed_slots.fillna(0).gt(0)
    if 'ecg_model_leads' in frame:
        frame['has_ecg'] &= frame.ecg_model_leads.fillna(0).gt(0)
    for short, column in (('cgm', 'cgm_observed_slots'), ('activity', 'activity_observed_slots')):
        frame['has_' + short] = frame.has_wearable & frame[column].fillna(0).gt(0)
    retinal, metadata['retinal'] = _retinal_view(
        prepared / 'retinal_model_views/retinal_model_view_index.csv',
        artifact_root=None, allow_missing=False, participant_ids=participant_ids)
    frame = frame.merge(retinal, on='person_id', how='left', validate='one_to_one')
    frame['retinal_asset_paths_json'] = frame.retinal_asset_paths_json.fillna('[]')
    for name in ('photography', 'oct', 'octa', 'flio'):
        column = 'retinal_' + name + '_asset_count'
        frame['has_retinal_' + name] = frame[column].fillna(0).gt(0)
    clinical = prepared / 'clinical/moca_binary'
    schema = json.loads((clinical / 'clinical_schema_moca_binary.json').read_text())
    frame['clinical_features_path'] = str(clinical / 'clinical_features_moca_binary_wide.csv')
    frame['task'] = 'moca_binary'  # Reference scaler namespace; no labels enter this cache.
    frame['task_config_hash'] = schema['task_config_hash']
    frame['has_clinical'] = False  # Clinical values are loaded per task during fitting.
    frame = frame.sort_values('person_id').reset_index(drop=True)
    cache = source / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache / 'index.csv', index=False)
    write_json(cache / 'index.metadata.json', dict(
        task='moca_binary', n_rows=len(frame), source_indexes=metadata,
        clinical_schema_hash=canonical_json_hash(schema),
        cohort_definition='All participants.tsv rows; no endpoint eligibility filter'))
    raw_ids = set(map(as_person_id, schema['raw_clinical_person_ids']))
    has_age = pd.to_numeric(people['age'], errors='coerce').notna()
    write_json(source / 'full_clinical_presence.json', dict(zip(
        people.person_id, (people.person_id.isin(raw_ids) | has_age).tolist())))
    return frame


def make_registry(cfg, cache_index):
    """Keep historical clinical views and their existing endpoint-exclusion rules."""
    from common import original_config, FULL_CONFIGS, HISTORY_TARGETS
    from artifact_helpers import as_person_id
    sys.path.insert(0, str(PREP))
    from build_history_task_labels_v1 import build

    source = Path(cfg['source_root'])
    entries = {}
    for task in FULL_CONFIGS:
        ds = original_config(task, {'artifact_root': cfg['prepared_root']})['dataset']
        entries[task] = dict(master=ds['master_csv'], scaler=ds['scaler_json'],
                            clinical_schema=ds['clinical_schema_json'],
                            clinical_csv=str(Path(cfg['prepared_root']) / 'clinical' / task /
                                             ('clinical_features_' + task + '_wide.csv')),
                            full_clinical_source_task=task)
    labels_dir = source / 'history_labels'
    clinical = Path(cfg['clinical_dir'])
    build(cfg['participants'], clinical / 'observation.csv',
          clinical / 'condition_occurrence.csv', labels_dir)
    index = cache_index.set_index('person_id')
    presence = json.loads((source / 'full_clinical_presence.json').read_text())
    for task in HISTORY_TARGETS:
        labels = pd.read_csv(labels_dir / (task + '_labels.csv'), dtype={'person_id': str})
        labels['person_id'] = labels.person_id.map(as_person_id)
        master = index.loc[labels.person_id].reset_index()
        if master.split.tolist() != labels.split.tolist():
            raise ValueError(task + ': history labels disagree with official participant splits')
        for column in ('task', 'task_config_hash', 'y', 'label_value', 'label_datetime'):
            master[column] = labels[column].to_numpy()
        master['has_clinical'] = master.person_id.map(presence)
        folder = source / 'inputs' / task
        folder.mkdir(parents=True, exist_ok=True)
        master.to_csv(folder / 'master.csv', index=False)
        entries[task] = {**entries['moca_binary'], 'master': str(folder / 'master.csv')}
    write_json(source / 'inputs.json', entries)


def cache_settings(cfg):
    return {**cfg, 'artifact_root': cfg['prepared_root'], 'output_root': cfg['source_root']}


def cache_shard(config, shard, count, cpu):
    cfg = load_config(config)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(cfg['cpu_threads'])
    import torch
    from data import build_cache
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_float32_matmul_precision('high')
    build_cache(cache_settings(cfg), shard, count, torch.device('cpu' if cpu else 'cuda:0'))


def run_cache(cfg, config_path, devices):
    source = Path(cfg['source_root'])
    source.mkdir(parents=True, exist_ok=True)
    index = make_cache_index(cfg)
    make_registry(cfg, index)
    runtime = source / 'preparation.yaml'
    runtime.write_text(yaml.safe_dump({**cfg, 'devices': devices}, sort_keys=False))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    processes = []
    try:
        for shard, gpu in enumerate(devices):
            env = os.environ.copy()
            if gpu != 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[int(gpu)] if visible else gpu
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(config_path),
                       '--cache-shard', str(shard), '--shard-count', str(len(devices))]
            if gpu == 'cpu':
                command.append('--cpu')
            processes.append(subprocess.Popen(command, env=env, cwd=HERE))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError('Cache workers failed; rerun --stage cache to reuse completed participants')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    from prepare_cardiovascular_inputs import prepare
    cardiovascular_root = prepare(dict(
        source_root=str(source), output_root=str(source),
        prepared_inputs_root=str(source / 'cardiovascular_inputs'),
        clinical_dir=cfg['clinical_dir'], participants_tsv=cfg['participants'],
        tasks=list(CURRENT_TASKS[1:3])))
    registry = json.loads((cardiovascular_root / 'inputs.json').read_text())
    # Expose just the six reported endpoints to downstream runs. The two legacy
    # clinical views remain intermediate inputs, not additional experiments.
    write_json(source / 'inputs.json', {task: registry[task] for task in CURRENT_TASKS})
    write_json(source / 'PREPARATION_COMPLETE.json', dict(
        participants=len(index), tasks=list(CURRENT_TASKS), cache=str(source / 'cache')))
    print('PREPARATION_COMPLETE: ' + str(source), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'pipeline.yaml')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--stage', choices=('all', 'raw', 'cache'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cache-shard', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--shard-count', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--cpu', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cache_shard is not None:
        cache_shard(args.config, args.cache_shard, args.shard_count, args.cpu)
        return
    cfg = load_config(args.config)
    if args.stage in ('all', 'raw'):
        run_raw(cfg, args.dry_run)
    if args.stage in ('all', 'cache'):
        if args.dry_run:
            print('Build label-free cache for all participants -> ' + cfg['source_root'])
            print('Prepare current six task labels and clinical input registry')
        else:
            run_cache(cfg, args.config.resolve(), list(map(str, args.devices or cfg.get('devices', [0, 1, 2, 3]))))


if __name__ == '__main__':
    main()
