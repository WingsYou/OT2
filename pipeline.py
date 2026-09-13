#!/usr/bin/env python3
"""Explicit preparation or existing-input training; no preprocessing by default."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

HERE = Path(__file__).resolve().parent
PREP = HERE / 'data_preprocessing_cluster'
ORIGINAL = ('moca_binary', 'cvd_broad_binary', 'cvd_strict_binary')


def load_paths(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ('participants', 'clinical_dir'):
        cfg[key] = cfg[key].format(dataset_root=cfg['dataset_root'])
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
    for task in ORIGINAL:
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('raw','labels','train','report','all'),
                   help='raw: preprocess; labels: bind history/concepts; train/report: existing inputs; all: raw through tables')
    p.add_argument('--pipeline-config', type=Path, default=HERE/'pipeline.yaml')
    p.add_argument('--study-config', type=Path, default=HERE/'study.yaml')
    p.add_argument('--devices', nargs='+', default=None)
    p.add_argument('--output-root', type=Path, help='Override only the model/report output directory')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(); cfg = load_paths(a.pipeline_config)
    study_path = a.study_config.resolve()
    if a.stage in ('raw','all'):
        missing = [k for k,v in cfg['manifests'].items() if not v]
        if missing and not a.dry_run:
            p.error('Set the actual manifest paths in pipeline.yaml: '+', '.join(missing))
        # Regeneration gets its own directory; do not overwrite previous processed data.
        if not a.dry_run:
            root = Path(cfg['prepared_root']); root.mkdir(parents=True, exist_ok=False)
            (root/'scalers').mkdir()
            (root/'model_full_assets_v1/inputs').mkdir(parents=True)
        for cmd in raw_commands(cfg): execute(cmd, a.dry_run)
        study_path = Path(cfg['prepared_root'])/'study.inputs.yaml'
        if not a.dry_run:
            study = yaml.safe_load(a.study_config.read_text()); study['artifact_root'] = cfg['prepared_root']
            study['output_root'] = cfg['new_output_root']
            study_path.write_text(yaml.safe_dump(study, sort_keys=False))
    if a.stage in ('labels','all'):
        target = Path(cfg['prepared_root'])/'study.prepared.yaml'
        print(f'Prepare history labels/masters and original12 train/val labels -> {target}', flush=True)
        if not a.dry_run:
            from prepare_study_inputs import prepare
            prepare(cfg, study_path, target)
        study_path = target
    if a.stage in ('train','report','all'):
        cmd = [sys.executable, '-u', str(HERE/'run.py'), '--config', str(study_path),
               '--stage', 'report' if a.stage=='report' else 'all']
        if a.devices: cmd += ['--devices', *a.devices]
        if a.output_root: cmd += ['--output-root', str(a.output_root)]
        execute(cmd, a.dry_run)
        if a.dry_run:
            print('Append concept definitions and F1 details to the resulting tables/SUMMARY_*.md')
        else:
            out = (a.output_root or Path(yaml.safe_load(study_path.read_text())['output_root'])).expanduser().resolve()
            execute([sys.executable, str(HERE/'report_update/enrich_summaries.py'),
                     '--results-root', str(out), '--out-dir', str(out/'tables')])


if __name__ == '__main__':
    main()
