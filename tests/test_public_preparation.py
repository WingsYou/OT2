"""Small synthetic checks for the public raw-to-cache bridge; no study data."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from prepare_public_inputs import HERE, load_config, make_cache_index


class PublicPreparationTest(unittest.TestCase):
    def test_relative_paths_and_dataset_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.yaml'
            config.write_text(yaml.safe_dump(dict(
                dataset_root='data/example', prepared_root='outputs/native',
                source_root='outputs/prepared',
                manifests={'ecg': '{dataset_root}/cardiac_ecg/manifest.tsv'})))
            settings = load_config(config)
            self.assertEqual(settings['participants'], str(HERE / 'data/example/participants.tsv'))
            self.assertEqual(settings['manifests']['ecg'], str(HERE / 'data/example/cardiac_ecg/manifest.tsv'))

    def test_cache_keeps_people_without_endpoint_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, source = root / 'preprocessed', root / 'prepared'
            participants = root / 'participants.tsv'
            pd.DataFrame(dict(person_id=['p1', 'p2', 'p3'],
                              recommended_split=['train', 'val', 'test'],
                              age=[51, 62, 70])).to_csv(participants, sep='\t', index=False)
            clinical = prepared / 'clinical/moca_binary'
            clinical.mkdir(parents=True)
            (clinical / 'clinical_schema_moca_binary.json').write_text(json.dumps(dict(
                task_config_hash='synthetic-task', raw_clinical_person_ids=['p1'])))

            def artifact(name):
                path = prepared / (name + '.pt')
                path.write_bytes(b'synthetic-index-target')
                return str(path)

            def index(path, rows):
                path = prepared / path
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(path, index=False)

            common = dict(status='processed', config_hash='synthetic-input')
            index('environment_v4/environment_5min_index.csv', [
                dict(common, person_id='p1', artifact_path=artifact('env1'),
                     n_observed_slots=10, slot_coverage_fraction=.5),
                dict(common, person_id='p2', artifact_path=artifact('env2'),
                     n_observed_slots=0, slot_coverage_fraction=0)])
            index('wearable_v5/wearable_5min_index.csv', [
                dict(common, person_id='p3', out_path=artifact('wear3'),
                     n_cgm_observed_slots=3, n_activity_observed_slots=0)])
            index('ecg/ecg_index.csv', [
                dict(common, person_id='p2', artifact_path=artifact('ecg2'), n_model_leads=0)])
            index('retinal_model_views/retinal_model_view_index.csv', [
                dict(common, person_id='p3', asset_id='one',
                     artifact_path=artifact('oct3'), modality='oct')])
            frame = make_cache_index(dict(prepared_root=str(prepared), source_root=str(source),
                                          participants=str(participants)))
            self.assertEqual(frame.person_id.tolist(), ['p1', 'p2', 'p3'])
            self.assertEqual(frame.split.tolist(), ['train', 'val', 'test'])
            self.assertNotIn('y', frame.columns)
            self.assertEqual(frame.has_environment.tolist(), [True, False, False])
            self.assertEqual(frame.has_ecg.tolist(), [False, False, False])
            self.assertEqual(frame.has_cgm.tolist(), [False, False, True])
            self.assertEqual(frame.has_activity.tolist(), [False, False, False])
            self.assertEqual(frame.has_retinal_oct.tolist(), [False, False, True])
            self.assertEqual(json.loads((source / 'cache/index.metadata.json').read_text())['n_rows'], 3)


if __name__ == '__main__':
    unittest.main()
