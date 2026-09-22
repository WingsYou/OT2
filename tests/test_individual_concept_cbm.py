"""Small synthetic checks for the new value types and restoration calculation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from common import Bootstrap
from individual_concept_labels import blocked, recorded_value
from individual_concept_model import prepare, restore, predict, date_days
from individual_concept_experiment import (metric_batch, restoration_batch, run_task, make_report,
                                           concept_id_groups, analysis_dir)


def fixture():
    rng = np.random.default_rng(42)
    definitions = [
        ('numeric', ['text'], '3004249'), ('numeric_repeat', ['text'], '3004249'),
        ('category', ['radio'], '2005200014'), ('constant', ['text'], '70001'),
        ('unobserved', ['text'], '70002'), ('date', ['text'], '4160030'),
        ('phase_named_date', ['radio'], '2005200014')]
    schema = {'inventory_ids': 5, 'source_pairs': 7, 'concepts': [dict(name=n, concept_id=cid,
        concept_name=n, source_field=n, field_types=t, unit='', domain='Observation')
        for n, t, cid in definitions]}
    labels = pd.DataFrame({'person_id': list(map(str, range(36))),
        'numeric': [str(i / 5) for i in range(36)], 'category': ['0', '1', '777'] * 12,
        'numeric_repeat': [str((i + 10) / 3) for i in range(36)],
        'constant': ['fixed metadata'] * 36, 'unobserved': [''] * 24 + ['new in test'] * 12,
        'date': [f'2023-01-{i % 20 + 1:02d}' for i in range(36)], 'phase_named_date': ['1', '2'] * 18})
    caches = {}
    for i, split in enumerate(('train', 'val', 'test')):
        caches[split] = {'person_ids': list(map(str, range(i*12, i*12+12))),
                         'embedding': torch.tensor(rng.normal(size=(12, 4)), dtype=torch.float32),
                         'y': torch.tensor([0, 1] * 6), 'original_logits': torch.linspace(-1, 1, 12)}
    return labels, schema, caches


class IndividualConceptTests(unittest.TestCase):
    def test_target_exclusions_and_missingness(self):
        p = {'excluded_source_fields': ['mhoccur_rnl'], 'excluded_source_patterns': ['moca'],
             'excluded_concept_ids': ['2005200017']}
        self.assertTrue(blocked('renamed_field', {'2005200014', '2005200017'}, p))
        self.assertTrue(blocked('mhoccur_rnl', {'2005200014'}, p))
        self.assertFalse(blocked('mhoccur_hbp', {'2005200014', '316866'}, p))
        self.assertEqual(recorded_value({'value_as_number': '0'}), ('0', 'value_as_number'))
        self.assertEqual(recorded_value({'value_as_number': 'nan', 'value_as_concept_id': '0'}), (None, ''))
        labels, schema, cache = fixture()
        spec, data = prepare(cache['train'], cache['val'], labels, schema)
        d = {c['name']: c for c in spec['items']}
        self.assertEqual(d['date']['encoding'], 'date_days')
        self.assertEqual(date_days('1/21/23 9:04'), date_days('2023-01-21 09:04'))
        self.assertEqual(d['phase_named_date']['type'], 'categorical')
        self.assertEqual(d['constant']['status'], 'constant')
        self.assertEqual(d['unobserved']['status'], 'no_training_label')
        self.assertAlmostEqual(d['numeric']['mean'], np.arange(12).mean()/5)

    def test_batched_metrics_with_ties(self):
        rng = np.random.default_rng(42); y = np.array([0, 1] * 15)
        p = rng.choice([.1, .3, .5, .8], size=(7, 30)); boot = Bootstrap(y, 35, 42, 'cpu')
        np.testing.assert_allclose(metric_batch(boot, p, .5), np.stack([boot.metrics(v, .5) for v in p]), atol=1e-12)

    def test_both_modes_train_restore_and_report(self):
        labels, schema, caches = fixture()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); task = 'moca_binary'; source = root/'source'; output = root/'out'
            basic = source/task/'basic'; basic.mkdir(parents=True)
            labeldir = output/'labels'/task; labeldir.mkdir(parents=True)
            labels.to_csv(labeldir/'labels.csv', index=False)
            (labeldir/'schema.json').write_text(json.dumps(schema))
            paths, masks = {}, {}
            for split, cache in caches.items():
                paths[split] = basic/f'features_{split}.pt'; torch.save(cache, paths[split])
                missing = {'embedding': cache['embedding']*.4, 'original_logits': cache['original_logits']*.4,
                           'eligible': torch.tensor([True]*10+[False]*2)}
                masks[split] = root/f'masked_{split}.pt'; torch.save(missing, masks[split])
                if split != 'train': torch.save({'retinal_all': missing}, basic/f'removals_{split}.pt')
            cfg = dict(output_root=str(source), individual_root=str(output), tasks=[task],
                       individual_modes=['basic','mixed'], seed=42, bootstrap_samples=20,
                       concept_epochs=2, head_epochs=2, joint_epochs=2, concept_patience=2)
            for mode in cfg['individual_modes']:
                cfg['individual_mode'] = mode
                with patch('individual_concept_experiment.feature_caches', return_value=(paths, masks)):
                    run_task(task, cfg, torch.device('cpu'))
                fit_folder = output/mode/task; folder = analysis_dir(output, mode, task)
                artifact = torch.load(fit_folder/'model.pt', weights_only=False)
                model = restore(artifact, 'cpu')
                full, p = predict(model, caches['val']['embedding'])
                missing, _ = predict(model, caches['val']['embedding']*.4)
                all_restored = restoration_batch(model, missing, full, [list(range(model.spec['width']))])[0]
                np.testing.assert_allclose(all_restored, p, atol=1e-7)
                c = next(c for c in model.spec['items'] if c['name']=='category')
                cols = list(range(c['start'],c['stop'])); direct = missing.clone(); direct[:, cols] = full[:, cols]
                expected = model.task_head(direct).squeeze(-1).sigmoid().detach().numpy()
                np.testing.assert_allclose(restoration_batch(model, missing, full, [cols])[0], expected, atol=1e-7)
                groups = concept_id_groups(model.spec['items'])
                self.assertEqual(groups['3004249']['members'], ['numeric','numeric_repeat'])
                self.assertEqual(len(groups['2005200014']['columns']), 5)
                self.assertEqual(sorted(i for g in groups.values() for i in g['columns']), list(range(model.spec['width'])))
                for cid in ('3004249','2005200014'):
                    cols = groups[cid]['columns']; direct = missing.clone(); direct[:,cols] = full[:,cols]
                    expected = model.task_head(direct).squeeze(-1).sigmoid().detach().numpy()
                    np.testing.assert_allclose(restoration_batch(model, missing, full, [cols])[0], expected, atol=1e-7)
                for c in model.spec['items']:
                    if not c['learnable']: torch.testing.assert_close(full[:,c['start']:c['stop']], missing[:,c['start']:c['stop']])
                results = pd.read_csv(folder/'restoration.csv')
                self.assertEqual(set(results.split), {'val','test'})
                self.assertEqual(set(results[results.split=='val'].concept), set(groups)|{'ALL_ITEMS','TOP3_JOINT'})
                selected = json.loads((folder/'VAL_TOP3.json').read_text())
                for source_name, concepts in selected.items():
                    got = set(results[(results.split=='test')&(results.removed_input==source_name)].concept)
                    self.assertEqual(got, set(concepts)|{'ALL_ITEMS','TOP3_JOINT'})
                    self.assertEqual(len(concepts), len(set(concepts)))
                # A completed field-level v1 run must not skip ID-level analysis
                # or force a refit. Its model stays at the original location.
                (fit_folder/'DONE.json').write_text('{"legacy_field_run": true}')
                (folder/'DONE.json').unlink()
                with patch('individual_concept_experiment.feature_caches', return_value=(paths, masks)), \
                     patch('individual_concept_experiment.fit', side_effect=AssertionError('must reuse weights')):
                    run_task(task, cfg, torch.device('cpu'))
                self.assertTrue((folder/'DONE.json').exists())
            make_report(cfg)
            summaries = list((output/'tables').glob('SUMMARY_CONCEPT_ID_*.md'))
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all('Source fields restored together' in p.read_text() for p in summaries))


if __name__ == '__main__': unittest.main()
