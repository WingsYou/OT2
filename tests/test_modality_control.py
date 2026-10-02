"""Exercise both historical-source routes without any trainable checkpoint."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from common import REMOVALS, save_torch, write_json
from run_modality_control import run_task, make_report, source_config


class ModalityControlTests(unittest.TestCase):
    def test_cached_two_source_control_and_frozen_thresholds(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); old = root/'old_cbms'; shared = root/'old_backbones'; new = root/'new_cvd'
            out = root/'si'; tasks = ['moca_binary', 'major_cardiovascular_events_binary']
            for folder, config in ((old, dict(backbone_root=str(shared))), (new, {})):
                folder.mkdir(); (folder/'runtime.yaml').write_text(yaml.safe_dump(config))
            cfg = dict(output_root=str(out), tasks=tasks, seed=42, bootstrap_samples=50,
                       source_runs=dict(existing_tasks=str(old), cardiovascular_tasks=str(new)))
            y = np.array([1, 0, 0, 1, 1, 0]); ids = list(map(str, range(6)))
            scores = {}; cuts = dict(full=.65, basic=.55)
            for task, backbone, source in ((tasks[0], shared, old), (tasks[1], new, new)):
                for variant in ('full', 'basic'):
                    folder = backbone/task/variant; folder.mkdir(parents=True)
                    write_json(folder/'TRAINING_DONE.json', dict(threshold=cuts[variant]))
                    p = torch.tensor([.8, .2, .3, .7, .6, .4] if variant=='full' else [.7, .5, .4, .9, .2, .1])
                    baseline = p.logit().sigmoid().numpy()
                    removed = {name:dict(probability=baseline.copy() if name=='environment' else np.roll(baseline,1),
                                         available=np.array([1,1,1,1,1,0],bool)) for name in REMOVALS}
                    scores[variant] = baseline, removed
                    for split in ('val', 'test'):
                        save_torch(folder/f'features_{split}.pt', dict(person_ids=ids, y=torch.tensor(y),
                            original_logits=p.logit(), masked_logits=(p*.5).logit()))
                        # Old Full ablations live beside CBM analyses; newer
                        # removals live alongside the original backbone.
                        cache = source/task/'analysis'/f'removals_{variant}_{split}.pt' if task==tasks[0] else folder/f'removals_{split}.pt'
                        save_torch(cache, removed)
            self.assertEqual(source_config(tasks[0],cfg)[0]['backbone_root'], str(shared))
            self.assertEqual(source_config(tasks[1],cfg)[0]['backbone_root'], str(new))
            with patch('train.FrozenFusionReadout', side_effect=AssertionError('Existing probabilities must be reused')):
                for task in tasks: run_task(task, cfg, 'cpu')
            make_report(cfg)
            metrics = dict(auroc=lambda p,c:roc_auc_score(y,p), f1=lambda p,c:f1_score(y,p>=c),
                           balanced_accuracy=lambda p,c:balanced_accuracy_score(y,p>=c))
            for variant in ('full', 'basic'):
                table = pd.read_csv(out/'tables'/f'MODALITY_ABLATION_{variant}_test.csv')
                self.assertEqual(len(table), len(tasks)*len(REMOVALS)*3)
                baseline, removed = scores[variant]
                for row in table.itertuples():
                    score = metrics[row.metric]; cut = cuts[variant]
                    self.assertEqual(row.threshold, cut)
                    self.assertEqual(row.n_available,5)
                    self.assertAlmostEqual(row.value, score(baseline,cut)-score(removed[row.removed_input]['probability'],cut))
                    if row.removed_input=='environment':
                        self.assertEqual((row.value,row.ci_lower,row.ci_upper),(0.,0.,0.))
                self.assertTrue(table[table.metric!='auroc'].q_value.notna().all())
                self.assertTrue(table[table.metric=='auroc'].q_value.isna().all())
            ref = pd.read_csv(out/'tables'/'DOWNSTREAM_REFERENCE_test.csv')
            self.assertEqual(len(ref),len(tasks)*3*3)
            self.assertTrue((ref[ref.model=='full_basic_inference'].threshold==cuts['full']).all())
            text = (out/'tables'/'SUMMARY_MODALITY_CONTROL_test.md').read_text()
            self.assertIn('Major cardiovascular events',text)
            self.assertIn('does not establish clinical causality',text)
            with patch('train.cached_removals', side_effect=AssertionError('Completed task must resume')):
                run_task(tasks[0],cfg,'cpu')


if __name__=='__main__': unittest.main()
