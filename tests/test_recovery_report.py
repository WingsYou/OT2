"""Check compact recovery exports preserve diagnostic selection and populations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cbm.recovery_report import CONDITIONS, SOURCES, make_recovery_report
from stats import auroc_selection


class RecoveryReportTests(unittest.TestCase):
    def test_exports_keep_all_ids_and_only_existing_all_participant_selection(self):
        task = 'moca_binary'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics = root/'tables/concept_diagnostics'
            diagnostics.mkdir(parents=True)
            for split in ('val', 'test'):
                performance = []
                for condition in CONDITIONS:
                    row = dict(task=task, split=split, condition=condition, n=20)
                    for metric in ('auroc', 'f1', 'ba', 'brier'):
                        row.update({metric: .6, metric+'_ci_lower': .5, metric+'_ci_upper': .7})
                    performance.append(row)
                pd.DataFrame(performance).to_csv(diagnostics/f'DOWNSTREAM_BRIER_{split}.csv', index=False)
                for source, _, _, legacy in SOURCES:
                    # A selected negative effect must be retained, with no positivity filter.
                    gains = np.array([-.1]*9 + [-.0012345678901234567, np.nan])
                    selection = auroc_selection(gains)
                    rows = []
                    for i, gain in enumerate(gains):
                        row = dict(task=task, split=split, source=source,
                                   population='all_participants', concept_id=f'{i:05d}',
                                   concept_name=f'Example {i}', n_eligible=12, n_complete=7,
                                   n_total=20, status='evaluated' if np.isfinite(gain) else 'no_observed_learnable_label')
                        row.update({name: values[i] for name, values in selection.items()})
                        for metric in ('auroc', 'f1', 'ba', 'brier'):
                            row.update({metric+'_gain': gain, metric+'_gain_ci_lower': gain-.01,
                                        metric+'_gain_ci_upper': gain+.01,
                                        'restored_'+metric: .6, 'restored_'+metric+'_ci_lower': .5,
                                        'restored_'+metric+'_ci_upper': .7})
                        rows.append(row)
                    if source == 'measured':
                        # This separately selected observed-subset row must never leak into main exports.
                        rows.append(dict(rows[0], population='observed_participants', auroc_selected=True,
                                         auroc_gain=.5, auroc_rank_within_split=1))
                    # An aggregate may contain a task excluded from the current configuration.
                    rows.append(dict(rows[0], task='kidney_history_binary', auroc_selected=True))
                    pd.DataFrame(rows).to_csv(diagnostics/f'{legacy}_{split}.csv', index=False)
            cfg = dict(output_root=str(root), tasks=[task], seed=42, bootstrap_samples=10000,
                       exclude_observation=True, concept_ranking='auroc_mean_plus_2sd')
            make_recovery_report(cfg)
            for split in ('val', 'test'):
                for source, prefix, _, legacy in SOURCES:
                    all_rows = pd.read_csv(root/f'tables/{prefix}_CONCEPT_RECOVERY_ALL_{split}.csv',
                                           dtype={'concept_id': str}, float_precision='round_trip')
                    selected = pd.read_csv(root/f'tables/{prefix}_CONCEPT_RECOVERY_SELECTED_{split}.csv',
                                           dtype={'concept_id': str}, float_precision='round_trip')
                    self.assertEqual(len(all_rows), 11)
                    self.assertEqual(set(all_rows.population), {'all_participants'})
                    self.assertEqual(set(all_rows.task), {task})
                    self.assertEqual(selected.concept_id.tolist(), ['00009'])
                    self.assertLess(selected.auroc_gain.iloc[0], 0)
                    original = pd.read_csv(diagnostics/f'{legacy}_{split}.csv',
                                           dtype={'concept_id': str}, float_precision='round_trip')
                    expected = original[(original.task == task) & (original.population == 'all_participants')]
                    pd.testing.assert_frame_equal(all_rows.sort_values('concept_id').reset_index(drop=True),
                                                  expected.sort_values('concept_id').reset_index(drop=True))
                    self.assertTrue(pd.isna(all_rows.iloc[-1].auroc_rank_within_split))
                summary = (root/f'tables/SUMMARY_CONCEPT_RECOVERY_{split}.md').read_text()
                self.assertIn(f'{split} ΔAUROC > μ + 2σ', summary)
                self.assertIn('without Observation', summary)
                self.assertIn('AUROC after recovery', summary)
                self.assertIn('12 / 7 / 20', summary)
                self.assertIn('Example 9', summary)
                self.assertNotIn('| Example 0 |', summary)
                self.assertIn('SOURCE_RUNS.json', summary)


if __name__ == '__main__':
    unittest.main()
