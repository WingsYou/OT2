"""English F1-only tables for coauthors, with fixed validation rankings."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import TASK_LABELS, INPUT_LABELS, read_json
from stats import bh_adjust

MODEL_NAMES = {'full':'Full Clinical model', 'full_basic_inference':'Full model: Basic Clinical at inference',
               'basic':'Basic Clinical model', 'basic_cbm':'Basic Clinical + CBM',
               'mixed_full':'Mixed CBM: full clinical inputs', 'mixed_basic':'Mixed CBM: basic clinical inputs'}
COMPARISON_NAMES = {'full_input_loss':'Full model: loss after removing nonbasic clinical inputs',
    'basic_vs_full':'Basic minus Full model', 'retraining_gain':'Basic retraining gain over Full model with basic-only clinical inputs',
    'basic_cbm_gain':'Basic CBM minus Basic model', 'mixed_cbm_gain':'Mixed CBM (basic inputs) minus Full model (basic inputs)',
    'mixed_input_loss':'Mixed CBM: loss after removing nonbasic clinical inputs'}


def md(headers, rows):
    def cell(value): return str(value).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]])+'\n'


def interval(row, signed=False):
    fmt = '+.4f' if signed else '.4f'
    return f'{row["value"]:{fmt}} [{row["ci_lower"]:{fmt}}, {row["ci_upper"]:{fmt}}]'


def prob(value): return f'{value:.4g}'


def make_report(cfg):
    root = Path(cfg['output_root']); out = root/'tables'; out.mkdir(exist_ok=True)
    completed = [t for t in cfg['tasks'] if (root/t/'analysis/DONE.json').exists()]
    if not completed: return
    frames = {}
    for name in ('classification','comparisons','ablation','restoration','cohorts'):
        df = pd.concat([pd.read_csv(root/t/'analysis'/(name+'.csv'),dtype={'concept_id':str}) for t in completed],ignore_index=True)
        if 'p_value' in df:
            df['q_value'] = np.nan
            for _,idx in df.groupby(['split','family']).groups.items():
                df.loc[idx,'q_value'] = bh_adjust(df.loc[idx,'p_value'])
        frames[name] = df
        df.to_csv(out/(name+'.csv'),index=False)
    for split in ('val','test'):
        c,e,a,r,co = [frames[name].query('split == @split') for name in ('classification','comparisons','ablation','restoration','cohorts')]
        lines = [f'# OT2 F1 study — {split}', '',
            f'Completed tasks: {len(completed)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}. '
            f'{cfg["bootstrap_samples"]:,} participant bootstrap resamples. Positive-class F1 throughout.', '',
            'Training uses exact class-balanced oversampling. Validation and test retain their original participants and class proportions. '
            'Checkpoint selection, F1 thresholds, the top three modality removals, and the top ten Concept IDs per removal use validation only. '
            'Test tables retain the validation ranks; a top-ranked effect can be zero or negative.', '',
            'Intervals are paired, outcome-stratified percentile 95% CIs. Differences are in F1 units. '
            'Two-sided p values use participant-level prediction swaps; q values use Benjamini–Hochberg adjustment within each analysis family '
            '(downstream comparisons, CBM comparisons, modality ablations, Basic restoration, Mixed restoration), across all completed tasks, separately per split. '
            'Validation restoration correction includes every screened learnable ID, not only the displayed top ten. '
            'Validation findings are exploratory. A single seed does not measure retraining variability. '
            'Previously inspected test data are not a new independent confirmatory cohort.', '',
            '## Evaluation cohorts', '',
            md(['Task','N','Positive','Negative','Prevalence','F1 if all predictions are positive'],
               [[TASK_LABELS[x.task],x.n,x.positive,x.negative,f'{x.prevalence:.3f}',f'{x.all_positive_f1:.4f}'] for x in co.itertuples()]),
            '## 1. Downstream classification', '', 'Cells show F1 [95% CI].', '']
        def cell(task,model):
            return interval(c[(c.task==task)&(c.model==model)].iloc[0])
        lines.append(md(['Task',*map(MODEL_NAMES.get,('full','full_basic_inference','basic'))],
            [[TASK_LABELS[t],*[cell(t,m) for m in ('full','full_basic_inference','basic')]] for t in completed]))
        lines += ['## 2. Basic-model modality ablation', '',
            'Cells show **F1 drop = unperturbed − removed** [95% CI]. Bold cells mark validation-selected top-three removals. '
            'Each comparison uses the same full task cohort; naturally absent sources remain absent. '
            'The CSV also reports how many participants had the removed source(s). Group and single-source removals can overlap.', '']
        matrix=[]
        for source in INPUT_LABELS:
            row=[INPUT_LABELS[source]]
            for t in completed:
                x=a[(a.task==t)&(a.removed_input==source)].iloc[0]; value=interval(x,True)
                row.append('**'+value+'**' if pd.notna(x.val_rank) else value)
            matrix.append(row)
        lines.append(md(['Removed input',*[TASK_LABELS[t] for t in completed]],matrix))
        lines.append(md(['Task','Validation rank 1','Validation rank 2','Validation rank 3'],
            [[TASK_LABELS[t],*[INPUT_LABELS[g] for g in read_json(root/t/'analysis/SELECTION.json')['top_modalities']]] for t in completed]))
        lines += ['## 3. Concept bottleneck classification', '',
            'Basic CBM uses the newly trained Basic backbone. Mixed CBM uses the newly trained Full backbone and is trained on paired full/basic-only inputs. '
            'Backbones are frozen during CBM fitting; concept predictor and classifier undergo joint readout tuning. '
            'The Mixed threshold is chosen on masked validation inputs and is shared by both Mixed conditions.', '',
            md(['Task',*[MODEL_NAMES[m] for m in ('basic_cbm','mixed_full','mixed_basic')]],
               [[TASK_LABELS[t],*[cell(t,m) for m in ('basic_cbm','mixed_full','mixed_basic')]] for t in completed]),
            md(['Task','Basic CBM selected fit','Mixed CBM selected fit'],
               [[TASK_LABELS[t],*[read_json(root/t/(m+'_cbm')/'DONE.json')['selection']['joint']['status']
                                  for m in ('basic','mixed')]] for t in completed]),
            '`fitted` denotes a selected joint-tuning epoch; `sequential_fallback` retains the concept-then-head model. '
            'Joint tuning is selected only when its validation F1/BCE improves while satisfying the concept-loss tolerance.', '',
            '## 4. Paired model comparisons', '',
            md(['Task','Comparison','F1 difference [95% CI]','p','q'],
               [[TASK_LABELS[x.task],COMPARISON_NAMES[x.comparison],interval(x._asdict(),True),prob(x.p_value),prob(x.q_value)] for x in e.itertuples()]),
            '## 5. Modality-ablation statistics', '',
            md(['Task','Removed input','F1 drop [95% CI]','p','q'],
               [[TASK_LABELS[x.task],INPUT_LABELS[x.removed_input],interval(x._asdict(),True),prob(x.p_value),prob(x.q_value)] for x in a.itertuples()]),
            '## 6. Concept restoration', '',
            '**Recovery = restored F1 − masked F1.** Each intervention replaces all channels of one Concept ID with the same participant’s '
            'unperturbed predicted channels. Source fields are listed together when they share that ID. '
            'This is a diagnostic substitution using a full-input reference, not autonomous reconstruction from missing inputs or replacement with measured labels. '
            'Only individual IDs are restored; the top ten are not restored jointly.', '']
        for mode in ('basic','mixed'):
            lines += [f'### {"Basic CBM: top-three modality removals" if mode=="basic" else "Mixed CBM: nonbasic clinical inputs removed"}', '']
            for t in completed:
                sources=read_json(root/t/'analysis/SELECTION.json')['top_modalities'] if mode=='basic' else ['nonbasic_clinical']
                for source in sources:
                    z=r[(r.task==t)&(r['mode']==mode)&(r.removed_input==source)&r.val_rank.notna()].sort_values('val_rank')
                    first=z.iloc[0]
                    label=INPUT_LABELS.get(source,'Nonbasic clinical inputs')
                    lines += [f'#### {TASK_LABELS[t]} — {label}', '',
                        f'Unperturbed CBM F1: {first.full_f1:.4f}; masked CBM F1: {first.masked_f1:.4f}.', '',
                        md(['Val rank','Concept ID','Concept (recorded terminology)','Source fields restored together','Restored F1','F1 recovery [95% CI]','p','q'],
                           [[int(x.val_rank),x.concept_id,x.concept_name,x.source_fields,f'{x.restored_f1:.4f}',interval(x._asdict(),True),prob(x.p_value),prob(x.q_value)] for x in z.itertuples()])]
        lines += ['## Thresholds', '',
            md(['Task','Model / input condition','Validation-selected threshold'],
               [[TASK_LABELS[x.task],MODEL_NAMES[x.model],f'{x.threshold:.6f}'] for x in c.itertuples()]),
            'The Concept-ID definitions, fitted encodings, labels and training logs are stored in the task/CBM directories. '
            'Task outcome exclusions, the four-survey restriction, and the four previously removed administrative IDs remain in effect. '
            'All prediction files are participant-level and remain in the local result directory.', '']
        (out/f'SUMMARY_{split}.md').write_text('\n'.join(lines))
    print(f'[tables] {len(completed)}/{len(cfg["tasks"])} tasks: {out}',flush=True)
