"""Separate English summaries for the no-retraining concept diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import TASK_LABELS, read_json
from stats import bh_adjust, auroc_selection
from report import md
from cbm.diagnostics import VERSION


def ci(row, metric, signed=False):
    value = row.get(metric,np.nan)
    if pd.isna(value): return 'NA'
    fmt = '+.4f' if signed else '.4f'
    return f'{value:{fmt}} [{row[metric+"_ci_lower"]:{fmt}}, {row[metric+"_ci_upper"]:{fmt}}]'


def number(x): return 'NA' if pd.isna(x) else f'{x:.4g}'


def make_diagnostics_report(cfg):
    root = Path(cfg['output_root']); out = root/'tables/concept_diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    tasks = [t for t in cfg['tasks'] if (root/t/'concept_diagnostics/DONE.json').exists()
             and read_json(root/t/'concept_diagnostics/DONE.json').get('version')==VERSION]
    if not tasks: return
    frames = {name:pd.concat([pd.read_csv(root/t/'concept_diagnostics'/(name+'.csv'),dtype={'concept_id':str})
                             for t in tasks],ignore_index=True)
              for name in ('quality','performance','comparisons','restoration')}
    r = frames['restoration']
    for metric in ('f1','ba','brier'):
        r[metric+'_q_value'] = np.nan
        eligible = r.learnable & (r.status=='evaluated') & r[metric+'_p_value'].notna()
        for _,idx in r[eligible].groupby(['split','source','population']).groups.items():
            r.loc[idx,metric+'_q_value'] = bh_adjust(r.loc[idx,metric+'_p_value'])
    # F1/BA remain reported outcomes. Brier ranks are supplementary/exploratory.
    eligible = r.learnable & (r.status=='evaluated') & r.brier_gain.notna()
    r['brier_rank_within_split'] = np.nan
    r.loc[eligible,'brier_rank_within_split'] = r[eligible].groupby(['task','split','source','population']).brier_gain.rank(method='min',ascending=False)
    for _, idx in r.groupby(['task','split','source','population']).groups.items():
        group = r.loc[idx]
        # Predicted restoration tests every single ID, including zero effects
        # from constant channels. Measured candidates need a usable target.
        evaluable = (group.source == 'full_input_prediction') | (group.status == 'evaluated')
        for key, values in auroc_selection(group.auroc_gain, evaluable).items():
            if key not in r: r[key] = False if key == 'auroc_selected' else np.nan
            r.loc[idx, key] = values
    rank_column = 'auroc_rank_within_split'
    frames['restoration'] = r.sort_values(['task','split','source','population',rank_column,'concept_id'],na_position='last')
    for split in ('val','test'):
        q,p,c,s = [frames[name].query('split == @split') for name in ('quality','performance','comparisons','restoration')]
        bank = 'without Observation' if cfg.get('exclude_observation') else 'with the four approved Observation surveys'
        lines = [f'# Mixed CBM concept diagnostics — {split}', '',
                 f'Concept bank: **{bank}**. Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. '
                 f'Seed {cfg["seed"]}; {cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'No retraining, preprocessing, concept-bank changes, or threshold optimization was performed. '
                 'Mixed full, masked and supplemented predictions use the saved masked-validation threshold. '
                 'The original model uses its own saved threshold. Nonbasic clinical inputs are removed; '
                 'Basic Clinical and the other eight modalities remain available.', '',
                 '## Downstream probability and classification performance', '',
                 'Brier is the mean squared error of the positive-class probability (lower is better). '
                 'AUROC evaluates discrimination independently of a classification threshold. '
                 'F1 and BA retain their existing definitions and thresholds.', '',
                 md(['Task','Model/input','AUROC [95% CI]','Brier [95% CI]','F1 [95% CI]','BA [95% CI]'],
                    [[TASK_LABELS[x.task],x.condition,ci(x._asdict(),'auroc'),ci(x._asdict(),'brier'),ci(x._asdict(),'f1'),ci(x._asdict(),'ba')] for x in p.itertuples()]),
                 '## Information loss after removing nonbasic clinical inputs', '',
                 'Positive values mean removal worsened performance: Brier(masked) − Brier(full); '
                 'AUROC/F1/BA(full) − AUROC/F1/BA(masked).', '',
                 md(['Task','Model','AUROC drop [95% CI]','Brier increase [95% CI]','F1 drop [95% CI]','BA drop [95% CI]'],
                    [[TASK_LABELS[x.task],x.model,ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),ci(x._asdict(),'f1_gain',True),
                      ci(x._asdict(),'ba_gain',True)] for x in c.itertuples()]),
                 '## Concept prediction quality', '',
                 'Quality is evaluated per source/qualifier/unit item, because one Concept ID may have multiple measurements. '
                 'Only recorded, encodable labels are evaluated. The CSV includes missing/unknown-category counts. '
                 'Continuous items report MAE/RMSE in their recorded units (days for date-encoded items) and MSE in training-standardized units. '
                 'Categorical items report accuracy, macro recall over observed categories, log loss, and multiclass Brier '
                 '(sum over categories, range 0–2). Reference predictors use the training mean or training category frequencies. '
                 'Constants and items without training labels are retained in the CSV but excluded from this summary.', '']
        quality_summary = []
        for task in tasks:
            z=q[(q.task==task)&q.learnable&(q.n_evaluable>0)]
            continuous=z[z.type=='continuous'];categorical=z[z.type=='categorical']
            def count(df,column):
                return int((df.get(column,pd.Series(dtype=float))>0).sum())
            cont_better=int((continuous.get('masked_standardized_mse',pd.Series(dtype=float)) < continuous.get('train_mean_standardized_mse',pd.Series(dtype=float))).sum())
            cat_better=int((categorical.get('masked_multiclass_brier',pd.Series(dtype=float)) < categorical.get('train_prior_multiclass_brier',pd.Series(dtype=float))).sum())
            quality_summary.append([TASK_LABELS[task],len(continuous),cont_better,count(continuous,'masking_standardized_mse_increase'),
                                    len(categorical),cat_better,count(categorical,'masking_multiclass_brier_increase')])
        lines += [md(['Task','Continuous items','Masked better than training mean','Continuous worse after masking',
                      'Categorical items','Masked better than training prior','Categorical worse after masking'],quality_summary),
                  'Counts are descriptive, not significance tests; see CONCEPT_QUALITY for individual values and sample sizes.', '',
                  '## Single-ID supplementation', '',
                  'Each ID is changed separately from the same masked baseline. Full-input-prediction restoration replaces '
                  'all channels of that ID with full-input predicted concepts. Measured supplementation replaces only observed, '
                  'encodable, learnable items with their recorded targets; missing items and unseen categories stay predicted. '
                  'Continuous targets use saved training normalization; categorical targets use the saved category order and one-hot encoding. '
                  'No absence is interpreted as a negative response. Constant/untrained channels are not supplemented.', '',
                  'All-participant effects include people without available measurements (unchanged predictions). '
                  'Observed-participant effects compare exactly the same eligible people before/after supplementation; '
                  'eligible means at least one learnable item of the ID is available, so some IDs are only partially supplemented. '
                  'n_complete in the CSV counts people with all learnable items of the ID available. '
                  'Observed cohorts can differ between IDs; their gains should not be compared as if they came from one common population. '
                  'BA and AUROC are NA for cohorts containing only one outcome class.', '',
                  'Brier gain = baseline Brier − supplemented Brier; AUROC/F1/BA gain = supplemented − baseline. '
                  'Positive values are improvements. Test rankings are exploratory, and their intervals do not account '
                  'for selecting the largest test effects.', '']
        lines += [f'Tables report single Concept IDs with {split} ΔAUROC > μ + 2σ, ranked directly on {split}. '
                  'μ and σ are computed separately for each task and intervention source across all evaluated individual IDs; '
                  'σ is the population standard deviation (ddof=0). Predicted restoration includes constant IDs with zero gain. '
                  'Measured supplementation requires usable measured targets. Joint/all-ID restoration does not enter the distribution. '
                  'This is a descriptive outlier rule, not a significance test. No fixed top-10 padding or extra positivity filter is applied; '
                  'a selected negative gain is still a deterioration. Test candidates are not selected using validation ranks. '
                  'The CSV retains every ID with its AUROC rank, selection flag, μ, σ and cutoff; unevaluable IDs appear last. '
                  'AUROC has paired bootstrap intervals without permutation p/q values.', '']
        selection_rows = []
        for (task,source), group in s[s.population=='all_participants'].groupby(['task','source'],sort=False):
            first = group.iloc[0]
            selection_rows.append([TASK_LABELS[task],source,int(first.auroc_selection_n),
                                   number(first.auroc_selection_mean),number(first.auroc_selection_std),
                                   number(first.auroc_selection_cutoff),int(group.auroc_selected.sum())])
        lines += [md(['Task','Intervention source','Evaluated IDs','Mean ΔAUROC (μ)','SD (σ)','μ + 2σ','Selected IDs'],selection_rows)]
        for source,title in [('full_input_prediction','Full-input predicted concept restoration'),('measured','Measured concept supplementation')]:
            lines += [f'### {title} — all participants', '']
            for task in tasks:
                selected = s.auroc_selected
                z=s[(s.task==task)&(s.source==source)&(s.population=='all_participants')&selected]
                if source=='measured':z=z[z.n_eligible>0]
                z=z.sort_values([rank_column,'concept_id'])
                lines += [f'#### {TASK_LABELS[task]}', '']
                if z.empty:
                    lines += [f'No single Concept ID exceeded the {split} μ + 2σ cutoff.', ''];continue
                lines += [md([split.title()+' AUROC rank','Concept ID','Concept','Source fields','Eligible N / total',
                              'ΔAUROC [95% CI]','ΔBrier [95% CI]','Brier q','ΔF1 [95% CI]','ΔBA [95% CI]','Corrected / newly wrong'],
                    [[int(getattr(x,rank_column)),x.concept_id,x.concept_name,x.source_fields,f'{x.n_eligible}/{x.n_total}',
                      ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),number(x.brier_q_value),ci(x._asdict(),'f1_gain',True),
                      ci(x._asdict(),'ba_gain',True),f'{int(x.corrected)}/{int(x.newly_wrong)}'] for x in z.itertuples()])]
                if source=='measured':
                    obs=s[(s.task==task)&(s.source==source)&(s.population=='observed_participants')&s.concept_id.isin(z.concept_id)].sort_values([rank_column,'concept_id'])
                    lines += [f'The same candidates, evaluated only among participants with usable measured labels, sorted by their {split} '+'AUROC'+' gain in that subset:', '',
                              md(['Concept ID','Observed N','Positive / negative','ΔAUROC [95% CI]','ΔBrier [95% CI]','ΔF1 [95% CI]','ΔBA [95% CI]'],
                                 [[x.concept_id,x.n,f'{x.positive}/{x.negative}',ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),
                                   ci(x._asdict(),'f1_gain',True),ci(x._asdict(),'ba_gain',True)] for x in obs.itertuples()])]
        lines += ['## Interpretation and exports', '',
                  'Measured-value supplementation is an additional intervention experiment; benefit is not guaranteed. '
                  'A classifier trained on predicted concepts may respond differently to measured targets. '
                  'Small observed cohorts, outcome imbalance, multiple testing and a single training seed limit certainty. '
                  'No concept was added or removed according to these results.', '',
                  'F1, BA and Brier restoration q values use BH correction over learnable, evaluated IDs across completed tasks within '
                  'each split, source, population and metric. Full-input-prediction restoration and measured supplementation '
                  'are separate families; all IDs, not only displayed candidates, enter correction. '
                  'P values use paired prediction swaps. Resampling retains the observed outcome counts in each evaluated cohort. '
                  'Final q values require all configured tasks; intervals do not account for selection or training-seed variability.', '',
                  f'CSV: DOWNSTREAM_BRIER_{split}.csv; INPUT_REMOVAL_BRIER_{split}.csv; CONCEPT_QUALITY_{split}.csv; '
                  f'PREDICTED_RESTORATION_BRIER_{split}.csv; MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv. '
                  'Per-patient measured probabilities and availability are saved under each task/concept_diagnostics/.', '']
        for name,df in [('DOWNSTREAM_BRIER',p),('INPUT_REMOVAL_BRIER',c),('CONCEPT_QUALITY',q),
                        ('PREDICTED_RESTORATION_BRIER',s[s.source=='full_input_prediction']),
                        ('MEASURED_CONCEPT_SUPPLEMENTATION',s[s.source=='measured'])]:
            df.to_csv(out/f'{name}_{split}.csv',index=False)
        (out/f'SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md').write_text('\n'.join(lines))
    print(f'[diagnostic tables] {len(tasks)}/{len(cfg["tasks"])} tasks: {out}',flush=True)
