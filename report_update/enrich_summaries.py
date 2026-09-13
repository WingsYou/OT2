#!/usr/bin/env python3
"""Add original12 definitions and optional F1 detail from existing saved predictions."""
import argparse
import csv
import struct
from pathlib import Path

TASKS={'moca_binary':'MoCA < 26','cvd_broad_binary':'CVD broad','cvd_strict_binary':'CVD strict',
       'osteoporosis_history_binary':'Osteoporosis history','kidney_history_binary':'Kidney-problem history',
       'lung_history_binary':'Lung-problem history'}
ARMS={'full':(('full','Full clinical','probability'),('basic_inference','Full: basic inference','basic_inference_probability')),
      'basic':(('basic','Basic clinical','probability'),),
      'cbm':(('cbm_joint_lambda_1','Basic + CBM','joint_lambda_1'),)}
START='<!-- original12-report-explanation:start -->'
END='<!-- original12-report-explanation:end -->'
METRIC_NOTE='''## Appendix B. Reading F1 alongside AUROC and balanced accuracy

F1 here is **positive-class F1**, calculated as `2 TP / (2 TP + FP + FN)`. It depends on both precision and recall, and does not reward true negatives. Positive prevalence therefore matters when comparing tasks. For illustration only, a cohort with 100 positives and 900 negatives, sensitivity 0.70 and specificity 0.70 has TP=70, FP=270, FN=30, balanced accuracy 0.70 and F1 only 0.3182. This is a numerical example, not this study's confusion matrix. See the [F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

The experiment selects thresholds using validation Youden J (`sensitivity + specificity − 1`), which maximizes validation balanced accuracy. It does not maximize F1. AUROC evaluates ranking across thresholds; F1 and balanced accuracy describe the selected operating point. Removal and restoration analyses retain the parent's validation threshold, so a score shift can change F1 even if ranking changes little. See [decision-threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

Low F1 can reflect false positives, false negatives, limited discrimination, or their combination. Precision, recall and TP/FP/FN/TN from saved predictions distinguish these explanations. Any separate F1-oriented threshold analysis should select its threshold using validation (or training-only cross-validation), then apply it unchanged to test; the current primary tables remain under their original threshold rule.
'''

def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])

def f1_details(root,split):
    rows=[]
    for task,label in TASKS.items():
        for variant,arms in ARMS.items():
            directory=root/task/variant;pred_path=directory/f'predictions_{split}.csv'
            metrics_path=directory/'classification.csv'
            if not pred_path.exists() or not metrics_path.exists():continue
            predictions=read_csv(pred_path);metrics=read_csv(metrics_path)
            y=[int(float(row['y'])) for row in predictions];n=len(y);positive=sum(y)
            for arm,arm_label,column in arms:
                cut=float(next(row['threshold'] for row in metrics if row['split']==split and row['model']==arm))
                # Pandas writes these float32 probabilities with short decimals.
                # Restore their original dtype before comparing an exact saved threshold.
                f32=lambda value:struct.unpack('f',struct.pack('f',float(value)))[0]
                pred=[f32(row[column])>=cut for row in predictions]
                tp=sum(t==1 and p for t,p in zip(y,pred));fp=sum(t==0 and p for t,p in zip(y,pred))
                fn=positive-tp;tn=n-positive-fp
                div=lambda a,b:a/b if b else 0.
                values=[div(tp,tp+fp),div(tp,tp+fn),div(tn,tn+fp),div(2*tp,2*tp+fp+fn)]
                rows.append([label,arm_label,n,f'{positive} ({100*positive/n:.1f}%)',f'{cut:.6f}',tp,fp,fn,tn,
                             *[f'{v:.4f}' for v in values]])
    if not rows:return None
    return f'# F1 operating-point details — {split}\n\n'+\
        'Computed from this run’s saved predictions at the existing validation-selected thresholds. No model fitting, inference or threshold optimization is performed. Precision is recorded as zero when no positives are predicted. Counts are aggregate; participant IDs are not included.\n\n'+\
        md_table(['Task','Model','N','Positive N (%)','Threshold','TP','FP','FN','TN','Precision','Recall / sensitivity','Specificity','F1'],rows)+'\n'

def enrich(text,reference,details_name=None):
    if START in text:
        before,tail=text.split(START,1);_,after=tail.split(END,1);text=before+after
    extra=reference.strip()+'\n\n'+METRIC_NOTE.strip()
    if details_name:extra+=f'\n\nExact operating-point counts and metrics: [F1 details]({details_name}).'
    return text.rstrip()+'\n\n'+START+'\n'+extra+'\n'+END+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,help='Folder containing SUMMARY_val.md and SUMMARY_test.md')
    p.add_argument('--results-root',type=Path,help='Completed study root; also enables F1 details from saved predictions')
    p.add_argument('--out-dir',type=Path,help='Default: <input-dir>/with_concepts')
    args=p.parse_args()
    if args.input_dir is None and args.results_root is None:p.error('Provide --input-dir or --results-root')
    source=args.input_dir or args.results_root/'tables';out=args.out_dir or source/'with_concepts';out.mkdir(parents=True,exist_ok=True)
    reference=(Path(__file__).parent/'CONCEPT_REFERENCE.md').read_text()
    for split in ('val','test'):
        detail=f1_details(args.results_root,split) if args.results_root else None
        name=f'F1_DETAILS_{split}.md' if detail else None
        if detail:(out/name).write_text(detail)
        path=out/f'SUMMARY_{split}.md'
        path.write_text(enrich((source/f'SUMMARY_{split}.md').read_text(),reference,name))
        print(path)
    (out/'CONCEPT_REFERENCE.md').write_text(reference)

if __name__=='__main__':main()
