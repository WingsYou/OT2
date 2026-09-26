"""Numerical and saved-artifact checks for no-retraining supplementation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, f1_score, balanced_accuracy_score
from stats import PairedF1
from common import save_torch, write_json
from cbm.model import IndividualCBM, state
from cbm.experiment import concept_id_groups, restoration_batch
from cbm.diagnostics import measured_targets, concept_quality, run_diagnostics, supplementation_statistics
from cbm.diagnostics_report import make_diagnostics_report


def specification():
    def item(name,cid,start,stop,kind,**kwargs):
        return dict(name=name,concept_id=cid,concept_name=name,source_field=name,
                    unit='',qualifier_concept_id='',domain='Measurement',type=kind,
                    encoding='number' if kind=='continuous' else 'raw_category',
                    start=start,stop=stop,mean=0.,std=1.,categories=[],learnable=True,
                    status='learned',n_train_observed=4,**kwargs)
    items=[item('first','101',0,1,'continuous'),item('second','101',1,2,'continuous'),
           item('category','102',2,4,'categorical'),item('constant','103',4,5,'continuous')]
    items[0].update(mean=10.,std=2.)
    items[1].update(mean=100.,std=10.)
    items[2].update(categories=['No','Yes'],domain='Observation')
    items[3].update(learnable=False,status='constant')
    return dict(input_dim=2,width=5,x_mean=[0.,0.],x_std=[1.,1.],items=items)


class ConceptDiagnosticsTests(unittest.TestCase):
    def test_brier_bootstrap_and_prediction_swaps(self):
        y=np.array([0,0,0,1,1]);a=np.array([.1,.6,.2,.7,.8]);b=np.array([.2,.5,.1,.6,.7])
        boot=PairedF1(y,100,13,'cpu')
        draws=boot.draws(np.stack([a,b]),.5,'brier')
        np.testing.assert_allclose(draws[:,0],[brier_score_loss(y,p) for p in (a,b)])
        for k in (1,35,100):
            sample=np.repeat(np.arange(len(y)),boot.w[k].numpy().astype(int))
            self.assertAlmostEqual(draws[0,k],brier_score_loss(y[sample],a[sample]))
        difference=brier_score_loss(y,a)-brier_score_loss(y,b)
        null=[brier_score_loss(y,np.where(s,b,a))-brier_score_loss(y,np.where(s,a,b))
              for s in boot.swaps.numpy().astype(bool)]
        expected=(1+sum(abs(v)>=abs(difference)-1e-12 for v in null))/101
        self.assertAlmostEqual(boot.pvalues(a,b,.5,.5,'brier')[0],expected)
        row=supplementation_statistics(boot,a,b,.5)[0]
        self.assertAlmostEqual(row['brier_gain'],difference)
        self.assertAlmostEqual(row['f1_gain'],f1_score(y,b>=.5)-f1_score(y,a>=.5))
        single=PairedF1(np.zeros(3,int),100,13,'cpu',allow_single_class=True)
        row=supplementation_statistics(single,a[:3],b[:3],.5)[0]
        self.assertTrue(np.isnan(row['ba_gain']));self.assertTrue(np.isnan(row['ba_p_value']))
        self.assertAlmostEqual(row['brier_gain'],np.square(a[:3]).mean()-np.square(b[:3]).mean())

    def test_measured_encoding_and_partial_id_restoration(self):
        torch.manual_seed(11);model=IndividualCBM(specification()).eval().requires_grad_(False)
        missing=torch.tensor([[.1,.2,.3,.7,.0],[.4,.5,.6,.4,.0],[.8,.9,.2,.8,.0]])
        labels=pd.DataFrame(dict(first=['12','',''],second=['','110',''],category=['Yes','unseen',''],constant=['8','9','10']))
        target,available=measured_targets(model,labels,missing)
        expected=missing.clone();expected[0,0]=1;expected[1,1]=1;expected[0,2:4]=torch.tensor([0.,1.])
        torch.testing.assert_close(target,expected)
        np.testing.assert_array_equal(available,[[1,0,1,0],[0,1,0,0],[0,0,0,0]])
        groups=concept_id_groups(model.spec['items'])
        direct=missing.clone();direct[:,:2]=target[:,:2]
        actual=restoration_batch(model,missing,target,[groups['101']['columns']])[0]
        np.testing.assert_allclose(actual,model.task_head(direct).squeeze(-1).sigmoid().numpy(),atol=1e-7)
        self.assertEqual(actual[2],model.task_head(missing).squeeze(-1).sigmoid().numpy()[2])
        train=pd.DataFrame(dict(first=['8','12','8','12'],second=['90','110','90','110'],
                                category=['No','Yes','No','Yes'],constant=['0']*4))
        quality=concept_quality(model.spec['items'],labels,train,target.numpy(),missing.numpy())
        self.assertEqual(quality[0]['full_rmse'],0.)
        self.assertAlmostEqual(quality[0]['masked_rmse'],1.8,places=6)
        self.assertEqual(quality[2]['n_unseen_category'],1)
        self.assertEqual(quality[2]['full_accuracy'],1.)
        self.assertAlmostEqual(quality[2]['train_prior_multiclass_brier'],.5)

    def test_existing_artifacts_both_banks_without_training(self):
        torch.set_num_threads(1);torch.manual_seed(11)
        task='moca_binary'
        with tempfile.TemporaryDirectory() as tmp:
            top=Path(tmp);backbone=top/'backbones'/task/'full'
            records=[]
            for split in ('train','val','test'):
                ids=[split+str(i) for i in range(6)]
                x=torch.tensor([[0.,1.],[1.,0.],[1.,2.],[2.,1.],[.5,.5],[2.,2.]])
                cache=dict(person_ids=ids,embedding=x,masked_embedding=x*.4,
                           y=torch.tensor([0,0,0,1,1,1]),original_logits=torch.arange(6.)-2.5,
                           masked_logits=torch.arange(6.)-1.5)
                save_torch(backbone/f'features_{split}.pt',cache)
                for i,pid in enumerate(ids):
                    records.append(dict(person_id=pid,first=str(8+2*i) if i!=1 else '',
                                        second=str(90+10*i) if i!=0 else '',
                                        category=('No' if i%2==0 else 'Yes') if split=='train' else ('No' if i<2 else 'unseen' if i==2 else ''),
                                        constant='0'))
            write_json(backbone/'TRAINING_DONE.json',dict(threshold=.8))
            for excluded in (True,False):
                root=top/('without_observation' if excluded else 'with_observation')
                cfg=dict(output_root=str(root),backbone_root=str(top/'backbones'),tasks=[task],
                         seed=42,bootstrap_samples=30,exclude_observation=excluded)
                spec=specification()
                if excluded:
                    spec['items']=spec['items'][:2];spec['width']=2
                model=IndividualCBM(spec).eval().requires_grad_(False)
                save_torch(root/task/'mixed_cbm/model.pt',dict(spec=spec,states={'joint':state(model)},threshold=.55))
                folder=root/'labels'/task;folder.mkdir(parents=True)
                pd.DataFrame(records).to_csv(folder/'labels.csv',index=False)
                with patch('cbm.model.fit',side_effect=AssertionError('must not train')):
                    run_diagnostics(task,cfg,torch.device('cpu'))
                make_diagnostics_report(cfg)
                self.assertFalse((root/task/'analysis').exists())
                table=root/'tables/concept_diagnostics'
                frame=pd.read_csv(table/'MEASURED_CONCEPT_SUPPLEMENTATION_test.csv',dtype={'concept_id':str})
                saved=pd.read_csv(root/task/'concept_diagnostics/measured_predictions_test.csv',float_precision='round_trip')
                for row in frame[frame.population=='all_participants'].itertuples():
                    p=saved[row.concept_id].to_numpy(np.float32)
                    self.assertEqual(row.threshold,.55)
                    self.assertAlmostEqual(row.restored_brier,brier_score_loss(saved.y,p),places=7)
                    self.assertAlmostEqual(row.restored_f1,f1_score(saved.y,p>=.55))
                    self.assertAlmostEqual(row.restored_ba,balanced_accuracy_score(saved.y,p>=.55))
                    self.assertEqual(row.changed,row.corrected+row.newly_wrong)
                if not excluded:
                    row=frame[(frame.concept_id=='102')&(frame.population=='observed_participants')].iloc[0]
                    self.assertEqual(row.n,2);self.assertEqual(row.positive,0);self.assertTrue(pd.isna(row.ba_gain))
                    self.assertTrue(pd.isna(row.ba_p_value));self.assertTrue(pd.isna(row.ba_q_value))
                    q=pd.read_csv(table/'CONCEPT_QUALITY_test.csv',dtype={'concept_id':str})
                    self.assertEqual(q[q.concept_id=='102'].iloc[0].n_unseen_category,1)
                text=(table/'SUMMARY_CONCEPT_DIAGNOSTICS_test.md').read_text()
                self.assertIn('without Observation' if excluded else 'with the four approved Observation surveys',text)
                self.assertIn('validation Brier rank',text)
                with patch('cbm.diagnostics.load',side_effect=AssertionError('must reuse')):
                    run_diagnostics(task,cfg,torch.device('cpu'))


if __name__=='__main__':unittest.main()
