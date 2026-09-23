"""Small synthetic checks for the changed sampling, inference, and full workflow."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, balanced_accuracy_score
from common import MODALITIES, REMOVALS, read_json, write_json, save_torch
from stats import balanced_indices, f1, threshold, PairedF1, bh_adjust
from train import run_model, load
from data import SOURCE_FLAG_BY_GROUP
from task2_v2_model import Task2V2ModelConfig
from cbm.labels import load_bank, FOUR_OBSERVATION_FORMS, SURVEY_ADMIN_IDS
from cbm.experiment import run_cbm, concept_id_groups, restoration_batch
from cbm.model import restore, predict
from analysis import run_analysis
from report import make_report


class F1StudyTests(unittest.TestCase):
    def test_balancing_threshold_and_paired_statistics(self):
        y=np.array([0]*7+[1]*3);idx=balanced_indices(y,42)
        self.assertEqual(np.bincount(y[idx]).tolist(),[7,7])
        self.assertEqual(set(idx),set(range(10)))
        p=np.array([.8,.2,.2,.4,.4,.1,.6,.6,.8,.9])
        cut=threshold(y,p)
        candidates=sorted(set(p),reverse=True)
        expected=max(candidates,key=lambda c:f1_score(y,p>=c))
        self.assertEqual(cut,expected)
        boot=PairedF1(y,500,42,'cpu');other=np.roll(p,2)
        draws=boot.draws(np.stack([p,other]),cut)
        np.testing.assert_allclose(draws[:,0],[f1_score(y,v>=cut) for v in (p,other)])
        weights=boot.w.numpy().astype(int)
        for i in (1,2,499):
            sample=np.repeat(np.arange(len(y)),weights[i])
            self.assertAlmostEqual(draws[0,i],f1_score(y[sample],p[sample]>=cut))
        effect=boot.effects(p,p,cut,cut)[0]
        self.assertEqual(effect,dict(value=0.,ci_lower=0.,ci_upper=0.,p_value=1.))
        self.assertAlmostEqual(boot.effects(p,other,cut,cut)[0]['p_value'],boot.effects(other,p,cut,cut)[0]['p_value'])
        np.testing.assert_allclose(bh_adjust([.01,.04,.03]),[.03,.04,.04])
        ba=boot.draws(np.stack([p,other]),cut,metric='balanced_accuracy')
        np.testing.assert_allclose(ba[:,0],[balanced_accuracy_score(y,v>=cut) for v in (p,other)])
        for i in (1,2,499):
            sample=np.repeat(np.arange(len(y)),weights[i])
            self.assertAlmostEqual(ba[0,i],balanced_accuracy_score(y[sample],p[sample]>=cut))
        np.testing.assert_allclose(boot.draws(np.ones(len(y)),.5,metric='balanced_accuracy'),.5)
        a,b=p>=cut,other>=cut
        observed=balanced_accuracy_score(y,a)-balanced_accuracy_score(y,b)
        null=[balanced_accuracy_score(y,np.where(s,b,a))-balanced_accuracy_score(y,np.where(s,a,b))
              for s in boot.swaps.cpu().numpy().astype(bool)]
        expected=(1+sum(abs(v)>=abs(observed)-1e-12 for v in null))/501
        self.assertAlmostEqual(boot.effects(p,other,cut,cut,metric='balanced_accuracy')[0]['p_value'],expected)

    def test_approved_inventory(self):
        bank=load_bank(observation_forms=FOUR_OBSERVATION_FORMS,excluded_ids=SURVEY_ADMIN_IDS)
        counts={'moca_binary':222,'cvd_broad_binary':217,'cvd_strict_binary':220,
                'osteoporosis_history_binary':224,'kidney_history_binary':224,'lung_history_binary':224}
        for task,n in counts.items():self.assertEqual(bank['tasks'][task]['inventory_ids'],n)
        self.assertFalse(set(SURVEY_ADMIN_IDS)&{c['concept_id'] for c in bank['concepts'].values()})
        obs=[c for c in bank['concepts'].values() if c['domain']=='Observation']
        self.assertEqual({c['form_name'] for c in obs},set(FOUR_OBSERVATION_FORMS))
        fields={bank['concepts'][k]['source_field'] for k in bank['tasks']['kidney_history_binary']['concept_keys']}
        self.assertNotIn('mhoccur_rnl',fields)

    def test_small_end_to_end_both_backbones_and_cbms(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        task='moca_binary';cfg=yaml.safe_load(Path('study.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';out=root/'results'
            (source/'cache/people').mkdir(parents=True);(out/'labels'/task).mkdir(parents=True)
            cfg.update(source_root=str(source),output_root=str(out),tasks=[task],num_workers=0,
                batch_size=4,max_epochs=2,patience=2,concept_epochs=2,head_epochs=2,joint_epochs=2,
                concept_patience=3,bootstrap_samples=30)
            dimensions=dict(clinical_dim=25,environment_dim=2,activity_dim=2,cgm_dim=2,ecg_waveform_dim=12,ecg_numeric_dim=8)
            write_json(source/'cache/dimensions.json',dimensions)
            write_json(source/'inputs.json',{task:dict(clinical_csv='synthetic',full_clinical_source_task=task)})
            y=np.array([0]*6+[1]*2+[0]*3+[1]*3+[0]*4+[1]*2)
            master=pd.DataFrame(dict(person_id=list(map(str,range(len(y)))),split=['train']*8+['val']*6+['test']*6,y=y))
            for row in master.itertuples():
                s=dict(person_id=row.person_id,source_present=torch.ones(9,dtype=torch.bool),
                    ecg_waveform=torch.randn(20,12),ecg_value_mask=torch.ones(20,12,dtype=torch.bool),
                    ecg_sample_mask=torch.ones(20,dtype=torch.bool),ecg_lead_mask=torch.ones(12,dtype=torch.bool),
                    ecg_numeric=torch.randn(8),ecg_numeric_mask=torch.ones(8,dtype=torch.bool))
                for flag in SOURCE_FLAG_BY_GROUP.values():s[flag]=torch.tensor(True)
                for stream in ('environment','activity','cgm'):
                    s[stream]=torch.randn(12,2);s[stream+'_feature_mask']=torch.ones(12,2,dtype=torch.bool)
                    keys=['slot_mask'] if stream=='environment' else ['observed_mask','valid_mask']
                    for k in keys:s[stream+'_'+k]=torch.ones(12,dtype=torch.bool)
                for g in MODALITIES[3:7]:s[g]=[dict(features=torch.randn(1,512),view_mask=torch.ones(1,dtype=torch.bool))]
                if row.person_id=='13':
                    s['source_present'][2]=False;s[SOURCE_FLAG_BY_GROUP['environment']]=torch.tensor(False)
                save_torch(source/'cache/people'/(row.person_id+'.pt'),s)
            values=torch.randn(len(y),25);mask=torch.ones_like(values,dtype=torch.bool)
            inputs=(master,values,mask,list(range(23)),[str(i) for i in range(25)],[])
            def small_config(cfg,dims):
                return Task2V2ModelConfig(**dims,d_model=8,num_heads=2,temporal_layers=1,fusion_layers=1,
                    retinal_backbone_dim=512,retinal_backbone_frozen=True)
            labels=pd.DataFrame({'person_id':master.person_id,**{'c'+str(i):[str((j+i)%5) for j in range(len(y))] for i in range(13)}})
            labels.to_csv(out/'labels'/task/'labels.csv',index=False)
            schema=dict(inventory_ids=12,concepts=[dict(name='c'+str(i),concept_id=str(100+i//2) if i<2 else str(100+i),
                source_field='field'+str(i),concept_name='Concept '+str(i),field_types=['text'],unit='',domain='Measurement') for i in range(13)])
            write_json(out/'labels'/task/'schema.json',schema)
            with patch('train.clinical_inputs',return_value=inputs),patch('train.cache_config',side_effect=small_config):
                for variant in ('full','basic'):
                    run_model(task,variant,cfg,torch.device('cpu'))
                    info=read_json(out/task/variant/'TRAINING_DONE.json')
                    self.assertEqual(info['epoch_size'],12)
                    self.assertEqual(info['n_train'],8)
                    for split in ('val','test'):
                        cache=load(out/task/variant/f'features_{split}.pt')
                        self.assertEqual(len(cache['person_ids']),6)
                        self.assertEqual(len(set(cache['person_ids'])),6)
                    mode='mixed' if variant=='full' else 'basic'
                    run_cbm(task,mode,cfg,torch.device('cpu'))
            cache=load(out/task/'basic/features_val.pt')
            removed=load(out/task/'basic/removals_val.pt')
            np.testing.assert_allclose(removed['environment']['probability'][-1],cache['original_logits'].sigmoid().numpy()[-1],atol=1e-7)
            run_analysis(task,cfg,torch.device('cpu'));make_report(cfg)
            # A completed F1-only run upgrades from CSVs without loading models.
            analysis=out/task/'analysis'; selected_before=(analysis/'SELECTION.json').read_bytes()
            original={}
            for name in ('classification','comparisons','ablation','restoration'):
                df=pd.read_csv(analysis/(name+'.csv'),dtype={'concept_id':str},float_precision='round_trip')
                original[name]=df
                cols=[c for c in df if not c.startswith('ba_') and c not in ('full_ba','removed_ba','masked_ba','restored_ba','restored_ba_ci_lower','restored_ba_ci_upper')]
                df[cols].to_csv(analysis/(name+'.csv'),index=False)
            done=read_json(analysis/'DONE.json');done.pop('reported_metrics');write_json(analysis/'DONE.json',done)
            with patch('analysis.restore',side_effect=AssertionError('must not load CBM')):
                run_analysis(task,cfg,torch.device('cpu'))
            self.assertEqual(selected_before,(analysis/'SELECTION.json').read_bytes())
            for name,before in original.items():
                after=pd.read_csv(analysis/(name+'.csv'),dtype={'concept_id':str},float_precision='round_trip')
                pd.testing.assert_frame_equal(before,after,check_exact=False,atol=1e-12,rtol=1e-12)
            self.assertIn('BA recovery [95% CI]',(out/'tables/SUMMARY_test.md').read_text())
            selection=read_json(out/task/'analysis/SELECTION.json')
            self.assertEqual(len(selection['top_modalities']),3)
            self.assertEqual(len(selection['concepts']),4)
            r=pd.read_csv(out/task/'analysis/restoration.csv',dtype={'concept_id':str})
            for key,ids in selection['concepts'].items():
                self.assertEqual(len(ids),10)
                mode,source_name=key.split('__',1)
                test=r[(r.split=='test')&(r['mode']==mode)&(r.removed_input==source_name)]
                self.assertEqual(set(test.concept_id),{c['concept_id'] for c in schema['concepts']}|{'ALL_CONCEPTS'})
                ranked=test[test.intervention=='individual'].sort_values('rank')
                self.assertTrue(np.all(np.diff(ranked.value.to_numpy())<=1e-12))
                joint=test[test.intervention=='all_concepts'].iloc[0]
                self.assertAlmostEqual(joint.restored_f1,joint.full_f1)
                self.assertAlmostEqual(joint.restored_ba,joint.full_ba)
            a=pd.read_csv(out/task/'analysis/ablation.csv')
            self.assertEqual(set(a.removed_input),set(REMOVALS));self.assertEqual(len(a),22)
            self.assertTrue((out/'tables/SUMMARY_test.md').exists())
            exported=pd.read_csv(out/'tables/CONCEPT_RESTORATION_ALL_test.csv',dtype={'concept_id':str})
            self.assertEqual(len(exported),4*12)
            self.assertEqual(len(pd.read_csv(out/'tables/ALL_CONCEPTS_RESTORED_test.csv')),4)
            # An older completed analysis must not suppress the expanded test sweep.
            done=read_json(analysis/'DONE.json');done.pop('restoration_version');write_json(analysis/'DONE.json',done)
            run_analysis(task,cfg,torch.device('cpu'))
            self.assertEqual(read_json(analysis/'DONE.json')['restoration_version'],2)
            artifact=load(out/task/'mixed_cbm/model.pt');model=restore(artifact,'cpu')
            cache=load(out/task/'full/features_val.pt')
            full_z,full_p=predict(model,cache['embedding']);masked_z,_=predict(model,cache['masked_embedding'])
            groups=concept_id_groups(artifact['spec']['items']);self.assertEqual(len(groups['100']['columns']),2)
            columns=groups['100']['columns'];direct=masked_z.clone();direct[:,columns]=full_z[:,columns]
            expected=model.task_head(direct).squeeze(-1).sigmoid().numpy()
            np.testing.assert_allclose(restoration_batch(model,masked_z,full_z,[columns])[0],expected,atol=1e-7)
            np.testing.assert_allclose(restoration_batch(model,masked_z,full_z,[list(range(model.spec['width']))])[0],full_p,atol=1e-7)


if __name__=='__main__':unittest.main()
