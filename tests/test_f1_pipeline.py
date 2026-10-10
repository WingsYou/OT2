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
from infosign.common import MODALITIES, REMOVALS, read_json, write_json, save_torch
from infosign.stats import balanced_indices, f1, threshold, PairedF1, bh_adjust
from infosign.train import run_model, load, cached_removals
from infosign.data import SOURCE_FLAG_BY_GROUP
from infosign.encoders import Task2V2ModelConfig
from infosign.cbm.labels import load_bank, SURVEY_ADMIN_IDS
from infosign.cbm.model import run_cbm, concept_id_groups, restoration_batch
from infosign.cbm.model import restore, predict
from infosign.analysis import run_analysis, restoration_details, ANALYSIS_VERSION
from infosign.report import make_report


class F1StudyTests(unittest.TestCase):
    def test_worker_mixed_only_still_trains_basic_backbone(self):
        from run import run_worker
        cfg=dict(cpu_threads=1)
        for variant in ('full','basic'):
            with self.subTest(variant=variant), \
                    patch('infosign.train.run_model') as train_model, \
                    patch('infosign.cbm.model.run_cbm') as train_cbm:
                run_worker(cfg,'moca_binary',variant,cpu=True)
                train_model.assert_called_once_with('moca_binary',variant,cfg,torch.device('cpu'))
                if variant=='full':
                    train_cbm.assert_called_once_with('moca_binary',cfg,torch.device('cpu'))
                else:
                    train_cbm.assert_not_called()

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

    def test_restoration_ties(self):
        y=np.array([1,0,1,0]);masked=np.array([.6,.4,.4,.6],dtype=np.float32)
        ps=np.array([[.7,.3,.45,.55],[.8,.2,.49,.51],[.7,.3,.45,.55],[.6,.6,.4,.4]],dtype=np.float32)
        d=restoration_details(y,ps,masked,.5,[True]*4)
        self.assertEqual([x['rank'] for x in d],[1]*4)
        self.assertEqual([x['tied_concepts'] for x in d],[4]*4)
        self.assertEqual(d[0]['decision_group'],d[1]['decision_group'])
        self.assertNotEqual(d[0]['probability_group'],d[1]['probability_group'])
        self.assertEqual(d[0]['probability_group'],d[2]['probability_group'])
        self.assertNotEqual(d[0]['decision_group'],d[3]['decision_group'])
        self.assertEqual([x['n_prediction_changes'] for x in d],[0,0,0,2])

    def test_approved_inventory(self):
        cfg=yaml.safe_load(Path('config.yaml').read_text())['study']
        self.assertTrue(cfg['exclude_observation'])
        bank=load_bank(observation_forms=(),excluded_ids=SURVEY_ADMIN_IDS)
        counts={'moca_binary':109,'cardiovascular_conditions_binary':107,
                'major_cardiovascular_events_binary':110,'osteoporosis_history_binary':111,
                'kidney_history_binary':111,'lung_history_binary':111}
        self.assertEqual(set(cfg['tasks']),set(counts))
        self.assertFalse(any(c['domain']=='Observation' for c in bank['concepts'].values()))
        self.assertFalse(set(SURVEY_ADMIN_IDS)&{c['concept_id'] for c in bank['concepts'].values()})
        targets={'moca_binary':{'mhoccur_ad','mhoccur_cogn','mocacmpdattim'},
                 'cardiovascular_conditions_binary':{'mhoccur_hbp','mhoccur_clsh','mhoccur_circ','mhoccur_rvo','mhoccur_cvdot'},
                 'major_cardiovascular_events_binary':{'mhoccur_mi','mhoccur_strk'},
                 'osteoporosis_history_binary':{'mhoccur_oa'},'kidney_history_binary':{'mhoccur_rnl'},
                 'lung_history_binary':{'mhoccur_plm'}}
        for task,n in counts.items():
            self.assertEqual(bank['tasks'][task]['inventory_ids'],n)
            keys=bank['tasks'][task]['concept_keys']
            fields={bank['concepts'][k]['source_field'] for k in keys}
            self.assertFalse(fields & targets[task])
            self.assertTrue(any(bank['concepts'][k]['domain']=='Condition' and
                                bank['concepts'][k]['form_name']=='health_survey' for k in keys))
        for task in ('cardiovascular_conditions_binary','major_cardiovascular_events_binary'):
            fields={bank['concepts'][k]['source_field'] for k in bank['tasks'][task]['concept_keys']}
            other='major_cardiovascular_events_binary' if task=='cardiovascular_conditions_binary' else 'cardiovascular_conditions_binary'
            self.assertTrue(targets[other] <= fields)

    def test_small_end_to_end_both_backbones_and_mixed_cbm(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        task='moca_binary';cfg=yaml.safe_load(Path('config.yaml').read_text())['study']
        cfg.pop('backbone_root',None)
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
            with patch('infosign.train.clinical_inputs',return_value=inputs),patch('infosign.train.cache_config',side_effect=small_config):
                for variant in ('full','basic'):
                    run_model(task,variant,cfg,torch.device('cpu'))
                    info=read_json(out/task/variant/'TRAINING_DONE.json')
                    self.assertEqual(info['epoch_size'],12)
                    self.assertEqual(info['n_train'],8)
                    for split in ('val','test'):
                        cache=load(out/task/variant/f'features_{split}.pt')
                        self.assertEqual(len(cache['person_ids']),6)
                        self.assertEqual(len(set(cache['person_ids'])),6)
                run_cbm(task,cfg,torch.device('cpu'))
            cache=load(out/task/'basic/features_val.pt')
            removed=load(out/task/'basic/removals_val.pt')
            np.testing.assert_allclose(removed['environment']['probability'][-1],cache['original_logits'].sigmoid().numpy()[-1],atol=1e-7)
            # Simulate upgrading the old run: Full ablations were never exported.
            expected_full={split:load(out/task/'full'/f'removals_{split}.pt') for split in ('val','test')}
            for split in ('val','test'):(out/task/'full'/f'removals_{split}.pt').unlink()
            write_json(out/task/'analysis/DONE.json',dict(restoration_version=2,reported_metrics=['f1','balanced_accuracy']))
            with patch('infosign.train.CachedDataset',side_effect=AssertionError('must not read raw tensors')):
                run_analysis(task,cfg,torch.device('cpu'))
            make_report(cfg)
            analysis=out/task/'analysis'
            for split in ('val','test'):
                actual=load(analysis/f'removals_full_{split}.pt')
                for source_name in REMOVALS:
                    np.testing.assert_allclose(actual[source_name]['probability'],expected_full[split][source_name]['probability'],atol=1e-7)
            self.assertEqual(read_json(analysis/'DONE.json')['analysis_version'],ANALYSIS_VERSION)
            with patch('infosign.analysis.restore',side_effect=AssertionError('completed analysis must reuse')):
                run_analysis(task,cfg,torch.device('cpu'))
            selection=read_json(analysis/'SELECTION.json')
            self.assertEqual(set(selection['top_modalities']),{'full','basic'})
            self.assertTrue(all(len(v)==3 for v in selection['top_modalities'].values()))
            r=pd.read_csv(analysis/'restoration.csv',dtype={'concept_id':str})
            self.assertEqual(set(r['mode']),{'mixed'})
            self.assertEqual(set(r.removed_input),{'nonbasic_clinical'})
            test=r[r.split=='test']
            self.assertEqual(set(test.concept_id),{c['concept_id'] for c in schema['concepts']}|{'ALL_CONCEPTS'})
            ranked=test[test.intervention=='individual'].sort_values('rank')
            self.assertTrue(np.all(np.diff(ranked.value.to_numpy())<=1e-12))
            joint=test[test.intervention=='all_concepts'].iloc[0]
            self.assertAlmostEqual(joint.restored_f1,joint.full_f1)
            self.assertAlmostEqual(joint.restored_ba,joint.full_ba)
            a=pd.read_csv(analysis/'ablation.csv')
            self.assertEqual(set(a.removed_input),set(REMOVALS));self.assertEqual(len(a),44)
            self.assertEqual(set(a.model),{'full','basic'})
            for x in a.itertuples():
                cache=load(out/task/x.model/f'features_{x.split}.pt');labels=cache['y'].numpy()
                removed=(expected_full[x.split] if x.model=='full' else load(out/task/x.model/f'removals_{x.split}.pt'))[x.removed_input]['probability']
                expected=f1_score(labels,cache['original_logits'].sigmoid().numpy()>=x.threshold)-f1_score(labels,removed>=x.threshold)
                self.assertAlmostEqual(x.value,expected)
                expected=balanced_accuracy_score(labels,cache['original_logits'].sigmoid().numpy()>=x.threshold)-balanced_accuracy_score(labels,removed>=x.threshold)
                self.assertAlmostEqual(x.ba_value,expected)
            text=(out/'tables/SUMMARY_test.md').read_text()
            self.assertIn('All clinical information',text);self.assertIn('Basic-only inputs + all concepts restored',text)
            self.assertNotIn('### Basic CBM: top-three',text)
            self.assertIn('all Observation-domain concepts excluded',text)
            exported=pd.read_csv(out/'tables/CONCEPT_RESTORATION_ALL_test.csv',dtype={'concept_id':str})
            self.assertEqual(len(exported),12)
            self.assertEqual(len(pd.read_csv(out/'tables/ALL_CONCEPTS_RESTORED_test.csv')),1)
            performance=pd.read_csv(out/'tables/MIXED_CBM_PERFORMANCE_test.csv')
            self.assertEqual(set(performance.model),{'mixed_full','mixed_basic','mixed_all_restored'})
            for metric in ('value','ba_value','auroc_value'):
                self.assertAlmostEqual(performance.loc[performance.model=='mixed_all_restored',metric].iloc[0],performance.loc[performance.model=='mixed_full',metric].iloc[0])
            # Compare diagnostics with the saved patient probabilities.
            saved=pd.read_csv(analysis/'restoration_mixed__nonbasic_clinical_test.csv',float_precision='round_trip')
            for row in exported.itertuples():
                prob=saved[row.concept_id].to_numpy(np.float32);masked=saved.masked_probability.to_numpy(np.float32)
                self.assertEqual(row.n_prediction_changes,int(np.count_nonzero((prob>=row.threshold)!=(masked>=row.threshold))))
                self.assertAlmostEqual(row.mean_abs_probability_change,float(np.abs(prob-masked).mean()))
            gains=exported.auroc_value.to_numpy()
            cutoff=gains.mean()+2*gains.std(ddof=0)
            np.testing.assert_allclose(exported.auroc_selection_cutoff,cutoff)
            np.testing.assert_array_equal(exported.auroc_selected,gains>cutoff)
            selected=pd.read_csv(out/'tables/CONCEPT_RESTORATION_SELECTED_AUROC_test.csv',dtype={'concept_id':str})
            self.assertEqual(set(selected.concept_id),set(exported.loc[gains>cutoff,'concept_id']))
            self.assertIn('population SD',(out/'tables/SUMMARY_test.md').read_text())
            # Separate output directory reads old backbones; no model retraining.
            import shutil
            reused=root/'reused'; (reused/task).mkdir(parents=True)
            for variant in ('full','basic'):shutil.move(str(out/task/variant),str(reused/task/variant))
            cfg['backbone_root']=str(reused)
            (analysis/'DONE.json').unlink()
            with patch('infosign.analysis.cached_removals',wraps=cached_removals):
                run_analysis(task,cfg,torch.device('cpu'))
            cfg['exclude_observation']=True; make_report(cfg)
            self.assertIn('all Observation-domain concepts excluded',(out/'tables/SUMMARY_test.md').read_text())
            artifact=load(out/task/'mixed_cbm/model.pt');model=restore(artifact,'cpu')
            cache=load(reused/task/'full/features_val.pt')
            full_z,full_p=predict(model,cache['embedding']);masked_z,_=predict(model,cache['masked_embedding'])
            groups=concept_id_groups(artifact['spec']['items']);self.assertEqual(len(groups['100']['columns']),2)
            columns=groups['100']['columns'];direct=masked_z.clone();direct[:,columns]=full_z[:,columns]
            expected=model.task_head(direct).squeeze(-1).sigmoid().numpy()
            np.testing.assert_allclose(restoration_batch(model,masked_z,full_z,[columns])[0],expected,atol=1e-7)
            np.testing.assert_allclose(restoration_batch(model,masked_z,full_z,[list(range(model.spec['width']))])[0],full_p,atol=1e-7)
            self.assertFalse((out/task/'basic_cbm').exists())
            classification=pd.read_csv(analysis/'classification.csv')
            self.assertEqual(set(classification.model),{'full','full_basic_inference','basic','mixed_full','mixed_basic'})
            comparisons=pd.read_csv(analysis/'comparisons.csv')
            self.assertNotIn('basic_cbm_gain',set(comparisons.comparison))
            text=(out/'tables/SUMMARY_test.md').read_text()
            self.assertNotIn('Basic CBM',text)
            self.assertNotIn('Basic Clinical + CBM',text)
            self.assertIn('Mixed CBM selected fit',text)
            self.assertIn('Basic Clinical model',text)
            with patch('infosign.analysis.restore',side_effect=AssertionError('completed Mixed-only analysis must reuse')):
                run_analysis(task,cfg,torch.device('cpu'))
            from infosign.cbm.diagnostics import run_diagnostics
            from run import make_reports
            run_diagnostics(task,cfg,torch.device('cpu'))
            make_reports(cfg)
            for split in ('val','test'):
                self.assertTrue((out/'tables'/f'SUMMARY_CONCEPT_RECOVERY_{split}.md').exists())
                for intervention in ('PREDICTED','MEASURED'):
                    all_ids=pd.read_csv(out/'tables'/f'{intervention}_CONCEPT_RECOVERY_ALL_{split}.csv',dtype={'concept_id':str})
                    self.assertEqual(set(all_ids.concept_id),set(groups))
                    self.assertEqual(set(all_ids.population),{'all_participants'})



if __name__=='__main__':unittest.main()
