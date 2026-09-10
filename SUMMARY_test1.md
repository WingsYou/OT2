# Clinical A/B/C/D summary

**Overall status: COMPLETE**

Exploratory test analysis (test has already been inspected); frozen seed-42 models. All four predictions are recomputed and paired on the same people. No training or threshold optimization.

This report contains aggregate results and feature names, without participant identifiers or predictions.

Runner: complete; phase: complete; updated UTC: 2026-09-10T04:30:25.125119+00:00.


## Conditions

| ID | Clinical | Other eight modalities |
|---|---|---|
| A | Full, as trained | Native availability |
| B | Full, as trained | All disabled |
| C | Agreed 23 basic columns | Native availability |
| D | Agreed 23 basic columns | All disabled |

Disabled in B/D: `cardiac_ecg`, `environment`, `retinal_flio`, `retinal_oct`, `retinal_octa`, `retinal_photography`, `wearable_activity_monitor`, `wearable_blood_glucose`. ECG removal includes waveform and machine-report values. Clinical values are preserved in A/B, including any Clinical records related to other modalities.

## Main comparison

| Task | N | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| MoCA | 349 | 0.7386 | 0.7317 | 0.6903 | 0.6854 | +0.0069 [-0.0060, +0.0204] | +0.0049 [-0.0577, +0.0678] |
| CVD broad | 348 | 0.7341 | 0.7214 | 0.6513 | 0.6257 | +0.0126 [-0.0075, +0.0349] | +0.0256 [-0.0332, +0.0843] |
| CVD strict | 342 | 0.7570 | 0.7422 | 0.6926 | 0.6582 | +0.0148 [-0.0018, +0.0337] | +0.0345 [-0.0073, +0.0782] |

Positive A−B/C−D means removing the other eight modalities worsened AUROC. A−B measures their collective contribution with full Clinical; C−D measures it with basic Clinical. 0.0100 equals one AUROC percentage point.

- These are input-removal controls on the existing multimodal model. B/D are not separately trained Clinical-only models.
- A/C retain naturally available nonclinical inputs; no complete-case restriction is used for the primary cohort.
- Pointwise 95% paired bootstrap intervals stratify participants by outcome; identical participant masks reuse identical intervals. They exclude training-seed uncertainty and multiplicity adjustment.
- An interval crossing zero does not establish equivalence/noninferiority. These results alone do not demonstrate reconstruction or the performance achievable after retraining.
- All conditions retain the original validation-selected threshold. Positive degradation means reference minus deleted for benefit metrics, and deleted minus reference for Brier/log loss.

## MoCA

Threshold: 0.45519860; bootstrap samples: 2000.

### All-participant metrics

| Metric | A | B | C | D | A − B degradation [95% CI] | C − D degradation [95% CI] |
|---|---:|---:|---:|---:|---|---|
| auroc | 0.7386 | 0.7317 | 0.6903 | 0.6854 | +0.0069 [-0.0060, +0.0204] | +0.0049 [-0.0577, +0.0678] |
| auprc | 0.6533 | 0.6370 | 0.6176 | 0.6095 | +0.0163 [-0.0017, +0.0372] | +0.0081 [-0.0710, +0.0880] |
| balanced_accuracy | 0.6722 | 0.6618 | 0.6478 | 0.5000 | +0.0104 [-0.0258, +0.0462] | +0.1478 [+0.1024, +0.1931] |
| sensitivity | 0.7027 | 0.8311 | 0.8378 | 1.0000 | -0.1284 [-0.1824, -0.0811] | -0.1622 [-0.2230, -0.1081] |
| specificity | 0.6418 | 0.4925 | 0.4577 | 0.0000 | +0.1493 [+0.1045, +0.1990] | +0.4577 [+0.3881, +0.5274] |
| log_loss | 0.6126 | 0.6542 | 0.6615 | 0.8349 | +0.0417 [+0.0253, +0.0575] | +0.1734 [+0.1381, +0.2071] |
| brier | 0.2117 | 0.2281 | 0.2341 | 0.3156 | +0.0165 [+0.0091, +0.0237] | +0.0815 [+0.0656, +0.0968] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 349 | 148 / 201 | 0.7386 | 0.7317 | 0.6903 | 0.6854 | +0.0069 [-0.0060, +0.0204] | +0.0049 [-0.0577, +0.0678] | 0/0/0/0 |
| basic_clinical_present | 349 | 148 / 201 | 0.7386 | 0.7317 | 0.6903 | 0.6854 | +0.0069 [-0.0060, +0.0204] | +0.0049 [-0.0577, +0.0678] | 0/0/0/0 |
| basic_and_nonclinical_present | 349 | 148 / 201 | 0.7386 | 0.7317 | 0.6903 | 0.6854 | +0.0069 [-0.0060, +0.0204] | +0.0049 [-0.0577, +0.0678] | 0/0/0/0 |

### Exact basic Clinical scope

A/B retain all 955 trained Clinical columns. C/D retain 23 and mask the other 932; both values and observed masks are cleared, with the original input width/scalers preserved.

Basic groups: age_related: 3, education: 9, language_background: 6, marital_status: 5.

| Retained encoded feature | Source table | Source key |
|---|---|---|
| person__age | participants | age |
| obse__brthyy__c4078999__u0sunknown | observation | brthyy |
| obse__cage__c4265453__u0sunknown | observation | cage |
| obse__years_of_education__c42528764__u0sunknown | observation | years_of_education |
| obsc__cl_maristat__response_c21499180 | observation | cl_maristat |
| obsc__cl_maristat__response_c45876756 | observation | cl_maristat |
| obsc__cl_maristat__response_c45879879 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883375 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883711 | observation | cl_maristat |
| obsc__pxed1__response_c45878298 | observation | pxed1 |
| obsc__pxed1__response_c45881674 | observation | pxed1 |
| obsc__pxed1__response_c45882538 | observation | pxed1 |
| obsc__pxed1__response_c45883176 | observation | pxed1 |
| obsc__pxed1__response_c45884155 | observation | pxed1 |
| obsc__pxed1__response_c45884462 | observation | pxed1 |
| obsc__pxed1__response_c45884463 | observation | pxed1 |
| obsc__pxed1__response_c45884464 | observation | pxed1 |
| obsc__pxep1__response_c45877994 | observation | pxep1 |
| obsc__pxep1__response_c45878245 | observation | pxep1 |
| obsc__pxep2__response_c37079013 | observation | pxep2 |
| obsc__pxep2__response_c45878245 | observation | pxep2 |
| obsc__pxep3__response_c45877705 | observation | pxep3 |
| obsc__pxep3__response_c45877707 | observation | pxep3 |

### Provenance

| Item | Value |
|---|---|
| analysis_role | exploratory_test_collective_nonclinical_reliance |
| checkpoint_path | /data/pzhao3/ot2_output/model_full_assets_v1/models/moca_binary/seed42/best_model.pt |
| checkpoint_sha256 | bc720e18565f8c4ff7699d27328cb27c1098266419e53aaf5962310f5339e154 |
| config_hash | d83c64cd36ed0ff2ec0130500d033873fd65bc97ac2d1cf187decb54f972299c |
| config_sha256 | d3385b42d5e5de51d0d9d2264cca4913f640e189622cf4c69696763cdfb6719f |
| clinical_schema_sha256 | da677fb50c38589626122374f712131d48ef44f1c1785211b0b066cfb67e7d26 |
| clinical_catalog_sha256 | 8e2b9ef2c72570ce6c32559e40f9dffbfe2abc52e31af82035ab63b6980fd2b9 |
| plan_sha256 | 29032f72905f3e230dd2e90289a04583a3b76d6036811021d9488fd65c63b58e |
| threshold_source | saved_full_teacher_validation_threshold |
| training_seed | 42 |
| bootstrap_seed | 42 |
| metadata_sha256 | ab20721849071bd7702551d35440d509f4008a35641c931bbabbefd87bec6de1 |
| basic_plan_sha256 | 53dd453ed415efa6b2acc95e1fbd9399b26a4752c91ef6493b50e2a4e4328258 |
| source_cache | {"content_sha256": "a977812a8c8aed92f0db91bd05186c9984fda35e41b5fa9810b46c2a212b7b7f", "file_sha256": "1eca5d1494bfd6e9e5fb9f34376e762f156914d21f6e75be2e56d4b8d87d8705", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "bc720e18565f8c4ff7699d27328cb27c1098266419e53aaf5962310f5339e154", "cohort_person_ids_sha256": "b14ecc6d927f2f3eb0f917154b68e0bf58ed5c77bb2610c047f7c5e00c50c14a", "config_hash": "d83c64cd36ed0ff2ec0130500d033873fd65bc97ac2d1cf187decb54f972299c", "encoder_batch_size": 2, "master_config_hash": "13683a6afc4927f4923147109bfc20e6f115d4154a08d8014deb6d89f133e820", "model_config_hash": "981300bd5b48c18b2b34142beac7e38f3c28339c0c8096708c29deb9124d42b2", "n_participants": 349, "scaler_config_hash": "a8d107610d6700cc000d74ce3acdd87aade399a4b3ea7de404bbba992760d827", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "test", "task": "moca_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1_test/moca_binary/seed42/test_domain/source_token_cache_test.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "27d3784c351b77e9cf54e9976eae3c4226c13f6cd865126869c071795167f320", "abcd_cohort_counts.csv": "be0e8e815c593a5f18f8c9b99475666018b581295d9a7f9e243a6f02948cbf85", "abcd_conditions.json": "0595453e5569b9cec9f8b890cd19f1abc4c057c4854fa2a2292a1441c977ce98", "abcd_metrics.csv": "f2674be8822a4cf142067f9923fba42a7c17728e81bff0374143626903ec7b24", "abcd_paired_predictions.csv": "f6eba93369f0613903cc748d000aa0b38c43f56c1b4a072c4e4017dcd017d0dd", "abcd_plan.json": "de7c0449a45c77b68b164ccf32bf9c2482669a5c076a965696ddc56ed666130d", "abcd_summary.csv": "37c8c81cb18e9505a6f1fa25a0289c7d836c303eb81081e03e374010dfd884b8"} |
| all_four_predictions_recomputed_and_paired | True |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| identical_cohort_intervals_reused | True |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
## CVD broad

Threshold: 0.91117971; bootstrap samples: 2000.

### All-participant metrics

| Metric | A | B | C | D | A − B degradation [95% CI] | C − D degradation [95% CI] |
|---|---:|---:|---:|---:|---|---|
| auroc | 0.7341 | 0.7214 | 0.6513 | 0.6257 | +0.0126 [-0.0075, +0.0349] | +0.0256 [-0.0332, +0.0843] |
| auprc | 0.9116 | 0.8913 | 0.8879 | 0.8544 | +0.0203 [+0.0021, +0.0371] | +0.0335 [-0.0004, +0.0668] |
| balanced_accuracy | 0.6380 | 0.6726 | 0.6489 | 0.5373 | -0.0346 [-0.0766, +0.0130] | +0.1117 [+0.0481, +0.1760] |
| sensitivity | 0.6192 | 0.8078 | 0.5516 | 0.9253 | -0.1886 [-0.2349, -0.1458] | -0.3737 [-0.4306, -0.3167] |
| specificity | 0.6567 | 0.5373 | 0.7463 | 0.1493 | +0.1194 [+0.0448, +0.1944] | +0.5970 [+0.4776, +0.7164] |
| log_loss | 0.4808 | 0.4742 | 0.5472 | 0.5723 | -0.0066 [-0.0431, +0.0303] | +0.0251 [-0.0285, +0.0773] |
| brier | 0.1507 | 0.1431 | 0.1788 | 0.1700 | -0.0076 [-0.0222, +0.0075] | -0.0088 [-0.0290, +0.0114] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 348 | 281 / 67 | 0.7341 | 0.7214 | 0.6513 | 0.6257 | +0.0126 [-0.0075, +0.0349] | +0.0256 [-0.0332, +0.0843] | 0/0/0/0 |
| basic_clinical_present | 348 | 281 / 67 | 0.7341 | 0.7214 | 0.6513 | 0.6257 | +0.0126 [-0.0075, +0.0349] | +0.0256 [-0.0332, +0.0843] | 0/0/0/0 |
| basic_and_nonclinical_present | 346 | 280 / 66 | 0.7358 | 0.7205 | 0.6576 | 0.6322 | +0.0152 [-0.0057, +0.0380] | +0.0254 [-0.0349, +0.0859] | 0/0/0/0 |

### Exact basic Clinical scope

A/B retain all 965 trained Clinical columns. C/D retain 23 and mask the other 942; both values and observed masks are cleared, with the original input width/scalers preserved.

Basic groups: age_related: 3, education: 9, language_background: 6, marital_status: 5.

| Retained encoded feature | Source table | Source key |
|---|---|---|
| person__age | participants | age |
| obse__brthyy__c4078999__u0sunknown | observation | brthyy |
| obse__cage__c4265453__u0sunknown | observation | cage |
| obse__years_of_education__c42528764__u0sunknown | observation | years_of_education |
| obsc__cl_maristat__response_c21499180 | observation | cl_maristat |
| obsc__cl_maristat__response_c45876756 | observation | cl_maristat |
| obsc__cl_maristat__response_c45879879 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883375 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883711 | observation | cl_maristat |
| obsc__pxed1__response_c45878298 | observation | pxed1 |
| obsc__pxed1__response_c45881674 | observation | pxed1 |
| obsc__pxed1__response_c45882538 | observation | pxed1 |
| obsc__pxed1__response_c45883176 | observation | pxed1 |
| obsc__pxed1__response_c45884155 | observation | pxed1 |
| obsc__pxed1__response_c45884462 | observation | pxed1 |
| obsc__pxed1__response_c45884463 | observation | pxed1 |
| obsc__pxed1__response_c45884464 | observation | pxed1 |
| obsc__pxep1__response_c45877994 | observation | pxep1 |
| obsc__pxep1__response_c45878245 | observation | pxep1 |
| obsc__pxep2__response_c37079013 | observation | pxep2 |
| obsc__pxep2__response_c45878245 | observation | pxep2 |
| obsc__pxep3__response_c45877705 | observation | pxep3 |
| obsc__pxep3__response_c45877707 | observation | pxep3 |

### Provenance

| Item | Value |
|---|---|
| analysis_role | exploratory_test_collective_nonclinical_reliance |
| checkpoint_path | /data/pzhao3/ot2_output/model_full_assets_v1/models/cvd_broad_binary/seed42/best_model.pt |
| checkpoint_sha256 | 0ec20efb87ce84368742af51a11949bfc9a11aaf3ccae77c54554230ad192cbf |
| config_hash | 0e8280aca239e98788b867020bdb2a28b6cd89b46e1f0c7a12af8e800d12f66c |
| config_sha256 | a424b7be9315822393be31730261da514549d0d6a8c129154f1cde297662e735 |
| clinical_schema_sha256 | 7e64a5b8e22a4ba570824742f96f9b5f0c9c39e9501a34c78a24c1b4c193fb1b |
| clinical_catalog_sha256 | 05a1f3e9acc27d814748eda9e89a60e9ceba0284e7e3f8bf377acafcf16ee79a |
| plan_sha256 | e0ab0cb6c931ff5ab824667e59cb6d49ba5346fe3567bf631a5ee08983edfa72 |
| threshold_source | saved_full_teacher_validation_threshold |
| training_seed | 42 |
| bootstrap_seed | 42 |
| metadata_sha256 | 510c745e6a010a9ee5d140e57f1600b4c04feac752ece04186352ea9df8bb52a |
| basic_plan_sha256 | 79a1648b6f6c457cd42394bbac004b6f3eb60549b11716cbe5c945ca4c88f217 |
| source_cache | {"content_sha256": "832fc4c4c433fa00484182351ea0c2d310fa0872baa1e1ae144b6db41cb77a77", "file_sha256": "300cef7211db2933369648a3021df069fce2a993723dcf632788ee62892d83ba", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "0ec20efb87ce84368742af51a11949bfc9a11aaf3ccae77c54554230ad192cbf", "cohort_person_ids_sha256": "ffb3cea4b7f77cefb486e181194edd8a12c3663d9918e772329fc7e59d5e4fcb", "config_hash": "0e8280aca239e98788b867020bdb2a28b6cd89b46e1f0c7a12af8e800d12f66c", "encoder_batch_size": 2, "master_config_hash": "bfb04d90d1a3815d58a4f7005f900a57e0695a243f5728e451ed8bf378087cde", "model_config_hash": "21fc1229e93a9566e19b525e98884d2fbfdbe62ef7e7a5a24b6f29b21ab6d1e6", "n_participants": 348, "scaler_config_hash": "785040fd28d09ea22bb9c226eb4f4c8907e28730554dd06fccfbef710f4a9c02", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "test", "task": "cvd_broad_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1_test/cvd_broad_binary/seed42/test_domain/source_token_cache_test.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "ca1c3710eb14823ece766ea9f7dec8fb00912d449c9af9b90b97f430157ae1f3", "abcd_cohort_counts.csv": "5ef6ca3ddf83cbecab853b204d1fb27125b6bb3d05e4f153eee2bf052309ea8d", "abcd_conditions.json": "fb5b0555c092a4c0e6e9875e62fe6ecc1b481a82c925e1c23a2ab171085e7005", "abcd_metrics.csv": "62c4291b48d028c4626e9c059dc8c43333f7ca2a707e30d275e9268fa40bea95", "abcd_paired_predictions.csv": "99c95f78d7c7f1095cf5e0c3f236b77270ea4edb7d1f518c6be90d0d0fbaa063", "abcd_plan.json": "4283d9aff983b40c28810646e7ed68dc29f23eac6afa0201e884c0214dd75a1a", "abcd_summary.csv": "dac8986c921df2eeee6eb94365d5d3f10c122d22c50402ecae5806ec6f40704c"} |
| all_four_predictions_recomputed_and_paired | True |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| identical_cohort_intervals_reused | True |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
## CVD strict

Threshold: 0.59784270; bootstrap samples: 2000.

### All-participant metrics

| Metric | A | B | C | D | A − B degradation [95% CI] | C − D degradation [95% CI] |
|---|---:|---:|---:|---:|---|---|
| auroc | 0.7570 | 0.7422 | 0.6926 | 0.6582 | +0.0148 [-0.0018, +0.0337] | +0.0345 [-0.0073, +0.0782] |
| auprc | 0.6473 | 0.6095 | 0.5814 | 0.4595 | +0.0379 [-0.0070, +0.0773] | +0.1219 [+0.0407, +0.1868] |
| balanced_accuracy | 0.7158 | 0.7287 | 0.6458 | 0.6140 | -0.0129 [-0.0433, +0.0133] | +0.0318 [-0.0162, +0.0841] |
| sensitivity | 0.6783 | 0.7304 | 0.6000 | 0.7478 | -0.0522 [-0.1043, -0.0087] | -0.1478 [-0.2261, -0.0609] |
| specificity | 0.7533 | 0.7269 | 0.6916 | 0.4802 | +0.0264 [+0.0000, +0.0573] | +0.2115 [+0.1498, +0.2731] |
| log_loss | 0.6286 | 0.6418 | 0.6881 | 0.7654 | +0.0132 [-0.0089, +0.0335] | +0.0773 [+0.0439, +0.1096] |
| brier | 0.2157 | 0.2241 | 0.2464 | 0.2850 | +0.0085 [+0.0000, +0.0166] | +0.0386 [+0.0240, +0.0524] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 342 | 115 / 227 | 0.7570 | 0.7422 | 0.6926 | 0.6582 | +0.0148 [-0.0018, +0.0337] | +0.0345 [-0.0073, +0.0782] | 0/0/0/0 |
| basic_clinical_present | 342 | 115 / 227 | 0.7570 | 0.7422 | 0.6926 | 0.6582 | +0.0148 [-0.0018, +0.0337] | +0.0345 [-0.0073, +0.0782] | 0/0/0/0 |
| basic_and_nonclinical_present | 340 | 114 / 226 | 0.7591 | 0.7448 | 0.6925 | 0.6573 | +0.0143 [-0.0025, +0.0320] | +0.0353 [-0.0068, +0.0787] | 0/0/0/0 |

### Exact basic Clinical scope

A/B retain all 974 trained Clinical columns. C/D retain 23 and mask the other 951; both values and observed masks are cleared, with the original input width/scalers preserved.

Basic groups: age_related: 3, education: 9, language_background: 6, marital_status: 5.

| Retained encoded feature | Source table | Source key |
|---|---|---|
| person__age | participants | age |
| obse__brthyy__c4078999__u0sunknown | observation | brthyy |
| obse__cage__c4265453__u0sunknown | observation | cage |
| obse__years_of_education__c42528764__u0sunknown | observation | years_of_education |
| obsc__cl_maristat__response_c21499180 | observation | cl_maristat |
| obsc__cl_maristat__response_c45876756 | observation | cl_maristat |
| obsc__cl_maristat__response_c45879879 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883375 | observation | cl_maristat |
| obsc__cl_maristat__response_c45883711 | observation | cl_maristat |
| obsc__pxed1__response_c45878298 | observation | pxed1 |
| obsc__pxed1__response_c45881674 | observation | pxed1 |
| obsc__pxed1__response_c45882538 | observation | pxed1 |
| obsc__pxed1__response_c45883176 | observation | pxed1 |
| obsc__pxed1__response_c45884155 | observation | pxed1 |
| obsc__pxed1__response_c45884462 | observation | pxed1 |
| obsc__pxed1__response_c45884463 | observation | pxed1 |
| obsc__pxed1__response_c45884464 | observation | pxed1 |
| obsc__pxep1__response_c45877994 | observation | pxep1 |
| obsc__pxep1__response_c45878245 | observation | pxep1 |
| obsc__pxep2__response_c37079013 | observation | pxep2 |
| obsc__pxep2__response_c45878245 | observation | pxep2 |
| obsc__pxep3__response_c45877705 | observation | pxep3 |
| obsc__pxep3__response_c45877707 | observation | pxep3 |

### Provenance

| Item | Value |
|---|---|
| analysis_role | exploratory_test_collective_nonclinical_reliance |
| checkpoint_path | /data/pzhao3/ot2_output/model_full_assets_v1/models/cvd_strict_binary/seed42/best_model.pt |
| checkpoint_sha256 | 8bb7c4416e6318b253f1bbb5f0f90aad912d1b3a4fe67ad0ae2f7df8b7bae1a3 |
| config_hash | 95fee85977cd6e00d6481da4f7d2302d44a0fe0bcb9a1ac7d8523a15029e1e45 |
| config_sha256 | bf4c2a3be1dcd02b2bb66acd907a1e5f10cbac2ca72f469b844a794fb7e08083 |
| clinical_schema_sha256 | 0a9fe93337a6a0073ebc7f56de25b65f4c433449d65d42317fba37d7e85bc32e |
| clinical_catalog_sha256 | 320131d74ee2508d95fdb35e9ae6de5ca022f6399dba0493ba3056d214aa7929 |
| plan_sha256 | 765cf467b2dbaa8d5c4a22018d9391f8619b5e04c4a8fe7094a4b840bc2fe776 |
| threshold_source | saved_full_teacher_validation_threshold |
| training_seed | 42 |
| bootstrap_seed | 42 |
| metadata_sha256 | a4ba731d716276de1cd6d863b75ee6bfb33c7a631a2dc8880e5bb1df34be57c5 |
| basic_plan_sha256 | 1715f7984a787f8b0edf88fadc30dca2991a19b00faba6a9aa40dcbbc3aeb69a |
| source_cache | {"content_sha256": "b63edf05eb38a256019f95596f3606f57c2b8f65e29ab6a7508b21a4aa8afd07", "file_sha256": "3320fa6c722a4e7e48d616c6d11940243a3cff9ce41fa8769815f86c704dc673", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "8bb7c4416e6318b253f1bbb5f0f90aad912d1b3a4fe67ad0ae2f7df8b7bae1a3", "cohort_person_ids_sha256": "05c649d8242ee4f9ffa5a642f9fd7e0cc265fdc1cc0a8d00dfdece053ae794f9", "config_hash": "95fee85977cd6e00d6481da4f7d2302d44a0fe0bcb9a1ac7d8523a15029e1e45", "encoder_batch_size": 2, "master_config_hash": "2f6c7be047082aadf347763b282a8a1c2c979f94f0391316381ebbf0dd7d5775", "model_config_hash": "a42dc3fbc1bee9562475396635898921f93d01b316b109983602bcffc3648674", "n_participants": 342, "scaler_config_hash": "f041d13116980d3c528715f30762737108271ac00c58fb69672f3bd2c9251844", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "test", "task": "cvd_strict_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1_test/cvd_strict_binary/seed42/test_domain/source_token_cache_test.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "3c2c144f2bef95a047f4d7faf9dcf97c3d0b1b5ad97a162a18190078a1f0c6b7", "abcd_cohort_counts.csv": "5f82705061ad50d84b5bfaf63a2f31c5c92d54c4d8ff9897ebab55829e73fafd", "abcd_conditions.json": "36879d41afc92f4525c9cbd8dd3f810b7a9bf4777967fefcbd816fd931a7c962", "abcd_metrics.csv": "caa9abc95ef848b13cc11e74fceb6417b005a857675ccabd1f78a717c3bfbeec", "abcd_paired_predictions.csv": "2113004356354aa54aae697bab30d7607426baca733dbfbfeb93ec558a53f5d8", "abcd_plan.json": "655f12fe793d444a633c29303a9a8dee15c01115d48f5650e1814a8ed9a33f4d", "abcd_summary.csv": "7b8cc54b7d0e18356ece1b68d9cab24858eda50c39c7b7dcda0852e053f7ccfa"} |
| all_four_predictions_recomputed_and_paired | True |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| identical_cohort_intervals_reused | True |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |

## Shared code snapshot

| File | SHA256 |
|---|---|
| abcd_inference_engine_v1.py | 096cb57d89f087eb2cbf64f701d9c9c9bb332f67acedbe892d47200e4a1d50c2 |
| analyze_clinical_abcd_v1.py | f00abeda64e19bba229d1885c16177ec1966b515072dbc00f089afd06eebd638 |
| analyze_modality_importance_v2.py | b20ebe3b7b08f7597b8b03a71d3cd257cf735b8ceaee4a38de9d4e809c1f32e8 |
| clinical_abcd_plan_v1.py | daee4d392a7d91ae66666b4f5c9268ebe2ae783a30b687e2d9180e65fe52f6f4 |
| clinical_basic_information_v1.py | 77b6feef189b14b498716e0a2f75f7af330c5966e63d90a53258a28c0a89a259 |
| clinical_overlap_inference_v1.py | 2360ec9a2500799c0727407902ab70b8c705f7b2bd2b639dc676fe73b155ef96 |
| fit_train_scalers_v2.py | 1c8138a3a66ff6ad922f4e907df95c0b0e5c67487042963a37ed9fce52102419 |
| preprocessing_v2.py | 0219f9a8e9c291f2b9c5091205e4476db73c17c6a232fee1bc789491e7b44e14 |
| run_clinical_abcd_v1.py | d57c128cc5d14b688b6dabf7c870e77a3c1a44ad65787133064d82bde0195a25 |
| summarize_clinical_abcd_v1.py | a30a6aefe444c309c6f2ae28fa621bc08e5b2a55762e70f8e27a458cb1167827 |
| task2_v2_config.py | 458848c8032a9d47fc54ca112043b2929b81e126cd80601c94ee20514e21ac06 |
| task2_v2_dataset.py | 24afc6b37058021b4f20bc17a50f16a5fe72e543768ac4365a3dc214fcb9fcef |
| task2_v2_engine.py | 63f8014d3d4c9fd7d7c2469d0496bf17245bfa9b8110503b39572d80a6286ac4 |
| task2_v2_model.py | 7f9a22f149306ef16412311e9f8d6275804fe19b66f03e2aedd031dd102c5463 |
| train_task2_v2.py | 34cf798448a025cc8d49a69a4df4eec549d798ca6a004e1f7e5435a4b08444a0 |
