# Clinical A/B/C/D summary

**Overall status: COMPLETE**

Validation analysis; frozen seed-42 models. All four predictions are recomputed and paired on the same people. No training or threshold optimization.

This report contains aggregate results and feature names, without participant identifiers or predictions.

Runner: complete; phase: complete; updated UTC: 2026-09-10T04:28:28.696214+00:00.


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
| MoCA | 351 | 0.7270 | 0.7272 | 0.6559 | 0.6729 | -0.0001 [-0.0149, +0.0153] | -0.0170 [-0.0786, +0.0450] |
| CVD broad | 352 | 0.7395 | 0.7152 | 0.6620 | 0.6046 | +0.0244 [+0.0027, +0.0490] | +0.0574 [-0.0004, +0.1157] |
| CVD strict | 351 | 0.7261 | 0.7257 | 0.5928 | 0.5569 | +0.0004 [-0.0187, +0.0207] | +0.0359 [-0.0097, +0.0824] |

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
| auroc | 0.7270 | 0.7272 | 0.6559 | 0.6729 | -0.0001 [-0.0149, +0.0153] | -0.0170 [-0.0786, +0.0450] |
| auprc | 0.6589 | 0.6633 | 0.5740 | 0.6101 | -0.0044 [-0.0320, +0.0277] | -0.0361 [-0.1131, +0.0441] |
| balanced_accuracy | 0.6865 | 0.6660 | 0.6006 | 0.5000 | +0.0205 [-0.0128, +0.0539] | +0.1006 [+0.0558, +0.1468] |
| sensitivity | 0.7372 | 0.8397 | 0.8013 | 1.0000 | -0.1026 [-0.1538, -0.0577] | -0.1987 [-0.2628, -0.1410] |
| specificity | 0.6359 | 0.4923 | 0.4000 | 0.0000 | +0.1436 [+0.0974, +0.1949] | +0.4000 [+0.3333, +0.4667] |
| log_loss | 0.6204 | 0.6507 | 0.6807 | 0.8177 | +0.0304 [+0.0123, +0.0484] | +0.1370 [+0.1013, +0.1722] |
| brier | 0.2141 | 0.2266 | 0.2420 | 0.3075 | +0.0125 [+0.0044, +0.0208] | +0.0655 [+0.0495, +0.0813] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 351 | 156 / 195 | 0.7270 | 0.7272 | 0.6559 | 0.6729 | -0.0001 [-0.0149, +0.0153] | -0.0170 [-0.0786, +0.0450] | 0/0/0/0 |
| basic_clinical_present | 351 | 156 / 195 | 0.7270 | 0.7272 | 0.6559 | 0.6729 | -0.0001 [-0.0149, +0.0153] | -0.0170 [-0.0786, +0.0450] | 0/0/0/0 |
| basic_and_nonclinical_present | 351 | 156 / 195 | 0.7270 | 0.7272 | 0.6559 | 0.6729 | -0.0001 [-0.0149, +0.0153] | -0.0170 [-0.0786, +0.0450] | 0/0/0/0 |

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
| analysis_role | validation_collective_nonclinical_reliance |
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
| metadata_sha256 | 260518a89cc2e90f015107255d70104fea98e285b077fc39b2bcacec41089981 |
| basic_plan_sha256 | 53dd453ed415efa6b2acc95e1fbd9399b26a4752c91ef6493b50e2a4e4328258 |
| source_cache | {"content_sha256": "bd2949d257f58437882b572968dc0321f7249c0012e940874a69178e2104ceea", "file_sha256": "bcc5e0533f99c0e4e51b8006bb8a6c11d1231ad0610271b7d4ef879ffd49f627", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "bc720e18565f8c4ff7699d27328cb27c1098266419e53aaf5962310f5339e154", "cohort_person_ids_sha256": "ee04e45f40889bf56a041444f0a734a87863bda812544e4369ca75565ce91da0", "config_hash": "d83c64cd36ed0ff2ec0130500d033873fd65bc97ac2d1cf187decb54f972299c", "encoder_batch_size": 2, "master_config_hash": "13683a6afc4927f4923147109bfc20e6f115d4154a08d8014deb6d89f133e820", "model_config_hash": "981300bd5b48c18b2b34142beac7e38f3c28339c0c8096708c29deb9124d42b2", "n_participants": 351, "scaler_config_hash": "a8d107610d6700cc000d74ce3acdd87aade399a4b3ea7de404bbba992760d827", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "moca_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1/moca_binary/seed42/val_domain/source_token_cache_val.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "d8b9a8909552669018d755c29f5f3e7bdcac60dddb01a3cf0aeeda6e0a84f965", "abcd_cohort_counts.csv": "e3861edc9aada02f4bf0932633d6627a7d348d50ef6172c83f015a5bd6f67c7b", "abcd_conditions.json": "0595453e5569b9cec9f8b890cd19f1abc4c057c4854fa2a2292a1441c977ce98", "abcd_metrics.csv": "294865fe5561e56c2faa971df87c469a8d3f1d2250a5af902eae0a6896a67e22", "abcd_paired_predictions.csv": "1f12d65df8710bc30bd733095f0731ec66127984701f07a978da4442ecba77bb", "abcd_plan.json": "de7c0449a45c77b68b164ccf32bf9c2482669a5c076a965696ddc56ed666130d", "abcd_summary.csv": "05b17d0135e1ef157f6d84a80462460abd4a01f8a50d18f6daed72e607296ba0"} |
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
| auroc | 0.7395 | 0.7152 | 0.6620 | 0.6046 | +0.0244 [+0.0027, +0.0490] | +0.0574 [-0.0004, +0.1157] |
| auprc | 0.8921 | 0.8889 | 0.8744 | 0.8260 | +0.0032 [-0.0150, +0.0242] | +0.0484 [+0.0161, +0.0800] |
| balanced_accuracy | 0.6866 | 0.6331 | 0.5897 | 0.5560 | +0.0534 [-0.0023, +0.1093] | +0.0338 [-0.0312, +0.0979] |
| sensitivity | 0.6423 | 0.8175 | 0.5000 | 0.9453 | -0.1752 [-0.2226, -0.1314] | -0.4453 [-0.5073, -0.3832] |
| specificity | 0.7308 | 0.4487 | 0.6795 | 0.1667 | +0.2821 [+0.1795, +0.3846] | +0.5128 [+0.4099, +0.6154] |
| log_loss | 0.5128 | 0.5571 | 0.5333 | 0.6500 | +0.0443 [+0.0142, +0.0749] | +0.1166 [+0.0737, +0.1595] |
| brier | 0.1638 | 0.1694 | 0.1694 | 0.1950 | +0.0056 [-0.0067, +0.0180] | +0.0256 [+0.0093, +0.0423] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 352 | 274 / 78 | 0.7395 | 0.7152 | 0.6620 | 0.6046 | +0.0244 [+0.0027, +0.0490] | +0.0574 [-0.0004, +0.1157] | 0/0/0/0 |
| basic_clinical_present | 352 | 274 / 78 | 0.7395 | 0.7152 | 0.6620 | 0.6046 | +0.0244 [+0.0027, +0.0490] | +0.0574 [-0.0004, +0.1157] | 0/0/0/0 |
| basic_and_nonclinical_present | 352 | 274 / 78 | 0.7395 | 0.7152 | 0.6620 | 0.6046 | +0.0244 [+0.0027, +0.0490] | +0.0574 [-0.0004, +0.1157] | 0/0/0/0 |

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
| analysis_role | validation_collective_nonclinical_reliance |
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
| metadata_sha256 | 9b7664097bdea5a2c95d3b0eb300c4837c51dd9870a3d1bd6d1e2da267c80014 |
| basic_plan_sha256 | 79a1648b6f6c457cd42394bbac004b6f3eb60549b11716cbe5c945ca4c88f217 |
| source_cache | {"content_sha256": "5134b1379353ec0e9255474563fa1a20daad2e3121972707e10d5dfc29673994", "file_sha256": "90a292a105698042352d1be47bc70f65ecf9311dd89bf817e7521c3464e1e219", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "0ec20efb87ce84368742af51a11949bfc9a11aaf3ccae77c54554230ad192cbf", "cohort_person_ids_sha256": "648b808a65bc627c36c062fbb849f026788445cdc5eab7b699b88d1bdf980287", "config_hash": "0e8280aca239e98788b867020bdb2a28b6cd89b46e1f0c7a12af8e800d12f66c", "encoder_batch_size": 2, "master_config_hash": "bfb04d90d1a3815d58a4f7005f900a57e0695a243f5728e451ed8bf378087cde", "model_config_hash": "21fc1229e93a9566e19b525e98884d2fbfdbe62ef7e7a5a24b6f29b21ab6d1e6", "n_participants": 352, "scaler_config_hash": "785040fd28d09ea22bb9c226eb4f4c8907e28730554dd06fccfbef710f4a9c02", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "cvd_broad_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1/cvd_broad_binary/seed42/val_domain/source_token_cache_val.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "6ffc2f6d07c2f7d6e020de696183986584fe74ffacbff77152830c25bb416c37", "abcd_cohort_counts.csv": "4b6de1b9739ba6eed8147dfb0dcb0671cc55d631248d4eac2a4f91e0a0be7985", "abcd_conditions.json": "fb5b0555c092a4c0e6e9875e62fe6ecc1b481a82c925e1c23a2ab171085e7005", "abcd_metrics.csv": "b203f640c09966b3c0bc862182c2055313062bb7b677beab65ba6e1fd0c71f23", "abcd_paired_predictions.csv": "be9f9e46f027665a846f49cfcd14927ae9cd44f273e46ed670acf46d639116a7", "abcd_plan.json": "4283d9aff983b40c28810646e7ed68dc29f23eac6afa0201e884c0214dd75a1a", "abcd_summary.csv": "81e8c081191698623bd6d1742dab0944761671acd976ba7d80139b3b098736a7"} |
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
| auroc | 0.7261 | 0.7257 | 0.5928 | 0.5569 | +0.0004 [-0.0187, +0.0207] | +0.0359 [-0.0097, +0.0824] |
| auprc | 0.5188 | 0.5223 | 0.4059 | 0.3279 | -0.0035 [-0.0407, +0.0353] | +0.0781 [+0.0041, +0.1501] |
| balanced_accuracy | 0.6935 | 0.6573 | 0.5519 | 0.5404 | +0.0363 [+0.0031, +0.0696] | +0.0115 [-0.0395, +0.0615] |
| sensitivity | 0.6768 | 0.6162 | 0.4848 | 0.6364 | +0.0606 [+0.0000, +0.1212] | -0.1515 [-0.2424, -0.0606] |
| specificity | 0.7103 | 0.6984 | 0.6190 | 0.4444 | +0.0119 [-0.0159, +0.0437] | +0.1746 [+0.1151, +0.2341] |
| log_loss | 0.6583 | 0.6595 | 0.7638 | 0.8092 | +0.0012 [-0.0227, +0.0234] | +0.0454 [+0.0086, +0.0790] |
| brier | 0.2294 | 0.2323 | 0.2795 | 0.3063 | +0.0029 [-0.0064, +0.0114] | +0.0268 [+0.0108, +0.0411] |

### Availability strata

All four conditions use the same people within each row. `basic_clinical_present` requires at least one observed retained Clinical column; `basic_and_nonclinical_present` also requires at least one other native modality.

| Cohort | N | Positive / negative | A AUROC | B AUROC | C AUROC | D AUROC | A − B [95% CI] | C − D [95% CI] | No input A/B/C/D |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| all_participants | 351 | 99 / 252 | 0.7261 | 0.7257 | 0.5928 | 0.5569 | +0.0004 [-0.0187, +0.0207] | +0.0359 [-0.0097, +0.0824] | 0/0/0/0 |
| basic_clinical_present | 351 | 99 / 252 | 0.7261 | 0.7257 | 0.5928 | 0.5569 | +0.0004 [-0.0187, +0.0207] | +0.0359 [-0.0097, +0.0824] | 0/0/0/0 |
| basic_and_nonclinical_present | 351 | 99 / 252 | 0.7261 | 0.7257 | 0.5928 | 0.5569 | +0.0004 [-0.0187, +0.0207] | +0.0359 [-0.0097, +0.0824] | 0/0/0/0 |

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
| analysis_role | validation_collective_nonclinical_reliance |
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
| metadata_sha256 | dc9855ea282d08ee412b4262e359a59b4e6dfdbc77393afbdd7b5938c30f5654 |
| basic_plan_sha256 | 1715f7984a787f8b0edf88fadc30dca2991a19b00faba6a9aa40dcbbc3aeb69a |
| source_cache | {"content_sha256": "300266ae4e94f84cd163be2120dc8249282f0c6ffd5599186e6d2e96e7fa4018", "file_sha256": "c79168d48d84d210209d9acba13554b540819ca9c892ff78dd49e74cf69ec55b", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "8bb7c4416e6318b253f1bbb5f0f90aad912d1b3a4fe67ad0ae2f7df8b7bae1a3", "cohort_person_ids_sha256": "396f9a4710d5d75770040275fa0e2b63c92ef3b3b4ebbe20694e0476a4e12241", "config_hash": "95fee85977cd6e00d6481da4f7d2302d44a0fe0bcb9a1ac7d8523a15029e1e45", "encoder_batch_size": 2, "master_config_hash": "2f6c7be047082aadf347763b282a8a1c2c979f94f0391316381ebbf0dd7d5775", "model_config_hash": "a42dc3fbc1bee9562475396635898921f93d01b316b109983602bcffc3648674", "n_participants": 351, "scaler_config_hash": "f041d13116980d3c528715f30762737108271ac00c58fb69672f3bd2c9251844", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "cvd_strict_binary"}, "path": "/data/pzhao3/ot2_output/model_full_assets_domain_ablation_v1/cvd_strict_binary/seed42/val_domain/source_token_cache_val.pt", "raw_dataset_passes_this_run": 0, "reused": true} |
| output_hashes | {"abcd_bootstrap_intervals.csv": "513b76a53ecd03181931835b0cebf6402b748cc26ea75601b97121af9b0bb237", "abcd_cohort_counts.csv": "e86c82ba4078fef7369759f0ae3b8edac411afca21d43c560079060aef465396", "abcd_conditions.json": "36879d41afc92f4525c9cbd8dd3f810b7a9bf4777967fefcbd816fd931a7c962", "abcd_metrics.csv": "8e1019378b18bd95b7e9f09024ce7531eabe7f2d97cbe1e762c488dc689ebc1c", "abcd_paired_predictions.csv": "b749ab4c92fd6a1937d0e6a2d9563a1eb37c6d245560bc781a772e42690a42b2", "abcd_plan.json": "655f12fe793d444a633c29303a9a8dee15c01115d48f5650e1814a8ed9a33f4d", "abcd_summary.csv": "b0ee6806fc12929b9d16ea578c6e9f219f6fa035742a1a8e370ee2257c53186d"} |
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
