# Domain ablation summary

**Overall status: COMPLETE**

Frozen seed-42 models; test split (exploratory); joint removal of model branches and prespecified Clinical information. This file contains aggregate results and field names, without participant-level predictions or identifiers.

Runner state: complete; phase: complete; updated UTC: 2026-09-10T03:52:25.054403+00:00.

| Task | Result status |
|---|---|
| MoCA | Verified |
| CVD broad | Verified |
| CVD strict | Verified |

## AUROC degradation across tasks

Positive = AUROC decreased after deletion; negative = AUROC increased. Values are absolute AUROC differences, followed by pointwise 95% paired bootstrap intervals. 0.0100 equals one AUROC percentage point.

| Deletion | MoCA | CVD broad | CVD strict |
|---|---|---|---|
| All Clinical data | +0.0794 [+0.0317, +0.1292] | +0.1370 [+0.0536, +0.2190] | +0.1055 [+0.0514, +0.1650] |
| ECG + cardiovascular Clinical | -0.0033 [-0.0134, +0.0060] | +0.0219 [+0.0021, +0.0452] | +0.0282 [+0.0066, +0.0511] |
| Environmental sensor | +0.0018 [-0.0013, +0.0051] | -0.0010 [-0.0046, +0.0025] | +0.0022 [-0.0006, +0.0053] |
| All retinal + ocular Clinical | +0.0185 [-0.0063, +0.0458] | -0.0089 [-0.0333, +0.0157] | -0.0021 [-0.0129, +0.0087] |
| Activity monitor + activity/sleep Clinical | +0.0036 [-0.0007, +0.0080] | +0.0059 [-0.0040, +0.0164] | +0.0011 [-0.0037, +0.0060] |
| CGM + metabolic Clinical | -0.0008 [-0.0113, +0.0100] | +0.0194 [-0.0006, +0.0380] | -0.0047 [-0.0140, +0.0047] |
| All wearable + associated Clinical | +0.0016 [-0.0100, +0.0126] | +0.0266 [+0.0040, +0.0483] | -0.0035 [-0.0137, +0.0074] |

## Descriptive findings

- MoCA: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.0794 [+0.0317, +0.1292]. Conditions with a pointwise interval entirely above zero: All Clinical data. This is exploratory test evidence.
- CVD broad: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.1370 [+0.0536, +0.2190]. Conditions with a pointwise interval entirely above zero: All Clinical data, ECG + cardiovascular Clinical, All wearable + associated Clinical. This is exploratory test evidence.
- CVD strict: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.1055 [+0.0514, +0.1650]. Conditions with a pointwise interval entirely above zero: All Clinical data, ECG + cardiovascular Clinical. This is exploratory test evidence.

## Full-input baseline

| Task | N | Prevalence | AUROC | AUPRC | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoCA | 349 | 0.4241 | 0.7386 | 0.6533 | 0.7027 | 0.6418 | 0.2117 |
| CVD broad | 348 | 0.8075 | 0.7341 | 0.9116 | 0.6192 | 0.6567 | 0.1507 |
| CVD strict | 342 | 0.3363 | 0.7570 | 0.6473 | 0.6783 | 0.7533 | 0.2157 |

## Interpretation and scope

- AUROC/AUPRC/recall/balanced-accuracy degradation = full minus deleted; Brier/log-loss degradation = deleted minus full. Positive always means worse.
- Intervals crossing zero indicate unresolved direction, not equivalence or noninferiority. Intervals are not adjusted for multiple comparisons and exclude training-seed uncertainty.
- Full and deleted predictions use identical people within each reported cohort and the original full-model validation threshold. Threshold metrics can change substantially under input removal.
- These are fitted-model joint information-removal effects; they are neither causal effects nor exclusive device contributions. Single-group effects need not add up to a joint effect.
- Missing-input performance does not demonstrate reconstruction or the performance achievable by retraining.
- The baseline is the unmodified frozen teacher; this run does not retrain or reconstruct inputs.
- Basic Clinical (23 encoded columns: age, education, marital status, language) remains except in All Clinical data.
- Retinal acquisition 85 + ocular function/form 27 + vision questionnaire 24 columns are supplemented with task-available ocular history.
- Activity/sleep additionally removes cm_slp and the entire CESD-10 total cestl to avoid retaining the ces7 component through a composite. The total also contains non-sleep information; this is a conservative broader deletion. Other CESD items remain.
- mhoccur_rvo belongs to both ocular and cardiovascular groups; joint removals use a deduplicated union.
- No additional Clinical fields are assigned to environmental sensors. Housing and neighborhood questionnaires remain.
- BMI/anthropometry, pulse (acquisition origin unresolved), CRP, general chemistry/CBC, family history, diet, other diabetes questionnaires (including PAID), psychology and cognition remain unless All Clinical data is removed. These are specified subsets, not removal of every biological correlate.
- Metabolic removal uses the four named laboratory measures, diabetes/prediabetes/elevated-A1c history, diagnosis age, and four cmtrt treatment/management items. Lipids are assigned to the cardiovascular group in this version.
- eos_ds is unchanged in the baseline and domain-specific conditions; it disappears only when the complete Clinical branch is disabled. The earlier EOS/time-window question is not changed by this experiment.
- Sources already excluded by a task's trained schema are recorded as absent, never reintroduced.

## Shared code snapshot

| File | SHA256 |
|---|---|
| analyze_domain_ablation_v1.py | 20e5da24a44903959d81d57d0ee9f828cc03ee6e72fdbfe7fd5f2cf877403c42 |
| analyze_modality_importance_v2.py | b20ebe3b7b08f7597b8b03a71d3cd257cf735b8ceaee4a38de9d4e809c1f32e8 |
| clinical_basic_information_v1.py | 77b6feef189b14b498716e0a2f75f7af330c5966e63d90a53258a28c0a89a259 |
| clinical_domain_groups_v1.py | b3a1a293ff315bdc6bcab45d4defc16affb629c6c3ab45c87829ea3c449a7862 |
| clinical_modality_overlap_v1.py | ecc616d62aa1e4aeac1ef3bde4f3c256b4543f0d87ea40eb765ea598c543acfd |
| clinical_overlap_inference_v1.py | 2360ec9a2500799c0727407902ab70b8c705f7b2bd2b639dc676fe73b155ef96 |
| domain_ablation_engine_v1.py | 1e94ec5d4374aa0831bcfe7300cdf1a9e49bbef38ab8da7952e2407b6e6114cd |
| fit_train_scalers_v2.py | 1c8138a3a66ff6ad922f4e907df95c0b0e5c67487042963a37ed9fce52102419 |
| preprocessing_v2.py | 0219f9a8e9c291f2b9c5091205e4476db73c17c6a232fee1bc789491e7b44e14 |
| run_domain_ablation_v1.py | cff0e4dcb6434e62fa5478e5f646647e872b5b25b21d2dac75094fb373428a0f |
| summarize_domain_ablation_v1.py | d1b98fe3c052ab1d9a5c932a614e7d37ed6f3946a534aeaf090b0262cb27230b |
| task2_v2_config.py | 458848c8032a9d47fc54ca112043b2929b81e126cd80601c94ee20514e21ac06 |
| task2_v2_dataset.py | 24afc6b37058021b4f20bc17a50f16a5fe72e543768ac4365a3dc214fcb9fcef |
| task2_v2_engine.py | 63f8014d3d4c9fd7d7c2469d0496bf17245bfa9b8110503b39572d80a6286ac4 |
| task2_v2_model.py | 7f9a22f149306ef16412311e9f8d6275804fe19b66f03e2aedd031dd102c5463 |
| train_task2_v2.py | 34cf798448a025cc8d49a69a4df4eec549d798ca6a004e1f7e5435a4b08444a0 |

## MoCA: detailed results

N = 349; fixed threshold = 0.45519860; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7386 | 0.6592 | +0.0794 [+0.0317, +0.1292] | ok |
| All Clinical data | auprc | 0.6533 | 0.5769 | +0.0764 [+0.0114, +0.1451] | ok |
| All Clinical data | balanced_accuracy | 0.6722 | 0.6129 | +0.0593 [+0.0092, +0.1134] | ok |
| All Clinical data | sensitivity | 0.7027 | 0.7432 | -0.0405 [-0.1216, +0.0473] | ok |
| All Clinical data | specificity | 0.6418 | 0.4826 | +0.1592 [+0.0945, +0.2239] | ok |
| All Clinical data | log_loss | 0.6126 | 0.6636 | +0.0510 [+0.0095, +0.0952] | ok |
| All Clinical data | brier | 0.2117 | 0.2350 | +0.0234 [+0.0055, +0.0423] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7386 | 0.7419 | -0.0033 [-0.0134, +0.0060] | ok |
| ECG + cardiovascular Clinical | auprc | 0.6533 | 0.6473 | +0.0060 [-0.0094, +0.0232] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.6722 | 0.6851 | -0.0128 [-0.0322, +0.0064] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.7027 | 0.7432 | -0.0405 [-0.0743, -0.0135] | ok |
| ECG + cardiovascular Clinical | specificity | 0.6418 | 0.6269 | +0.0149 [-0.0050, +0.0398] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.6126 | 0.6132 | +0.0006 [-0.0097, +0.0104] | ok |
| ECG + cardiovascular Clinical | brier | 0.2117 | 0.2111 | -0.0006 [-0.0049, +0.0036] | ok |
| Environmental sensor | auroc | 0.7386 | 0.7368 | +0.0018 [-0.0013, +0.0051] | ok |
| Environmental sensor | auprc | 0.6533 | 0.6540 | -0.0007 [-0.0079, +0.0088] | ok |
| Environmental sensor | balanced_accuracy | 0.6722 | 0.6747 | -0.0025 [-0.0135, +0.0076] | ok |
| Environmental sensor | sensitivity | 0.7027 | 0.7027 | +0.0000 [-0.0203, +0.0203] | ok |
| Environmental sensor | specificity | 0.6418 | 0.6468 | -0.0050 [-0.0149, +0.0000] | ok |
| Environmental sensor | log_loss | 0.6126 | 0.6142 | +0.0017 [-0.0005, +0.0038] | ok |
| Environmental sensor | brier | 0.2117 | 0.2123 | +0.0006 [-0.0003, +0.0015] | ok |
| All retinal + ocular Clinical | auroc | 0.7386 | 0.7201 | +0.0185 [-0.0063, +0.0458] | ok |
| All retinal + ocular Clinical | auprc | 0.6533 | 0.6574 | -0.0041 [-0.0385, +0.0384] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.6722 | 0.6520 | +0.0203 [-0.0142, +0.0557] | ok |
| All retinal + ocular Clinical | sensitivity | 0.7027 | 0.6622 | +0.0405 [-0.0135, +0.1014] | ok |
| All retinal + ocular Clinical | specificity | 0.6418 | 0.6418 | +0.0000 [-0.0448, +0.0448] | ok |
| All retinal + ocular Clinical | log_loss | 0.6126 | 0.6175 | +0.0050 [-0.0207, +0.0310] | ok |
| All retinal + ocular Clinical | brier | 0.2117 | 0.2148 | +0.0031 [-0.0081, +0.0145] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7386 | 0.7350 | +0.0036 [-0.0007, +0.0080] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.6533 | 0.6510 | +0.0023 [-0.0038, +0.0117] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.6722 | 0.6660 | +0.0062 [-0.0180, +0.0306] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.7027 | 0.7500 | -0.0473 [-0.0811, -0.0135] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.6418 | 0.5821 | +0.0597 [+0.0299, +0.0945] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.6126 | 0.6241 | +0.0115 [+0.0059, +0.0173] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.2117 | 0.2165 | +0.0048 [+0.0023, +0.0074] | ok |
| CGM + metabolic Clinical | auroc | 0.7386 | 0.7394 | -0.0008 [-0.0113, +0.0100] | ok |
| CGM + metabolic Clinical | auprc | 0.6533 | 0.6550 | -0.0017 [-0.0179, +0.0154] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.6722 | 0.6619 | +0.0103 [-0.0114, +0.0322] | ok |
| CGM + metabolic Clinical | sensitivity | 0.7027 | 0.6622 | +0.0405 [+0.0135, +0.0743] | ok |
| CGM + metabolic Clinical | specificity | 0.6418 | 0.6617 | -0.0199 [-0.0498, +0.0100] | ok |
| CGM + metabolic Clinical | log_loss | 0.6126 | 0.6092 | -0.0034 [-0.0148, +0.0075] | ok |
| CGM + metabolic Clinical | brier | 0.2117 | 0.2106 | -0.0010 [-0.0061, +0.0038] | ok |
| All wearable + associated Clinical | auroc | 0.7386 | 0.7370 | +0.0016 [-0.0100, +0.0126] | ok |
| All wearable + associated Clinical | auprc | 0.6533 | 0.6544 | -0.0011 [-0.0177, +0.0195] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.6722 | 0.6699 | +0.0023 [-0.0205, +0.0240] | ok |
| All wearable + associated Clinical | sensitivity | 0.7027 | 0.7230 | -0.0203 [-0.0541, +0.0135] | ok |
| All wearable + associated Clinical | specificity | 0.6418 | 0.6169 | +0.0249 [+0.0000, +0.0498] | ok |
| All wearable + associated Clinical | log_loss | 0.6126 | 0.6174 | +0.0048 [-0.0071, +0.0164] | ok |
| All wearable + associated Clinical | brier | 0.2117 | 0.2136 | +0.0020 [-0.0033, +0.0068] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.6592 | +0.0794 [+0.0317, +0.1292] |
| All Clinical data | native_present_only | 349 | 148 / 201 | 0 | 0.7386 | 0.6592 | +0.0794 [+0.0311, +0.1272] |
| All Clinical data | native_present_with_remaining_input | 349 | 148 / 201 | 0 | 0.7386 | 0.6592 | +0.0794 [+0.0311, +0.1299] |
| All Clinical data | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.6592 | +0.0794 [+0.0287, +0.1279] |
| ECG + cardiovascular Clinical | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7419 | -0.0033 [-0.0134, +0.0060] |
| ECG + cardiovascular Clinical | native_present_only | 346 | 147 / 199 | 0 | 0.7366 | 0.7398 | -0.0031 [-0.0131, +0.0062] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 346 | 147 / 199 | 0 | 0.7366 | 0.7398 | -0.0031 [-0.0128, +0.0068] |
| ECG + cardiovascular Clinical | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7419 | -0.0033 [-0.0127, +0.0060] |
| Environmental sensor | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7368 | +0.0018 [-0.0013, +0.0051] |
| Environmental sensor | native_present_only | 348 | 147 / 201 | 0 | 0.7374 | 0.7355 | +0.0018 [-0.0012, +0.0048] |
| Environmental sensor | native_present_with_remaining_input | 348 | 147 / 201 | 0 | 0.7374 | 0.7355 | +0.0018 [-0.0013, +0.0050] |
| Environmental sensor | any_target_information_present | 348 | 147 / 201 | 0 | 0.7374 | 0.7355 | +0.0018 [-0.0012, +0.0050] |
| All retinal + ocular Clinical | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7201 | +0.0185 [-0.0063, +0.0458] |
| All retinal + ocular Clinical | any_target_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7201 | +0.0185 [-0.0081, +0.0441] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 349 | 148 / 201 | 0 | 0.7386 | 0.7201 | +0.0185 [-0.0071, +0.0460] |
| All retinal + ocular Clinical | all_targets_present | 288 | 113 / 175 | 0 | 0.7607 | 0.7420 | +0.0186 [-0.0076, +0.0474] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 288 | 113 / 175 | 0 | 0.7607 | 0.7420 | +0.0186 [-0.0087, +0.0475] |
| All retinal + ocular Clinical | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7201 | +0.0185 [-0.0068, +0.0432] |
| Activity monitor + activity/sleep Clinical | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7350 | +0.0036 [-0.0007, +0.0080] |
| Activity monitor + activity/sleep Clinical | native_present_only | 326 | 136 / 190 | 0 | 0.7284 | 0.7247 | +0.0037 [-0.0009, +0.0084] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 326 | 136 / 190 | 0 | 0.7284 | 0.7247 | +0.0037 [-0.0010, +0.0083] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7350 | +0.0036 [-0.0006, +0.0081] |
| CGM + metabolic Clinical | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7394 | -0.0008 [-0.0113, +0.0100] |
| CGM + metabolic Clinical | native_present_only | 344 | 145 / 199 | 0 | 0.7386 | 0.7398 | -0.0012 [-0.0117, +0.0094] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 344 | 145 / 199 | 0 | 0.7386 | 0.7398 | -0.0012 [-0.0117, +0.0098] |
| CGM + metabolic Clinical | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7394 | -0.0008 [-0.0114, +0.0106] |
| All wearable + associated Clinical | all_participants | 349 | 148 / 201 | 0 | 0.7386 | 0.7370 | +0.0016 [-0.0100, +0.0126] |
| All wearable + associated Clinical | any_target_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7370 | +0.0016 [-0.0098, +0.0126] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 349 | 148 / 201 | 0 | 0.7386 | 0.7370 | +0.0016 [-0.0102, +0.0130] |
| All wearable + associated Clinical | all_targets_present | 321 | 133 / 188 | 0 | 0.7282 | 0.7295 | -0.0013 [-0.0137, +0.0104] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 321 | 133 / 188 | 0 | 0.7282 | 0.7295 | -0.0013 [-0.0132, +0.0103] |
| All wearable + associated Clinical | any_target_information_present | 349 | 148 / 201 | 0 | 0.7386 | 0.7370 | +0.0016 [-0.0097, +0.0129] |

### Exact Clinical source scope

Counts below are encoded input columns, not independent examinations. Every encoding of a listed source is removed. The full ordered encoded-column list and indices are preserved in domain_plan.json and domain_conditions.json.

**All Clinical data** — 955 columns from 381 sources; disabled branches: clinical_data.

Removed source keys: `age`, `bmi_vsorres`, `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `brthyy`, `cage`, `ces1`, `ces10`, `ces2`, `ces3`, `ces4`, `ces5`, `ces6`, `ces7`, `ces8`, `ces9`, `cestl`, `cl_maristat`, `cm_act`, `cm_ant`, `cm_asp`, `cm_dcg`, `cm_ibp`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `diet1`, `diet2`, `diet3`, `diet4`, `diet5`, `diet6`, `diet7`, `diet8`, `diet9`, `dietscore`, `dmlact`, `dmledu`, `dmlfeet`, `dmlfrveg`, `dmlgrain`, `dmlhex`, `dmlmd`, `dmlpor`, `dmlsugar`, `dmlvex`, `dri1`, `dri2`, `eos_ds`, `fh_dm2pt`, `fh_dm2ptsp`, `fh_dm2sb`, `fh_dm2sbsp`, `food_insecurity_change_diet_frequency`, `height_vsorres`, `hip_vsorres`, `import_a_g_ratio`, `import_albumin`, `import_alkaline_phosphatase`, `import_alt_got`, `import_ast_got`, `import_bilirubin_total`, `import_bun`, `import_buncreatinineratio`, `import_c_peptide`, `import_calcium`, `import_carbon_dioxide_total`, `import_chloride`, `import_creatinine`, `import_crp_hs`, `import_globulin_total`, `import_glucose`, `import_hba1c`, `import_hdl_cholesterol`, `import_insulin`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_potassium`, `import_protein_total`, `import_sodium`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`, `import_urine_albumin`, `import_urine_creatinine`, `lbscat_a1c`, `lbscat_hct`, `lbscat_mch`, `lbscat_mchc`, `lbscat_mcv`, `lbscat_plt`, `lbscat_rbc`, `lbscat_rdw`, `lbscat_wbc`, `mh_a1c`, `mh_dm_age`, `mhoccur_amd`, `mhoccur_ca`, `mhoccur_circ`, `mhoccur_clsh`, `mhoccur_cns`, `mhoccur_crt`, `mhoccur_cvdot`, `mhoccur_ded`, `mhoccur_ear`, `mhoccur_fall`, `mhoccur_fallot`, `mhoccur_gi`, `mhoccur_glc`, `mhoccur_hbp`, `mhoccur_lbp`, `mhoccur_mi`, `mhoccur_ms`, `mhoccur_oa`, `mhoccur_obs`, `mhoccur_pd`, `mhoccur_pdr`, `mhoccur_plm`, `mhoccur_ra`, `mhoccur_rnl`, `mhoccur_rvo`, `mhoccur_strk`, `mhoccur_ua`, `mhoccur_yn`, `mhterm_dm2`, `mhterm_predm`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `msslf1`, `msslf10`, `msslf2`, `msslf3`, `msslf4`, `msslf5`, `msslf6`, `msslf7`, `msslf8`, `msslf9`, `msslffl`, `msslftt`, `mssrf1`, `mssrf10`, `mssrf2`, `mssrf3`, `mssrf4`, `mssrf5`, `mssrf6`, `mssrf7`, `mssrf8`, `mssrf9`, `mssrffl`, `mssrftt`, `paate`, `paid_cml`, `paid_dpr`, `paid_eng`, `paid_scrd`, `paid_wr`, `paidscore`, `pdrx9`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `pulse_vsorres`, `pulse_vsorres_2`, `pxahc1`, `pxahc10`, `pxahc2`, `pxahc3`, `pxahc4`, `pxahc5`, `pxahc7`, `pxahc8`, `pxahc9`, `pxdhc1`, `pxdhc2`, `pxdhc3`, `pxdhc4`, `pxdhc5`, `pxdhc6`, `pxdhc7`, `pxed1`, `pxep1`, `pxep2`, `pxep3`, `pxfi1`, `pxfi2`, `pxfi3`, `pxfi4`, `pxfi5`, `pxhi1`, `pxhi2`, `pxhic1`, `pxhic2`, `pxhic3`, `pxhic4`, `pxhic5`, `pxhic7`, `pxhic8`, `pxji1`, `pxji2`, `pxne1`, `pxne10`, `pxne11`, `pxne12`, `pxne13`, `pxne14`, `pxne15`, `pxne16`, `pxne17`, `pxne2`, `pxne3`, `pxne4`, `pxne5`, `pxne6`, `pxne7`, `pxne8`, `pxne9`, `pxpa1`, `pxpa2`, `pxpa3`, `pxpa4`, `pxrd1`, `pxrd10`, `pxrd11`, `pxrd12`, `pxrd13`, `pxrd14`, `pxrd15`, `pxrd16`, `pxrd17`, `pxrd18`, `pxrd19`, `pxrd2`, `pxrd20`, `pxrd21`, `pxrd22`, `pxrd23`, `pxrd24`, `pxrd25`, `pxrd26`, `pxrd27`, `pxrd28`, `pxrd29`, `pxrd3`, `pxrd30`, `pxrd31`, `pxrd32`, `pxrd33`, `pxrd34`, `pxrd35`, `pxrd36`, `pxrd37`, `pxrd38`, `pxrd39`, `pxrd4`, `pxrd40`, `pxrd41`, `pxrd42`, `pxrd43`, `pxrd44`, `pxrd45`, `pxrd46`, `pxrd47`, `pxrd48`, `pxrd49`, `pxrd5`, `pxrd50`, `pxrd51`, `pxrd52`, `pxrd53`, `pxrd54`, `pxrd6`, `pxrd7`, `pxrd8`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `sualcage`, `sualcdrv`, `sualckncf`, `subrdosfr`, `subrkncf`, `sulqdosfr`, `sulqkncf`, `sumrjcdur`, `sumrjceddur`, `sumrjfr`, `sumrjkncf`, `sumrjstage`, `sumrjtrt`, `susmkcdur`, `susmkdosfr`, `susmkncf`, `susmkstaage`, `susmkstoage`, `suvpcdur`, `suvpdosfr`, `suvpkncf1`, `suvpstage`, `suvptoage`, `suvptoyr`, `suwndosfr`, `suwnkncf`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`, `waist_vsorres`, `weight_vsorres`, `whr_vsorres`, `years_of_education`.

**ECG + cardiovascular Clinical** — 34 columns from 18 sources; disabled branches: cardiac_ecg.

Removed source keys: `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `import_hdl_cholesterol`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`, `mhoccur_circ`, `mhoccur_clsh`, `mhoccur_cvdot`, `mhoccur_hbp`, `mhoccur_lbp`, `mhoccur_mi`, `mhoccur_rvo`, `mhoccur_strk`.

**Environmental sensor** — 0 columns from 0 sources; disabled branches: environment.

Removed source keys: None.

**All retinal + ocular Clinical** — 154 columns from 83 sources; disabled branches: retinal_flio, retinal_oct, retinal_octa, retinal_photography.

Removed source keys: `mhoccur_amd`, `mhoccur_crt`, `mhoccur_ded`, `mhoccur_glc`, `mhoccur_pdr`, `mhoccur_rvo`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`.

**Activity monitor + activity/sleep Clinical** — 19 columns from 6 sources; disabled branches: wearable_activity_monitor.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `dmlact`, `dmlhex`, `dmlvex`.

**CGM + metabolic Clinical** — 22 columns from 12 sources; disabled branches: wearable_blood_glucose.

Removed source keys: `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

**All wearable + associated Clinical** — 41 columns from 18 sources; disabled branches: wearable_activity_monitor, wearable_blood_glucose.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `dmlact`, `dmlhex`, `dmlvex`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

### Provenance and verification

| Item | Value |
|---|---|
| checkpoint | /data/pzhao3/ot2_output/model_full_assets_v1/models/moca_binary/seed42/best_model.pt |
| checkpoint SHA256 | bc720e18565f8c4ff7699d27328cb27c1098266419e53aaf5962310f5339e154 |
| config hash | d83c64cd36ed0ff2ec0130500d033873fd65bc97ac2d1cf187decb54f972299c |
| config file SHA256 | d3385b42d5e5de51d0d9d2264cca4913f640e189622cf4c69696763cdfb6719f |
| Clinical schema SHA256 | da677fb50c38589626122374f712131d48ef44f1c1785211b0b066cfb67e7d26 |
| Clinical catalog SHA256 | 8e2b9ef2c72570ce6c32559e40f9dffbfe2abc52e31af82035ab63b6980fd2b9 |
| deletion plan SHA256 | 635273f839e4f45e4e3b9c96e7f909b0b62361ace61d9d6f9ac501984820ea15 |
| metadata SHA256 | 568c789e371b8ce1fbb410290ac9b7c76f1b34187020260b5d7926b39593ab50 |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | 1eca5d1494bfd6e9e5fb9f34376e762f156914d21f6e75be2e56d4b8d87d8705 |
| evaluation cohort hash | b14ecc6d927f2f3eb0f917154b68e0bf58ed5c77bb2610c047f7c5e00c50c14a |
| master config hash | 13683a6afc4927f4923147109bfc20e6f115d4154a08d8014deb6d89f133e820 |
| scaler config hash | a8d107610d6700cc000d74ce3acdd87aade399a4b3ea7de404bbba992760d827 |
| resolved model config hash | 981300bd5b48c18b2b34142beac7e38f3c28339c0c8096708c29deb9124d42b2 |
| result files | All six file hashes verified before report generation |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
| paired_participant_alignment | True |


## CVD broad: detailed results

N = 348; fixed threshold = 0.91117971; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7341 | 0.5970 | +0.1370 [+0.0536, +0.2190] | ok |
| All Clinical data | auprc | 0.9116 | 0.8647 | +0.0470 [+0.0095, +0.0845] | ok |
| All Clinical data | balanced_accuracy | 0.6380 | 0.5682 | +0.0697 [-0.0078, +0.1533] | ok |
| All Clinical data | sensitivity | 0.6192 | 0.4947 | +0.1246 [+0.0569, +0.1957] | ok |
| All Clinical data | specificity | 0.6567 | 0.6418 | +0.0149 [-0.1343, +0.1642] | ok |
| All Clinical data | log_loss | 0.4808 | 0.5525 | +0.0717 [-0.0035, +0.1441] | ok |
| All Clinical data | brier | 0.1507 | 0.1777 | +0.0269 [+0.0003, +0.0530] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7341 | 0.7121 | +0.0219 [+0.0021, +0.0452] | ok |
| ECG + cardiovascular Clinical | auprc | 0.9116 | 0.8977 | +0.0140 [+0.0011, +0.0282] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.6380 | 0.6555 | -0.0175 [-0.0548, +0.0227] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.6192 | 0.7438 | -0.1246 [-0.1638, -0.0819] | ok |
| ECG + cardiovascular Clinical | specificity | 0.6567 | 0.5672 | +0.0896 [+0.0299, +0.1642] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.4808 | 0.4763 | -0.0045 [-0.0281, +0.0192] | ok |
| ECG + cardiovascular Clinical | brier | 0.1507 | 0.1456 | -0.0052 [-0.0151, +0.0048] | ok |
| Environmental sensor | auroc | 0.7341 | 0.7350 | -0.0010 [-0.0046, +0.0025] | ok |
| Environmental sensor | auprc | 0.9116 | 0.9121 | -0.0005 [-0.0031, +0.0022] | ok |
| Environmental sensor | balanced_accuracy | 0.6380 | 0.6394 | -0.0014 [-0.0142, +0.0174] | ok |
| Environmental sensor | sensitivity | 0.6192 | 0.6370 | -0.0178 [-0.0356, -0.0036] | ok |
| Environmental sensor | specificity | 0.6567 | 0.6418 | +0.0149 [+0.0000, +0.0448] | ok |
| Environmental sensor | log_loss | 0.4808 | 0.4722 | -0.0086 [-0.0136, -0.0038] | ok |
| Environmental sensor | brier | 0.1507 | 0.1472 | -0.0035 [-0.0056, -0.0015] | ok |
| All retinal + ocular Clinical | auroc | 0.7341 | 0.7429 | -0.0089 [-0.0333, +0.0157] | ok |
| All retinal + ocular Clinical | auprc | 0.9116 | 0.9158 | -0.0041 [-0.0176, +0.0098] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.6380 | 0.6452 | -0.0073 [-0.0603, +0.0539] | ok |
| All retinal + ocular Clinical | sensitivity | 0.6192 | 0.8577 | -0.2384 [-0.2883, -0.1886] | ok |
| All retinal + ocular Clinical | specificity | 0.6567 | 0.4328 | +0.2239 [+0.1343, +0.3284] | ok |
| All retinal + ocular Clinical | log_loss | 0.4808 | 0.4896 | +0.0089 [-0.0304, +0.0479] | ok |
| All retinal + ocular Clinical | brier | 0.1507 | 0.1476 | -0.0032 [-0.0184, +0.0122] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7341 | 0.7281 | +0.0059 [-0.0040, +0.0164] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.9116 | 0.9062 | +0.0055 [-0.0015, +0.0132] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.6380 | 0.6308 | +0.0071 [-0.0018, +0.0178] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.6192 | 0.6050 | +0.0142 [-0.0036, +0.0356] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.6567 | 0.6567 | +0.0000 [+0.0000, +0.0000] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.4808 | 0.4693 | -0.0115 [-0.0205, -0.0028] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.1507 | 0.1466 | -0.0041 [-0.0076, -0.0010] | ok |
| CGM + metabolic Clinical | auroc | 0.7341 | 0.7146 | +0.0194 [-0.0006, +0.0380] | ok |
| CGM + metabolic Clinical | auprc | 0.9116 | 0.9081 | +0.0035 [-0.0050, +0.0117] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.6380 | 0.6291 | +0.0089 [-0.0267, +0.0412] | ok |
| CGM + metabolic Clinical | sensitivity | 0.6192 | 0.6014 | +0.0178 [-0.0178, +0.0534] | ok |
| CGM + metabolic Clinical | specificity | 0.6567 | 0.6567 | +0.0000 [-0.0597, +0.0597] | ok |
| CGM + metabolic Clinical | log_loss | 0.4808 | 0.5145 | +0.0338 [+0.0097, +0.0564] | ok |
| CGM + metabolic Clinical | brier | 0.1507 | 0.1643 | +0.0136 [+0.0045, +0.0224] | ok |
| All wearable + associated Clinical | auroc | 0.7341 | 0.7075 | +0.0266 [+0.0040, +0.0483] | ok |
| All wearable + associated Clinical | auprc | 0.9116 | 0.9015 | +0.0102 [-0.0004, +0.0205] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.6380 | 0.6351 | +0.0029 [-0.0377, +0.0420] | ok |
| All wearable + associated Clinical | sensitivity | 0.6192 | 0.5836 | +0.0356 [+0.0000, +0.0747] | ok |
| All wearable + associated Clinical | specificity | 0.6567 | 0.6866 | -0.0299 [-0.1045, +0.0302] | ok |
| All wearable + associated Clinical | log_loss | 0.4808 | 0.5000 | +0.0192 [-0.0040, +0.0429] | ok |
| All wearable + associated Clinical | brier | 0.1507 | 0.1587 | +0.0079 [-0.0006, +0.0167] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 348 | 281 / 67 | 2 | 0.7341 | 0.5970 | +0.1370 [+0.0536, +0.2190] |
| All Clinical data | native_present_only | 348 | 281 / 67 | 2 | 0.7341 | 0.5970 | +0.1370 [+0.0499, +0.2245] |
| All Clinical data | native_present_with_remaining_input | 346 | 280 / 66 | 0 | 0.7358 | 0.5996 | +0.1362 [+0.0468, +0.2202] |
| All Clinical data | any_target_information_present | 348 | 281 / 67 | 2 | 0.7341 | 0.5970 | +0.1370 [+0.0521, +0.2285] |
| ECG + cardiovascular Clinical | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7121 | +0.0219 [+0.0021, +0.0452] |
| ECG + cardiovascular Clinical | native_present_only | 343 | 277 / 66 | 0 | 0.7360 | 0.7134 | +0.0226 [+0.0016, +0.0451] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 343 | 277 / 66 | 0 | 0.7360 | 0.7134 | +0.0226 [+0.0028, +0.0444] |
| ECG + cardiovascular Clinical | any_target_information_present | 348 | 281 / 67 | 0 | 0.7341 | 0.7121 | +0.0219 [+0.0011, +0.0434] |
| Environmental sensor | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7350 | -0.0010 [-0.0046, +0.0025] |
| Environmental sensor | native_present_only | 345 | 279 / 66 | 0 | 0.7361 | 0.7370 | -0.0009 [-0.0046, +0.0024] |
| Environmental sensor | native_present_with_remaining_input | 345 | 279 / 66 | 0 | 0.7361 | 0.7370 | -0.0009 [-0.0046, +0.0029] |
| Environmental sensor | any_target_information_present | 345 | 279 / 66 | 0 | 0.7361 | 0.7370 | -0.0009 [-0.0046, +0.0028] |
| All retinal + ocular Clinical | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7429 | -0.0089 [-0.0333, +0.0157] |
| All retinal + ocular Clinical | any_target_present | 346 | 280 / 66 | 0 | 0.7358 | 0.7429 | -0.0071 [-0.0307, +0.0187] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 346 | 280 / 66 | 0 | 0.7358 | 0.7429 | -0.0071 [-0.0309, +0.0176] |
| All retinal + ocular Clinical | all_targets_present | 288 | 231 / 57 | 0 | 0.7496 | 0.7532 | -0.0036 [-0.0287, +0.0230] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 288 | 231 / 57 | 0 | 0.7496 | 0.7532 | -0.0036 [-0.0294, +0.0234] |
| All retinal + ocular Clinical | any_target_information_present | 348 | 281 / 67 | 0 | 0.7341 | 0.7429 | -0.0089 [-0.0335, +0.0165] |
| Activity monitor + activity/sleep Clinical | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7281 | +0.0059 [-0.0040, +0.0164] |
| Activity monitor + activity/sleep Clinical | native_present_only | 323 | 262 / 61 | 0 | 0.7237 | 0.7181 | +0.0055 [-0.0045, +0.0162] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 323 | 262 / 61 | 0 | 0.7237 | 0.7181 | +0.0055 [-0.0048, +0.0168] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 348 | 281 / 67 | 0 | 0.7341 | 0.7281 | +0.0059 [-0.0037, +0.0165] |
| CGM + metabolic Clinical | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7146 | +0.0194 [-0.0006, +0.0380] |
| CGM + metabolic Clinical | native_present_only | 342 | 276 / 66 | 0 | 0.7342 | 0.7165 | +0.0176 [-0.0023, +0.0377] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 342 | 276 / 66 | 0 | 0.7342 | 0.7165 | +0.0176 [-0.0019, +0.0375] |
| CGM + metabolic Clinical | any_target_information_present | 348 | 281 / 67 | 0 | 0.7341 | 0.7146 | +0.0194 [-0.0008, +0.0397] |
| All wearable + associated Clinical | all_participants | 348 | 281 / 67 | 0 | 0.7341 | 0.7075 | +0.0266 [+0.0040, +0.0483] |
| All wearable + associated Clinical | any_target_present | 346 | 280 / 66 | 0 | 0.7358 | 0.7110 | +0.0248 [+0.0025, +0.0479] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 346 | 280 / 66 | 0 | 0.7358 | 0.7110 | +0.0248 [+0.0033, +0.0478] |
| All wearable + associated Clinical | all_targets_present | 319 | 258 / 61 | 0 | 0.7219 | 0.6990 | +0.0229 [-0.0008, +0.0452] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 319 | 258 / 61 | 0 | 0.7219 | 0.6990 | +0.0229 [-0.0001, +0.0449] |
| All wearable + associated Clinical | any_target_information_present | 348 | 281 / 67 | 0 | 0.7341 | 0.7075 | +0.0266 [+0.0050, +0.0488] |

### Exact Clinical source scope

Counts below are encoded input columns, not independent examinations. Every encoding of a listed source is removed. The full ordered encoded-column list and indices are preserved in domain_plan.json and domain_conditions.json.

**All Clinical data** — 965 columns from 405 sources; disabled branches: clinical_data.

Removed source keys: `age`, `bmi_vsorres`, `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `brthyy`, `cage`, `ces1`, `ces10`, `ces2`, `ces3`, `ces4`, `ces5`, `ces6`, `ces7`, `ces8`, `ces9`, `cestl`, `cl_maristat`, `clock_visuospatial_executive`, `clock_visuospatial_executive_time`, `cm_act`, `cm_ant`, `cm_asp`, `cm_dcg`, `cm_ibp`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `cube_visuospatial_executive`, `cube_visuospatial_executive_time`, `delayed_recall_with_no_clue`, `delayed_recall_with_no_clue_time`, `diet1`, `diet2`, `diet3`, `diet4`, `diet5`, `diet6`, `diet7`, `diet8`, `diet9`, `dietscore`, `digitspan`, `digitspan_time`, `dmlact`, `dmledu`, `dmlfeet`, `dmlfrveg`, `dmlgrain`, `dmlhex`, `dmlmd`, `dmlpor`, `dmlsugar`, `dmlvex`, `dri1`, `dri2`, `eos_ds`, `fh_dm2pt`, `fh_dm2ptsp`, `fh_dm2sb`, `fh_dm2sbsp`, `fluency_language`, `food_insecurity_change_diet_frequency`, `height_vsorres`, `hip_vsorres`, `import_a_g_ratio`, `import_albumin`, `import_alkaline_phosphatase`, `import_alt_got`, `import_ast_got`, `import_bilirubin_total`, `import_bun`, `import_buncreatinineratio`, `import_c_peptide`, `import_calcium`, `import_carbon_dioxide_total`, `import_chloride`, `import_creatinine`, `import_crp_hs`, `import_globulin_total`, `import_glucose`, `import_hba1c`, `import_hdl_cholesterol`, `import_insulin`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_potassium`, `import_protein_total`, `import_sodium`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`, `import_urine_albumin`, `import_urine_creatinine`, `lbscat_a1c`, `lbscat_hct`, `lbscat_mch`, `lbscat_mchc`, `lbscat_mcv`, `lbscat_plt`, `lbscat_rbc`, `lbscat_rdw`, `lbscat_wbc`, `lettera`, `lettera_time`, `memory_trial1`, `memory_trial1_time`, `memory_trial2`, `memory_trial2_time`, `mh_a1c`, `mh_dm_age`, `mhoccur_ad`, `mhoccur_amd`, `mhoccur_ca`, `mhoccur_cns`, `mhoccur_cogn`, `mhoccur_crt`, `mhoccur_ded`, `mhoccur_ear`, `mhoccur_fall`, `mhoccur_fallot`, `mhoccur_gi`, `mhoccur_glc`, `mhoccur_ms`, `mhoccur_oa`, `mhoccur_obs`, `mhoccur_pd`, `mhoccur_pdr`, `mhoccur_plm`, `mhoccur_ra`, `mhoccur_rnl`, `mhoccur_ua`, `mhoccur_yn`, `mhterm_dm2`, `mhterm_predm`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `moca_abstraction`, `moca_abstraction_time`, `moca_combined_mis_score`, `moca_orientation`, `moca_orientation_time`, `moca_total_score`, `moca_total_score_time`, `msslf1`, `msslf10`, `msslf2`, `msslf3`, `msslf4`, `msslf5`, `msslf6`, `msslf7`, `msslf8`, `msslf9`, `msslffl`, `msslftt`, `mssrf1`, `mssrf10`, `mssrf2`, `mssrf3`, `mssrf4`, `mssrf5`, `mssrf6`, `mssrf7`, `mssrf8`, `mssrf9`, `mssrffl`, `mssrftt`, `naming`, `naming_time`, `paate`, `paid_cml`, `paid_dpr`, `paid_eng`, `paid_scrd`, `paid_wr`, `paidscore`, `pdrx9`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `pulse_vsorres`, `pulse_vsorres_2`, `pxahc1`, `pxahc10`, `pxahc2`, `pxahc3`, `pxahc4`, `pxahc5`, `pxahc7`, `pxahc8`, `pxahc9`, `pxdhc1`, `pxdhc2`, `pxdhc3`, `pxdhc4`, `pxdhc5`, `pxdhc6`, `pxdhc7`, `pxed1`, `pxep1`, `pxep2`, `pxep3`, `pxfi1`, `pxfi2`, `pxfi3`, `pxfi4`, `pxfi5`, `pxhi1`, `pxhi2`, `pxhic1`, `pxhic2`, `pxhic3`, `pxhic4`, `pxhic5`, `pxhic7`, `pxhic8`, `pxji1`, `pxji2`, `pxne1`, `pxne10`, `pxne11`, `pxne12`, `pxne13`, `pxne14`, `pxne15`, `pxne16`, `pxne17`, `pxne2`, `pxne3`, `pxne4`, `pxne5`, `pxne6`, `pxne7`, `pxne8`, `pxne9`, `pxpa1`, `pxpa2`, `pxpa3`, `pxpa4`, `pxrd1`, `pxrd10`, `pxrd11`, `pxrd12`, `pxrd13`, `pxrd14`, `pxrd15`, `pxrd16`, `pxrd17`, `pxrd18`, `pxrd19`, `pxrd2`, `pxrd20`, `pxrd21`, `pxrd22`, `pxrd23`, `pxrd24`, `pxrd25`, `pxrd26`, `pxrd27`, `pxrd28`, `pxrd29`, `pxrd3`, `pxrd30`, `pxrd31`, `pxrd32`, `pxrd33`, `pxrd34`, `pxrd35`, `pxrd36`, `pxrd37`, `pxrd38`, `pxrd39`, `pxrd4`, `pxrd40`, `pxrd41`, `pxrd42`, `pxrd43`, `pxrd44`, `pxrd45`, `pxrd46`, `pxrd47`, `pxrd48`, `pxrd49`, `pxrd5`, `pxrd50`, `pxrd51`, `pxrd52`, `pxrd53`, `pxrd54`, `pxrd6`, `pxrd7`, `pxrd8`, `repetition`, `repetition_time`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `sualcage`, `sualcdrv`, `sualckncf`, `subrdosfr`, `subrkncf`, `subtraction`, `subtraction_time`, `sulqdosfr`, `sulqkncf`, `sumrjcdur`, `sumrjceddur`, `sumrjfr`, `sumrjkncf`, `sumrjstage`, `sumrjtrt`, `susmkcdur`, `susmkdosfr`, `susmkncf`, `susmkstaage`, `susmkstoage`, `suvpcdur`, `suvpdosfr`, `suvpkncf1`, `suvpstage`, `suvptoage`, `suvptoyr`, `suwndosfr`, `suwnkncf`, `trails_visuospatial_executive`, `trails_visuospatial_executive_time`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`, `waist_vsorres`, `weight_vsorres`, `whr_vsorres`, `years_of_education`.

**ECG + cardiovascular Clinical** — 10 columns from 10 sources; disabled branches: cardiac_ecg.

Removed source keys: `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `import_hdl_cholesterol`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`.

Allowlisted but absent from this trained task (not reintroduced): `mhoccur_circ`, `mhoccur_clsh`, `mhoccur_cvdot`, `mhoccur_hbp`, `mhoccur_lbp`, `mhoccur_mi`, `mhoccur_rvo`, `mhoccur_strk`.

**Environmental sensor** — 0 columns from 0 sources; disabled branches: environment.

Removed source keys: None.

**All retinal + ocular Clinical** — 151 columns from 82 sources; disabled branches: retinal_flio, retinal_oct, retinal_octa, retinal_photography.

Removed source keys: `mhoccur_amd`, `mhoccur_crt`, `mhoccur_ded`, `mhoccur_glc`, `mhoccur_pdr`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`.

Allowlisted but absent from this trained task (not reintroduced): `mhoccur_rvo`.

**Activity monitor + activity/sleep Clinical** — 19 columns from 6 sources; disabled branches: wearable_activity_monitor.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `dmlact`, `dmlhex`, `dmlvex`.

**CGM + metabolic Clinical** — 22 columns from 12 sources; disabled branches: wearable_blood_glucose.

Removed source keys: `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

**All wearable + associated Clinical** — 41 columns from 18 sources; disabled branches: wearable_activity_monitor, wearable_blood_glucose.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `dmlact`, `dmlhex`, `dmlvex`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

### Provenance and verification

| Item | Value |
|---|---|
| checkpoint | /data/pzhao3/ot2_output/model_full_assets_v1/models/cvd_broad_binary/seed42/best_model.pt |
| checkpoint SHA256 | 0ec20efb87ce84368742af51a11949bfc9a11aaf3ccae77c54554230ad192cbf |
| config hash | 0e8280aca239e98788b867020bdb2a28b6cd89b46e1f0c7a12af8e800d12f66c |
| config file SHA256 | a424b7be9315822393be31730261da514549d0d6a8c129154f1cde297662e735 |
| Clinical schema SHA256 | 7e64a5b8e22a4ba570824742f96f9b5f0c9c39e9501a34c78a24c1b4c193fb1b |
| Clinical catalog SHA256 | 05a1f3e9acc27d814748eda9e89a60e9ceba0284e7e3f8bf377acafcf16ee79a |
| deletion plan SHA256 | 56b46300bf12c44725e37cc743c9cbc210e49c7d7a67990a97538ba3ed2ff487 |
| metadata SHA256 | d58358ce4272b8b4cd6388e175ef86c0ac4389a1324c81478acf7f758311aff8 |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | 300cef7211db2933369648a3021df069fce2a993723dcf632788ee62892d83ba |
| evaluation cohort hash | ffb3cea4b7f77cefb486e181194edd8a12c3663d9918e772329fc7e59d5e4fcb |
| master config hash | bfb04d90d1a3815d58a4f7005f900a57e0695a243f5728e451ed8bf378087cde |
| scaler config hash | 785040fd28d09ea22bb9c226eb4f4c8907e28730554dd06fccfbef710f4a9c02 |
| resolved model config hash | 21fc1229e93a9566e19b525e98884d2fbfdbe62ef7e7a5a24b6f29b21ab6d1e6 |
| result files | All six file hashes verified before report generation |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
| paired_participant_alignment | True |


## CVD strict: detailed results

N = 342; fixed threshold = 0.59784270; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7570 | 0.6516 | +0.1055 [+0.0514, +0.1650] | ok |
| All Clinical data | auprc | 0.6473 | 0.5510 | +0.0963 [+0.0326, +0.1591] | ok |
| All Clinical data | balanced_accuracy | 0.7158 | 0.6126 | +0.1032 [+0.0488, +0.1580] | ok |
| All Clinical data | sensitivity | 0.6783 | 0.2957 | +0.3826 [+0.2870, +0.4783] | ok |
| All Clinical data | specificity | 0.7533 | 0.9295 | -0.1762 [-0.2292, -0.1233] | ok |
| All Clinical data | log_loss | 0.6286 | 0.6026 | -0.0260 [-0.0790, +0.0278] | ok |
| All Clinical data | brier | 0.2157 | 0.2061 | -0.0096 [-0.0326, +0.0126] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7570 | 0.7289 | +0.0282 [+0.0066, +0.0511] | ok |
| ECG + cardiovascular Clinical | auprc | 0.6473 | 0.5645 | +0.0828 [+0.0186, +0.1341] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.7158 | 0.6748 | +0.0410 [+0.0082, +0.0759] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.6783 | 0.5478 | +0.1304 [+0.0696, +0.2000] | ok |
| ECG + cardiovascular Clinical | specificity | 0.7533 | 0.8018 | -0.0485 [-0.0837, -0.0176] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.6286 | 0.5984 | -0.0302 [-0.0558, -0.0052] | ok |
| ECG + cardiovascular Clinical | brier | 0.2157 | 0.2040 | -0.0117 [-0.0224, -0.0017] | ok |
| Environmental sensor | auroc | 0.7570 | 0.7548 | +0.0022 [-0.0006, +0.0053] | ok |
| Environmental sensor | auprc | 0.6473 | 0.6423 | +0.0051 [-0.0008, +0.0120] | ok |
| Environmental sensor | balanced_accuracy | 0.7158 | 0.7044 | +0.0114 [-0.0148, +0.0355] | ok |
| Environmental sensor | sensitivity | 0.6783 | 0.7304 | -0.0522 [-0.0957, -0.0174] | ok |
| Environmental sensor | specificity | 0.7533 | 0.6784 | +0.0749 [+0.0441, +0.1101] | ok |
| Environmental sensor | log_loss | 0.6286 | 0.6673 | +0.0387 [+0.0328, +0.0449] | ok |
| Environmental sensor | brier | 0.2157 | 0.2318 | +0.0161 [+0.0135, +0.0190] | ok |
| All retinal + ocular Clinical | auroc | 0.7570 | 0.7591 | -0.0021 [-0.0129, +0.0087] | ok |
| All retinal + ocular Clinical | auprc | 0.6473 | 0.6522 | -0.0049 [-0.0243, +0.0136] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.7158 | 0.7042 | +0.0116 [-0.0211, +0.0444] | ok |
| All retinal + ocular Clinical | sensitivity | 0.6783 | 0.7652 | -0.0870 [-0.1391, -0.0435] | ok |
| All retinal + ocular Clinical | specificity | 0.7533 | 0.6432 | +0.1101 [+0.0705, +0.1542] | ok |
| All retinal + ocular Clinical | log_loss | 0.6286 | 0.6815 | +0.0529 [+0.0379, +0.0672] | ok |
| All retinal + ocular Clinical | brier | 0.2157 | 0.2398 | +0.0242 [+0.0176, +0.0305] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7570 | 0.7559 | +0.0011 [-0.0037, +0.0060] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.6473 | 0.6474 | -0.0001 [-0.0085, +0.0084] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.7158 | 0.7128 | +0.0029 [-0.0343, +0.0379] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.6783 | 0.7913 | -0.1130 [-0.1739, -0.0609] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.7533 | 0.6344 | +0.1189 [+0.0793, +0.1630] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.6286 | 0.6903 | +0.0617 [+0.0511, +0.0732] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.2157 | 0.2419 | +0.0263 [+0.0215, +0.0316] | ok |
| CGM + metabolic Clinical | auroc | 0.7570 | 0.7617 | -0.0047 [-0.0140, +0.0047] | ok |
| CGM + metabolic Clinical | auprc | 0.6473 | 0.6511 | -0.0037 [-0.0162, +0.0076] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.7158 | 0.6963 | +0.0195 [-0.0046, +0.0479] | ok |
| CGM + metabolic Clinical | sensitivity | 0.6783 | 0.6261 | +0.0522 [+0.0087, +0.1043] | ok |
| CGM + metabolic Clinical | specificity | 0.7533 | 0.7665 | -0.0132 [-0.0396, +0.0132] | ok |
| CGM + metabolic Clinical | log_loss | 0.6286 | 0.5981 | -0.0304 [-0.0417, -0.0193] | ok |
| CGM + metabolic Clinical | brier | 0.2157 | 0.2033 | -0.0123 [-0.0173, -0.0074] | ok |
| All wearable + associated Clinical | auroc | 0.7570 | 0.7606 | -0.0035 [-0.0137, +0.0074] | ok |
| All wearable + associated Clinical | auprc | 0.6473 | 0.6548 | -0.0075 [-0.0222, +0.0075] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.7158 | 0.7045 | +0.0113 [-0.0170, +0.0377] | ok |
| All wearable + associated Clinical | sensitivity | 0.6783 | 0.7217 | -0.0435 [-0.0870, +0.0000] | ok |
| All wearable + associated Clinical | specificity | 0.7533 | 0.6872 | +0.0661 [+0.0352, +0.1013] | ok |
| All wearable + associated Clinical | log_loss | 0.6286 | 0.6494 | +0.0208 [+0.0081, +0.0341] | ok |
| All wearable + associated Clinical | brier | 0.2157 | 0.2252 | +0.0096 [+0.0041, +0.0154] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 342 | 115 / 227 | 2 | 0.7570 | 0.6516 | +0.1055 [+0.0514, +0.1650] |
| All Clinical data | native_present_only | 342 | 115 / 227 | 2 | 0.7570 | 0.6516 | +0.1055 [+0.0478, +0.1656] |
| All Clinical data | native_present_with_remaining_input | 340 | 114 / 226 | 0 | 0.7591 | 0.6512 | +0.1079 [+0.0483, +0.1666] |
| All Clinical data | any_target_information_present | 342 | 115 / 227 | 2 | 0.7570 | 0.6516 | +0.1055 [+0.0462, +0.1625] |
| ECG + cardiovascular Clinical | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7289 | +0.0282 [+0.0066, +0.0511] |
| ECG + cardiovascular Clinical | native_present_only | 337 | 111 / 226 | 0 | 0.7630 | 0.7340 | +0.0290 [+0.0072, +0.0513] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 337 | 111 / 226 | 0 | 0.7630 | 0.7340 | +0.0290 [+0.0071, +0.0507] |
| ECG + cardiovascular Clinical | any_target_information_present | 342 | 115 / 227 | 0 | 0.7570 | 0.7289 | +0.0282 [+0.0074, +0.0498] |
| Environmental sensor | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7548 | +0.0022 [-0.0006, +0.0053] |
| Environmental sensor | native_present_only | 339 | 114 / 225 | 0 | 0.7600 | 0.7578 | +0.0022 [-0.0004, +0.0050] |
| Environmental sensor | native_present_with_remaining_input | 339 | 114 / 225 | 0 | 0.7600 | 0.7578 | +0.0022 [-0.0003, +0.0050] |
| Environmental sensor | any_target_information_present | 339 | 114 / 225 | 0 | 0.7600 | 0.7578 | +0.0022 [-0.0004, +0.0051] |
| All retinal + ocular Clinical | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7591 | -0.0021 [-0.0129, +0.0087] |
| All retinal + ocular Clinical | any_target_present | 340 | 114 / 226 | 0 | 0.7591 | 0.7619 | -0.0028 [-0.0138, +0.0090] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 340 | 114 / 226 | 0 | 0.7591 | 0.7619 | -0.0028 [-0.0146, +0.0086] |
| All retinal + ocular Clinical | all_targets_present | 285 | 91 / 194 | 0 | 0.7878 | 0.7897 | -0.0019 [-0.0133, +0.0084] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 285 | 91 / 194 | 0 | 0.7878 | 0.7897 | -0.0019 [-0.0130, +0.0087] |
| All retinal + ocular Clinical | any_target_information_present | 342 | 115 / 227 | 0 | 0.7570 | 0.7591 | -0.0021 [-0.0136, +0.0090] |
| Activity monitor + activity/sleep Clinical | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7559 | +0.0011 [-0.0037, +0.0060] |
| Activity monitor + activity/sleep Clinical | native_present_only | 317 | 108 / 209 | 0 | 0.7516 | 0.7499 | +0.0017 [-0.0027, +0.0060] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 317 | 108 / 209 | 0 | 0.7516 | 0.7499 | +0.0017 [-0.0026, +0.0058] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 342 | 115 / 227 | 0 | 0.7570 | 0.7559 | +0.0011 [-0.0038, +0.0060] |
| CGM + metabolic Clinical | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7617 | -0.0047 [-0.0140, +0.0047] |
| CGM + metabolic Clinical | native_present_only | 336 | 114 / 222 | 0 | 0.7602 | 0.7646 | -0.0044 [-0.0139, +0.0054] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 336 | 114 / 222 | 0 | 0.7602 | 0.7646 | -0.0044 [-0.0142, +0.0046] |
| CGM + metabolic Clinical | any_target_information_present | 342 | 115 / 227 | 0 | 0.7570 | 0.7617 | -0.0047 [-0.0148, +0.0050] |
| All wearable + associated Clinical | all_participants | 342 | 115 / 227 | 0 | 0.7570 | 0.7606 | -0.0035 [-0.0137, +0.0074] |
| All wearable + associated Clinical | any_target_present | 340 | 114 / 226 | 0 | 0.7591 | 0.7626 | -0.0035 [-0.0140, +0.0075] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 340 | 114 / 226 | 0 | 0.7591 | 0.7626 | -0.0035 [-0.0146, +0.0074] |
| All wearable + associated Clinical | all_targets_present | 313 | 108 / 205 | 0 | 0.7526 | 0.7550 | -0.0024 [-0.0136, +0.0093] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 313 | 108 / 205 | 0 | 0.7526 | 0.7550 | -0.0024 [-0.0134, +0.0083] |
| All wearable + associated Clinical | any_target_information_present | 342 | 115 / 227 | 0 | 0.7570 | 0.7606 | -0.0035 [-0.0148, +0.0068] |

### Exact Clinical source scope

Counts below are encoded input columns, not independent examinations. Every encoding of a listed source is removed. The full ordered encoded-column list and indices are preserved in domain_plan.json and domain_conditions.json.

**All Clinical data** — 974 columns from 408 sources; disabled branches: clinical_data.

Removed source keys: `age`, `bmi_vsorres`, `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `brthyy`, `cage`, `ces1`, `ces10`, `ces2`, `ces3`, `ces4`, `ces5`, `ces6`, `ces7`, `ces8`, `ces9`, `cestl`, `cl_maristat`, `clock_visuospatial_executive`, `clock_visuospatial_executive_time`, `cm_act`, `cm_ant`, `cm_asp`, `cm_dcg`, `cm_ibp`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `cube_visuospatial_executive`, `cube_visuospatial_executive_time`, `delayed_recall_with_no_clue`, `delayed_recall_with_no_clue_time`, `diet1`, `diet2`, `diet3`, `diet4`, `diet5`, `diet6`, `diet7`, `diet8`, `diet9`, `dietscore`, `digitspan`, `digitspan_time`, `dmlact`, `dmledu`, `dmlfeet`, `dmlfrveg`, `dmlgrain`, `dmlhex`, `dmlmd`, `dmlpor`, `dmlsugar`, `dmlvex`, `dri1`, `dri2`, `eos_ds`, `fh_dm2pt`, `fh_dm2ptsp`, `fh_dm2sb`, `fh_dm2sbsp`, `fluency_language`, `food_insecurity_change_diet_frequency`, `height_vsorres`, `hip_vsorres`, `import_a_g_ratio`, `import_albumin`, `import_alkaline_phosphatase`, `import_alt_got`, `import_ast_got`, `import_bilirubin_total`, `import_bun`, `import_buncreatinineratio`, `import_c_peptide`, `import_calcium`, `import_carbon_dioxide_total`, `import_chloride`, `import_creatinine`, `import_crp_hs`, `import_globulin_total`, `import_glucose`, `import_hba1c`, `import_hdl_cholesterol`, `import_insulin`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_potassium`, `import_protein_total`, `import_sodium`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`, `import_urine_albumin`, `import_urine_creatinine`, `lbscat_a1c`, `lbscat_hct`, `lbscat_mch`, `lbscat_mchc`, `lbscat_mcv`, `lbscat_plt`, `lbscat_rbc`, `lbscat_rdw`, `lbscat_wbc`, `lettera`, `lettera_time`, `memory_trial1`, `memory_trial1_time`, `memory_trial2`, `memory_trial2_time`, `mh_a1c`, `mh_dm_age`, `mhoccur_ad`, `mhoccur_amd`, `mhoccur_ca`, `mhoccur_clsh`, `mhoccur_cns`, `mhoccur_cogn`, `mhoccur_crt`, `mhoccur_ded`, `mhoccur_ear`, `mhoccur_fall`, `mhoccur_fallot`, `mhoccur_gi`, `mhoccur_glc`, `mhoccur_hbp`, `mhoccur_lbp`, `mhoccur_ms`, `mhoccur_oa`, `mhoccur_obs`, `mhoccur_pd`, `mhoccur_pdr`, `mhoccur_plm`, `mhoccur_ra`, `mhoccur_rnl`, `mhoccur_ua`, `mhoccur_yn`, `mhterm_dm2`, `mhterm_predm`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `moca_abstraction`, `moca_abstraction_time`, `moca_combined_mis_score`, `moca_orientation`, `moca_orientation_time`, `moca_total_score`, `moca_total_score_time`, `msslf1`, `msslf10`, `msslf2`, `msslf3`, `msslf4`, `msslf5`, `msslf6`, `msslf7`, `msslf8`, `msslf9`, `msslffl`, `msslftt`, `mssrf1`, `mssrf10`, `mssrf2`, `mssrf3`, `mssrf4`, `mssrf5`, `mssrf6`, `mssrf7`, `mssrf8`, `mssrf9`, `mssrffl`, `mssrftt`, `naming`, `naming_time`, `paate`, `paid_cml`, `paid_dpr`, `paid_eng`, `paid_scrd`, `paid_wr`, `paidscore`, `pdrx9`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `pulse_vsorres`, `pulse_vsorres_2`, `pxahc1`, `pxahc10`, `pxahc2`, `pxahc3`, `pxahc4`, `pxahc5`, `pxahc7`, `pxahc8`, `pxahc9`, `pxdhc1`, `pxdhc2`, `pxdhc3`, `pxdhc4`, `pxdhc5`, `pxdhc6`, `pxdhc7`, `pxed1`, `pxep1`, `pxep2`, `pxep3`, `pxfi1`, `pxfi2`, `pxfi3`, `pxfi4`, `pxfi5`, `pxhi1`, `pxhi2`, `pxhic1`, `pxhic2`, `pxhic3`, `pxhic4`, `pxhic5`, `pxhic7`, `pxhic8`, `pxji1`, `pxji2`, `pxne1`, `pxne10`, `pxne11`, `pxne12`, `pxne13`, `pxne14`, `pxne15`, `pxne16`, `pxne17`, `pxne2`, `pxne3`, `pxne4`, `pxne5`, `pxne6`, `pxne7`, `pxne8`, `pxne9`, `pxpa1`, `pxpa2`, `pxpa3`, `pxpa4`, `pxrd1`, `pxrd10`, `pxrd11`, `pxrd12`, `pxrd13`, `pxrd14`, `pxrd15`, `pxrd16`, `pxrd17`, `pxrd18`, `pxrd19`, `pxrd2`, `pxrd20`, `pxrd21`, `pxrd22`, `pxrd23`, `pxrd24`, `pxrd25`, `pxrd26`, `pxrd27`, `pxrd28`, `pxrd29`, `pxrd3`, `pxrd30`, `pxrd31`, `pxrd32`, `pxrd33`, `pxrd34`, `pxrd35`, `pxrd36`, `pxrd37`, `pxrd38`, `pxrd39`, `pxrd4`, `pxrd40`, `pxrd41`, `pxrd42`, `pxrd43`, `pxrd44`, `pxrd45`, `pxrd46`, `pxrd47`, `pxrd48`, `pxrd49`, `pxrd5`, `pxrd50`, `pxrd51`, `pxrd52`, `pxrd53`, `pxrd54`, `pxrd6`, `pxrd7`, `pxrd8`, `repetition`, `repetition_time`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `sualcage`, `sualcdrv`, `sualckncf`, `subrdosfr`, `subrkncf`, `subtraction`, `subtraction_time`, `sulqdosfr`, `sulqkncf`, `sumrjcdur`, `sumrjceddur`, `sumrjfr`, `sumrjkncf`, `sumrjstage`, `sumrjtrt`, `susmkcdur`, `susmkdosfr`, `susmkncf`, `susmkstaage`, `susmkstoage`, `suvpcdur`, `suvpdosfr`, `suvpkncf1`, `suvpstage`, `suvptoage`, `suvptoyr`, `suwndosfr`, `suwnkncf`, `trails_visuospatial_executive`, `trails_visuospatial_executive_time`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`, `waist_vsorres`, `weight_vsorres`, `whr_vsorres`, `years_of_education`.

**ECG + cardiovascular Clinical** — 19 columns from 13 sources; disabled branches: cardiac_ecg.

Removed source keys: `bp1_diabp_vsorres`, `bp1_sysbp_vsorres`, `bp2_diabp_vsorres`, `bp2_sysbp_vsorres`, `import_hdl_cholesterol`, `import_ldl_cholesterol`, `import_nt_probnp`, `import_total_cholesterol`, `import_triglycerides`, `import_troponin_t`, `mhoccur_clsh`, `mhoccur_hbp`, `mhoccur_lbp`.

Allowlisted but absent from this trained task (not reintroduced): `mhoccur_circ`, `mhoccur_cvdot`, `mhoccur_mi`, `mhoccur_rvo`, `mhoccur_strk`.

**Environmental sensor** — 0 columns from 0 sources; disabled branches: environment.

Removed source keys: None.

**All retinal + ocular Clinical** — 151 columns from 82 sources; disabled branches: retinal_flio, retinal_oct, retinal_octa, retinal_photography.

Removed source keys: `mhoccur_amd`, `mhoccur_crt`, `mhoccur_ded`, `mhoccur_glc`, `mhoccur_pdr`, `mlcsodfcl`, `mlcsodlog`, `mlcsodmiss`, `mlcsosfcl`, `mlcsoslog`, `mlcsosmiss`, `plcsodfcl`, `plcsodlog`, `plcsodmiss`, `plcsosfcl`, `plcsoslog`, `plcsosmiss`, `rtci_odd1`, `rtci_odd2`, `rtci_odm1`, `rtci_odm2`, `rtci_osd1`, `rtci_osd2`, `rtci_osm1`, `rtci_osm2`, `rtei_odfaf`, `rtei_odir`, `rtei_osfaf`, `rtei_osir`, `rtfl_od`, `rtfl_os`, `rtma_odm`, `rtma_odm1`, `rtma_odw`, `rtma_osm`, `rtma_osm1`, `rtma_osw`, `rtop_odd`, `rtop_odm`, `rtop_osd`, `rtop_osm`, `rtsm_odc`, `rtsm_odm`, `rtsm_odn`, `rtsm_odt`, `rtsm_osc`, `rtsm_osm`, `rtsm_osn`, `rtsm_ost`, `rtso_odm`, `rtso_odo`, `rtso_odp`, `rtso_osm`, `rtso_oso`, `rtso_osp`, `rttr_odm1`, `rttr_odm2`, `rttr_odr`, `rttr_osm1`, `rttr_osm2`, `rttr_osr`, `via1`, `via2`, `via3`, `via4`, `via5`, `via6`, `viaodaxi`, `viaodcyl`, `viaodmlog`, `viaodmscore`, `viaodplog`, `viaodpscore`, `viaodsph`, `viaomthd`, `viaosaxi`, `viaoscyl`, `viaosmlog`, `viaosmscore`, `viaosplog`, `viaospscore`, `viaossph`.

Allowlisted but absent from this trained task (not reintroduced): `mhoccur_rvo`.

**Activity monitor + activity/sleep Clinical** — 19 columns from 6 sources; disabled branches: wearable_activity_monitor.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `dmlact`, `dmlhex`, `dmlvex`.

**CGM + metabolic Clinical** — 22 columns from 12 sources; disabled branches: wearable_blood_glucose.

Removed source keys: `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

**All wearable + associated Clinical** — 41 columns from 18 sources; disabled branches: wearable_activity_monitor, wearable_blood_glucose.

Removed source keys: `ces7`, `cestl`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `dmlact`, `dmlhex`, `dmlvex`, `import_c_peptide`, `import_glucose`, `import_hba1c`, `import_insulin`, `mh_a1c`, `mh_dm_age`, `mhterm_dm2`, `mhterm_predm`.

### Provenance and verification

| Item | Value |
|---|---|
| checkpoint | /data/pzhao3/ot2_output/model_full_assets_v1/models/cvd_strict_binary/seed42/best_model.pt |
| checkpoint SHA256 | 8bb7c4416e6318b253f1bbb5f0f90aad912d1b3a4fe67ad0ae2f7df8b7bae1a3 |
| config hash | 95fee85977cd6e00d6481da4f7d2302d44a0fe0bcb9a1ac7d8523a15029e1e45 |
| config file SHA256 | bf4c2a3be1dcd02b2bb66acd907a1e5f10cbac2ca72f469b844a794fb7e08083 |
| Clinical schema SHA256 | 0a9fe93337a6a0073ebc7f56de25b65f4c433449d65d42317fba37d7e85bc32e |
| Clinical catalog SHA256 | 320131d74ee2508d95fdb35e9ae6de5ca022f6399dba0493ba3056d214aa7929 |
| deletion plan SHA256 | ef670e17a925a16bccc738ec7c71ea689cd9df33aa825f9ea54a495b6e536d90 |
| metadata SHA256 | 6d74e202f3085bade1ab490225a7bae1554282e3451ca500c5fb03451ab3d8c5 |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | 3320fa6c722a4e7e48d616c6d11940243a3cff9ce41fa8769815f86c704dc673 |
| evaluation cohort hash | 05c649d8242ee4f9ffa5a642f9fd7e0cc265fdc1cc0a8d00dfdece053ae794f9 |
| master config hash | 2f6c7be047082aadf347763b282a8a1c2c979f94f0391316381ebbf0dd7d5775 |
| scaler config hash | f041d13116980d3c528715f30762737108271ac00c58fb69672f3bd2c9251844 |
| resolved model config hash | a42dc3fbc1bee9562475396635898921f93d01b316b109983602bcffc3648674 |
| result files | All six file hashes verified before report generation |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
| paired_participant_alignment | True |

