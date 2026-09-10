# Domain ablation summary

**Overall status: COMPLETE**

Frozen seed-42 models; validation split; joint removal of model branches and prespecified Clinical information. This file contains aggregate results and field names, without participant-level predictions or identifiers.

Runner state: complete; phase: complete; updated UTC: 2026-09-10T03:45:36.653804+00:00.

| Task | Result status |
|---|---|
| MoCA | Verified |
| CVD broad | Verified |
| CVD strict | Verified |

## AUROC degradation across tasks

Positive = AUROC decreased after deletion; negative = AUROC increased. Values are absolute AUROC differences, followed by pointwise 95% paired bootstrap intervals. 0.0100 equals one AUROC percentage point.

| Deletion | MoCA | CVD broad | CVD strict |
|---|---|---|---|
| All Clinical data | +0.1013 [+0.0480, +0.1546] | +0.0875 [+0.0071, +0.1695] | +0.1454 [+0.0836, +0.2078] |
| ECG + cardiovascular Clinical | -0.0061 [-0.0158, +0.0029] | +0.0307 [+0.0107, +0.0513] | +0.0293 [+0.0023, +0.0575] |
| Environmental sensor | +0.0006 [-0.0022, +0.0034] | +0.0002 [-0.0039, +0.0042] | -0.0033 [-0.0069, +0.0002] |
| All retinal + ocular Clinical | +0.0257 [-0.0029, +0.0535] | -0.0062 [-0.0322, +0.0202] | -0.0060 [-0.0183, +0.0060] |
| Activity monitor + activity/sleep Clinical | +0.0043 [-0.0004, +0.0086] | +0.0021 [-0.0075, +0.0117] | -0.0057 [-0.0121, +0.0002] |
| CGM + metabolic Clinical | -0.0006 [-0.0142, +0.0129] | +0.0101 [-0.0204, +0.0386] | +0.0000 [-0.0118, +0.0126] |
| All wearable + associated Clinical | +0.0044 [-0.0103, +0.0193] | +0.0138 [-0.0180, +0.0432] | -0.0042 [-0.0180, +0.0102] |

## Descriptive findings

- MoCA: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.1013 [+0.0480, +0.1546]. Conditions with a pointwise interval entirely above zero: All Clinical data. This is descriptive validation evidence.
- CVD broad: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.0875 [+0.0071, +0.1695]. Conditions with a pointwise interval entirely above zero: All Clinical data, ECG + cardiovascular Clinical. This is descriptive validation evidence.
- CVD strict: largest observed AUROC degradation among the selected conditions: All Clinical data, +0.1454 [+0.0836, +0.2078]. Conditions with a pointwise interval entirely above zero: All Clinical data, ECG + cardiovascular Clinical. This is descriptive validation evidence.

## Full-input baseline

| Task | N | Prevalence | AUROC | AUPRC | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoCA | 351 | 0.4444 | 0.7270 | 0.6589 | 0.7372 | 0.6359 | 0.2141 |
| CVD broad | 352 | 0.7784 | 0.7395 | 0.8921 | 0.6423 | 0.7308 | 0.1638 |
| CVD strict | 351 | 0.2821 | 0.7261 | 0.5188 | 0.6768 | 0.7103 | 0.2294 |

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
| analyze_domain_ablation_v1.py | af224cfe03befbef6378dc9425fc0fa57d519d1b1485a4f327651faaee00f5de |
| analyze_modality_importance_v2.py | b20ebe3b7b08f7597b8b03a71d3cd257cf735b8ceaee4a38de9d4e809c1f32e8 |
| clinical_basic_information_v1.py | 77b6feef189b14b498716e0a2f75f7af330c5966e63d90a53258a28c0a89a259 |
| clinical_domain_groups_v1.py | b3a1a293ff315bdc6bcab45d4defc16affb629c6c3ab45c87829ea3c449a7862 |
| clinical_modality_overlap_v1.py | ecc616d62aa1e4aeac1ef3bde4f3c256b4543f0d87ea40eb765ea598c543acfd |
| clinical_overlap_inference_v1.py | 2360ec9a2500799c0727407902ab70b8c705f7b2bd2b639dc676fe73b155ef96 |
| domain_ablation_engine_v1.py | f9c7d12ea24b5fb62d0429c461f47b0e517d7d3cb62b044032d0afd70c2bfb6d |
| fit_train_scalers_v2.py | 1c8138a3a66ff6ad922f4e907df95c0b0e5c67487042963a37ed9fce52102419 |
| preprocessing_v2.py | 0219f9a8e9c291f2b9c5091205e4476db73c17c6a232fee1bc789491e7b44e14 |
| run_domain_ablation_v1.py | bf8e6851fde0043f88de33ea5a64f0d790b7b224e428e0c251788fc7afcbdc1c |
| summarize_domain_ablation_v1.py | 98238708dd3650849ee2ae0acdefef3faf13dfcb1b2658dfd38426f1da0b0d56 |
| task2_v2_config.py | 458848c8032a9d47fc54ca112043b2929b81e126cd80601c94ee20514e21ac06 |
| task2_v2_dataset.py | 24afc6b37058021b4f20bc17a50f16a5fe72e543768ac4365a3dc214fcb9fcef |
| task2_v2_engine.py | 63f8014d3d4c9fd7d7c2469d0496bf17245bfa9b8110503b39572d80a6286ac4 |
| task2_v2_model.py | 7f9a22f149306ef16412311e9f8d6275804fe19b66f03e2aedd031dd102c5463 |
| train_task2_v2.py | 34cf798448a025cc8d49a69a4df4eec549d798ca6a004e1f7e5435a4b08444a0 |

## MoCA: detailed results

N = 351; fixed threshold = 0.45519860; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7270 | 0.6257 | +0.1013 [+0.0480, +0.1546] | ok |
| All Clinical data | auprc | 0.6589 | 0.5483 | +0.1106 [+0.0488, +0.1672] | ok |
| All Clinical data | balanced_accuracy | 0.6865 | 0.5795 | +0.1071 [+0.0494, +0.1635] | ok |
| All Clinical data | sensitivity | 0.7372 | 0.6667 | +0.0705 [-0.0128, +0.1538] | ok |
| All Clinical data | specificity | 0.6359 | 0.4923 | +0.1436 [+0.0615, +0.2256] | ok |
| All Clinical data | log_loss | 0.6204 | 0.6862 | +0.0659 [+0.0212, +0.1103] | ok |
| All Clinical data | brier | 0.2141 | 0.2443 | +0.0302 [+0.0107, +0.0500] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7270 | 0.7331 | -0.0061 [-0.0158, +0.0029] | ok |
| ECG + cardiovascular Clinical | auprc | 0.6589 | 0.6718 | -0.0130 [-0.0264, +0.0034] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.6865 | 0.6917 | -0.0051 [-0.0269, +0.0160] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.7372 | 0.7628 | -0.0256 [-0.0577, +0.0000] | ok |
| ECG + cardiovascular Clinical | specificity | 0.6359 | 0.6205 | +0.0154 [-0.0154, +0.0462] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.6204 | 0.6187 | -0.0017 [-0.0113, +0.0078] | ok |
| ECG + cardiovascular Clinical | brier | 0.2141 | 0.2131 | -0.0010 [-0.0053, +0.0032] | ok |
| Environmental sensor | auroc | 0.7270 | 0.7264 | +0.0006 [-0.0022, +0.0034] | ok |
| Environmental sensor | auprc | 0.6589 | 0.6610 | -0.0021 [-0.0095, +0.0056] | ok |
| Environmental sensor | balanced_accuracy | 0.6865 | 0.6891 | -0.0026 [-0.0167, +0.0115] | ok |
| Environmental sensor | sensitivity | 0.7372 | 0.7372 | +0.0000 [-0.0192, +0.0192] | ok |
| Environmental sensor | specificity | 0.6359 | 0.6410 | -0.0051 [-0.0256, +0.0154] | ok |
| Environmental sensor | log_loss | 0.6204 | 0.6204 | +0.0000 [-0.0021, +0.0021] | ok |
| Environmental sensor | brier | 0.2141 | 0.2140 | -0.0001 [-0.0010, +0.0009] | ok |
| All retinal + ocular Clinical | auroc | 0.7270 | 0.7013 | +0.0257 [-0.0029, +0.0535] | ok |
| All retinal + ocular Clinical | auprc | 0.6589 | 0.6447 | +0.0142 [-0.0243, +0.0525] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.6865 | 0.6500 | +0.0365 [-0.0013, +0.0756] | ok |
| All retinal + ocular Clinical | sensitivity | 0.7372 | 0.6538 | +0.0833 [+0.0321, +0.1346] | ok |
| All retinal + ocular Clinical | specificity | 0.6359 | 0.6462 | -0.0103 [-0.0667, +0.0513] | ok |
| All retinal + ocular Clinical | log_loss | 0.6204 | 0.6344 | +0.0141 [-0.0142, +0.0409] | ok |
| All retinal + ocular Clinical | brier | 0.2141 | 0.2206 | +0.0065 [-0.0060, +0.0181] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7270 | 0.7227 | +0.0043 [-0.0004, +0.0086] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.6589 | 0.6545 | +0.0044 [-0.0015, +0.0109] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.6865 | 0.6609 | +0.0256 [+0.0032, +0.0481] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.7372 | 0.7628 | -0.0256 [-0.0513, -0.0064] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.6359 | 0.5590 | +0.0769 [+0.0410, +0.1128] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.6204 | 0.6326 | +0.0123 [+0.0063, +0.0179] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.2141 | 0.2194 | +0.0053 [+0.0026, +0.0079] | ok |
| CGM + metabolic Clinical | auroc | 0.7270 | 0.7276 | -0.0006 [-0.0142, +0.0129] | ok |
| CGM + metabolic Clinical | auprc | 0.6589 | 0.6679 | -0.0091 [-0.0328, +0.0194] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.6865 | 0.6865 | +0.0000 [-0.0224, +0.0231] | ok |
| CGM + metabolic Clinical | sensitivity | 0.7372 | 0.7115 | +0.0256 [+0.0000, +0.0577] | ok |
| CGM + metabolic Clinical | specificity | 0.6359 | 0.6615 | -0.0256 [-0.0615, +0.0103] | ok |
| CGM + metabolic Clinical | log_loss | 0.6204 | 0.6144 | -0.0060 [-0.0202, +0.0075] | ok |
| CGM + metabolic Clinical | brier | 0.2141 | 0.2118 | -0.0023 [-0.0085, +0.0035] | ok |
| All wearable + associated Clinical | auroc | 0.7270 | 0.7226 | +0.0044 [-0.0103, +0.0193] | ok |
| All wearable + associated Clinical | auprc | 0.6589 | 0.6609 | -0.0021 [-0.0277, +0.0292] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.6865 | 0.6744 | +0.0122 [-0.0109, +0.0365] | ok |
| All wearable + associated Clinical | sensitivity | 0.7372 | 0.7436 | -0.0064 [-0.0321, +0.0192] | ok |
| All wearable + associated Clinical | specificity | 0.6359 | 0.6051 | +0.0308 [-0.0051, +0.0718] | ok |
| All wearable + associated Clinical | log_loss | 0.6204 | 0.6249 | +0.0045 [-0.0108, +0.0189] | ok |
| All wearable + associated Clinical | brier | 0.2141 | 0.2162 | +0.0021 [-0.0043, +0.0082] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.6257 | +0.1013 [+0.0480, +0.1546] |
| All Clinical data | native_present_only | 351 | 156 / 195 | 0 | 0.7270 | 0.6257 | +0.1013 [+0.0486, +0.1567] |
| All Clinical data | native_present_with_remaining_input | 351 | 156 / 195 | 0 | 0.7270 | 0.6257 | +0.1013 [+0.0459, +0.1558] |
| All Clinical data | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.6257 | +0.1013 [+0.0484, +0.1558] |
| ECG + cardiovascular Clinical | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7331 | -0.0061 [-0.0158, +0.0029] |
| ECG + cardiovascular Clinical | native_present_only | 346 | 155 / 191 | 0 | 0.7353 | 0.7407 | -0.0053 [-0.0144, +0.0038] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 346 | 155 / 191 | 0 | 0.7353 | 0.7407 | -0.0053 [-0.0139, +0.0038] |
| ECG + cardiovascular Clinical | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7331 | -0.0061 [-0.0151, +0.0032] |
| Environmental sensor | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7264 | +0.0006 [-0.0022, +0.0034] |
| Environmental sensor | native_present_only | 343 | 151 / 192 | 0 | 0.7266 | 0.7258 | +0.0007 [-0.0023, +0.0036] |
| Environmental sensor | native_present_with_remaining_input | 343 | 151 / 192 | 0 | 0.7266 | 0.7258 | +0.0007 [-0.0023, +0.0035] |
| Environmental sensor | any_target_information_present | 343 | 151 / 192 | 0 | 0.7266 | 0.7258 | +0.0007 [-0.0022, +0.0036] |
| All retinal + ocular Clinical | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7013 | +0.0257 [-0.0029, +0.0535] |
| All retinal + ocular Clinical | any_target_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7013 | +0.0257 [+0.0001, +0.0528] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 351 | 156 / 195 | 0 | 0.7270 | 0.7013 | +0.0257 [-0.0031, +0.0549] |
| All retinal + ocular Clinical | all_targets_present | 293 | 125 / 168 | 0 | 0.7066 | 0.6832 | +0.0234 [-0.0080, +0.0550] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 293 | 125 / 168 | 0 | 0.7066 | 0.6832 | +0.0234 [-0.0085, +0.0583] |
| All retinal + ocular Clinical | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7013 | +0.0257 [-0.0022, +0.0525] |
| Activity monitor + activity/sleep Clinical | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7227 | +0.0043 [-0.0004, +0.0086] |
| Activity monitor + activity/sleep Clinical | native_present_only | 319 | 138 / 181 | 0 | 0.7392 | 0.7365 | +0.0027 [-0.0015, +0.0072] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 319 | 138 / 181 | 0 | 0.7392 | 0.7365 | +0.0027 [-0.0017, +0.0072] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7227 | +0.0043 [-0.0000, +0.0089] |
| CGM + metabolic Clinical | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7276 | -0.0006 [-0.0142, +0.0129] |
| CGM + metabolic Clinical | native_present_only | 343 | 151 / 192 | 0 | 0.7226 | 0.7225 | +0.0001 [-0.0143, +0.0152] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 343 | 151 / 192 | 0 | 0.7226 | 0.7225 | +0.0001 [-0.0138, +0.0141] |
| CGM + metabolic Clinical | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7276 | -0.0006 [-0.0139, +0.0136] |
| All wearable + associated Clinical | all_participants | 351 | 156 / 195 | 0 | 0.7270 | 0.7226 | +0.0044 [-0.0103, +0.0193] |
| All wearable + associated Clinical | any_target_present | 346 | 152 / 194 | 0 | 0.7254 | 0.7207 | +0.0046 [-0.0095, +0.0194] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 346 | 152 / 194 | 0 | 0.7254 | 0.7207 | +0.0046 [-0.0093, +0.0197] |
| All wearable + associated Clinical | all_targets_present | 316 | 137 / 179 | 0 | 0.7363 | 0.7328 | +0.0035 [-0.0112, +0.0185] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 316 | 137 / 179 | 0 | 0.7363 | 0.7328 | +0.0035 [-0.0122, +0.0174] |
| All wearable + associated Clinical | any_target_information_present | 351 | 156 / 195 | 0 | 0.7270 | 0.7226 | +0.0044 [-0.0094, +0.0179] |

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
| metadata SHA256 | 304273cb0ad06277a7adca33ffdf3fc2764b8bad09fcabdb1e4facd8a113c8be |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | bcc5e0533f99c0e4e51b8006bb8a6c11d1231ad0610271b7d4ef879ffd49f627 |
| evaluation cohort hash | ee04e45f40889bf56a041444f0a734a87863bda812544e4369ca75565ce91da0 |
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

N = 352; fixed threshold = 0.91117971; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7395 | 0.6520 | +0.0875 [+0.0071, +0.1695] | ok |
| All Clinical data | auprc | 0.8921 | 0.8668 | +0.0253 [-0.0186, +0.0704] | ok |
| All Clinical data | balanced_accuracy | 0.6866 | 0.5989 | +0.0876 [+0.0153, +0.1599] | ok |
| All Clinical data | sensitivity | 0.6423 | 0.4927 | +0.1496 [+0.0766, +0.2226] | ok |
| All Clinical data | specificity | 0.7308 | 0.7051 | +0.0256 [-0.1026, +0.1410] | ok |
| All Clinical data | log_loss | 0.5128 | 0.5501 | +0.0373 [-0.0348, +0.1080] | ok |
| All Clinical data | brier | 0.1638 | 0.1777 | +0.0139 [-0.0118, +0.0394] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7395 | 0.7088 | +0.0307 [+0.0107, +0.0513] | ok |
| ECG + cardiovascular Clinical | auprc | 0.8921 | 0.8789 | +0.0132 [-0.0032, +0.0294] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.6866 | 0.6387 | +0.0478 [-0.0007, +0.0973] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.6423 | 0.7518 | -0.1095 [-0.1460, -0.0730] | ok |
| ECG + cardiovascular Clinical | specificity | 0.7308 | 0.5256 | +0.2051 [+0.1154, +0.2949] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.5128 | 0.5494 | +0.0366 [+0.0144, +0.0586] | ok |
| ECG + cardiovascular Clinical | brier | 0.1638 | 0.1685 | +0.0047 [-0.0042, +0.0136] | ok |
| Environmental sensor | auroc | 0.7395 | 0.7393 | +0.0002 [-0.0039, +0.0042] | ok |
| Environmental sensor | auprc | 0.8921 | 0.8945 | -0.0023 [-0.0058, +0.0008] | ok |
| Environmental sensor | balanced_accuracy | 0.6866 | 0.6829 | +0.0037 [-0.0137, +0.0248] | ok |
| Environmental sensor | sensitivity | 0.6423 | 0.6606 | -0.0182 [-0.0365, -0.0036] | ok |
| Environmental sensor | specificity | 0.7308 | 0.7051 | +0.0256 [+0.0000, +0.0641] | ok |
| Environmental sensor | log_loss | 0.5128 | 0.5092 | -0.0036 [-0.0092, +0.0015] | ok |
| Environmental sensor | brier | 0.1638 | 0.1611 | -0.0027 [-0.0051, -0.0005] | ok |
| All retinal + ocular Clinical | auroc | 0.7395 | 0.7457 | -0.0062 [-0.0322, +0.0202] | ok |
| All retinal + ocular Clinical | auprc | 0.8921 | 0.9004 | -0.0083 [-0.0243, +0.0085] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.6866 | 0.6395 | +0.0471 [-0.0105, +0.1029] | ok |
| All retinal + ocular Clinical | sensitivity | 0.6423 | 0.8431 | -0.2007 [-0.2518, -0.1496] | ok |
| All retinal + ocular Clinical | specificity | 0.7308 | 0.4359 | +0.2949 [+0.1923, +0.3974] | ok |
| All retinal + ocular Clinical | log_loss | 0.5128 | 0.5360 | +0.0233 [-0.0136, +0.0587] | ok |
| All retinal + ocular Clinical | brier | 0.1638 | 0.1628 | -0.0010 [-0.0165, +0.0131] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7395 | 0.7375 | +0.0021 [-0.0075, +0.0117] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.8921 | 0.8929 | -0.0008 [-0.0089, +0.0059] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.6866 | 0.6646 | +0.0219 [-0.0055, +0.0512] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.6423 | 0.6241 | +0.0182 [-0.0073, +0.0438] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.7308 | 0.7051 | +0.0256 [-0.0256, +0.0769] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.5128 | 0.5079 | -0.0049 [-0.0135, +0.0039] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.1638 | 0.1618 | -0.0019 [-0.0053, +0.0013] | ok |
| CGM + metabolic Clinical | auroc | 0.7395 | 0.7294 | +0.0101 [-0.0204, +0.0386] | ok |
| CGM + metabolic Clinical | auprc | 0.8921 | 0.9083 | -0.0162 [-0.0458, +0.0111] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.6866 | 0.6446 | +0.0420 [-0.0001, +0.0841] | ok |
| CGM + metabolic Clinical | sensitivity | 0.6423 | 0.5584 | +0.0839 [+0.0438, +0.1241] | ok |
| CGM + metabolic Clinical | specificity | 0.7308 | 0.7308 | +0.0000 [-0.0769, +0.0769] | ok |
| CGM + metabolic Clinical | log_loss | 0.5128 | 0.5251 | +0.0123 [-0.0117, +0.0359] | ok |
| CGM + metabolic Clinical | brier | 0.1638 | 0.1704 | +0.0066 [-0.0019, +0.0153] | ok |
| All wearable + associated Clinical | auroc | 0.7395 | 0.7257 | +0.0138 [-0.0180, +0.0432] | ok |
| All wearable + associated Clinical | auprc | 0.8921 | 0.9069 | -0.0148 [-0.0442, +0.0134] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.6866 | 0.6436 | +0.0429 [+0.0027, +0.0859] | ok |
| All wearable + associated Clinical | sensitivity | 0.6423 | 0.5693 | +0.0730 [+0.0328, +0.1131] | ok |
| All wearable + associated Clinical | specificity | 0.7308 | 0.7179 | +0.0128 [-0.0641, +0.0897] | ok |
| All wearable + associated Clinical | log_loss | 0.5128 | 0.5166 | +0.0038 [-0.0209, +0.0284] | ok |
| All wearable + associated Clinical | brier | 0.1638 | 0.1665 | +0.0027 [-0.0060, +0.0114] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.6520 | +0.0875 [+0.0071, +0.1695] |
| All Clinical data | native_present_only | 352 | 274 / 78 | 0 | 0.7395 | 0.6520 | +0.0875 [+0.0043, +0.1665] |
| All Clinical data | native_present_with_remaining_input | 352 | 274 / 78 | 0 | 0.7395 | 0.6520 | +0.0875 [+0.0034, +0.1667] |
| All Clinical data | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.6520 | +0.0875 [+0.0097, +0.1656] |
| ECG + cardiovascular Clinical | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7088 | +0.0307 [+0.0107, +0.0513] |
| ECG + cardiovascular Clinical | native_present_only | 347 | 270 / 77 | 0 | 0.7405 | 0.7082 | +0.0322 [+0.0127, +0.0533] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 347 | 270 / 77 | 0 | 0.7405 | 0.7082 | +0.0322 [+0.0128, +0.0536] |
| ECG + cardiovascular Clinical | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7088 | +0.0307 [+0.0093, +0.0520] |
| Environmental sensor | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7393 | +0.0002 [-0.0039, +0.0042] |
| Environmental sensor | native_present_only | 344 | 269 / 75 | 0 | 0.7568 | 0.7560 | +0.0008 [-0.0030, +0.0046] |
| Environmental sensor | native_present_with_remaining_input | 344 | 269 / 75 | 0 | 0.7568 | 0.7560 | +0.0008 [-0.0029, +0.0046] |
| Environmental sensor | any_target_information_present | 344 | 269 / 75 | 0 | 0.7568 | 0.7560 | +0.0008 [-0.0028, +0.0047] |
| All retinal + ocular Clinical | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7457 | -0.0062 [-0.0322, +0.0202] |
| All retinal + ocular Clinical | any_target_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7457 | -0.0062 [-0.0316, +0.0210] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 352 | 274 / 78 | 0 | 0.7395 | 0.7457 | -0.0062 [-0.0311, +0.0193] |
| All retinal + ocular Clinical | all_targets_present | 293 | 228 / 65 | 0 | 0.7690 | 0.7695 | -0.0005 [-0.0268, +0.0261] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 293 | 228 / 65 | 0 | 0.7690 | 0.7695 | -0.0005 [-0.0254, +0.0277] |
| All retinal + ocular Clinical | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7457 | -0.0062 [-0.0325, +0.0228] |
| Activity monitor + activity/sleep Clinical | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7375 | +0.0021 [-0.0075, +0.0117] |
| Activity monitor + activity/sleep Clinical | native_present_only | 320 | 251 / 69 | 0 | 0.7571 | 0.7581 | -0.0009 [-0.0096, +0.0080] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 320 | 251 / 69 | 0 | 0.7571 | 0.7581 | -0.0009 [-0.0098, +0.0084] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7375 | +0.0021 [-0.0075, +0.0116] |
| CGM + metabolic Clinical | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7294 | +0.0101 [-0.0204, +0.0386] |
| CGM + metabolic Clinical | native_present_only | 344 | 269 / 75 | 0 | 0.7558 | 0.7368 | +0.0189 [-0.0084, +0.0443] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 344 | 269 / 75 | 0 | 0.7558 | 0.7368 | +0.0189 [-0.0076, +0.0442] |
| CGM + metabolic Clinical | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7294 | +0.0101 [-0.0212, +0.0388] |
| All wearable + associated Clinical | all_participants | 352 | 274 / 78 | 0 | 0.7395 | 0.7257 | +0.0138 [-0.0180, +0.0432] |
| All wearable + associated Clinical | any_target_present | 347 | 272 / 75 | 0 | 0.7566 | 0.7357 | +0.0209 [-0.0104, +0.0492] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 347 | 272 / 75 | 0 | 0.7566 | 0.7357 | +0.0209 [-0.0086, +0.0499] |
| All wearable + associated Clinical | all_targets_present | 317 | 248 / 69 | 0 | 0.7564 | 0.7371 | +0.0192 [-0.0108, +0.0491] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 317 | 248 / 69 | 0 | 0.7564 | 0.7371 | +0.0192 [-0.0106, +0.0477] |
| All wearable + associated Clinical | any_target_information_present | 352 | 274 / 78 | 0 | 0.7395 | 0.7257 | +0.0138 [-0.0162, +0.0430] |

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
| metadata SHA256 | cad9fb2821b21d4295f26fb1efe14635f96c1297d469dd6cdbccf1cfd1a62fc5 |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | 90a292a105698042352d1be47bc70f65ecf9311dd89bf817e7521c3464e1e219 |
| evaluation cohort hash | 648b808a65bc627c36c062fbb849f026788445cdc5eab7b699b88d1bdf980287 |
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

N = 351; fixed threshold = 0.59784270; bootstrap samples = 2000.

### All-participant metrics

| Deletion | Metric | Full input | After deletion | Degradation [95% CI] | CI status |
|---|---|---:|---:|---|---|
| All Clinical data | auroc | 0.7261 | 0.5806 | +0.1454 [+0.0836, +0.2078] | ok |
| All Clinical data | auprc | 0.5188 | 0.4015 | +0.1173 [+0.0490, +0.1890] | ok |
| All Clinical data | balanced_accuracy | 0.6935 | 0.5664 | +0.1272 [+0.0698, +0.1847] | ok |
| All Clinical data | sensitivity | 0.6768 | 0.2121 | +0.4646 [+0.3636, +0.5657] | ok |
| All Clinical data | specificity | 0.7103 | 0.9206 | -0.2103 [-0.2698, -0.1508] | ok |
| All Clinical data | log_loss | 0.6583 | 0.6217 | -0.0367 [-0.0909, +0.0180] | ok |
| All Clinical data | brier | 0.2294 | 0.2137 | -0.0157 [-0.0390, +0.0082] | ok |
| ECG + cardiovascular Clinical | auroc | 0.7261 | 0.6967 | +0.0293 [+0.0023, +0.0575] | ok |
| ECG + cardiovascular Clinical | auprc | 0.5188 | 0.4671 | +0.0517 [-0.0062, +0.1035] | ok |
| ECG + cardiovascular Clinical | balanced_accuracy | 0.6935 | 0.6566 | +0.0370 [-0.0022, +0.0772] | ok |
| ECG + cardiovascular Clinical | sensitivity | 0.6768 | 0.5354 | +0.1414 [+0.0707, +0.2121] | ok |
| ECG + cardiovascular Clinical | specificity | 0.7103 | 0.7778 | -0.0675 [-0.1071, -0.0317] | ok |
| ECG + cardiovascular Clinical | log_loss | 0.6583 | 0.6199 | -0.0384 [-0.0644, -0.0131] | ok |
| ECG + cardiovascular Clinical | brier | 0.2294 | 0.2148 | -0.0146 [-0.0254, -0.0042] | ok |
| Environmental sensor | auroc | 0.7261 | 0.7294 | -0.0033 [-0.0069, +0.0002] | ok |
| Environmental sensor | auprc | 0.5188 | 0.5234 | -0.0046 [-0.0119, +0.0031] | ok |
| Environmental sensor | balanced_accuracy | 0.6935 | 0.6699 | +0.0236 [+0.0016, +0.0435] | ok |
| Environmental sensor | sensitivity | 0.6768 | 0.6970 | -0.0202 [-0.0505, +0.0000] | ok |
| Environmental sensor | specificity | 0.7103 | 0.6429 | +0.0675 [+0.0357, +0.0992] | ok |
| Environmental sensor | log_loss | 0.6583 | 0.6972 | +0.0389 [+0.0329, +0.0454] | ok |
| Environmental sensor | brier | 0.2294 | 0.2451 | +0.0158 [+0.0130, +0.0188] | ok |
| All retinal + ocular Clinical | auroc | 0.7261 | 0.7321 | -0.0060 [-0.0183, +0.0060] | ok |
| All retinal + ocular Clinical | auprc | 0.5188 | 0.5327 | -0.0139 [-0.0381, +0.0089] | ok |
| All retinal + ocular Clinical | balanced_accuracy | 0.6935 | 0.6510 | +0.0426 [+0.0177, +0.0666] | ok |
| All retinal + ocular Clinical | sensitivity | 0.6768 | 0.6869 | -0.0101 [-0.0505, +0.0202] | ok |
| All retinal + ocular Clinical | specificity | 0.7103 | 0.6151 | +0.0952 [+0.0595, +0.1310] | ok |
| All retinal + ocular Clinical | log_loss | 0.6583 | 0.7203 | +0.0620 [+0.0475, +0.0764] | ok |
| All retinal + ocular Clinical | brier | 0.2294 | 0.2561 | +0.0267 [+0.0203, +0.0332] | ok |
| Activity monitor + activity/sleep Clinical | auroc | 0.7261 | 0.7317 | -0.0057 [-0.0121, +0.0002] | ok |
| Activity monitor + activity/sleep Clinical | auprc | 0.5188 | 0.5225 | -0.0037 [-0.0125, +0.0060] | ok |
| Activity monitor + activity/sleep Clinical | balanced_accuracy | 0.6935 | 0.6641 | +0.0294 [+0.0013, +0.0554] | ok |
| Activity monitor + activity/sleep Clinical | sensitivity | 0.6768 | 0.7172 | -0.0404 [-0.0808, -0.0101] | ok |
| Activity monitor + activity/sleep Clinical | specificity | 0.7103 | 0.6111 | +0.0992 [+0.0675, +0.1349] | ok |
| Activity monitor + activity/sleep Clinical | log_loss | 0.6583 | 0.7254 | +0.0670 [+0.0559, +0.0782] | ok |
| Activity monitor + activity/sleep Clinical | brier | 0.2294 | 0.2569 | +0.0275 [+0.0223, +0.0326] | ok |
| CGM + metabolic Clinical | auroc | 0.7261 | 0.7260 | +0.0000 [-0.0118, +0.0126] | ok |
| CGM + metabolic Clinical | auprc | 0.5188 | 0.5275 | -0.0087 [-0.0280, +0.0115] | ok |
| CGM + metabolic Clinical | balanced_accuracy | 0.6935 | 0.6890 | +0.0045 [-0.0226, +0.0326] | ok |
| CGM + metabolic Clinical | sensitivity | 0.6768 | 0.6162 | +0.0606 [+0.0202, +0.1111] | ok |
| CGM + metabolic Clinical | specificity | 0.7103 | 0.7619 | -0.0516 [-0.0794, -0.0278] | ok |
| CGM + metabolic Clinical | log_loss | 0.6583 | 0.6294 | -0.0290 [-0.0417, -0.0162] | ok |
| CGM + metabolic Clinical | brier | 0.2294 | 0.2162 | -0.0132 [-0.0185, -0.0080] | ok |
| All wearable + associated Clinical | auroc | 0.7261 | 0.7302 | -0.0042 [-0.0180, +0.0102] | ok |
| All wearable + associated Clinical | auprc | 0.5188 | 0.5304 | -0.0116 [-0.0322, +0.0096] | ok |
| All wearable + associated Clinical | balanced_accuracy | 0.6935 | 0.6728 | +0.0207 [-0.0031, +0.0447] | ok |
| All wearable + associated Clinical | sensitivity | 0.6768 | 0.6869 | -0.0101 [-0.0505, +0.0202] | ok |
| All wearable + associated Clinical | specificity | 0.7103 | 0.6587 | +0.0516 [+0.0198, +0.0873] | ok |
| All wearable + associated Clinical | log_loss | 0.6583 | 0.6890 | +0.0307 [+0.0164, +0.0456] | ok |
| All wearable + associated Clinical | brier | 0.2294 | 0.2417 | +0.0123 [+0.0062, +0.0189] | ok |

### Availability strata (paired AUROC comparisons)

`native_present_only` means the target device is present. For families, `any_target_present` and `all_targets_present` distinguish at least one versus every target device. `*_with_remaining_input` additionally requires some input after cleanup. `any_target_information_present` also includes people with observed selected Clinical companions despite absent target devices.

| Deletion | Cohort | N | Positive / negative | No remaining input | Full AUROC | After deletion | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| All Clinical data | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.5806 | +0.1454 [+0.0836, +0.2078] |
| All Clinical data | native_present_only | 351 | 99 / 252 | 0 | 0.7261 | 0.5806 | +0.1454 [+0.0844, +0.2053] |
| All Clinical data | native_present_with_remaining_input | 351 | 99 / 252 | 0 | 0.7261 | 0.5806 | +0.1454 [+0.0816, +0.2053] |
| All Clinical data | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.5806 | +0.1454 [+0.0854, +0.2090] |
| ECG + cardiovascular Clinical | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.6967 | +0.0293 [+0.0023, +0.0575] |
| ECG + cardiovascular Clinical | native_present_only | 346 | 99 / 247 | 0 | 0.7303 | 0.7011 | +0.0292 [+0.0029, +0.0568] |
| ECG + cardiovascular Clinical | native_present_with_remaining_input | 346 | 99 / 247 | 0 | 0.7303 | 0.7011 | +0.0292 [+0.0029, +0.0563] |
| ECG + cardiovascular Clinical | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.6967 | +0.0293 [+0.0026, +0.0577] |
| Environmental sensor | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.7294 | -0.0033 [-0.0069, +0.0002] |
| Environmental sensor | native_present_only | 343 | 99 / 244 | 0 | 0.7300 | 0.7318 | -0.0018 [-0.0054, +0.0015] |
| Environmental sensor | native_present_with_remaining_input | 343 | 99 / 244 | 0 | 0.7300 | 0.7318 | -0.0018 [-0.0052, +0.0012] |
| Environmental sensor | any_target_information_present | 343 | 99 / 244 | 0 | 0.7300 | 0.7318 | -0.0018 [-0.0052, +0.0015] |
| All retinal + ocular Clinical | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.7321 | -0.0060 [-0.0183, +0.0060] |
| All retinal + ocular Clinical | any_target_present | 351 | 99 / 252 | 0 | 0.7261 | 0.7321 | -0.0060 [-0.0182, +0.0059] |
| All retinal + ocular Clinical | any_target_present_with_remaining_input | 351 | 99 / 252 | 0 | 0.7261 | 0.7321 | -0.0060 [-0.0184, +0.0060] |
| All retinal + ocular Clinical | all_targets_present | 292 | 79 / 213 | 0 | 0.7279 | 0.7308 | -0.0029 [-0.0158, +0.0101] |
| All retinal + ocular Clinical | all_targets_present_with_remaining_input | 292 | 79 / 213 | 0 | 0.7279 | 0.7308 | -0.0029 [-0.0154, +0.0099] |
| All retinal + ocular Clinical | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.7321 | -0.0060 [-0.0185, +0.0064] |
| Activity monitor + activity/sleep Clinical | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.7317 | -0.0057 [-0.0121, +0.0002] |
| Activity monitor + activity/sleep Clinical | native_present_only | 319 | 95 / 224 | 0 | 0.7323 | 0.7320 | +0.0003 [-0.0049, +0.0056] |
| Activity monitor + activity/sleep Clinical | native_present_with_remaining_input | 319 | 95 / 224 | 0 | 0.7323 | 0.7320 | +0.0003 [-0.0048, +0.0058] |
| Activity monitor + activity/sleep Clinical | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.7317 | -0.0057 [-0.0123, +0.0005] |
| CGM + metabolic Clinical | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.7260 | +0.0000 [-0.0118, +0.0126] |
| CGM + metabolic Clinical | native_present_only | 343 | 97 / 246 | 0 | 0.7271 | 0.7294 | -0.0023 [-0.0147, +0.0107] |
| CGM + metabolic Clinical | native_present_with_remaining_input | 343 | 97 / 246 | 0 | 0.7271 | 0.7294 | -0.0023 [-0.0146, +0.0102] |
| CGM + metabolic Clinical | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.7260 | +0.0000 [-0.0125, +0.0130] |
| All wearable + associated Clinical | all_participants | 351 | 99 / 252 | 0 | 0.7261 | 0.7302 | -0.0042 [-0.0180, +0.0102] |
| All wearable + associated Clinical | any_target_present | 346 | 99 / 247 | 0 | 0.7311 | 0.7365 | -0.0055 [-0.0183, +0.0084] |
| All wearable + associated Clinical | any_target_present_with_remaining_input | 346 | 99 / 247 | 0 | 0.7311 | 0.7365 | -0.0055 [-0.0179, +0.0088] |
| All wearable + associated Clinical | all_targets_present | 316 | 93 / 223 | 0 | 0.7281 | 0.7277 | +0.0005 [-0.0133, +0.0145] |
| All wearable + associated Clinical | all_targets_present_with_remaining_input | 316 | 93 / 223 | 0 | 0.7281 | 0.7277 | +0.0005 [-0.0125, +0.0145] |
| All wearable + associated Clinical | any_target_information_present | 351 | 99 / 252 | 0 | 0.7261 | 0.7302 | -0.0042 [-0.0181, +0.0097] |

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
| metadata SHA256 | df9b66ee7d5479f6de01688dd6046e399aa209a362e3613d6444e416dbbfc3e8 |
| threshold origin | saved_full_teacher_validation_threshold |
| training seed | 42 |
| bootstrap seed | 42 |
| source cache SHA256 | c79168d48d84d210209d9acba13554b540819ca9c892ff78dd49e74cf69ec55b |
| evaluation cohort hash | 396f9a4710d5d75770040275fa0e2b63c92ef3b3b4ebbe20694e0476a4e12241 |
| master config hash | 2f6c7be047082aadf347763b282a8a1c2c979f94f0391316381ebbf0dd7d5775 |
| scaler config hash | f041d13116980d3c528715f30762737108271ac00c58fb69672f3bd2c9251844 |
| resolved model config hash | a42dc3fbc1bee9562475396635898921f93d01b316b109983602bcffc3648674 |
| result files | All six file hashes verified before report generation |
| clinical_values_and_observed_masks_removed | True |
| full_forward_equivalence | passed; atol=1e-6, rtol=1e-6 |
| native_clinical_equivalence | passed; atol=1e-6, rtol=1e-6 |
| original_cache_and_clinical_bank_unchanged | True |
| paired_participant_alignment | True |

