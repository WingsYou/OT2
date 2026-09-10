# Basic Clinical retraining and modality importance

**Overall status: COMPLETE**

New seed-42 task models trained with only the agreed 23 Clinical columns plus the other eight modalities. Original splits and retained train-fitted scaler statistics are preserved. No test inference is included.

All nonclinical preprocessing, model settings and ImageNet retinal initialization follow the source experiment. The previous full-Clinical task checkpoint is not loaded. Clinical input width is physically 23 during training and inference.

This aggregate report contains no participant identifiers or predictions.

Runner: complete; phase: complete; updated UTC: 2026-09-10T07:40:48.218000+00:00.

| Task | Training | Importance |
|---|---|---|
| MoCA | Verified | Verified |
| CVD broad | Verified | Verified |
| CVD strict | Verified | Verified |

## New-model validation baseline

| Task | Best epoch | Epochs | Threshold | AUROC | AUPRC | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoCA | 11 | 18 | 0.5100 | 0.7012 | 0.6244 | 0.6603 | 0.7026 | 0.2316 |
| CVD broad | 10 | 17 | 0.4765 | 0.6934 | 0.8861 | 0.7153 | 0.5897 | 0.2247 |
| CVD strict | 9 | 16 | 0.4743 | 0.6231 | 0.4094 | 0.6364 | 0.5873 | 0.2593 |

## AUROC degradation after deletion

Positive = new model performs worse after removal. These are absolute AUROC differences with pointwise paired 95% bootstrap intervals, not percentages of importance.

| Removed input | MoCA | CVD broad | CVD strict |
|---|---|---|---|
| clinical_data | +0.0532 [+0.0182, +0.0907] | +0.0009 [-0.0455, +0.0465] | +0.0049 [-0.0295, +0.0401] |
| cardiac_ecg | -0.0071 [-0.0196, +0.0058] | +0.0383 [+0.0141, +0.0621] | +0.0247 [-0.0001, +0.0507] |
| environment | +0.0003 [-0.0168, +0.0159] | +0.0032 [-0.0046, +0.0111] | +0.0019 [-0.0119, +0.0154] |
| retinal_flio | +0.0091 [-0.0043, +0.0232] | -0.0001 [-0.0125, +0.0120] | -0.0101 [-0.0238, +0.0038] |
| retinal_oct | +0.0045 [-0.0041, +0.0130] | +0.0030 [-0.0045, +0.0102] | +0.0013 [-0.0054, +0.0085] |
| retinal_octa | -0.0000 [-0.0081, +0.0076] | +0.0011 [-0.0024, +0.0043] | -0.0034 [-0.0108, +0.0036] |
| retinal_photography | +0.0207 [+0.0065, +0.0349] | +0.0030 [-0.0007, +0.0070] | -0.0032 [-0.0084, +0.0021] |
| wearable_activity_monitor | +0.0117 [-0.0019, +0.0250] | +0.0213 [-0.0011, +0.0437] | +0.0212 [-0.0064, +0.0487] |
| wearable_blood_glucose | -0.0036 [-0.0186, +0.0110] | +0.0005 [-0.0113, +0.0124] | +0.0091 [-0.0107, +0.0298] |
| retinal_all | +0.0506 [+0.0206, +0.0800] | +0.0051 [-0.0116, +0.0225] | -0.0159 [-0.0326, -0.0003] |
| wearable_all | +0.0112 [-0.0094, +0.0310] | +0.0235 [-0.0012, +0.0500] | +0.0340 [-0.0000, +0.0685] |
| nonclinical_all | +0.0450 [+0.0015, +0.0899] | +0.0727 [+0.0238, +0.1207] | +0.0493 [+0.0010, +0.0981] |

`clinical_data` means only the 23 Basic Clinical columns. `nonclinical_all` removes the other eight together and leaves Basic Clinical; it tests whether the newly trained model collectively uses the other modalities.

- Clinical information reduction does not guarantee every remaining modality is used, or that reconstruction has been learned.
- All deletion conditions use the new model's full-input validation threshold. AUROC, probability errors and threshold-based metrics should be interpreted together.
- Identical participant masks reuse identical bootstrap intervals. Intervals exclude training-seed uncertainty and are not adjusted for multiplicity. Crossing zero does not establish equivalence or noninferiority.
- Leave-one-out measures this newly fitted model's reliance under missing inputs; it is not leave-one-modality-out retraining or a causal effect.

## MoCA details

Validation N=351; bootstrap samples=2000; threshold=0.51000843.

| Deletion | Metric | Baseline | After deletion | Degradation [95% CI] |
|---|---|---:|---:|---|
| clinical_data | auroc | 0.7012 | 0.6479 | +0.0532 [+0.0182, +0.0907] |
| clinical_data | auprc | 0.6244 | 0.5745 | +0.0499 [+0.0135, +0.0858] |
| clinical_data | balanced_accuracy | 0.6814 | 0.5897 | +0.0917 [+0.0481, +0.1372] |
| clinical_data | sensitivity | 0.6603 | 0.5128 | +0.1474 [+0.0769, +0.2244] |
| clinical_data | specificity | 0.7026 | 0.6667 | +0.0359 [-0.0154, +0.0872] |
| clinical_data | log_loss | 0.6751 | 0.7008 | +0.0257 [-0.0197, +0.0707] |
| clinical_data | brier | 0.2316 | 0.2467 | +0.0151 [-0.0028, +0.0332] |
| cardiac_ecg | auroc | 0.7012 | 0.7083 | -0.0071 [-0.0196, +0.0058] |
| cardiac_ecg | auprc | 0.6244 | 0.6377 | -0.0134 [-0.0372, +0.0094] |
| cardiac_ecg | balanced_accuracy | 0.6814 | 0.6558 | +0.0256 [-0.0109, +0.0628] |
| cardiac_ecg | sensitivity | 0.6603 | 0.7885 | -0.1282 [-0.1795, -0.0769] |
| cardiac_ecg | specificity | 0.7026 | 0.5231 | +0.1795 [+0.1282, +0.2308] |
| cardiac_ecg | log_loss | 0.6751 | 0.7276 | +0.0525 [+0.0246, +0.0799] |
| cardiac_ecg | brier | 0.2316 | 0.2520 | +0.0204 [+0.0086, +0.0322] |
| environment | auroc | 0.7012 | 0.7009 | +0.0003 [-0.0168, +0.0159] |
| environment | auprc | 0.6244 | 0.6334 | -0.0090 [-0.0372, +0.0186] |
| environment | balanced_accuracy | 0.6814 | 0.6846 | -0.0032 [-0.0308, +0.0237] |
| environment | sensitivity | 0.6603 | 0.6410 | +0.0192 [-0.0256, +0.0641] |
| environment | specificity | 0.7026 | 0.7282 | -0.0256 [-0.0564, +0.0000] |
| environment | log_loss | 0.6751 | 0.6641 | -0.0111 [-0.0334, +0.0103] |
| environment | brier | 0.2316 | 0.2274 | -0.0042 [-0.0134, +0.0047] |
| retinal_flio | auroc | 0.7012 | 0.6921 | +0.0091 [-0.0043, +0.0232] |
| retinal_flio | auprc | 0.6244 | 0.6067 | +0.0177 [-0.0069, +0.0459] |
| retinal_flio | balanced_accuracy | 0.6814 | 0.6692 | +0.0122 [-0.0154, +0.0397] |
| retinal_flio | sensitivity | 0.6603 | 0.6667 | -0.0064 [-0.0449, +0.0321] |
| retinal_flio | specificity | 0.7026 | 0.6718 | +0.0308 [-0.0051, +0.0667] |
| retinal_flio | log_loss | 0.6751 | 0.7000 | +0.0249 [+0.0059, +0.0462] |
| retinal_flio | brier | 0.2316 | 0.2407 | +0.0091 [+0.0014, +0.0173] |
| retinal_oct | auroc | 0.7012 | 0.6967 | +0.0045 [-0.0041, +0.0130] |
| retinal_oct | auprc | 0.6244 | 0.6214 | +0.0030 [-0.0154, +0.0209] |
| retinal_oct | balanced_accuracy | 0.6814 | 0.6660 | +0.0154 [-0.0147, +0.0449] |
| retinal_oct | sensitivity | 0.6603 | 0.7372 | -0.0769 [-0.1218, -0.0385] |
| retinal_oct | specificity | 0.7026 | 0.5949 | +0.1077 [+0.0667, +0.1538] |
| retinal_oct | log_loss | 0.6751 | 0.7127 | +0.0375 [+0.0197, +0.0557] |
| retinal_oct | brier | 0.2316 | 0.2462 | +0.0147 [+0.0070, +0.0221] |
| retinal_octa | auroc | 0.7012 | 0.7012 | -0.0000 [-0.0081, +0.0076] |
| retinal_octa | auprc | 0.6244 | 0.6202 | +0.0042 [-0.0100, +0.0185] |
| retinal_octa | balanced_accuracy | 0.6814 | 0.6737 | +0.0077 [-0.0212, +0.0378] |
| retinal_octa | sensitivity | 0.6603 | 0.7372 | -0.0769 [-0.1218, -0.0321] |
| retinal_octa | specificity | 0.7026 | 0.6103 | +0.0923 [+0.0513, +0.1333] |
| retinal_octa | log_loss | 0.6751 | 0.6922 | +0.0171 [+0.0009, +0.0324] |
| retinal_octa | brier | 0.2316 | 0.2380 | +0.0065 [-0.0003, +0.0130] |
| retinal_photography | auroc | 0.7012 | 0.6805 | +0.0207 [+0.0065, +0.0349] |
| retinal_photography | auprc | 0.6244 | 0.5930 | +0.0314 [+0.0023, +0.0609] |
| retinal_photography | balanced_accuracy | 0.6814 | 0.6718 | +0.0096 [-0.0269, +0.0449] |
| retinal_photography | sensitivity | 0.6603 | 0.7692 | -0.1090 [-0.1603, -0.0641] |
| retinal_photography | specificity | 0.7026 | 0.5744 | +0.1282 [+0.0769, +0.1846] |
| retinal_photography | log_loss | 0.6751 | 0.7292 | +0.0541 [+0.0301, +0.0789] |
| retinal_photography | brier | 0.2316 | 0.2507 | +0.0191 [+0.0087, +0.0295] |
| wearable_activity_monitor | auroc | 0.7012 | 0.6894 | +0.0117 [-0.0019, +0.0250] |
| wearable_activity_monitor | auprc | 0.6244 | 0.6125 | +0.0119 [-0.0095, +0.0314] |
| wearable_activity_monitor | balanced_accuracy | 0.6814 | 0.6712 | +0.0103 [-0.0160, +0.0333] |
| wearable_activity_monitor | sensitivity | 0.6603 | 0.6859 | -0.0256 [-0.0705, +0.0128] |
| wearable_activity_monitor | specificity | 0.7026 | 0.6564 | +0.0462 [+0.0154, +0.0769] |
| wearable_activity_monitor | log_loss | 0.6751 | 0.6724 | -0.0027 [-0.0223, +0.0158] |
| wearable_activity_monitor | brier | 0.2316 | 0.2336 | +0.0020 [-0.0057, +0.0089] |
| wearable_blood_glucose | auroc | 0.7012 | 0.7048 | -0.0036 [-0.0186, +0.0110] |
| wearable_blood_glucose | auprc | 0.6244 | 0.6352 | -0.0108 [-0.0340, +0.0166] |
| wearable_blood_glucose | balanced_accuracy | 0.6814 | 0.6660 | +0.0154 [-0.0173, +0.0474] |
| wearable_blood_glucose | sensitivity | 0.6603 | 0.7115 | -0.0513 [-0.0897, -0.0128] |
| wearable_blood_glucose | specificity | 0.7026 | 0.6205 | +0.0821 [+0.0359, +0.1333] |
| wearable_blood_glucose | log_loss | 0.6751 | 0.6730 | -0.0021 [-0.0235, +0.0189] |
| wearable_blood_glucose | brier | 0.2316 | 0.2325 | +0.0009 [-0.0082, +0.0096] |
| retinal_all | auroc | 0.7012 | 0.6506 | +0.0506 [+0.0206, +0.0800] |
| retinal_all | auprc | 0.6244 | 0.5453 | +0.0791 [+0.0303, +0.1262] |
| retinal_all | balanced_accuracy | 0.6814 | 0.5808 | +0.1006 [+0.0474, +0.1494] |
| retinal_all | sensitivity | 0.6603 | 0.9359 | -0.2756 [-0.3526, -0.2115] |
| retinal_all | specificity | 0.7026 | 0.2256 | +0.4769 [+0.4051, +0.5487] |
| retinal_all | log_loss | 0.6751 | 0.9777 | +0.3026 [+0.2464, +0.3608] |
| retinal_all | brier | 0.2316 | 0.3400 | +0.1084 [+0.0853, +0.1316] |
| wearable_all | auroc | 0.7012 | 0.6900 | +0.0112 [-0.0094, +0.0310] |
| wearable_all | auprc | 0.6244 | 0.6239 | +0.0005 [-0.0307, +0.0308] |
| wearable_all | balanced_accuracy | 0.6814 | 0.6494 | +0.0321 [-0.0045, +0.0705] |
| wearable_all | sensitivity | 0.6603 | 0.7756 | -0.1154 [-0.1667, -0.0705] |
| wearable_all | specificity | 0.7026 | 0.5231 | +0.1795 [+0.1231, +0.2410] |
| wearable_all | log_loss | 0.6751 | 0.6867 | +0.0116 [-0.0199, +0.0433] |
| wearable_all | brier | 0.2316 | 0.2418 | +0.0103 [-0.0028, +0.0233] |
| nonclinical_all | auroc | 0.7012 | 0.6562 | +0.0450 [+0.0015, +0.0899] |
| nonclinical_all | auprc | 0.6244 | 0.5683 | +0.0561 [-0.0005, +0.1185] |
| nonclinical_all | balanced_accuracy | 0.6814 | 0.5000 | +0.1814 [+0.1308, +0.2263] |
| nonclinical_all | sensitivity | 0.6603 | 1.0000 | -0.3397 [-0.4167, -0.2692] |
| nonclinical_all | specificity | 0.7026 | 0.0000 | +0.7026 [+0.6359, +0.7692] |
| nonclinical_all | log_loss | 0.6751 | 1.3348 | +0.6596 [+0.5883, +0.7284] |
| nonclinical_all | brier | 0.2316 | 0.4557 | +0.2241 [+0.1948, +0.2516] |

### Availability strata

| Deletion | Cohort | N | Positive / negative | No remaining input | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---|
| clinical_data | all_participants | 351 | 156 / 195 | 0 | +0.0532 [+0.0182, +0.0907] |
| clinical_data | any_target_present | 351 | 156 / 195 | 0 | +0.0532 [+0.0182, +0.0907] |
| clinical_data | all_targets_present | 351 | 156 / 195 | 0 | +0.0532 [+0.0182, +0.0907] |
| clinical_data | any_target_present_with_remaining_input | 351 | 156 / 195 | 0 | +0.0532 [+0.0182, +0.0907] |
| cardiac_ecg | all_participants | 351 | 156 / 195 | 0 | -0.0071 [-0.0196, +0.0058] |
| cardiac_ecg | any_target_present | 346 | 155 / 191 | 0 | -0.0050 [-0.0170, +0.0066] |
| cardiac_ecg | all_targets_present | 346 | 155 / 191 | 0 | -0.0050 [-0.0170, +0.0066] |
| cardiac_ecg | any_target_present_with_remaining_input | 346 | 155 / 191 | 0 | -0.0050 [-0.0170, +0.0066] |
| environment | all_participants | 351 | 156 / 195 | 0 | +0.0003 [-0.0168, +0.0159] |
| environment | any_target_present | 343 | 151 / 192 | 0 | +0.0005 [-0.0162, +0.0177] |
| environment | all_targets_present | 343 | 151 / 192 | 0 | +0.0005 [-0.0162, +0.0177] |
| environment | any_target_present_with_remaining_input | 343 | 151 / 192 | 0 | +0.0005 [-0.0162, +0.0177] |
| retinal_flio | all_participants | 351 | 156 / 195 | 0 | +0.0091 [-0.0043, +0.0232] |
| retinal_flio | any_target_present | 293 | 125 / 168 | 0 | +0.0084 [-0.0076, +0.0250] |
| retinal_flio | all_targets_present | 293 | 125 / 168 | 0 | +0.0084 [-0.0076, +0.0250] |
| retinal_flio | any_target_present_with_remaining_input | 293 | 125 / 168 | 0 | +0.0084 [-0.0076, +0.0250] |
| retinal_oct | all_participants | 351 | 156 / 195 | 0 | +0.0045 [-0.0041, +0.0130] |
| retinal_oct | any_target_present | 350 | 156 / 194 | 0 | +0.0047 [-0.0034, +0.0126] |
| retinal_oct | all_targets_present | 350 | 156 / 194 | 0 | +0.0047 [-0.0034, +0.0126] |
| retinal_oct | any_target_present_with_remaining_input | 350 | 156 / 194 | 0 | +0.0047 [-0.0034, +0.0126] |
| retinal_octa | all_participants | 351 | 156 / 195 | 0 | -0.0000 [-0.0081, +0.0076] |
| retinal_octa | any_target_present | 350 | 156 / 194 | 0 | +0.0000 [-0.0083, +0.0081] |
| retinal_octa | all_targets_present | 350 | 156 / 194 | 0 | +0.0000 [-0.0083, +0.0081] |
| retinal_octa | any_target_present_with_remaining_input | 350 | 156 / 194 | 0 | +0.0000 [-0.0083, +0.0081] |
| retinal_photography | all_participants | 351 | 156 / 195 | 0 | +0.0207 [+0.0065, +0.0349] |
| retinal_photography | any_target_present | 351 | 156 / 195 | 0 | +0.0207 [+0.0065, +0.0349] |
| retinal_photography | all_targets_present | 351 | 156 / 195 | 0 | +0.0207 [+0.0065, +0.0349] |
| retinal_photography | any_target_present_with_remaining_input | 351 | 156 / 195 | 0 | +0.0207 [+0.0065, +0.0349] |
| wearable_activity_monitor | all_participants | 351 | 156 / 195 | 0 | +0.0117 [-0.0019, +0.0250] |
| wearable_activity_monitor | any_target_present | 319 | 138 / 181 | 0 | +0.0091 [-0.0046, +0.0238] |
| wearable_activity_monitor | all_targets_present | 319 | 138 / 181 | 0 | +0.0091 [-0.0046, +0.0238] |
| wearable_activity_monitor | any_target_present_with_remaining_input | 319 | 138 / 181 | 0 | +0.0091 [-0.0046, +0.0238] |
| wearable_blood_glucose | all_participants | 351 | 156 / 195 | 0 | -0.0036 [-0.0186, +0.0110] |
| wearable_blood_glucose | any_target_present | 343 | 151 / 192 | 0 | -0.0036 [-0.0184, +0.0116] |
| wearable_blood_glucose | all_targets_present | 343 | 151 / 192 | 0 | -0.0036 [-0.0184, +0.0116] |
| wearable_blood_glucose | any_target_present_with_remaining_input | 343 | 151 / 192 | 0 | -0.0036 [-0.0184, +0.0116] |
| retinal_all | all_participants | 351 | 156 / 195 | 0 | +0.0506 [+0.0206, +0.0800] |
| retinal_all | any_target_present | 351 | 156 / 195 | 0 | +0.0506 [+0.0206, +0.0800] |
| retinal_all | all_targets_present | 293 | 125 / 168 | 0 | +0.0512 [+0.0191, +0.0846] |
| retinal_all | any_target_present_with_remaining_input | 351 | 156 / 195 | 0 | +0.0506 [+0.0206, +0.0800] |
| wearable_all | all_participants | 351 | 156 / 195 | 0 | +0.0112 [-0.0094, +0.0310] |
| wearable_all | any_target_present | 346 | 152 / 194 | 0 | +0.0111 [-0.0085, +0.0314] |
| wearable_all | all_targets_present | 316 | 137 / 179 | 0 | +0.0068 [-0.0143, +0.0279] |
| wearable_all | any_target_present_with_remaining_input | 346 | 152 / 194 | 0 | +0.0111 [-0.0085, +0.0314] |
| nonclinical_all | all_participants | 351 | 156 / 195 | 0 | +0.0450 [+0.0015, +0.0899] |
| nonclinical_all | any_target_present | 351 | 156 / 195 | 0 | +0.0450 [+0.0015, +0.0899] |
| nonclinical_all | all_targets_present | 257 | 107 / 150 | 0 | +0.0537 [+0.0008, +0.1050] |
| nonclinical_all | any_target_present_with_remaining_input | 351 | 156 / 195 | 0 | +0.0450 [+0.0015, +0.0899] |

### Exact retained Clinical columns

| Encoded column | Table | Source key |
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
| checkpoint_sha256 | "35f9380da53cf6de440d20375a43f256d3fbf3341ed18165908df9e0ab7adf6f" |
| config_hash | "d13e745eede590b94ef6afe9b3f0ac3a6640aaa335f282b66909fb5e169ed7dc" |
| prepared_sha256 | "9d743ea4d2455a0a1199b13f3168a55141f4c2b7b978f5a28d44b43863cdf433" |
| trained_sha256 | "a94615169d6815559581e14be1c29f4d39eda7d80f1d87f731a4430d9f2a51fc" |
| basic_plan_sha256 | "53dd453ed415efa6b2acc95e1fbd9399b26a4752c91ef6493b50e2a4e4328258" |
| threshold_source | "this_retrained_models_validation_selected_threshold" |
| source_token_cache | {"content_sha256": "86e940b8ce6f09eb7bddf66914c382d3dcd35669ea45a841329cd59a03ba3249", "file_sha256": "116cbba685c7bd6dbdfe30865dd76f6c9ae36dff973f943abb18ba498127e7e8", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "35f9380da53cf6de440d20375a43f256d3fbf3341ed18165908df9e0ab7adf6f", "cohort_person_ids_sha256": "ee04e45f40889bf56a041444f0a734a87863bda812544e4369ca75565ce91da0", "config_hash": "d13e745eede590b94ef6afe9b3f0ac3a6640aaa335f282b66909fb5e169ed7dc", "encoder_batch_size": 2, "master_config_hash": "ac420a562c4c24eb80ec7d577b622520bea247e2dcbd0673bbcc9ca6b6a5a53b", "model_config_hash": "cb1dd473b8dd2fae1dffbc652f1367c26e0d547bdd2a4dc384833321017f6186", "n_participants": 351, "scaler_config_hash": "f42d42cda3ac3f0f1ce69d96a63e23e5485803c550d84d86667cfcc701058b40", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "moca_binary"}, "path": "/data/pzhao3/ot2_output/model_basic_clinical_retrain_v1_fix1/caches/moca_binary/35f9380da53cf6de440d20375a43f256d3fbf3341ed18165908df9e0ab7adf6f_val.pt", "reused": false} |
| output_sha256 | {"importance_bootstrap_intervals.csv": "7d868c7aa52b29e5d49773065c74c59d0189310ac785dace5517ddc0e90fe99e", "importance_cohort_counts.csv": "3feaca473c731b84d5effdee3645cb5ae9083dc4004fb04fec2fb6501de80053", "importance_paired_predictions.csv": "ac242a1b461a48e804aff9521b60af51c0be49fa3792f56c3ba46e37b695cce6", "importance_summary.csv": "3d022caf69366c4d6a943ab5e7c3cfd1f00efd2edac8588ffa815fe85725786b"} |
## CVD broad details

Validation N=352; bootstrap samples=2000; threshold=0.47645785.

| Deletion | Metric | Baseline | After deletion | Degradation [95% CI] |
|---|---|---:|---:|---|
| clinical_data | auroc | 0.6934 | 0.6925 | +0.0009 [-0.0455, +0.0465] |
| clinical_data | auprc | 0.8861 | 0.8850 | +0.0011 [-0.0177, +0.0200] |
| clinical_data | balanced_accuracy | 0.6525 | 0.6352 | +0.0173 [-0.0413, +0.0731] |
| clinical_data | sensitivity | 0.7153 | 0.6679 | +0.0474 [-0.0036, +0.0985] |
| clinical_data | specificity | 0.5897 | 0.6026 | -0.0128 [-0.1154, +0.0897] |
| clinical_data | log_loss | 0.6448 | 0.6374 | -0.0074 [-0.0405, +0.0248] |
| clinical_data | brier | 0.2247 | 0.2239 | -0.0008 [-0.0148, +0.0130] |
| cardiac_ecg | auroc | 0.6934 | 0.6551 | +0.0383 [+0.0141, +0.0621] |
| cardiac_ecg | auprc | 0.8861 | 0.8636 | +0.0224 [+0.0073, +0.0381] |
| cardiac_ecg | balanced_accuracy | 0.6525 | 0.5900 | +0.0625 [+0.0102, +0.1165] |
| cardiac_ecg | sensitivity | 0.7153 | 0.8467 | -0.1314 [-0.1752, -0.0912] |
| cardiac_ecg | specificity | 0.5897 | 0.3333 | +0.2564 [+0.1667, +0.3462] |
| cardiac_ecg | log_loss | 0.6448 | 0.5659 | -0.0790 [-0.1020, -0.0562] |
| cardiac_ecg | brier | 0.2247 | 0.1894 | -0.0353 [-0.0455, -0.0249] |
| environment | auroc | 0.6934 | 0.6902 | +0.0032 [-0.0046, +0.0111] |
| environment | auprc | 0.8861 | 0.8854 | +0.0006 [-0.0045, +0.0058] |
| environment | balanced_accuracy | 0.6525 | 0.6113 | +0.0413 [+0.0065, +0.0788] |
| environment | sensitivity | 0.7153 | 0.7482 | -0.0328 [-0.0584, -0.0146] |
| environment | specificity | 0.5897 | 0.4744 | +0.1154 [+0.0513, +0.1923] |
| environment | log_loss | 0.6448 | 0.6138 | -0.0310 [-0.0371, -0.0247] |
| environment | brier | 0.2247 | 0.2114 | -0.0133 [-0.0162, -0.0105] |
| retinal_flio | auroc | 0.6934 | 0.6936 | -0.0001 [-0.0125, +0.0120] |
| retinal_flio | auprc | 0.8861 | 0.8885 | -0.0024 [-0.0106, +0.0050] |
| retinal_flio | balanced_accuracy | 0.6525 | 0.6425 | +0.0100 [-0.0165, +0.0347] |
| retinal_flio | sensitivity | 0.7153 | 0.6825 | +0.0328 [+0.0073, +0.0584] |
| retinal_flio | specificity | 0.5897 | 0.6026 | -0.0128 [-0.0516, +0.0256] |
| retinal_flio | log_loss | 0.6448 | 0.6642 | +0.0194 [+0.0113, +0.0278] |
| retinal_flio | brier | 0.2247 | 0.2335 | +0.0088 [+0.0052, +0.0127] |
| retinal_oct | auroc | 0.6934 | 0.6904 | +0.0030 [-0.0045, +0.0102] |
| retinal_oct | auprc | 0.8861 | 0.8838 | +0.0022 [-0.0028, +0.0074] |
| retinal_oct | balanced_accuracy | 0.6525 | 0.5993 | +0.0532 [+0.0102, +0.0991] |
| retinal_oct | sensitivity | 0.7153 | 0.7883 | -0.0730 [-0.1058, -0.0438] |
| retinal_oct | specificity | 0.5897 | 0.4103 | +0.1795 [+0.1026, +0.2692] |
| retinal_oct | log_loss | 0.6448 | 0.5744 | -0.0704 [-0.0815, -0.0593] |
| retinal_oct | brier | 0.2247 | 0.1944 | -0.0303 [-0.0354, -0.0254] |
| retinal_octa | auroc | 0.6934 | 0.6924 | +0.0011 [-0.0024, +0.0043] |
| retinal_octa | auprc | 0.8861 | 0.8856 | +0.0005 [-0.0017, +0.0028] |
| retinal_octa | balanced_accuracy | 0.6525 | 0.6223 | +0.0303 [+0.0000, +0.0641] |
| retinal_octa | sensitivity | 0.7153 | 0.7445 | -0.0292 [-0.0511, -0.0109] |
| retinal_octa | specificity | 0.5897 | 0.5000 | +0.0897 [+0.0385, +0.1538] |
| retinal_octa | log_loss | 0.6448 | 0.6221 | -0.0227 [-0.0268, -0.0187] |
| retinal_octa | brier | 0.2247 | 0.2153 | -0.0094 [-0.0112, -0.0077] |
| retinal_photography | auroc | 0.6934 | 0.6904 | +0.0030 [-0.0007, +0.0070] |
| retinal_photography | auprc | 0.8861 | 0.8827 | +0.0034 [+0.0001, +0.0078] |
| retinal_photography | balanced_accuracy | 0.6525 | 0.6204 | +0.0321 [+0.0019, +0.0660] |
| retinal_photography | sensitivity | 0.7153 | 0.7409 | -0.0255 [-0.0474, -0.0073] |
| retinal_photography | specificity | 0.5897 | 0.5000 | +0.0897 [+0.0385, +0.1538] |
| retinal_photography | log_loss | 0.6448 | 0.6164 | -0.0285 [-0.0340, -0.0232] |
| retinal_photography | brier | 0.2247 | 0.2128 | -0.0119 [-0.0142, -0.0097] |
| wearable_activity_monitor | auroc | 0.6934 | 0.6722 | +0.0213 [-0.0011, +0.0437] |
| wearable_activity_monitor | auprc | 0.8861 | 0.8705 | +0.0155 [+0.0026, +0.0285] |
| wearable_activity_monitor | balanced_accuracy | 0.6525 | 0.6050 | +0.0475 [+0.0127, +0.0859] |
| wearable_activity_monitor | sensitivity | 0.7153 | 0.6460 | +0.0693 [+0.0365, +0.1058] |
| wearable_activity_monitor | specificity | 0.5897 | 0.5641 | +0.0256 [-0.0385, +0.0897] |
| wearable_activity_monitor | log_loss | 0.6448 | 0.6816 | +0.0368 [+0.0183, +0.0543] |
| wearable_activity_monitor | brier | 0.2247 | 0.2402 | +0.0155 [+0.0074, +0.0233] |
| wearable_blood_glucose | auroc | 0.6934 | 0.6929 | +0.0005 [-0.0113, +0.0124] |
| wearable_blood_glucose | auprc | 0.8861 | 0.8869 | -0.0008 [-0.0076, +0.0059] |
| wearable_blood_glucose | balanced_accuracy | 0.6525 | 0.6297 | +0.0228 [-0.0138, +0.0585] |
| wearable_blood_glucose | sensitivity | 0.7153 | 0.6825 | +0.0328 [+0.0073, +0.0620] |
| wearable_blood_glucose | specificity | 0.5897 | 0.5769 | +0.0128 [-0.0513, +0.0769] |
| wearable_blood_glucose | log_loss | 0.6448 | 0.6633 | +0.0184 [+0.0101, +0.0267] |
| wearable_blood_glucose | brier | 0.2247 | 0.2324 | +0.0077 [+0.0041, +0.0114] |
| retinal_all | auroc | 0.6934 | 0.6883 | +0.0051 [-0.0116, +0.0225] |
| retinal_all | auprc | 0.8861 | 0.8815 | +0.0046 [-0.0074, +0.0169] |
| retinal_all | balanced_accuracy | 0.6525 | 0.6147 | +0.0378 [-0.0098, +0.0882] |
| retinal_all | sensitivity | 0.7153 | 0.8577 | -0.1423 [-0.1826, -0.1022] |
| retinal_all | specificity | 0.5897 | 0.3718 | +0.2179 [+0.1282, +0.3077] |
| retinal_all | log_loss | 0.6448 | 0.5441 | -0.1008 [-0.1210, -0.0806] |
| retinal_all | brier | 0.2247 | 0.1813 | -0.0434 [-0.0522, -0.0348] |
| wearable_all | auroc | 0.6934 | 0.6699 | +0.0235 [-0.0012, +0.0500] |
| wearable_all | auprc | 0.8861 | 0.8721 | +0.0139 [+0.0013, +0.0274] |
| wearable_all | balanced_accuracy | 0.6525 | 0.6170 | +0.0356 [-0.0038, +0.0740] |
| wearable_all | sensitivity | 0.7153 | 0.6314 | +0.0839 [+0.0474, +0.1241] |
| wearable_all | specificity | 0.5897 | 0.6026 | -0.0128 [-0.0769, +0.0513] |
| wearable_all | log_loss | 0.6448 | 0.7006 | +0.0558 [+0.0371, +0.0753] |
| wearable_all | brier | 0.2247 | 0.2483 | +0.0236 [+0.0154, +0.0322] |
| nonclinical_all | auroc | 0.6934 | 0.6207 | +0.0727 [+0.0238, +0.1207] |
| nonclinical_all | auprc | 0.8861 | 0.8360 | +0.0500 [+0.0182, +0.0803] |
| nonclinical_all | balanced_accuracy | 0.6525 | 0.5000 | +0.1525 [+0.0911, +0.2129] |
| nonclinical_all | sensitivity | 0.7153 | 1.0000 | -0.2847 [-0.3394, -0.2299] |
| nonclinical_all | specificity | 0.5897 | 0.0000 | +0.5897 [+0.4744, +0.6923] |
| nonclinical_all | log_loss | 0.6448 | 0.5225 | -0.1224 [-0.1658, -0.0790] |
| nonclinical_all | brier | 0.2247 | 0.1697 | -0.0550 [-0.0745, -0.0357] |

### Availability strata

| Deletion | Cohort | N | Positive / negative | No remaining input | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---|
| clinical_data | all_participants | 352 | 274 / 78 | 0 | +0.0009 [-0.0455, +0.0465] |
| clinical_data | any_target_present | 352 | 274 / 78 | 0 | +0.0009 [-0.0455, +0.0465] |
| clinical_data | all_targets_present | 352 | 274 / 78 | 0 | +0.0009 [-0.0455, +0.0465] |
| clinical_data | any_target_present_with_remaining_input | 352 | 274 / 78 | 0 | +0.0009 [-0.0455, +0.0465] |
| cardiac_ecg | all_participants | 352 | 274 / 78 | 0 | +0.0383 [+0.0141, +0.0621] |
| cardiac_ecg | any_target_present | 347 | 270 / 77 | 0 | +0.0393 [+0.0152, +0.0649] |
| cardiac_ecg | all_targets_present | 347 | 270 / 77 | 0 | +0.0393 [+0.0152, +0.0649] |
| cardiac_ecg | any_target_present_with_remaining_input | 347 | 270 / 77 | 0 | +0.0393 [+0.0152, +0.0649] |
| environment | all_participants | 352 | 274 / 78 | 0 | +0.0032 [-0.0046, +0.0111] |
| environment | any_target_present | 344 | 269 / 75 | 0 | +0.0059 [-0.0016, +0.0140] |
| environment | all_targets_present | 344 | 269 / 75 | 0 | +0.0059 [-0.0016, +0.0140] |
| environment | any_target_present_with_remaining_input | 344 | 269 / 75 | 0 | +0.0059 [-0.0016, +0.0140] |
| retinal_flio | all_participants | 352 | 274 / 78 | 0 | -0.0001 [-0.0125, +0.0120] |
| retinal_flio | any_target_present | 293 | 228 / 65 | 0 | +0.0001 [-0.0154, +0.0142] |
| retinal_flio | all_targets_present | 293 | 228 / 65 | 0 | +0.0001 [-0.0154, +0.0142] |
| retinal_flio | any_target_present_with_remaining_input | 293 | 228 / 65 | 0 | +0.0001 [-0.0154, +0.0142] |
| retinal_oct | all_participants | 352 | 274 / 78 | 0 | +0.0030 [-0.0045, +0.0102] |
| retinal_oct | any_target_present | 350 | 274 / 76 | 0 | +0.0051 [-0.0018, +0.0119] |
| retinal_oct | all_targets_present | 350 | 274 / 76 | 0 | +0.0051 [-0.0018, +0.0119] |
| retinal_oct | any_target_present_with_remaining_input | 350 | 274 / 76 | 0 | +0.0051 [-0.0018, +0.0119] |
| retinal_octa | all_participants | 352 | 274 / 78 | 0 | +0.0011 [-0.0024, +0.0043] |
| retinal_octa | any_target_present | 350 | 274 / 76 | 0 | +0.0015 [-0.0016, +0.0050] |
| retinal_octa | all_targets_present | 350 | 274 / 76 | 0 | +0.0015 [-0.0016, +0.0050] |
| retinal_octa | any_target_present_with_remaining_input | 350 | 274 / 76 | 0 | +0.0015 [-0.0016, +0.0050] |
| retinal_photography | all_participants | 352 | 274 / 78 | 0 | +0.0030 [-0.0007, +0.0070] |
| retinal_photography | any_target_present | 352 | 274 / 78 | 0 | +0.0030 [-0.0007, +0.0070] |
| retinal_photography | all_targets_present | 352 | 274 / 78 | 0 | +0.0030 [-0.0007, +0.0070] |
| retinal_photography | any_target_present_with_remaining_input | 352 | 274 / 78 | 0 | +0.0030 [-0.0007, +0.0070] |
| wearable_activity_monitor | all_participants | 352 | 274 / 78 | 0 | +0.0213 [-0.0011, +0.0437] |
| wearable_activity_monitor | any_target_present | 320 | 251 / 69 | 0 | +0.0189 [-0.0069, +0.0421] |
| wearable_activity_monitor | all_targets_present | 320 | 251 / 69 | 0 | +0.0189 [-0.0069, +0.0421] |
| wearable_activity_monitor | any_target_present_with_remaining_input | 320 | 251 / 69 | 0 | +0.0189 [-0.0069, +0.0421] |
| wearable_blood_glucose | all_participants | 352 | 274 / 78 | 0 | +0.0005 [-0.0113, +0.0124] |
| wearable_blood_glucose | any_target_present | 344 | 269 / 75 | 0 | +0.0008 [-0.0122, +0.0135] |
| wearable_blood_glucose | all_targets_present | 344 | 269 / 75 | 0 | +0.0008 [-0.0122, +0.0135] |
| wearable_blood_glucose | any_target_present_with_remaining_input | 344 | 269 / 75 | 0 | +0.0008 [-0.0122, +0.0135] |
| retinal_all | all_participants | 352 | 274 / 78 | 0 | +0.0051 [-0.0116, +0.0225] |
| retinal_all | any_target_present | 352 | 274 / 78 | 0 | +0.0051 [-0.0116, +0.0225] |
| retinal_all | all_targets_present | 293 | 228 / 65 | 0 | +0.0057 [-0.0135, +0.0259] |
| retinal_all | any_target_present_with_remaining_input | 352 | 274 / 78 | 0 | +0.0051 [-0.0116, +0.0225] |
| wearable_all | all_participants | 352 | 274 / 78 | 0 | +0.0235 [-0.0012, +0.0500] |
| wearable_all | any_target_present | 347 | 272 / 75 | 0 | +0.0226 [-0.0026, +0.0498] |
| wearable_all | all_targets_present | 317 | 248 / 69 | 0 | +0.0260 [+0.0003, +0.0543] |
| wearable_all | any_target_present_with_remaining_input | 347 | 272 / 75 | 0 | +0.0226 [-0.0026, +0.0498] |
| nonclinical_all | all_participants | 352 | 274 / 78 | 0 | +0.0727 [+0.0238, +0.1207] |
| nonclinical_all | any_target_present | 352 | 274 / 78 | 0 | +0.0727 [+0.0238, +0.1207] |
| nonclinical_all | all_targets_present | 257 | 199 / 58 | 0 | +0.0926 [+0.0364, +0.1487] |
| nonclinical_all | any_target_present_with_remaining_input | 352 | 274 / 78 | 0 | +0.0727 [+0.0238, +0.1207] |

### Exact retained Clinical columns

| Encoded column | Table | Source key |
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
| checkpoint_sha256 | "35fcd9f67b6cfaddfcd9124b93ff796d04c436041ad6bab32c6fb115f2995d12" |
| config_hash | "26623239cf639c691b40ba63553b0f3fd63811a654756af35583d6e6a5c3888f" |
| prepared_sha256 | "848aabef63286eda7453dfe04c7a9de9548ea919ba86fcdfc67d1ce215757558" |
| trained_sha256 | "7a2050071e1211a010241661ca139a8497c79efcd5a955da359f14f43bbf6327" |
| basic_plan_sha256 | "79a1648b6f6c457cd42394bbac004b6f3eb60549b11716cbe5c945ca4c88f217" |
| threshold_source | "this_retrained_models_validation_selected_threshold" |
| source_token_cache | {"content_sha256": "74f6a0b51ff332d68e2ce73751473088422578feaad32e0bc2d68bc73faa6dde", "file_sha256": "8561d2af252ca7a8558794aebd336f8eaba3cd8e16940a748d9d578b0362a5a3", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "35fcd9f67b6cfaddfcd9124b93ff796d04c436041ad6bab32c6fb115f2995d12", "cohort_person_ids_sha256": "648b808a65bc627c36c062fbb849f026788445cdc5eab7b699b88d1bdf980287", "config_hash": "26623239cf639c691b40ba63553b0f3fd63811a654756af35583d6e6a5c3888f", "encoder_batch_size": 2, "master_config_hash": "a6a28e10b779d4e5f0519eb94b2f7022ce39dda44751c9f4457c9b7baa6003e9", "model_config_hash": "cb1dd473b8dd2fae1dffbc652f1367c26e0d547bdd2a4dc384833321017f6186", "n_participants": 352, "scaler_config_hash": "7d3633200aa88114037a4ea96fbb4bdef20bbefedad83580fcfd7c07bf5025d8", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "cvd_broad_binary"}, "path": "/data/pzhao3/ot2_output/model_basic_clinical_retrain_v1_fix1/caches/cvd_broad_binary/35fcd9f67b6cfaddfcd9124b93ff796d04c436041ad6bab32c6fb115f2995d12_val.pt", "reused": false} |
| output_sha256 | {"importance_bootstrap_intervals.csv": "341706bcd11600d252d6030014225c205e9ebc988f6f51f231889b87e0e594af", "importance_cohort_counts.csv": "163e4c08d75e24d6938f1a258c2cb6f6836c3a9df0cf85a2a3432163027325d9", "importance_paired_predictions.csv": "ff6e521c5fc008170b8c9e7661bffbef4af82ecaf6c715b2623a10214646b97f", "importance_summary.csv": "3edfd2b678fc57f5efa8716a69a5364b17ef2bc433e3f69e496324d48ff5ad71"} |
## CVD strict details

Validation N=351; bootstrap samples=2000; threshold=0.47426592.

| Deletion | Metric | Baseline | After deletion | Degradation [95% CI] |
|---|---|---:|---:|---|
| clinical_data | auroc | 0.6231 | 0.6182 | +0.0049 [-0.0295, +0.0401] |
| clinical_data | auprc | 0.4094 | 0.4029 | +0.0065 [-0.0260, +0.0407] |
| clinical_data | balanced_accuracy | 0.6118 | 0.5795 | +0.0323 [-0.0079, +0.0732] |
| clinical_data | sensitivity | 0.6364 | 0.6869 | -0.0505 [-0.1111, +0.0101] |
| clinical_data | specificity | 0.5873 | 0.4722 | +0.1151 [+0.0635, +0.1667] |
| clinical_data | log_loss | 0.7274 | 0.7603 | +0.0329 [+0.0009, +0.0646] |
| clinical_data | brier | 0.2593 | 0.2744 | +0.0151 [+0.0014, +0.0288] |
| cardiac_ecg | auroc | 0.6231 | 0.5984 | +0.0247 [-0.0001, +0.0507] |
| cardiac_ecg | auprc | 0.4094 | 0.3783 | +0.0312 [-0.0211, +0.0844] |
| cardiac_ecg | balanced_accuracy | 0.6118 | 0.5759 | +0.0359 [-0.0013, +0.0720] |
| cardiac_ecg | sensitivity | 0.6364 | 0.6162 | +0.0202 [-0.0404, +0.0808] |
| cardiac_ecg | specificity | 0.5873 | 0.5357 | +0.0516 [+0.0159, +0.0873] |
| cardiac_ecg | log_loss | 0.7274 | 0.7555 | +0.0281 [+0.0039, +0.0509] |
| cardiac_ecg | brier | 0.2593 | 0.2726 | +0.0132 [+0.0033, +0.0229] |
| environment | auroc | 0.6231 | 0.6213 | +0.0019 [-0.0119, +0.0154] |
| environment | auprc | 0.4094 | 0.4034 | +0.0060 [-0.0115, +0.0234] |
| environment | balanced_accuracy | 0.6118 | 0.5907 | +0.0211 [-0.0103, +0.0550] |
| environment | sensitivity | 0.6364 | 0.6061 | +0.0303 [-0.0303, +0.0909] |
| environment | specificity | 0.5873 | 0.5754 | +0.0119 [-0.0119, +0.0357] |
| environment | log_loss | 0.7274 | 0.7350 | +0.0075 [-0.0038, +0.0189] |
| environment | brier | 0.2593 | 0.2625 | +0.0032 [-0.0018, +0.0080] |
| retinal_flio | auroc | 0.6231 | 0.6332 | -0.0101 [-0.0238, +0.0038] |
| retinal_flio | auprc | 0.4094 | 0.4208 | -0.0113 [-0.0289, +0.0090] |
| retinal_flio | balanced_accuracy | 0.6118 | 0.6095 | +0.0023 [-0.0366, +0.0456] |
| retinal_flio | sensitivity | 0.6364 | 0.4848 | +0.1515 [+0.0808, +0.2222] |
| retinal_flio | specificity | 0.5873 | 0.7341 | -0.1468 [-0.1905, -0.1032] |
| retinal_flio | log_loss | 0.7274 | 0.6418 | -0.0856 [-0.1038, -0.0677] |
| retinal_flio | brier | 0.2593 | 0.2231 | -0.0362 [-0.0442, -0.0280] |
| retinal_oct | auroc | 0.6231 | 0.6218 | +0.0013 [-0.0054, +0.0085] |
| retinal_oct | auprc | 0.4094 | 0.4108 | -0.0013 [-0.0168, +0.0151] |
| retinal_oct | balanced_accuracy | 0.6118 | 0.5815 | +0.0303 [+0.0011, +0.0574] |
| retinal_oct | sensitivity | 0.6364 | 0.6869 | -0.0505 [-0.1010, -0.0101] |
| retinal_oct | specificity | 0.5873 | 0.4762 | +0.1111 [+0.0754, +0.1508] |
| retinal_oct | log_loss | 0.7274 | 0.7973 | +0.0698 [+0.0594, +0.0809] |
| retinal_oct | brier | 0.2593 | 0.2878 | +0.0285 [+0.0237, +0.0335] |
| retinal_octa | auroc | 0.6231 | 0.6265 | -0.0034 [-0.0108, +0.0036] |
| retinal_octa | auprc | 0.4094 | 0.4176 | -0.0082 [-0.0194, +0.0052] |
| retinal_octa | balanced_accuracy | 0.6118 | 0.5804 | +0.0314 [+0.0043, +0.0574] |
| retinal_octa | sensitivity | 0.6364 | 0.6768 | -0.0404 [-0.0808, -0.0101] |
| retinal_octa | specificity | 0.5873 | 0.4841 | +0.1032 [+0.0675, +0.1429] |
| retinal_octa | log_loss | 0.7274 | 0.7785 | +0.0510 [+0.0421, +0.0600] |
| retinal_octa | brier | 0.2593 | 0.2808 | +0.0215 [+0.0174, +0.0259] |
| retinal_photography | auroc | 0.6231 | 0.6263 | -0.0032 [-0.0084, +0.0021] |
| retinal_photography | auprc | 0.4094 | 0.4115 | -0.0020 [-0.0109, +0.0104] |
| retinal_photography | balanced_accuracy | 0.6118 | 0.5934 | +0.0184 [-0.0058, +0.0447] |
| retinal_photography | sensitivity | 0.6364 | 0.5758 | +0.0606 [+0.0202, +0.1111] |
| retinal_photography | specificity | 0.5873 | 0.6111 | -0.0238 [-0.0437, -0.0079] |
| retinal_photography | log_loss | 0.7274 | 0.7120 | -0.0154 [-0.0209, -0.0100] |
| retinal_photography | brier | 0.2593 | 0.2527 | -0.0067 [-0.0091, -0.0042] |
| wearable_activity_monitor | auroc | 0.6231 | 0.6019 | +0.0212 [-0.0064, +0.0487] |
| wearable_activity_monitor | auprc | 0.4094 | 0.3967 | +0.0127 [-0.0198, +0.0464] |
| wearable_activity_monitor | balanced_accuracy | 0.6118 | 0.5765 | +0.0354 [-0.0070, +0.0768] |
| wearable_activity_monitor | sensitivity | 0.6364 | 0.6768 | -0.0404 [-0.1111, +0.0303] |
| wearable_activity_monitor | specificity | 0.5873 | 0.4762 | +0.1111 [+0.0675, +0.1548] |
| wearable_activity_monitor | log_loss | 0.7274 | 0.8117 | +0.0842 [+0.0555, +0.1142] |
| wearable_activity_monitor | brier | 0.2593 | 0.2942 | +0.0349 [+0.0222, +0.0478] |
| wearable_blood_glucose | auroc | 0.6231 | 0.6140 | +0.0091 [-0.0107, +0.0298] |
| wearable_blood_glucose | auprc | 0.4094 | 0.4057 | +0.0037 [-0.0162, +0.0222] |
| wearable_blood_glucose | balanced_accuracy | 0.6118 | 0.5615 | +0.0503 [+0.0018, +0.1019] |
| wearable_blood_glucose | sensitivity | 0.6364 | 0.3333 | +0.3030 [+0.2222, +0.3939] |
| wearable_blood_glucose | specificity | 0.5873 | 0.7897 | -0.2024 [-0.2540, -0.1508] |
| wearable_blood_glucose | log_loss | 0.7274 | 0.6323 | -0.0951 [-0.1204, -0.0697] |
| wearable_blood_glucose | brier | 0.2593 | 0.2188 | -0.0405 [-0.0522, -0.0288] |
| retinal_all | auroc | 0.6231 | 0.6390 | -0.0159 [-0.0326, -0.0003] |
| retinal_all | auprc | 0.4094 | 0.4339 | -0.0244 [-0.0441, -0.0039] |
| retinal_all | balanced_accuracy | 0.6118 | 0.5945 | +0.0173 [-0.0126, +0.0462] |
| retinal_all | sensitivity | 0.6364 | 0.6970 | -0.0606 [-0.1111, -0.0202] |
| retinal_all | specificity | 0.5873 | 0.4921 | +0.0952 [+0.0635, +0.1311] |
| retinal_all | log_loss | 0.7274 | 0.7431 | +0.0156 [+0.0021, +0.0313] |
| retinal_all | brier | 0.2593 | 0.2676 | +0.0083 [+0.0021, +0.0152] |
| wearable_all | auroc | 0.6231 | 0.5891 | +0.0340 [-0.0000, +0.0685] |
| wearable_all | auprc | 0.4094 | 0.3938 | +0.0157 [-0.0180, +0.0505] |
| wearable_all | balanced_accuracy | 0.6118 | 0.5601 | +0.0518 [-0.0025, +0.1054] |
| wearable_all | sensitivity | 0.6364 | 0.3939 | +0.2424 [+0.1515, +0.3333] |
| wearable_all | specificity | 0.5873 | 0.7262 | -0.1389 [-0.1905, -0.0912] |
| wearable_all | log_loss | 0.7274 | 0.6653 | -0.0621 [-0.0951, -0.0279] |
| wearable_all | brier | 0.2593 | 0.2328 | -0.0265 [-0.0413, -0.0109] |
| nonclinical_all | auroc | 0.6231 | 0.5739 | +0.0493 [+0.0010, +0.0981] |
| nonclinical_all | auprc | 0.4094 | 0.3340 | +0.0754 [+0.0152, +0.1411] |
| nonclinical_all | balanced_accuracy | 0.6118 | 0.5447 | +0.0671 [+0.0058, +0.1313] |
| nonclinical_all | sensitivity | 0.6364 | 0.4545 | +0.1818 [+0.0707, +0.2929] |
| nonclinical_all | specificity | 0.5873 | 0.6349 | -0.0476 [-0.1071, +0.0119] |
| nonclinical_all | log_loss | 0.7274 | 0.6583 | -0.0691 [-0.1129, -0.0283] |
| nonclinical_all | brier | 0.2593 | 0.2332 | -0.0261 [-0.0441, -0.0083] |

### Availability strata

| Deletion | Cohort | N | Positive / negative | No remaining input | AUROC degradation [95% CI] |
|---|---|---:|---|---:|---|
| clinical_data | all_participants | 351 | 99 / 252 | 0 | +0.0049 [-0.0295, +0.0401] |
| clinical_data | any_target_present | 351 | 99 / 252 | 0 | +0.0049 [-0.0295, +0.0401] |
| clinical_data | all_targets_present | 351 | 99 / 252 | 0 | +0.0049 [-0.0295, +0.0401] |
| clinical_data | any_target_present_with_remaining_input | 351 | 99 / 252 | 0 | +0.0049 [-0.0295, +0.0401] |
| cardiac_ecg | all_participants | 351 | 99 / 252 | 0 | +0.0247 [-0.0001, +0.0507] |
| cardiac_ecg | any_target_present | 346 | 99 / 247 | 0 | +0.0253 [+0.0003, +0.0521] |
| cardiac_ecg | all_targets_present | 346 | 99 / 247 | 0 | +0.0253 [+0.0003, +0.0521] |
| cardiac_ecg | any_target_present_with_remaining_input | 346 | 99 / 247 | 0 | +0.0253 [+0.0003, +0.0521] |
| environment | all_participants | 351 | 99 / 252 | 0 | +0.0019 [-0.0119, +0.0154] |
| environment | any_target_present | 343 | 99 / 244 | 0 | +0.0020 [-0.0110, +0.0150] |
| environment | all_targets_present | 343 | 99 / 244 | 0 | +0.0020 [-0.0110, +0.0150] |
| environment | any_target_present_with_remaining_input | 343 | 99 / 244 | 0 | +0.0020 [-0.0110, +0.0150] |
| retinal_flio | all_participants | 351 | 99 / 252 | 0 | -0.0101 [-0.0238, +0.0038] |
| retinal_flio | any_target_present | 292 | 79 / 213 | 0 | -0.0025 [-0.0127, +0.0082] |
| retinal_flio | all_targets_present | 292 | 79 / 213 | 0 | -0.0025 [-0.0127, +0.0082] |
| retinal_flio | any_target_present_with_remaining_input | 292 | 79 / 213 | 0 | -0.0025 [-0.0127, +0.0082] |
| retinal_oct | all_participants | 351 | 99 / 252 | 0 | +0.0013 [-0.0054, +0.0085] |
| retinal_oct | any_target_present | 349 | 99 / 250 | 0 | +0.0020 [-0.0054, +0.0086] |
| retinal_oct | all_targets_present | 349 | 99 / 250 | 0 | +0.0020 [-0.0054, +0.0086] |
| retinal_oct | any_target_present_with_remaining_input | 349 | 99 / 250 | 0 | +0.0020 [-0.0054, +0.0086] |
| retinal_octa | all_participants | 351 | 99 / 252 | 0 | -0.0034 [-0.0108, +0.0036] |
| retinal_octa | any_target_present | 349 | 99 / 250 | 0 | -0.0029 [-0.0099, +0.0041] |
| retinal_octa | all_targets_present | 349 | 99 / 250 | 0 | -0.0029 [-0.0099, +0.0041] |
| retinal_octa | any_target_present_with_remaining_input | 349 | 99 / 250 | 0 | -0.0029 [-0.0099, +0.0041] |
| retinal_photography | all_participants | 351 | 99 / 252 | 0 | -0.0032 [-0.0084, +0.0021] |
| retinal_photography | any_target_present | 351 | 99 / 252 | 0 | -0.0032 [-0.0084, +0.0021] |
| retinal_photography | all_targets_present | 351 | 99 / 252 | 0 | -0.0032 [-0.0084, +0.0021] |
| retinal_photography | any_target_present_with_remaining_input | 351 | 99 / 252 | 0 | -0.0032 [-0.0084, +0.0021] |
| wearable_activity_monitor | all_participants | 351 | 99 / 252 | 0 | +0.0212 [-0.0064, +0.0487] |
| wearable_activity_monitor | any_target_present | 319 | 95 / 224 | 0 | +0.0282 [-0.0004, +0.0556] |
| wearable_activity_monitor | all_targets_present | 319 | 95 / 224 | 0 | +0.0282 [-0.0004, +0.0556] |
| wearable_activity_monitor | any_target_present_with_remaining_input | 319 | 95 / 224 | 0 | +0.0282 [-0.0004, +0.0556] |
| wearable_blood_glucose | all_participants | 351 | 99 / 252 | 0 | +0.0091 [-0.0107, +0.0298] |
| wearable_blood_glucose | any_target_present | 343 | 97 / 246 | 0 | +0.0094 [-0.0100, +0.0293] |
| wearable_blood_glucose | all_targets_present | 343 | 97 / 246 | 0 | +0.0094 [-0.0100, +0.0293] |
| wearable_blood_glucose | any_target_present_with_remaining_input | 343 | 97 / 246 | 0 | +0.0094 [-0.0100, +0.0293] |
| retinal_all | all_participants | 351 | 99 / 252 | 0 | -0.0159 [-0.0326, -0.0003] |
| retinal_all | any_target_present | 351 | 99 / 252 | 0 | -0.0159 [-0.0326, -0.0003] |
| retinal_all | all_targets_present | 292 | 79 / 213 | 0 | -0.0048 [-0.0174, +0.0069] |
| retinal_all | any_target_present_with_remaining_input | 351 | 99 / 252 | 0 | -0.0159 [-0.0326, -0.0003] |
| wearable_all | all_participants | 351 | 99 / 252 | 0 | +0.0340 [-0.0000, +0.0685] |
| wearable_all | any_target_present | 346 | 99 / 247 | 0 | +0.0317 [+0.0001, +0.0647] |
| wearable_all | all_targets_present | 316 | 93 / 223 | 0 | +0.0458 [+0.0101, +0.0798] |
| wearable_all | any_target_present_with_remaining_input | 346 | 99 / 247 | 0 | +0.0317 [+0.0001, +0.0647] |
| nonclinical_all | all_participants | 351 | 99 / 252 | 0 | +0.0493 [+0.0010, +0.0981] |
| nonclinical_all | any_target_present | 351 | 99 / 252 | 0 | +0.0493 [+0.0010, +0.0981] |
| nonclinical_all | all_targets_present | 256 | 75 / 181 | 0 | +0.0746 [+0.0211, +0.1294] |
| nonclinical_all | any_target_present_with_remaining_input | 351 | 99 / 252 | 0 | +0.0493 [+0.0010, +0.0981] |

### Exact retained Clinical columns

| Encoded column | Table | Source key |
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
| checkpoint_sha256 | "265a20aa6b6a039e4276826fb43a1ad78fb2d042414c930a07b4850acd7ad914" |
| config_hash | "24c9bfbbe0a655eb82ed808070c57c6d03f5164898d4bfc1d997a27d7a7d57fc" |
| prepared_sha256 | "11cace5f31af05500fd394f956a24ee91a9b98528f0c5be08d9d3ee6c9bb6686" |
| trained_sha256 | "dc8cc795e3ce3b4a7668b06fa4aa17a411c2083b69e4deabefeda4c4a45d83f4" |
| basic_plan_sha256 | "1715f7984a787f8b0edf88fadc30dca2991a19b00faba6a9aa40dcbbc3aeb69a" |
| threshold_source | "this_retrained_models_validation_selected_threshold" |
| source_token_cache | {"content_sha256": "862ef8f9f653e7db8b7578385d0164ce1a8123245e1236b0d21dd4b93583c3c2", "file_sha256": "6cff0b9302de0418b0a36055708af00ae32f2f0d9d785f5b6e04800fbe1538ef", "identity": {"amp_mode": "bfloat16_autocast", "checkpoint_sha256": "265a20aa6b6a039e4276826fb43a1ad78fb2d042414c930a07b4850acd7ad914", "cohort_person_ids_sha256": "396f9a4710d5d75770040275fa0e2b63c92ef3b3b4ebbe20694e0476a4e12241", "config_hash": "24c9bfbbe0a655eb82ed808070c57c6d03f5164898d4bfc1d997a27d7a7d57fc", "encoder_batch_size": 2, "master_config_hash": "37bfced0213962e9d988f897f8733d0722403cb2a2e3fceabad912e4deba5e52", "model_config_hash": "cb1dd473b8dd2fae1dffbc652f1367c26e0d547bdd2a4dc384833321017f6186", "n_participants": 351, "scaler_config_hash": "199cd41f95a121da5c4d473498f517e24ed90d2dfbf19fe67c5dcf189625ed80", "source_order": ["clinical_data", "cardiac_ecg", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography", "wearable_activity_monitor", "wearable_blood_glucose"], "split": "val", "task": "cvd_strict_binary"}, "path": "/data/pzhao3/ot2_output/model_basic_clinical_retrain_v1_fix1/caches/cvd_strict_binary/265a20aa6b6a039e4276826fb43a1ad78fb2d042414c930a07b4850acd7ad914_val.pt", "reused": false} |
| output_sha256 | {"importance_bootstrap_intervals.csv": "ca1841542d19f57c0c8a320f7b080dcd1ee1684ea214db6d6ddbf9cb768dbce6", "importance_cohort_counts.csv": "0e85e321553512d275ebd543c497c1526abd6a466f39ed56284d841d8d92c60e", "importance_paired_predictions.csv": "b6f20b4fb686932140c6c17cee723ccc1a9c725893c2812698e54408fb824ea8", "importance_summary.csv": "8d99032b1dc3ec730fb7ce6231656a3fb46e8287af27ed1a676e112bcbb03be5"} |

## Code snapshot

| File | SHA256 |
|---|---|
| analyze_basic_retrained_v1.py | 4737a936c52c2690ccca45d5b83e4c0f647c95ac5486153e6484bbf07b6fa9c3 |
| analyze_modality_importance_v2.py | b20ebe3b7b08f7597b8b03a71d3cd257cf735b8ceaee4a38de9d4e809c1f32e8 |
| basic_retrain_contract_v1.py | 29c5cf993c689fbc8adbe8ae37bb8de7a3785e8d3a2f7af496e3b8d18c16f490 |
| clinical_basic_information_v1.py | 77b6feef189b14b498716e0a2f75f7af330c5966e63d90a53258a28c0a89a259 |
| fit_train_scalers_v2.py | 1c8138a3a66ff6ad922f4e907df95c0b0e5c67487042963a37ed9fce52102419 |
| prepare_basic_clinical_training_v1.py | f1e302451b20b6f2a5c95a9d19e079d2f009a67fe5f2aa39cae7f19fe3e795a1 |
| preprocessing_v2.py | 0219f9a8e9c291f2b9c5091205e4476db73c17c6a232fee1bc789491e7b44e14 |
| run_basic_retrain_v1.py | 7469050094f87dd6f705b1b86e77ae455906c23833628ab91d89a77194df1a1c |
| summarize_basic_retrain_v1.py | 4d13c7dd7e4d1a8f95e983f797131c9cb2fe876f1f1ef3f5b8b17b96f5ce7ebc |
| task2_v2_config.py | 458848c8032a9d47fc54ca112043b2929b81e126cd80601c94ee20514e21ac06 |
| task2_v2_dataset.py | 24afc6b37058021b4f20bc17a50f16a5fe72e543768ac4365a3dc214fcb9fcef |
| task2_v2_engine.py | 63f8014d3d4c9fd7d7c2469d0496bf17245bfa9b8110503b39572d80a6286ac4 |
| task2_v2_model.py | 7f9a22f149306ef16412311e9f8d6275804fe19b66f03e2aedd031dd102c5463 |
| train_basic_clinical_v1.py | d37e5e8917bc2fa6238d02dc005257ab3675b49b72a49957c8a75acd01dcbe20 |
| train_task2_v2.py | 34cf798448a025cc8d49a69a4df4eec549d798ca6a004e1f7e5435a4b08444a0 |
