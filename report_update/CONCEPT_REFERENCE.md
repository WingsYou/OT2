## Appendix A. What the 12 concepts represent

All six tasks use the same original12 definitions: 10 continuous concepts and 2 categorical concepts, organized into 7 reporting groups. The 13th concept (CES-D-10 total) is not used.

| Concept group | Measurement / label definition | Source field(s) in data | Concept ID |
|---|---|---|---|
| BMI | Body mass index (kg/m²) | `bmi_vsorres` | 4245997 |
| Blood pressure | Systolic blood pressure (mmHg) | `bp1_sysbp_vsorres`, `bp2_sysbp_vsorres` | 3004249 |
| Blood pressure | Diastolic blood pressure (mmHg) | `bp1_diabp_vsorres`, `bp2_diabp_vsorres` | 3012888 |
| HbA1c | Glycated hemoglobin (%) | `import_hba1c` | 3004410 |
| Lipids | HDL cholesterol (mg/dL) | `import_hdl_cholesterol` | 3007070 |
| Lipids | LDL cholesterol (mg/dL) | `import_ldl_cholesterol` | 3028288 |
| Visual function | Right-eye photopic visual acuity (LogMAR) | `viaodpscore` | 2005200052 |
| Visual function | Left-eye photopic visual acuity (LogMAR) | `viaospscore` | 2005200053 |
| Visual function | Right-eye photopic log contrast sensitivity | `plcsodlog` | 2005200155 |
| Visual function | Left-eye photopic log contrast sensitivity | `plcsoslog` | 2005200156 |
| Activity | Self-reported activity: inactive / somewhat active / active | `dmlact` | 44808782 |
| Restless sleep | CES-D restless-sleep item: <1 / 1–2 / 3–4 / 5–7 days in the past week | `ces7` | 1761695 |

“Source field” names identify the variables used to extract the observed supervision labels; they are not downstream task labels. The IDs above identify the measurement or observation concept, not an answer code. The first 10 concepts come from `measurement.csv`; Activity and Restless sleep come from `observation.csv`.

Visual function contains four separately supervised concepts; Blood pressure and Lipids contain two each. Eligible BP1/BP2 recordings are combined within the recorded calendar date; right and left eyes remain separate. The joint-six group restores BMI, systolic BP, diastolic BP, HbA1c, HDL and LDL together.

Activity uses model classes 0/1/2 for inactive/somewhat active/active; the corresponding recorded scores are 0/5/10. Restless sleep uses classes 0/1/2/3 for the four frequency categories above. These are self-reported questionnaire items, not wearable-derived activity or an overall sleep-quality score. Unknown/refused labels are masked.

These measured labels supervise the concept predictor during fitting. Table 4 restores full-input **predicted** concept channels, not these measured values.

Definitions and source IDs follow the existing original12 experiment registry, cross-referenced to the [AI-READI v3 clinical/OMOP mapping](https://docs.aireadi.org/v3-omopAndClinicalTable).
