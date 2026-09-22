# Preserved input preparation

The new F1 experiment runner (`run.py`) uses existing processed inputs and the existing native/ResNet cache. **Do not rerun preprocessing for this study.** `data_preprocessing_cluster/` is preserved unchanged, including label definitions, clinical feature construction, scaler fitting and all nonclinical processing scripts.

The historical preparation helpers remain separate:

- `pipeline.py raw --dry-run` displays the existing preprocessing recipe. Actual regeneration requires the original manifests in `pipeline.yaml` and a fresh prepared directory.
- `pipeline.py labels` uses `prepare_study_inputs.py` to reproduce the historical history/original12 label binding. This is an archival preparation recipe; the current CBM supervision is extracted by `cbm/labels.py` instead.
- `preprocessing_study.yaml` preserves the previous preparation settings, independent of the new `study.yaml`.

`pipeline.py` no longer starts the retired experiments or old reports. Use `python -u run.py --devices 0 1 2 3` for the F1 study. Newly regenerated preparation outputs would require their own source cache and input binding before being used as `source_root`; that is outside the present rerun.
