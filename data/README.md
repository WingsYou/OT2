# Local AI-READI data

This directory intentionally contains no participant data.

Obtain AI-READI v3.0.0 from the [official dataset portal](https://doi.org/10.60775/fairhub.3) under its access requirements. Place the downloaded dataset contents in `aireadi/`, preserving `participants.tsv`, `clinical_data/`, all eight nonclinical source directories, their manifests, and referenced files. See the root [README](../README.md) for commands and the expected layout.

Alternatively, point `dataset_root` in `pipeline.yaml` and `clinical_dir` in `study.yaml` to an existing dataset outside the repository. Nothing needs to be copied here.

Raw data and generated participant-level artifacts stay local. Git ignores this directory except the instructions and empty placeholder.
