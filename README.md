# STURM output-path study

Reproducibility materials for native output, progressive decoder and parameter-matched FPN paths on a shared lightweight YOLO26n body, with single-sensor S1 or S2 inputs.

## Recompute saved-count analysis

Download and extract `reproducibility_materials_20261010.zip` into a working directory. Install Python and NumPy, then run from that extracted directory:

```bash
python scripts/manuscript_existing_counts_audit_20261010.py
```

This command reads saved JSON reports only. It performs no training, model inference or test-dataset loading. It writes per-event foreground-extent counts, test-event paired differences and combined validation/test pooled diagnostics to `results/manuscript_existing_counts_audit_20261010_v1/`. The command was executed in the portable review packet and the outputs were compared with the project summaries.

## Recorded scope

- Nine frozen best checkpoints; training seeds 2026, 2027 and 2028; evaluation seed 12026.
- Core test reports use one sample-specific perturbation stream for all systems.
- Validation sensor counts are included. Validation event counts are not available yet; the frozen-val supplementation script is prepared but has not run.
- Combined validation/test pooled statistics contain validation data used for checkpoint selection. They are diagnostics, not an independent 24-event test.
- Foreground extent errors count water pixels in evaluation tile domains. They are not deduplicated flood-only geographic areas in km².
- GPU runtime: four systems, three weights, two sensors and five batch sizes in one GPU session; 120 units and 36,000 raw timings. No I/O or H2D timing.

## Model and GPU inference code

`src/` contains the selected model, data and loss implementations. Server scripts preserve the original experiment paths and require the matching experiment environment, dataset and frozen weights. They are source records, not turnkey independent retraining commands. The raw imagery and checkpoint files are not redistributed. Checkpoint SHA256 identities are in `reproduction_manifest.json`.

The dataset is available from [STURM-Flood](https://github.com/STURM-WEO/STURM-Flood), and the external S1 dataset from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Follow their respective data and asset terms. The complete manuscript remains a local review draft; this repository publishes numerical/code materials.

## License status

A project-wide license for author-written code has not been selected. Upstream dependency notices are preserved under `third_party_notices/`; in particular Ultralytics is distributed under its recorded AGPL license. Publishing this repository does not replace upstream terms or assert that dataset or weight redistribution is permitted.
