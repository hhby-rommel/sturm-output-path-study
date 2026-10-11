# STURM output-path research source release

License: **AGPL-3.0-or-later** for author-controlled source, except where otherwise noted. Third-party components, datasets and model assets retain their terms.

This release expands the earlier audit-only snapshot into an ordinary source tree: model integration, training, loaders/losses, evaluation, spatial/extent analysis and forward-cost measurement. Numerical evidence and reported results are unchanged. Historical formal variants remain under prepared_controls/ to preserve their source identity. The former ZIP is an immutable historical snapshot, not the only location of source.

## Quick source map

- **Core paper files:** [output models](src/models/), [training and losses](src/training/), [data loader](src/dataset/), [evaluation](src/evaluation/) and [registered configurations](configs/).
- **Historical formal variants:** [prepared_controls/](prepared_controls/); the earlier selected-source ZIP remains an immutable snapshot.
- **Third-party material:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [retained notices](third_party_notices/) and [recorded upstream model YAML](upstream_configs/ultralytics/yolo26-sem.yaml).
- **Release interfaces:** [train](scripts/train.py), [evaluate](scripts/evaluate.py), [benchmark](scripts/benchmark.py) and [metadata restoration](scripts/prepare_data.py); these interfaces do not execute by default.

Full per-file origins and hashes: [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) / [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Source and environment

Read SOURCE_PROVENANCE.md, LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md. Python3.10/PyTorch2.1.0+cu121 and the recorded dependency constraints are documented in environment_record.json/environment.yml. Reconstruction choices are distinct from captured historical versions. A clean installation and full retraining have not been executed for this packaging release. Original guards may require RTX4090D; no backend fallback is inserted.

## Data and external assets

Download STURM-Flood and Sen1Floods11 from their original sources; no image, label cache or checkpoint binaries are redistributed. Keep STURM's Dataset hierarchy under data/raw/Dataset. Event splits and compressed original manifest metadata are included. Restore compressed manifest metadata and sampling/index records without reading images:

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --write
```

Sen1's official train/val/test file manifests, sampling arrays and source for cache preparation are under prepared_controls/ and scripts/prepare_sen1_full_pool*_20261004.py. Regenerate image/target/valid caches from legally obtained data; do not reuse historical approval files as permission. Data-download/cache routines are explicit source entrypoints, not executed by installation or dry-run.

## Inspect commands before running

```bash
python scripts/train.py --config configs/sturm_native.json --dry-run
python scripts/train.py --config configs/sturm_decoder.json --dry-run
python scripts/train.py --config configs/sturm_fpn.json --dry-run
python scripts/train.py --config configs/sen1_fpn.json --dry-run
python scripts/evaluate.py --config configs/fixed_input_evaluation.json --dry-run
python scripts/benchmark.py --config configs/runtime.json --dry-run
```

These interfaces print the historical entry and arguments; they never run by default. To execute a prepared run, use --execute. An entry that can access test also requires --allow-test. Native/decoder legacy training includes its original final test; the dispatcher makes that scope explicit rather than silently splitting its numerical recipe. FPN/Sen1 and frozen-evaluation entries retain protocol, asset, checkpoint and source-hash assertions. Prepare run-specific records/caches/checkpoints and reconcile source locks with the recorded header/path-only packaging changes before execution. The original lock records remain historical identities, not new success receipts. This source release supplies preferred source form and explicit interfaces; it does not claim a one-command rerun of every historical experiment.

Use new output directories. Do not reset heads, alter BN/optimizer/sampling/selection or bypass missing-asset guards to make a reproduction start. The registered YOLO budget is max50/patience10. Scripts preserve failures and one-shot test claims. No automatic queue, model download, training or test is launched by this release.

## Saved numerical evidence

```bash
python recompute_public_evidence_20261011.py
```

This reads the saved JSON only; it does not load images or weights. See source_release_validation.json for exactly what was checked locally. Historical MIT permissions remain available at their earlier commit and are not retroactively withdrawn. No license statement promises full historical numerical reproduction on another platform.

Metadata correction **v1.0.1-paper-release** supersedes v1.0-paper-release for current manifest and citation metadata. The previous source tag remains fixed. No experimental results, source computation, weights or test access changed.
