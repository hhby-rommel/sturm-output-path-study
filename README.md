# STURM output-path study

Numerical and code evidence for controlled comparison of the native output and parameter-matched decoder/FPN paths on a common lightweight YOLO26n body.

## Current evidence and recomputation

The current index is `reproduction_manifest.json` (v18-20261011).
- `evidence_20261011_v17.json`: Sen1-FPN, BN controls, direct event pairing, Otsu, threshold transfer, cost and precision reports; source records are embedded under `source_records`.
- `val_event_counts_20261011.json`: nine frozen-model validation event-count reports, already evaluated. Validation was used for checkpoint selection.
- `recompute_public_evidence_20261011.py`: recomputes seed means, sample SD and direct event pairing using saved JSON only.

For numerical recomputation, install NumPy and run:

```bash
python recompute_public_evidence_20261011.py
```

This command performs no training, inference or dataset/test access. Combined val/test event statistics are descriptive; they are not an independent 24-event test. Water-extent counts refer to valid tile domains and include permanent water, not deduplicated flood-only km².

## Archived selected source and notices

`reproducibility_materials_20261010.zip` contains the archived selected `src/` tree (six model/data/loss files) and `third_party_notices/`; these directories are inside the ZIP, not root directories of this repository. Its content hashes are indexed by `archive_inventory.json`. It also contains the earlier saved-count analysis and its source. Historical snapshots remain unchanged; the root manifest and license files describe the current status.

Additional server-source records are embedded in the evidence JSON. They retain original experiment paths and require matching data, dependencies and frozen weights. The selected source archive is not a complete turnkey retraining project.

## Environment

`environment_record.json` distinguishes recorded runtime values, archived project constraints and reconstruction choices. `environment.yml` and `requirements.txt` provide a reconstruction scaffold for Python 3.10.20 and PyTorch 2.1.0+cu121. The torchvision 0.16.0+cu121 pin follows the official PyTorch compatibility pair; its historical installed version was not recorded. Runtime GDAL version and a complete historical pip freeze are unavailable. This scaffold has not been reinstall-tested and is not claimed to reproduce historical training bitwise.

From a checkout containing both files:

```bash
conda env create -f environment.yml
```

## Data, weights and license

Data sources: [STURM-Flood](https://github.com/STURM-WEO/STURM-Flood) and [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Raw imagery and checkpoint binaries are not redistributed. Checkpoint SHA identities are recorded, but hashes do not provide weight-download access.

Independently authored code uses MIT (`LICENSE.author-code`); third-party and derivative code retains its applicable terms (`LICENSE_SCOPE.md`). Upstream notices, including the recorded Ultralytics license, are inside the ZIP. This repository publishes numerical/code evidence, not the manuscript. `CITATION.cff` identifies the artifact by its public account handle; no paper-author names or paper DOI are inferred.
