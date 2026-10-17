# Five-seed submission evidence and EMSR551 worked case

This fixed supplement extends the historical three-seed release with five-seed (2026–2030) fixed-input evidence, evaluation seed 12026. Historical tags are unchanged. No training, model inference or test evaluation is performed by this packaging operation.

- `checkpoint_identities.json`: 15 formal checkpoint identities, reused from existing receipts, excluding recovery diagnostics and the superseded Decoder2029 run.
- `source_data/`: the source tables supplied with the v33 manuscript, including core five-seed results, event sensitivity, validation diagnostics, threshold transfer and EMSR551 spatial counts.
- `results/`: completed validation/threshold merged outputs, six-checkpoint supplementary evaluation records and frozen choices where supplied.
- `evidence/EMSR551_spatial/`: 30 reconciled checkpoint-condition groups, original two-batch accounting and execution records. 60 batch-forward calls processed 960 tile instances; 270 target predictions were retained. Prediction and image binaries are not redistributed.
- `scripts/`: captured scientific analysis/evaluation sources. Historical execution scripts retain their recorded workspace paths and asset guards; they are provenance sources, not a claim of a one-command retraining workflow on any platform.

`python recompute_spatial_tables.py` checks the saved per-seed counts and reproduces the four published E.7.6 tables without loading models or images. Run it from any directory; it locates inputs relative to this file. The original broad saved-evidence recomputation remains at the repository root.

The legacy validation-wide spatial/boundary and hardware cohorts remain three seeds/checkpoints. The new case-level EMSR551 spatial analysis uses all 15 checkpoints. Five-fold training is not represented as completed evidence. Source tables retain clean/perturbed input conditions; raw execution JSON may use the original `degraded` key.

Author-controlled source follows the repository AGPL-3.0-or-later license. Third-party material retains upstream terms. No original imagery, checkpoint binaries, passwords or tokens are intentionally included.
