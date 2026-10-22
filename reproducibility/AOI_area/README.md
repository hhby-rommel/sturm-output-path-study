# Fixed-AOI area recomputation Tables G.4 and G.5

Release: [v1.1.2-aoi-five-seed](https://github.com/hhby-rommel/sturm-output-path-study/releases/tag/v1.1.2-aoi-five-seed). Use the commit targeted by this fixed tag (`git rev-parse v1.1.2-aoi-five-seed`). Earlier release tags retain their original commits.

From the repository root, run:

```sh
python reproducibility/AOI_area/recompute_AOI_area.py --output-dir AOI_recomputed
```

Python standard library only. A new output directory is required. This regenerates the six seed-2027 rows of Supplement Table G.4, the 30 per-seed rows for seeds 2026–2030 and the six mean ± sample-standard-deviation summaries of Table G.5. Confusion counts are reconciled against per-seed scene reports and verified grid geometry. Negative signed area bias means underprediction relative to reference water area.

The fixed-AOI sensitivity check uses EMSR292/01CHRISOUPOLI, clean inputs, and frozen native/decoder/FPN checkpoints. Seed 2027 reuses the original scene outputs; the other four seeds were evaluated without retraining or checkpoint reselection. S1 and S2 have different valid-coverage footprints and are not combined. These results do not isolate a sensor effect or establish cross-event generalization. Areas include permanent water.

The numerical script uses archived confusion counts and geometry records. It does not rerun satellite-image preprocessing, model inference or georeferenced stitching. Execution scripts are retained as historical source, not as an automatic rerun interface. Original satellite imagery and checkpoint binaries are not redistributed.
