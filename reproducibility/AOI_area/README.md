# Fixed AOI area recomputation

From repository root, run:

```sh
python reproducibility/AOI_area/recompute_AOI_area.py --output-dir AOI_recomputed
```

Python standard library only. The command reads source_data/fixed_AOI_area/, reconciles the two scene reports, original six-row counts, geometry and archived area results, then writes the recomputed CSV and execution record to a new directory. It does not load models or imagery. All six rows use seed 2027, clean inputs and the prespecified EMSR292/01CHRISOUPOLI selection. S1/S2 footprints differ; each sensor has common valid coverage across paths and zero within-sensor valid-pixel overlap. Water includes permanent water.
