# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Read frozen scene outputs only; nearest display resampling, no model/data inference."""
from pathlib import Path
import rasterio, numpy as np, json
from rasterio.enums import Resampling
p=Path('/root/autodl-tmp/sturm_codex_20260919/results/scene_demo_20261010_v1')
data={}; receipt={}
for s in ['S1','S2']:
 for f in ['input_channel0','GT','main_only_FP_FN','decoder_FP_FN','fpn_FP_FN']:
  with rasterio.open(p/s/(f+'.tif')) as d:
   factor=min(1,700/max(d.height,d.width)); shape=(max(1,round(d.height*factor)),max(1,round(d.width*factor)))
   a=d.read(1,out_shape=shape,resampling=Resampling.nearest); mask=d.read_masks(1,out_shape=shape,resampling=Resampling.nearest)==0
   if f=='input_channel0':
    full=d.read(1); mm=d.read_masks(1)>0; lo,hi=np.percentile(full[mm],[2,98]); a=np.uint8(np.clip((a-lo)/(hi-lo),0,1)*254)
   a[mask]=255; data[s+'_'+f]=a.astype(np.uint8);receipt[s+'_'+f]={'source':str(p/s/(f+'.tif')),'original_shape':[d.height,d.width],'display_shape':list(shape),'resampling':'nearest'}
np.savez_compressed(p.parent/'scene_display_arrays_20261010.npz',**data)
(p.parent/'scene_display_arrays_20261010.json').write_text(json.dumps(receipt,indent=2))
print('display arrays ready')
