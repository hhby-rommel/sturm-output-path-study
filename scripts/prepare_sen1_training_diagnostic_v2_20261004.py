# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""New S1-only training study: metadata audit and train/val diagnostic assets."""
from pathlib import Path
import csv, hashlib, json, zipfile, collections, concurrent.futures, shutil
import requests
import numpy as np
import rasterio

R=Path(__file__).resolve().parents[1]
A=R/'results/sen1floods11_compatibility_audit_20261004_v1'
O=R/'results/sen1_training_preparation_20261004_v2'
P=R/'prepared_controls/sen1_training_diagnostic_20261004_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not O.exists() and not P.exists()
O.mkdir();P.mkdir();(O/'assets').mkdir()
shutil.copy2(R/'results/sen1_training_preparation_20261004_v1/official_catalog.zip',O/'official_catalog.zip')
z=zipfile.ZipFile(O/'official_catalog.zip')
items=[json.loads(z.read(n)) for n in z.namelist() if n.startswith('catalog/sen1floods11_hand_labeled_source/') and n.endswith('.json')]
items={x['id']:x for x in items if x.get('type')=='Feature'}
splits={p:list(csv.reader((A/'official_splits'/f'flood_{f}_data.csv').open(encoding='utf-8-sig'))) for p,f in [('train','train'),('val','valid'),('test','test'),('Bolivia','bolivia')]}
assert {k:len(v) for k,v in splits.items()}=={'train':252,'val':89,'test':90,'Bolivia':15}
def ident(row):return row[0].removesuffix('_S1Hand.tif')
assert all(ident(r) in items for rows in splits.values() for r in rows)
assert len({r[0] for rows in splits.values() for r in rows})==446
overlap=[]
for pa,pb in [('train','val'),('train','test'),('val','test'),('train','Bolivia'),('val','Bolivia'),('test','Bolivia')]:
    for a in splits[pa]:
        u=items[ident(a)]['bbox']
        for b in splits[pb]:
            v=items[ident(b)]['bbox']
            dx=min(u[2],v[2])-max(u[0],v[0]);dy=min(u[3],v[3])-max(u[1],v[1])
            if dx>1e-9 and dy>1e-9:overlap.append({'partition_a':pa,'partition_b':pb,'a':a[0],'b':b[0],'bbox_area_degrees2':dx*dy})
audit={'split_counts':{k:len(v) for k,v in splits.items()},'filename_disjoint':True,
       'group_counts':{p:dict(collections.Counter(r[0].split('_')[0] for r in rs)) for p,rs in splits.items()},
       'event_isolation':False,'group_definition':'Official filename prefix; physical Mekong alias not relabeled',
       'spatial_positive_area_bbox_pairs':overlap,'spatial_tolerance_degrees':1e-9,
       'spatial_scope':'Complete 446 S1 source STAC bounding boxes. Geometry screen, not pixel-level duplication, acquisition equivalence, or every TIFF-header verification.',
       'test_imagery_or_labels_read':False,'catalog_sha256':sha(O/'official_catalog.zip')}
dump(O/'split_and_spatial_audit.json',audit)
assert not overlap,'Stop before diagnostic if metadata shows cross-split footprint overlap'
selection=[]
for part in ['train','val']:
    groups=sorted({r[0].split('_')[0] for r in splits[part]})
    for group in groups:
        row=min((r for r in splits[part] if r[0].split('_')[0]==group),key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())
        selection.append({'partition':part,'group':group,'image':row[0],'label':row[1],'source':items[ident(row)]})
dump(P/'selection_before_download.json',{'rule':'One chip per filename flood-group per official train/val, smallest SHA256(filename), no label/performance selection','chips':selection})
def fetch(row):
    rec=[]
    for key,folder in [('image','S1Hand'),('label','LabelHand')]:
        name=row[key];url=f'https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/{folder}/{name}'
        p=O/'assets'/name;shutil.copy2(R/'results/sen1_training_preparation_20261004_v1/assets'/name,p)
        rec.append({'file':name,'url':url,'sha256':sha(p),'bytes':p.stat().st_size,'partition':row['partition'],'reused_from':'preparation_v1_download'})
    return rec
receipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for records in executor.map(fetch,selection):receipts.extend(records)
dump(O/'asset_receipts.json',receipts)
arrays=[];profiles=[]
for row in selection:
    with rasterio.open(O/'assets'/row['image']) as ds:
        x=ds.read();nd=ds.nodata;sig=(ds.crs,ds.transform,ds.width,ds.height);descs=ds.descriptions;bounds=list(ds.bounds)
    with rasterio.open(O/'assets'/row['label']) as ds:
        y=ds.read(1);sig2=(ds.crs,ds.transform,ds.width,ds.height)
    assert x.shape==(2,512,512) and x.dtype==np.float32 and y.shape==(512,512)
    assert sig[0]==sig2[0] and sig[2:]==sig2[2:]
    maxdiff=max(max(abs(u-v) for u,v in zip(sig[1]*corner,sig2[1]*corner)) for corner in [(0,0),(512,0),(0,512),(512,512)])/min(abs(sig[1].a),abs(sig[1].e))
    assert maxdiff<=1e-6 and np.isin(y,[-1,0,1]).all()
    assert 'VV' in descs[0] and 'VH' in descs[1],descs
    valid=np.isfinite(x).all(0)&np.isin(y,[0,1])
    if nd is not None and np.isfinite(nd):valid&=(x!=nd).all(0)
    # Keep predetermined empty-domain chip; no label/performance-based replacement.
    prof={'partition':row['partition'],'image':row['image'],'group':row['group'],'channels':list(descs),'nodata':nd if nd is None or np.isfinite(nd) else None,'valid_pixels':int(valid.sum()),'foreground_pixels':int(((y==1)&valid).sum()),'label_codes':np.unique(y).tolist(),'grid_max_corner_error_pixels':maxdiff,'tiff_bounds':bounds,'stored_channel_quantiles':[[float(v) for v in np.percentile(c[valid],[0,1,25,50,75,99,100])] for c in x] if valid.any() else None}
    profiles.append(prof)
    x=x.copy();x[:,~valid]=0
    for yy in range(0,512,128):
        for xx in range(0,512,128):
            sl=np.s_[yy:yy+128,xx:xx+128]
            arrays.append({'partition':row['partition'],'group':row['group'],'chip':row['image'],'y':yy,'x':xx,'image':x[:,sl[0],sl[1]],'target':y[sl].astype(np.int64),'valid':valid[sl]})
np.savez_compressed(O/'train_val_tiles.npz',image=np.stack([v['image'] for v in arrays]),target=np.stack([v['target'] for v in arrays]),valid=np.stack([v['valid'] for v in arrays]))
dump(O/'tile_index.json',[{k:v for k,v in a.items() if k not in ['image','target','valid']} for a in arrays])
dump(O/'input_label_profiles.json',profiles)
W=R/'prepared_controls/decoder_reproduction_validation_20261004_v1/workspace_public_v3'
numeric={'batch':16,'clip':5,'supervision':'clean CE+Dice plus original S1 degraded CE+Dice, summed; no new class weights/aux outputs','BN':'YOLO running stats frozen; U-Net has no BN','max_epochs':{'decoder':50,'main_only':50,'unet':30},'patience':10,'selection':'degraded val F1 then exact clean tie then earlier epoch','sampler':'Uniform replacement over original-resolution train windows; one sensor only','input':'Raw received VV/VH dB float32; no normalization/scaling/clipping or statistics fitted to val/test','label':'-1 ignored; 0 background; 1 water','output':'128x128; main_only bilinear logits align_corners=False, label never interpolated'}
plan={'status':'LOCAL_DIAGNOSTIC_FROZEN_FORMAL_LAUNCH_DISABLED','dataset':'Sen1Floods11 v1.1 S1 hand labels','study':'Training on a second data source, not STURM zero-shot transfer or a sharing-mechanism test','official_split_scope':audit['split_counts'],'not_event_isolated':True,'diagnostic_seed':12026,'formal_seeds_proposed':[2026,2027,2028],'diagnostic_budget':{'updates_per_model':64,'evaluate_at':[0,16,32,64],'train_chips':10,'val_chips':10,'train_windows':160,'val_windows':160,'sample_sequence':'Same seeded fixed 64x16 replacement draw sequence in all arms; degradation seeded by update and independent of model dropout','no_score_based_extension':True},'numeric':numeric,
      'optimization':{'decoder':{'lr':.0003,'weight_decay':.0001,'scheduler':'none','exclude_norm_bias_decay':False},'main_only':{'lr':.0003,'weight_decay':.0001,'scheduler':'none','exclude_norm_bias_decay':False},'unet':{'lr':.0001,'weight_decay':.0001,'scheduler':'warmup_cosine','warmup_fraction':.05,'minimum_lr_ratio':.01,'exclude_norm_bias_decay':True,'total_steps':252*30,'scope':'Schedule sized to full official training windows/30 epochs, not compressed to 64 diagnostic steps'}},
      'initialization':'YOLO uses original Cityscapes-compatible checkpoint transfer, same-seed common tensors verified; U-Net native S1 expert base64/dropout.30/noBN random Xavier as original. No STURM-trained best used.',
      'model_workspace':W.relative_to(R).as_posix(),'pretrained':(W/'models/pretrained/yolo26n-sem.pt').relative_to(R).as_posix(),
      'cache':(O/'train_val_tiles.npz').relative_to(R).as_posix(),'tile_index':(O/'tile_index.json').relative_to(R).as_posix(),
      'formal_budget_proposed':{'train_windows':252*16,'steps_per_epoch':252,'YOLO_max_steps':12600,'UNet_max_steps':7560,'tasks':9,'early_stopping':'max50/patience10 YOLO; original U-Net max30/patience10; not equal tuning or total optimization budget'},
      'guards':{'local_only':True,'test':False,'formal_training':False,'empty_domain_chip':'retain with explicit zero-count domain; no replacement','overwrite':False,'retry_or_resume':False,'any_failure':'stop and record; no silent retuning'},'locked_files':[]}
for p in [O/'train_val_tiles.npz',O/'tile_index.json',P/'selection_before_download.json',W/'models/pretrained/yolo26n-sem.pt',R/'scripts/sen1_training_diagnostic_20261004.py']+list((W/'src').rglob('*.py')):
    plan['locked_files'].append({'path':p.relative_to(R).as_posix(),'sha256':sha(p)})
dump(P/'protocol.json',plan);(P/'protocol.sha256').write_text(sha(P/'protocol.json')+'\n')
dump(O/'completion.json',{'status':'METADATA_SPATIAL_SCREEN_AND_TRAIN_VAL_DIAGNOSTIC_ASSETS_READY','selected_chips':20,'tile_count':320,'spatial_positive_overlap_pairs':len(overlap),'training_started':False,'test_pixels_accessed':False})
print('READY: all446 metadata geometry screened,20 train/val chips,320 original128 windows; no test pixels')
