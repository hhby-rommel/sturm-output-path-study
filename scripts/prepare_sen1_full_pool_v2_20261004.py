# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Acquire only official train/val; build full-resolution mmap diagnostic inputs."""
from pathlib import Path
import concurrent.futures, csv, hashlib, json, shutil, time
import requests
import numpy as np
import rasterio
R=Path(__file__).resolve().parents[1]
O=R/'results/sen1_full_pool_preparation_20261004_v2'
A=R/'results/sen1floods11_compatibility_audit_20261004_v1/official_splits'
PREVIOUS=R/'results/sen1_full_pool_preparation_20261004_v1/assets'
OLD=R/'results/sen1_training_preparation_20261004_v2/assets'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
assert not O.exists(),'No overwrite or automatic resume'
O.mkdir();(O/'assets').mkdir()
started=time.perf_counter()
try:
 rows=[]
 for part,suffix,n in [('train','train',252),('val','valid',89)]:
  path=A/f'flood_{suffix}_data.csv'
  pairs=list(csv.reader(path.open(encoding='utf-8-sig')));assert len(pairs)==n
  rows.extend({'partition':part,'image':x,'label':y,'group':x.split('_')[0]} for x,y in pairs)
 dump(O/'selection_before_download.json',{'scope':'ALL official train and val, no test/Bolivia images or labels','rows':rows})
 def fetch(row):
  records=[]
  for key,folder in [('image','S1Hand'),('label','LabelHand')]:
   name=row[key];assert Path(name).name==name
   dst=O/'assets'/name;url=f'https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/{folder}/{name}'
   cached=(PREVIOUS/name) if (PREVIOUS/name).is_file() else OLD/name
   reused=cached.is_file()
   if reused:shutil.copy2(cached,dst)
   else:
    temp=dst.with_suffix(dst.suffix+'.partial')
    with requests.get(url,stream=True,timeout=(20,300)) as response:
     response.raise_for_status()
     with temp.open('wb') as file:
      for chunk in response.iter_content(1024*1024):file.write(chunk)
    temp.replace(dst)
   records.append({'partition':row['partition'],'file':name,'url':url,'bytes':dst.stat().st_size,'sha256':sha(dst),'reused_local_asset':reused})
  return records
 receipts=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for n,record in enumerate(pool.map(fetch,rows),1):
   receipts.extend(record)
   if n%10==0 or n==341:
    dump(O/'download_progress.json',{'completed_chips':n,'total_chips':341,'bytes_collected':sum(q['bytes'] for q in receipts),'elapsed_seconds':time.perf_counter()-started,'test_access':False})
    dump(O/'asset_receipts_partial.json',receipts)
    print('CHIPS',n,'/341',flush=True)
 dump(O/'asset_receipts.json',receipts)
 count=len(rows)*16
 xs=np.lib.format.open_memmap(O/'images.npy',mode='w+',dtype=np.float32,shape=(count,2,128,128))
 ys=np.lib.format.open_memmap(O/'targets.npy',mode='w+',dtype=np.int8,shape=(count,128,128))
 vs=np.lib.format.open_memmap(O/'valid.npy',mode='w+',dtype=np.bool_,shape=(count,128,128))
 index=[];profiles=[];cursor=0
 for row in rows:
  with rasterio.open(O/'assets'/row['image']) as ds:
   x=ds.read();nd=ds.nodata;sig=(ds.crs,ds.transform,ds.width,ds.height);descs=ds.descriptions
  with rasterio.open(O/'assets'/row['label']) as ds:y=ds.read(1);sig2=(ds.crs,ds.transform,ds.width,ds.height)
  assert x.shape==(2,512,512) and x.dtype==np.float32 and y.shape==(512,512)
  assert sig[0]==sig2[0] and sig[2:]==sig2[2:]
  error=max(max(abs(a-b) for a,b in zip(sig[1]*corner,sig2[1]*corner)) for corner in [(0,0),(512,0),(0,512),(512,512)])/min(abs(sig[1].a),abs(sig[1].e))
  assert error<=1e-6 and np.isin(y,[-1,0,1]).all()
  assert descs[0] is not None and 'VV' in descs[0] and descs[1] is not None and 'VH' in descs[1],descs
  valid=np.isfinite(x).all(0)&np.isin(y,[0,1])
  if nd is not None and np.isfinite(nd):valid&=(x!=nd).all(0)
  profiles.append({**row,'valid_pixels':int(valid.sum()),'foreground_pixels':int((valid&(y==1)).sum()),'label_codes':np.unique(y).tolist(),'channel_descriptions':list(descs),'grid_max_corner_error_pixels':error})
  x=x.copy();x[:,~valid]=0
  for yy in range(0,512,128):
   for xx in range(0,512,128):
    sl=np.s_[yy:yy+128,xx:xx+128];xs[cursor]=x[:,sl[0],sl[1]];ys[cursor]=y[sl];vs[cursor]=valid[sl]
    index.append({'partition':row['partition'],'group':row['group'],'chip':row['image'],'y':yy,'x':xx});cursor+=1
 xs.flush();ys.flush();vs.flush();del xs,ys,vs
 dump(O/'tile_index.json',index);dump(O/'input_label_profiles.json',profiles)
 cache_files=['images.npy','targets.npy','valid.npy','tile_index.json','input_label_profiles.json','asset_receipts.json','selection_before_download.json']
 dump(O/'cache_receipt.json',{'files':[{'path':name,'bytes':(O/name).stat().st_size,'sha256':sha(O/name)} for name in cache_files],
  'label_storage':'int8 (-1,0,1), converted to int64 only for batch; no change to label values',
  'invalid_handling':'same parent rule: invalid image positions zeroed, labels retained',
  'train_chips':252,'val_chips':89,'train_windows':4032,'val_windows':1424,'test_access':False})
 dump(O/'completion.json',{'status':'FULL_OFFICIAL_TRAIN_VAL_CACHE_READY','train_chips':252,'val_chips':89,'train_windows':4032,'val_windows':1424,'test_access':False,'training_started':False,'elapsed_seconds':time.perf_counter()-started})
 print('FULL_POOL_PREPARATION_COMPLETE',flush=True)
except Exception as e:
 dump(O/'failure.json',{'type':type(e).__name__,'message':str(e),'test_access':False,'no_automatic_retry':True})
 raise
