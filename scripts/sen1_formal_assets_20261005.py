# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,hashlib,shutil,concurrent.futures,time
import requests,numpy as np,rasterio

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def download_test(R,P):
 O=R/'results/sen1_formal_test_assets_20261005_v1';assert not O.exists(),'No repeat preparation/restart'
 assert shutil.disk_usage(R).free>int(json.loads((P/'storage_budget.json').read_text())['required_additional_free_bytes'])
 O.mkdir();(O/'raw').mkdir();(O/'network_attempts').mkdir()
 manifest=json.loads((P/'test_manifest.json').read_text());assert len(manifest['rows'])==90 and manifest['Bolivia_included'] is False
 pairs=[]
 for row in manifest['rows']:
  for kind in ['S1','label']:pairs.append((row[kind+'_file'],row[kind+'_url']))
 def fetch(pair):
  name,url=pair;assert Path(name).name==name and 'Bolivia' not in name
  target=O/'raw'/name;t=target.with_suffix('.tif.partial');errors=[]
  for attempt in range(1,4):
   try:
    assert shutil.disk_usage(R).free>2*1024**3
    with requests.get(url,stream=True,timeout=(20,60)) as res:
     res.raise_for_status();expected=res.headers.get('Content-Length')
     with t.open('wb') as f:
      for data in res.iter_content(65536):
       if data:f.write(data)
     if expected:assert t.stat().st_size==int(expected),'Incomplete response'
    t.replace(target);return {'file':name,'url':url,'bytes':target.stat().st_size,'sha256':sha(target)}
   except Exception as e:
    errors.append({'attempt':attempt,'type':type(e).__name__,'message':str(e)});dump(O/'network_attempts'/(name+'.json'),errors)
    if attempt==3:raise
    time.sleep(5*attempt)
 records=[]
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for future in concurrent.futures.as_completed([pool.submit(fetch,p) for p in pairs]):
    records.append(future.result());dump(O/'download_progress.json',{'files':len(records),'total':180,'bytes':sum(x['bytes'] for x in records)});print('TEST_ASSET_FILES',len(records),'/180',flush=True)
  header=[]
  for row in manifest['rows']:
   with rasterio.open(O/'raw'/row['S1_file']) as x,rasterio.open(O/'raw'/row['label_file']) as y:
    assert x.count==2 and x.width==x.height==512 and y.count==1 and y.width==y.height==512
    assert x.crs==y.crs and 'VV' in (x.descriptions[0] or '') and 'VH' in (x.descriptions[1] or '')
    error=max(max(abs(a-b) for a,b in zip(x.transform*corner,y.transform*corner)) for corner in [(0,0),(512,0),(0,512),(512,512)])/min(abs(x.transform.a),abs(x.transform.e));assert error<=1e-6
    header.append({'chip':row['S1_file'],'grid_error_pixels':error,'channels':x.descriptions})
  dump(O/'asset_receipts.json',sorted(records,key=lambda x:x['file']));dump(O/'header_grid_check.json',header)
  dump(O/'completion.json',{'status':'OFFICIAL90_TEST_ASSETS_ONLY_READY','manifest_sha256':sha(P/'test_manifest.json'),'files':180,'test_model_inference':False,'test_pixel_cache_built':False,'Bolivia':False})
 except Exception as e:dump(O/'failure.json',{'type':type(e).__name__,'message':str(e),'no_automatic_restart':True});raise
 return O
def build_test_cache(R,P):
 O=R/'results/sen1_formal_test_cache_20261005_v1';A=R/'results/sen1_formal_test_assets_20261005_v1'
 if O.exists():
  assert (O/'completion.json').exists(),'Incomplete cache; stop without rebuild'
  assert json.loads((O/'completion.json').read_text())['manifest_sha256']==sha(P/'test_manifest.json');return O
 assert json.loads((A/'completion.json').read_text())['manifest_sha256']==sha(P/'test_manifest.json')
 assert shutil.disk_usage(R).free>2*1024**3+237000000;O.mkdir()
 rows=json.loads((P/'test_manifest.json').read_text())['rows'];receipts=json.loads((A/'asset_receipts.json').read_text())
 for f in receipts:assert sha(A/'raw'/f['file'])==f['sha256']
 xs=np.lib.format.open_memmap(O/'images.npy',mode='w+',dtype=np.float32,shape=(1440,2,128,128));ys=np.lib.format.open_memmap(O/'targets.npy',mode='w+',dtype=np.int8,shape=(1440,128,128));vs=np.lib.format.open_memmap(O/'valid.npy',mode='w+',dtype=np.bool_,shape=(1440,128,128));index=[];profiles=[];cursor=0
 for row in rows:
  with rasterio.open(A/'raw'/row['S1_file']) as f:x=f.read();nd=f.nodata
  with rasterio.open(A/'raw'/row['label_file']) as f:y=f.read(1)
  assert x.shape==(2,512,512) and x.dtype==np.float32 and y.shape==(512,512) and np.isin(y,[-1,0,1]).all()
  valid=np.isfinite(x).all(0)&np.isin(y,[0,1])
  if nd is not None and np.isfinite(nd):valid&=(x!=nd).all(0)
  x=x.copy();x[:,~valid]=0
  profiles.append({'chip':row['S1_file'],'valid_pixels':int(valid.sum()),'foreground_pixels':int((valid&(y==1)).sum()),'label_codes':np.unique(y).tolist()})
  for yy in range(0,512,128):
   for xx in range(0,512,128):
    sl=np.s_[yy:yy+128,xx:xx+128];xs[cursor]=x[:,sl[0],sl[1]];ys[cursor]=y[sl];vs[cursor]=valid[sl];index.append({'partition':'test','chip':row['S1_file'],'group':row['S1_file'].split('_')[0],'y':yy,'x':xx});cursor+=1
 xs.flush();ys.flush();vs.flush();del xs,ys,vs;assert cursor==1440
 assert len({(q['chip'],q['y'],q['x']) for q in index})==1440
 dump(O/'tile_index.json',index);dump(O/'pixel_profiles.json',profiles)
 names=['images.npy','targets.npy','valid.npy','tile_index.json','pixel_profiles.json']
 dump(O/'cache_receipt.json',{'files':[{'path':n,'bytes':(O/n).stat().st_size,'sha256':sha(O/n)} for n in names]})
 dump(O/'completion.json',{'status':'OFFICIAL90_TEST_CACHE_READY','manifest_sha256':sha(P/'test_manifest.json'),'chips':90,'windows':1440,'Bolivia':False,'created_inside_first_task_test_claim':True})
 return O
if __name__=='__main__':
 R=Path(__file__).resolve().parents[1];download_test(R,R/'prepared_controls/sen1_formal_20261005_v1')
