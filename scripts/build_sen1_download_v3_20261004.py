# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'scripts/prepare_sen1_full_pool_v2_20261004.py').read_text(encoding='utf-8')
s=s.replace('import requests','import requests, os, datetime')
s=s.replace("O=R/'results/sen1_full_pool_preparation_20261004_v2'","O=R/'results/sen1_full_pool_preparation_20261004_v3'")
s=s.replace("PREVIOUS=R/'results/sen1_full_pool_preparation_20261004_v1/assets'","PREVIOUS=R/'results/sen1_full_pool_preparation_20261004_v2/assets'\nEARLIER=R/'results/sen1_full_pool_preparation_20261004_v1/assets'")
s=s.replace("O.mkdir();(O/'assets').mkdir()", "O.mkdir();(O/'assets').mkdir();(O/'network_attempts').mkdir()\ndump(O/'process.json',{'pid':os.getpid(),'script':str(Path(__file__).resolve()),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'official S1 train/val assets only; no training/test'})")
s=s.replace("cached=(PREVIOUS/name) if (PREVIOUS/name).is_file() else OLD/name", "cached=next((root/name for root in [PREVIOUS,EARLIER,OLD] if (root/name).is_file()),OLD/name)")
old="""    with requests.get(url,stream=True,timeout=(20,300)) as response:
     response.raise_for_status()
     with temp.open('wb') as file:
      for chunk in response.iter_content(1024*1024):file.write(chunk)
    temp.replace(dst)"""
new="""    errors=[]
    for attempt in range(1,4):
     try:
      with requests.get(url,stream=True,timeout=(20,60)) as response:
       response.raise_for_status()
       expected=response.headers.get('Content-Length')
       with temp.open('wb') as file:
        for chunk in response.iter_content(64*1024):
         if chunk:file.write(chunk)
       if expected is not None:assert temp.stat().st_size==int(expected),'Incomplete HTTP body'
      temp.replace(dst)
      break
     except (requests.RequestException,AssertionError) as error:
      errors.append({'attempt':attempt,'type':type(error).__name__,'message':str(error),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
      dump(O/'network_attempts'/(name+'.json'),{'url':url,'errors':errors,'max_attempts':3})
      if attempt==3:raise
      time.sleep(5*attempt)"""
assert old in s;s=s.replace(old,new)
s=s.replace("for n,record in enumerate(pool.map(fetch,rows),1):", "for n,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,row) for row in rows]),1):\n   record=future.result()")
s=s.replace("'training_started':False,'elapsed_seconds':time.perf_counter()-started", "'training_started':False,'elapsed_seconds':time.perf_counter()-started,'process_pid':os.getpid()")
entry=R/'scripts/prepare_sen1_full_pool_v3_20261004.py';assert not entry.exists();compile(s,str(entry),'exec');entry.write_text(s,encoding='utf-8')
draft=R/'scripts/freeze_sen1_full_pool_diagnostic_20261004.py'
assert not (R/'prepared_controls/sen1_full_pool_diagnostic_20261004_v1').exists(),'Do not change a frozen plan'
draft.write_text(draft.read_text(encoding='utf-8').replace('sen1_full_pool_preparation_20261004_v2','sen1_full_pool_preparation_20261004_v3'),encoding='utf-8')
print('DOWNLOAD_V3_READY: reuse previous complete files; max3 attempts/file; no training/test')
