"""Reproduce Supplement Tables G.4 and G.5 from archived numerical records.

Python standard library only; no imagery, model loading or inference.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,importlib.util,json,math,statistics,sys
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'source_data/fixed_AOI_area');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 src=a.data_dir;out=a.output_dir;assert not out.exists(),'Choose a new output directory'
 legacy=Path(__file__).with_name('recompute_single_seed_legacy.py')
 spec=importlib.util.spec_from_file_location('single_seed_recomputation',legacy);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 oldargv=sys.argv;sys.argv=[str(legacy),'--data-dir',str(src),'--output-dir',str(out)]
 try:module.main()
 finally:sys.argv=oldargv
 (out/'execution_record.json').rename(out/'single_seed_execution_record.json')
 geometry=json.loads((src/'fixed_AOI_geometry_verified.json').read_text(encoding='utf-8-sig'))
 archived=readcsv(src/'AOI_area_all_five_seeds.csv');expected=readcsv(src/'AOI_area_five_seed_summary.csv');assert len(archived)==30 and len(expected)==6
 assert {(q['sensor'],q['system'],int(q['seed'])) for q in archived}=={(sensor,system,seed) for sensor in ('S1','S2') for system in ('main_only','decoder','fpn') for seed in range(2026,2031)}
 rebuilt=[]
 for q in archived:
  sensor=q['sensor'];seed=int(q['seed']);system=q['system'];g=geometry[sensor]
  report=json.loads((src/'execution_records'/f'seed{seed}'/sensor/'scene_report.json').read_text())
  assert report['selection']['event']=='EMSR292' and report['selection']['AOI']=='01CHRISOUPOLI'
  assert report['CRS']==g['CRS']=='EPSG:32635' and report['overlap_pixels']==0
  assert int(q['valid_pixels'])==report['valid_covered_pixels']==g['unique_valid_pixels'] and int(q['pixel_area_m2'])==g['projected_pixel_area_m2']==100
  counts=report['models'][system];TP,FP,FN=(int(q[k]) for k in ('TP','FP','FN'));assert all(int(q[k])==counts[k] for k in ('TP','FP','FN'))
  ref=TP+FN;pred=TP+FP;assert ref==int(q['reference_water_pixels'])==g['reference_foreground_tile_pixels'] and pred==int(q['predicted_water_pixels'])
  r=dict(q);r.update(valid_coverage_km2=int(q['valid_pixels'])*.0001,reference_water_area_km2=ref*.0001,predicted_water_area_km2=pred*.0001,signed_area_bias_km2=(pred-ref)*.0001,absolute_area_error_km2=abs(pred-ref)*.0001,signed_area_bias_percent=100*(pred-ref)/ref,absolute_area_error_percent=100*abs(pred-ref)/ref,F1=2*TP/(2*TP+FP+FN))
  for k in r:
   if k in ('TP','FP','FN','valid_pixels','reference_water_pixels','predicted_water_pixels','pixel_area_m2','seed') or isinstance(r[k],float):assert math.isclose(float(r[k]),float(q[k]),rel_tol=1e-12,abs_tol=1e-12),(sensor,system,seed,k)
  rebuilt.append(r)
 for q in readcsv(out/'AOI_area_evaluation.csv'):
  r=next(n for n in rebuilt if n['sensor']==q['sensor'] and n['system']==q['system'] and int(n['seed'])==2027)
  for k in ('TP','FP','FN','F1','predicted_water_area_km2','absolute_area_error_km2'):assert math.isclose(float(r[k]),float(q[k]),abs_tol=1e-10)
 summary=[];metrics=['predicted_water_area_km2','signed_area_bias_km2','absolute_area_error_km2','signed_area_bias_percent','absolute_area_error_percent','F1']
 for e in expected:
  group=[q for q in rebuilt if q['sensor']==e['sensor'] and q['system']==e['system']];assert len(group)==5
  r=dict(e)
  for k in metrics:
   x=[float(q[k]) for q in group];r[k+'_mean']=statistics.mean(x);r[k+'_sample_SD']=statistics.stdev(x)
   for suffix in ('_mean','_sample_SD'):assert math.isclose(r[k+suffix],float(e[k+suffix]),rel_tol=1e-12,abs_tol=1e-12),(e['sensor'],e['system'],k+suffix)
  summary.append(r)
 writecsv(out/'AOI_area_all_five_seeds.csv',rebuilt);writecsv(out/'AOI_area_five_seed_summary.csv',summary)
 receipt=dict(status='PASS',tables={'G.4':6,'G.5':6},per_seed_rows=30,seeds=list(range(2026,2031)),checks=['Per-seed confusion counts match scene reports','Common original-grid geometry and valid coverage','Six seed2027 rows agree with original Table G.4 records','Area and F1 regenerated from confusion counts','Six five-seed mean and sample-SD rows reproduced'],training=False,inference=False,imagery_access=False,python=sys.version.split()[0],time_utc=datetime.now(timezone.utc).isoformat())
 (out/'execution_record.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print('PASS: Table G.4 six rows, 30 per-seed rows, Table G.5 six summaries')
if __name__=='__main__':main()
