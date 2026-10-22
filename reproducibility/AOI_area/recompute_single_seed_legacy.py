"""Recompute Supplement Table G.4 using only Python's standard library.

Run from an archive checkout:
python reproducibility/AOI_area/recompute_AOI_area.py --output-dir AOI_recomputed
"""
from pathlib import Path
import argparse,csv,json,math,sys
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'source_data/fixed_AOI_area')
 p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();src=a.data_dir;out=a.output_dir
 assert not out.exists(),'Choose a new output directory; existing results are preserved'
 geometry=json.loads((src/'fixed_AOI_geometry_verified.json').read_text(encoding='utf-8-sig'))
 with (src/'fixed_AOI_existing_workflow_results.csv').open(encoding='utf-8-sig',newline='') as f:counts=list(csv.DictReader(f))
 with (src/'AOI_area_evaluation.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;expected=list(reader)
 assert len(counts)==len(expected)==6
 rows=[]
 for q,e in zip(counts,expected):
  sensor=q['sensor'];g=geometry[sensor];report=json.loads((src/f'{sensor}_scene_report.json').read_text(encoding='utf-8-sig'));model=report['models'][q['system']]
  assert g['CRS']==report['CRS']=='EPSG:32635' and g['projected_unit']=='metre'
  factor=g['projected_pixel_area_m2']/1e6
  assert g['projected_pixel_area_m2']==g['pixel_width_m']*g['pixel_height_m']==100
  assert g['unique_valid_pixels']==g['valid_tile_pixels']==report['valid_covered_pixels']
  assert g['overlap_valid_pixels']==report['overlap_pixels']==0
  assert g['max_overlap']==report['max_overlap']==1
  TP,FP,FN=(int(q[k]) for k in ['TP','FP','FN']);assert all(int(q[k])==model[k] for k in ['TP','FP','FN'])
  ref=TP+FN;pred=TP+FP
  assert ref==int(q['reference_water_pixels'])==g['reference_foreground_tile_pixels'] and pred==int(q['predicted_water_pixels'])
  F1=2*TP/(2*TP+FP+FN);bias=100*(pred-ref)/ref
  assert math.isclose(F1,float(q['F1']),abs_tol=1e-12) and math.isclose(F1,model['F1'],abs_tol=1e-12)
  assert math.isclose(bias,float(q['signed_extent_bias_percent']),abs_tol=1e-10)
  r=dict(sensor=sensor,event_id=q['event_id'],AOI=q['AOI'],system=q['system'],seed=int(q['seed']),condition=q['condition'],CRS=g['CRS'],pixel_area_m2=g['projected_pixel_area_m2'],valid_pixels=g['unique_valid_pixels'],reference_water_pixels=ref,predicted_water_pixels=pred,TP=TP,FP=FP,FN=FN,valid_coverage_km2=g['unique_valid_pixels']*factor,reference_water_area_km2=ref*factor,predicted_water_area_km2=pred*factor,signed_area_bias_km2=(pred-ref)*factor,absolute_area_error_km2=abs(pred-ref)*factor,signed_area_bias_percent=bias,absolute_area_error_percent=abs(bias),F1=F1)
  assert set(r)==set(fields)
  for key,value in r.items():
   if isinstance(value,(float,int)):assert math.isclose(value,float(e[key]),rel_tol=1e-12,abs_tol=1e-12),(sensor,q['system'],key)
   else:assert value==e[key]
  rows.append(r)
 for sensor in geometry:
  r=[x for x in rows if x['sensor']==sensor];assert len(r)==3 and len({x['valid_pixels'] for x in r})==1 and len({x['reference_water_pixels'] for x in r})==1
 out.mkdir(parents=True)
 with (out/'AOI_area_evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 receipt=dict(status='PASS',rows=6,table='Supplement Table G.4',checks=['Pixel area and metre-based CRS','Unique valid coverage and zero overlap','Three-way TP/FP/FN reconciliation','Reference/predicted area and signed/absolute error','Common per-sensor coverage across three paths','All archived result columns reproduced'],python=sys.version.split()[0],executed_at=datetime.now(timezone.utc).isoformat(),training=False,inference=False,raw_image_or_label_access=False)
 (out/'execution_record.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 print('PASS: all six AOI rows and Table G.4 values reproduced')
if __name__=='__main__':main()
