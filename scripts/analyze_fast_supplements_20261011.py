# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,statistics,hashlib,sys,numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R=Path(__file__).resolve().parents[1];V=R/'results/extent_calibration_val_20261011_v1';C=R/'results/cpu_matched_repeat_20261011_v1';O=R/'results/fast_supplement_analysis_20261011_v1';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((V/'reevaluation_complete.json').read_text())['baseline_val_event_counts_equal']
assert json.loads((C/'completion.json').read_text())['units']==18
receipt=json.loads((V/'threshold_freeze_receipt.json').read_text());assert receipt['selection_sha256']==sha(V/'selected_thresholds_frozen.json')
assert receipt['counts_sha256']==sha(V/'event_threshold_counts.json')
grid=json.loads((V/'event_threshold_counts.json').read_text());selected=json.loads((V/'selected_thresholds_frozen.json').read_text())['thresholds'];rows=[]
for model,views in grid['counts'].items():
 system,seed=model.rsplit('_',1)
 for view,es in views.items():
  for strategy in ['baseline','extent_selected','F1_selected']:
   events={}
   for event_key,counts in es.items():
    sensor,event=event_key.split('|');s=next(q for q in selected if q['model']==model and q['view']==view and q['sensor']==sensor)
    tau=.5 if strategy=='baseline' else s[strategy]['tau'];j=grid['taus'].index(tau);q=counts[j]
    e=events.setdefault(event,dict(TP=0,FP=0,FN=0))
    for k in e:e[k]+=q[k]
   assert len(events)==12
   valid=[q for q in events.values() if q['TP']+q['FN']>0];bias=[100*(q['FP']-q['FN'])/(q['TP']+q['FN']) for q in valid]
   tp=sum(q['TP'] for q in events.values());fp=sum(q['FP'] for q in events.values());fn=sum(q['FN'] for q in events.values())
   row=dict(system=system,seed=int(seed),view=view,strategy=strategy,pooled_F1=2*tp/(2*tp+fp+fn),event_macro_F1=statistics.mean(2*q['TP']/(2*q['TP']+q['FP']+q['FN']) if 2*q['TP']+q['FP']+q['FN'] else 0 for q in events.values()),signed_bias_percent=statistics.mean(bias),MARE_percent=statistics.mean(abs(x) for x in bias),median_absolute_error_percent=float(np.median(np.abs(bias))),events=events)
   rows.append(row)
def stats(xs):return dict(mean=statistics.mean(xs),sample_SD=statistics.stdev(xs),values=xs)
summary={}
for view in ['clean','degraded']:
 summary[view]={}
 for system in ['main_only','decoder','fpn']:
  summary[view][system]={}
  for strategy in ['baseline','extent_selected','F1_selected']:
   rs=sorted([q for q in rows if (q['view'],q['system'],q['strategy'])==(view,system,strategy)],key=lambda q:q['seed']);assert len(rs)==3
   metrics={k:stats([q[k] for q in rs]) for k in ['pooled_F1','event_macro_F1','signed_bias_percent','MARE_percent','median_absolute_error_percent']}
   if strategy!='baseline':
    metrics['paired_change_vs_baseline']={k:stats([q[k]-next(b[k] for b in rows if (b['view'],b['system'],b['seed'],b['strategy'])==(view,system,q['seed'],'baseline')) for q in rs]) for k in ['pooled_F1','event_macro_F1','MARE_percent']}
   summary[view][system][strategy]=metrics
cpu=json.loads((C/'summary.json').read_text())
output=dict(val_summary=summary,val_rows=rows,cpu=cpu,threshold_freeze=receipt,scope='Threshold selection and evaluation both on val; in-sample diagnostic, not held-out calibration result; CPU one session three weights, five blocks x50 repeats',training=False,test_access=False,sources={str(p):sha(p) for p in [V/'event_threshold_counts.json',V/'selected_thresholds_frozen.json',C/'summary.json']})
(O/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
t='# 快速补充实验：CPU复测与val阈值校正\n\n无训练、无test访问。阈值0.5的val逐事件与传感器计数、输入流和权重SHA与原评价一致。阈值规则在运行前固定，36项主阈值及次目标阈值已保存SHA。\n\n## CPU同批次\n\n四线程、batch1、FP32；同一进程，三份权重，各预热20次，5块×50次。均值和SD跨三份权重汇总；原始计时与块均值保留。\n\n| 系统 | S1 ms均值±权重SD | S2 ms均值±权重SD |\n| --- | ---: | ---: |\n'
for system in ['main_only','decoder','fpn']:
 a,b=cpu[system+'/S1'],cpu[system+'/S2'];t+=f"| {system} | {a['mean_ms']:.3f}±{a['weight_sample_SD_ms']:.3f} | {b['mean_ms']:.3f}±{b['weight_sample_SD_ms']:.3f} |\n"
t+='\n## val阈值诊断\n\n按传感器选择阈值后，将同一事件的S1/S2计数相加，再对十二事件等权。校正收益为选阈值集内结果；test转移效果未检验。MARE为有效瓦片域水体前景平均绝对相对误差，非去重km²洪水面积。\n\n| 条件／系统／策略 | MARE %均值±SD | pooled F1均值 | event-macro F1均值 |\n| --- | ---: | ---: | ---: |\n'
for view,ss in summary.items():
 for system,strategies in ss.items():
  for strategy,q in strategies.items():t+=f"| {view}／{system}／{strategy} | {q['MARE_percent']['mean']:.2f}±{q['MARE_percent']['sample_SD']:.2f} | {q['pooled_F1']['mean']:.4f} | {q['event_macro_F1']['mean']:.4f} |\n"
t+='\n## 阈值及待执行范围\n\n| 条件／模型／传感器 | 范围目标τ | F1目标τ | 有效相对误差事件数 |\n| --- | ---: | ---: | ---: |\n'
for q in selected:t+=f"| {q['view']}／{q['model']}／{q['sensor']} | {q['extent_selected']['tau']:.2f} | {q['F1_selected']['tau']:.2f} | {q['extent_selected']['relative_error_events']} |\n"
t+=f"\n冻结选择SHA256：`{receipt['selection_sha256']}`。\n\n尚未执行test校正、Sen1-FPN训练、FP16/FLOPs、Otsu或新增AOI。既有稿件与历史正式结果保留，本报告独立存放。\n"
(O/'report.md').write_text(t,encoding='utf-8');print(t)
