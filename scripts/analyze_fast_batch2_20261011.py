# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,statistics,hashlib,sys,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parents[1];T=R/'results/extent_calibration_test_20261011_v1';H=R/'results/fp16_flops_20261011_v1';O=R/'results/fast_batch2_analysis_20261011_v1';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
completion=json.loads((T/'reevaluation_complete.json').read_text());assert completion['baseline_fixed_test_counts_equal']
assert completion['selection_sha256']==sha(R/'results/extent_calibration_val_20261011_v1/selected_thresholds_frozen.json')
CONT=R/'results/fp16_flops_continue_20261011_v1'
precision_completion=json.loads((CONT/'completion.json').read_text());assert precision_completion['units_accounted']==72
invalid=json.loads((CONT/'invalid_units.json').read_text())
cal=json.loads((T/'calibrated_counts.json').read_text())['calibrated'];rows=[]
for model,views in cal.items():
 system,seed=model.rsplit('_',1);prior=json.loads((T/f'{model}_reevaluation.json').read_text())['metrics']
 for view,strategies in views.items():
  for strategy in ['baseline','extent_selected','F1_selected']:
   if strategy=='baseline':
    events={e:{'TP':m['confusion_matrix'][1][1],'FP':m['confusion_matrix'][0][1],'FN':m['confusion_matrix'][1][0]} for e,m in prior[view]['per_event'].items()}
   else:
    events={}
    for key,q in strategies[strategy].items():
     sensor,event=key.split('|');e=events.setdefault(event,dict(TP=0,FP=0,FN=0))
     for k in e:e[k]+=q[k]
    for e,q in events.items():
     cm=prior[view]['per_event'][e]['confusion_matrix'];assert q['TP']+q['FN']==cm[1][1]+cm[1][0]
   assert len(events)==12
   valid=[q for q in events.values() if q['TP']+q['FN']>0];bias=[100*(q['FP']-q['FN'])/(q['TP']+q['FN']) for q in valid]
   a=sum(q['TP'] for q in events.values());b=sum(q['FP'] for q in events.values());c=sum(q['FN'] for q in events.values())
   rows.append(dict(system=system,seed=int(seed),view=view,strategy=strategy,pooled_F1=2*a/(2*a+b+c),event_macro_F1=statistics.mean(2*q['TP']/(2*q['TP']+q['FP']+q['FN']) if 2*q['TP']+q['FP']+q['FN'] else 0 for q in events.values()),MARE_percent=statistics.mean(abs(x) for x in bias),signed_bias_percent=statistics.mean(bias),median_percent=float(np.median(np.abs(bias))),Q1_percent=float(np.quantile(np.abs(bias),.25)),Q3_percent=float(np.quantile(np.abs(bias),.75)),events=events))
def stats(a):return dict(mean=statistics.mean(a),sample_SD=statistics.stdev(a),values=a)
summary={}
for view in ['clean','degraded']:
 summary[view]={}
 for system in ['main_only','decoder','fpn']:
  summary[view][system]={}
  for strategy in ['baseline','extent_selected','F1_selected']:
   rs=sorted([r for r in rows if (r['view'],r['system'],r['strategy'])==(view,system,strategy)],key=lambda r:r['seed']);assert len(rs)==3
   metrics={k:stats([r[k] for r in rs]) for k in ['MARE_percent','signed_bias_percent','pooled_F1','event_macro_F1','median_percent','Q1_percent','Q3_percent']}
   if strategy!='baseline':
    metrics['paired_change_vs_baseline']={k:stats([r[k]-next(b[k] for b in rows if (b['view'],b['system'],b['seed'],b['strategy'])==(view,system,r['seed'],'baseline')) for r in rs]) for k in ['MARE_percent','pooled_F1','event_macro_F1']}
   summary[view][system][strategy]=metrics
half=json.loads((CONT/'summary.json').read_text());flops={}
for f in H.glob('flops_*.json'):
 q=json.loads(f.read_text());flops[q['system']+'/'+q['sensor']]=q['major_operator_GFLOPs']
for f in list(H.glob('*_202*_b*_*.json'))+list(CONT.glob('*_202*_b*_*.json')):
 q=json.loads(f.read_text());
 if 'blocks_raw_ms' not in q:continue
 raw=sum(q['blocks_raw_ms'],[]);assert len(raw)==150 and abs(statistics.mean(raw)-q['mean_ms'])<1e-10
output=dict(test_summary=summary,test_rows=rows,precision_cost=half,invalid_FP16_units=invalid,precision_completion=precision_completion,major_operator_GFLOPs=flops,selection_sha256=completion['selection_sha256'],training=False,test_access_in_experiment=True,test_selection=False,scope='Fixed val-selected threshold transfer; FP16 costs on zeros only, not segmentation accuracy; FLOPs excludes other operators',sources={str(p):sha(p) for p in [T/'calibrated_counts.json',CONT/'summary.json']})
(O/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
t='# 第二批快速补充：冻结阈值test转移、FP16成本与主要算子计数\n\n## 校正结果\n\n九份冻结权重对test执行一次遍历，使用已固定的val阈值；没有test网格或重新选阈值。阈值0.5的全部计数与原固定输入test一致，输入流及权重不变。MARE为有效瓦片域水体前景的事件平均绝对相对误差，单位%；不是去重km²洪水面积。下表均值/SD跨三个种子；median/Q1/Q3的每种子结果与逐事件计数保留在analysis.json。\n\n| 条件／系统／策略 | MARE %均值±SD | 平均有符号偏差 % | pooled F1 | event-macro F1 |\n| --- | ---: | ---: | ---: | ---: |\n'
for view,ss in summary.items():
 for system,strategies in ss.items():
  for strategy,q in strategies.items():t+=f"| {view}／{system}／{strategy} | {q['MARE_percent']['mean']:.2f}±{q['MARE_percent']['sample_SD']:.2f} | {q['signed_bias_percent']['mean']:+.2f} | {q['pooled_F1']['mean']:.4f} | {q['event_macro_F1']['mean']:.4f} |\n"
t+='\n## GPU精度模式的前向成本\n\nRTX4090D，同一会话；三权重、两传感器、batch1/16、FP32与FP16 autocast，共72个预定单元；原批次在FP16非有限输出时停止，历史计时保留、缺失单元在第二会话补齐。预热20次、3块×50次。两种模式均计入原生上采样，输入为零张量。无效FP16单元无计时；下表显式列有效权重数，不能将两权重通过写成三权重安全。SD及全部原始计时在数据文件中。本批FP32不与历史批次混算。零值fixture的logit差仅是数值诊断，未评价FP16的分割精度。\n\n| 系统／传感器 | FP32 B1 ms | FP16 B1 ms | FP32 B16 ms | FP16 B16 ms |\n| --- | ---: | ---: | ---: | ---: |\n'
for system in ['main_only','decoder','fpn']:
 for sensor in ['S1','S2']:
  a=[half[f'{system}/{sensor}/b{b}/{p}'] for b,p in [(1,'FP32'),(1,'FP16_autocast'),(16,'FP32'),(16,'FP16_autocast')]]
  t+=f'| {system}／{sensor} | '+' | '.join(f"{x['mean_ms']:.3f}（n={x['valid_weights']}）" if x['mean_ms'] is not None else '无有效值' for x in a)+' |\n'
t+='\n无效单元：'+', '.join(q['unit'] for q in invalid)+'。原始失败记录及诊断重放保留；未修改权重、autocast模式或阈值。\n'
t+='\n## 主要算子GFLOPs\n\nbatch1、128×128：卷积/转置卷积/线性MAC×2，加mm/bmm/addmm的profiler FLOPs。BN、激活、池化、逐元素加法、softmax、插值未计；这是主要算子计数，不能称完整总FLOPs。\n\n| 系统 | S1 GFLOPs | S2 GFLOPs |\n| --- | ---: | ---: |\n'
for system in ['main_only','decoder','fpn']:t+=f"| {system} | {flops[system+'/S1']:.4f} | {flops[system+'/S2']:.4f} |\n"
t+='\nSen1-FPN与其他训练均未启动，历史正式结果和失败记录保留。本报告独立存放，尚未将新结果写入论文或发布仓库。\n'
(O/'report.md').write_text(t,encoding='utf-8');print(t)
