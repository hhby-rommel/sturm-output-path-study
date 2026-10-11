# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,statistics,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parents[1];D=R/'results/quick_diagnostics_20261011_v1';P=R/'results/attention_numeric_probe_20261011_v2';O=R/'results/quick_diagnostics_analysis_20261011_v1';O.mkdir(exist_ok=True)
assert json.loads((D/'completion.json').read_text())['runtime_failed']==0
numerics=json.loads((D/'numerics.json').read_text());assert len(numerics)==270
cross=json.loads((D/'fvcore_crosscheck.json').read_text());assert len(cross)==6 and all(q['relative_difference']==0 for q in cross)
costs=json.loads((D/'size_batch_costs.json').read_text());assert len(costs)==144;summary={}
for system in ['main_only','decoder','fpn']:
 for sensor in ['S1','S2']:
  for size,batch in [(128,1),(128,16),(128,32),(128,64),(256,1),(256,8),(512,1),(512,4)]:
   rows=sorted([q for q in costs if (q['system'],q['sensor'],q['size'],q['batch'])==(system,sensor,size,batch)],key=lambda q:q['seed']);assert len(rows)==3
   for row in rows:
    raw=sum(row['blocks_raw_ms'],[]);assert len(raw)==60 and abs(statistics.mean(raw)-row['mean_ms'])<1e-12
   a=[q['mean_ms'] for q in rows];summary[f'{system}/{sensor}/{size}/b{batch}']=dict(mean_ms=statistics.mean(a),sample_SD_ms=statistics.stdev(a),weight_means_ms=a)
counts={}
for system in ['main_only','decoder','fpn']:
 for precision in ['FP32','FP16_autocast','BF16_autocast']:
  r=[q for q in numerics if q['system']==system and q['precision']==precision];assert len(r)==30
  counts[system+'/'+precision]=dict(units=len(r),nonfinite=sum(not q['finite'] for q in r),real_val_units=sum(q['fixture']!='zero' for q in r),real_val_nonfinite=sum(q['fixture']!='zero' and not q['finite'] for q in r))
probe=json.loads((P/'probe.json').read_text());assert len(probe['rows'])==6
for row in probe['rows']:
 x=row['attention_QK'][0]
 assert row['output_finite']==(x['actual_pre_scaled_QK_FP16_inf']==0)
 if not row['output_finite']:assert x['qkv_finite'] and x['QK_raw_FP32_finite'] and x['actual_pre_scaled_softmax_nan']>0
grid=json.loads((R/'results/extent_calibration_val_20261011_v1/event_threshold_counts.json').read_text())['counts'];bias={}
for system in ['main_only','decoder','fpn']:
 for view in ['clean','degraded']:
  for sensor in ['S1','S2']:
   means=[]
   for seed in [2026,2027,2028]:
    rs=[rows[9] for key,rows in grid[f'{system}_{seed}'][view].items() if key.startswith(sensor+'|')]
    means.append(statistics.mean(100*(r['FP']-r['FN'])/(r['TP']+r['FN']) for r in rs if r['TP']+r['FN']>0))
   bias[f'{system}/{view}/{sensor}']=dict(mean_signed_bias_percent=statistics.mean(means),values=means)
hybrid=json.loads((R/'results/hybrid_sensor_diagnostic_20261011_v1/analysis.json').read_text())
result=dict(hybrid_pooled_diagnostic=hybrid,numeric_gate=counts,size_batch_summary=summary,fvcore_crosscheck=cross,attention_probe=probe,val_sensor_signed_bias=bias,training=False,test_access=False,scope='Fixed clean val fixtures and synthetic cost inputs; no full precision accuracy evaluation; runtime scope is specified dimensions/batches; intermediate reconstruction follows actual pre-scaling order',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'numerics.json',D/'size_batch_costs.json',P/'probe.json']})
(O/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
t='# 精度数值、算子计数与尺寸/batch核查\n\n本轮无训练、无test访问。九份冻结权重、八张规则选定的clean val瓦片（每传感器四个不同事件首瓦片）与两个零值夹具，FP32/FP16/BF16共270个数值单元。有效值检查不是完整精度评价。\n\n## 1. 数值检查\n\n| 系统／模式 | 总单元 | 非有限单元 | 真实val非有限／真实val单元 |\n| --- | ---: | ---: | --- |\n'
for key,q in counts.items():t+=f"| {key} | {q['units']} | {q['nonfinite']} | {q['real_val_nonfinite']}／{q['real_val_units']} |\n"
t+='''\nFP16失败包括FPN2026的五张真实val输入、decoder2028的五张真实val输入，以及FPN2028的S1零值输入。首个hook可见异常为共同主体core.model.10.m.0.attn.proj.conv，其输入已非有限；问题不是FPN专有的输出头故障。源码采用先缩放Q再QK相乘，第二版中间值探针按实际顺序重建该计算：失败案例的QKV有限，预缩放后的FP16 QK出现Inf，随后softmax出现NaN；同一QKV用FP32重建有限。原未缩放QK诊断保留，不用它替代实际计算顺序。

BF16在固定夹具上全部有限，但尚未评估完整val的BF16分割F1，不能直接宣称部署精度无损。权重及训练状态未改，未将注意力改成FP32来规避失败。\n'''
t+='\n## 2. fvcore交叉核对\n\n独立fvcore算子图计数与原hook/profiler在六个系统×传感器单元上完全一致。比较口径为卷积、转置卷积、线性与矩阵乘法的MAC×2；其他支持/不支持算子另存。FPN的高分辨率34通道3×3平滑卷积单独贡献0.452782 GFLOPs，解释其主要算子量较大；参数匹配不等于算子量匹配。\n'
t+='\n## 3. FP32尺寸/batch同批测量\n\n同一进程、九份权重、两传感器、八个预定组合，共144单元。每单元预热10次、3块×20次，零张量输入，返回与输入同尺寸二类logits；不是更大尺寸的精度验证或地理拼接。均值±SD跨三份权重，所有原始计时保留。\n\n| 传感器／尺寸／batch | 原生ms | decoder ms | FPN ms | FPN相对decoder耗时差% |\n| --- | ---: | ---: | ---: | ---: |\n'
for sensor in ['S1','S2']:
 for size,batch in [(128,1),(128,16),(128,32),(128,64),(256,1),(256,8),(512,1),(512,4)]:
  a=[summary[f'{sys}/{sensor}/{size}/b{batch}'] for sys in ['main_only','decoder','fpn']]
  t+=f'| {sensor}／{size}／{batch} | '+' | '.join(f"{x['mean_ms']:.3f}±{x['sample_SD_ms']:.3f}" for x in a)+f" | {100*(a[2]['mean_ms']/a[1]['mean_ms']-1):+.2f} |\n"
t+='''\n新批次没有显示“输入越大FPN必然更慢”的单调规律：S1的128输入下FPN略慢，512下略快；S2在多数单元中FPN较快，128/b64略慢。小幅差异随条件与记录批次变化，不能以部分算子量直接推导延迟，也不能由这些计时单独认定kernel launch是主因。

## 4. 阈值解释范围\n\n未校正val的扰动S2平均有符号偏差为原生+49.92%、decoder+41.32%、FPN+47.66%；S1为−7.63%、−11.74%、+0.96%。仅按“val整体过估”解释并不准确；上调S2阈值与该分项过估方向相符。test基线总体低估与校正后低估加重并存，仍不能单凭前景比例确认分布转移的因果解释。

原test阶段只保存冻结阈值与0.5基线，没有完整test τ网格；未再访问test。另用已保存的传感器总计数构造S1-only/S2-only的探索性pooled替换诊断，不作为校正方法或阈值选择；事件层面的0.5传感器细分计数不足，因此不报告该替换的MARE。“本方案未迁移”不能扩大为所有阈值方法均不可校正，“0.5优于本次冻结策略”不能称为test上全网格最优。\n'''
t+='\n只对S2应用范围阈值时，扰动pooled F1相对0.5平均下降为原生1.12、decoder1.49、FPN1.09 pp；只对S1则为0.47、0.60、2.15 pp。FPN的S1替换下降更大，不能笼统将全部退化归于S2。F1非线性，替换差不作可加的因果贡献分解。\n'
(O/'report.md').write_text(t,encoding='utf-8');print(t)
