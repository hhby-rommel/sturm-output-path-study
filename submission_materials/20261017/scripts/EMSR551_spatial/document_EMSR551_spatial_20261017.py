"""Attach exposure denominators and completed-execution handoff to the report."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,shutil
R=Path(__file__).resolve().parents[1];O=R/'results/EMSR551_spatial_analysis_20261017_v1'
S=R/'results/EMSR551_spatial_20261017_v1';P=R/'results/spatial_operational_preparation_20261017_v1'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
data=rows(S/'event_region_counts.csv');exposure=[]
for view in ('clean','degraded'):
    for radius in (2,1,3):
        rr=[q for q in data if q['system']=='main_only' and q['seed']=='2026' and q['condition']==view and int(q['radius'])==radius]
        assert len(rr)==3
        valid=sum(int(q['valid_pixels']) for q in rr);ref=sum(int(q['reference_foreground']) for q in rr)
        for q in rr:
            exposure.append(dict(condition='perturbed' if view=='degraded' else view,radius=radius,region=q['region'],
                valid_pixels=int(q['valid_pixels']),valid_domain_share_percent=100*int(q['valid_pixels'])/valid,
                reference_foreground=int(q['reference_foreground']),reference_foreground_share_percent=100*int(q['reference_foreground'])/ref))
with (O/'region_exposure.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(exposure[0]));w.writeheader();w.writerows(exposure)
lines=['','## 分区本身的覆盖量（避免把错误份额当作错误密度）','',
    '| 条件 | ±2 pixels分区 | 评价有效域份额 (%) | reference水体份额 (%) |',
    '|---|---|---:|---:|']
for q in exposure:
    if q['radius']==2:lines.append(f"| {q['condition']} | {q['region']} | {q['valid_domain_share_percent']:.2f} | {q['reference_foreground_share_percent']:.2f} |")
lines+=['',
    '边界带约占9%的有效域，却包含约78%的reference水体。因此，79–82%的FN在边界带，不能单独证明模型在边界处特别容易出错。off-boundary约占82%的有效域，其较高FP份额也不等同于更高错误密度。', '',
    '## 可以从计数直接读出的变化', '',
    '- Perturbed Decoder−Native：FP均值增加2148.0，FN减少695.2；总体FP−FN均值从−3850.0移至−1006.8。off-boundary的ΔFP为+1697.2、ΔFN为−255.2，边界带为+232.6和−440.2。extent error从54.26%降至28.57%，4/5 seeds改善。这是误检与遗漏平衡变化，不能只写成边界恢复或漏报减少。',
    '- Perturbed FPN−Native：FP均值增加4445.6，FN减少1036.6；off-boundary的ΔFP为+3694.2。FPN五seed signed bias仍跨越负值与正值（−48.90%到+119.17%）。总体absolute extent error均值增加0.78pp，但4/5个seed实际下降；应同时报告均值与seed分布。',
    '- Clean FPN−Native：FP均值增加2760.2，FN减少993.6，absolute extent error平均下降23.42pp，5/5 seeds改善。Clean Decoder的FP和FN分别减少302.0和285.4，absolute extent error均值增加0.20pp，3/5 seeds改善。',
    '- 这些是EMSR551当前checkpoint与条件下的空间位置和计数解释，不证明物理原因，也不外推为其他事件的相同机制。半径1和3的全部预设结果保留在CSV中；不据此搜索更有利的边界定义。', '',
    '本轮终点：完成EMSR551后停止。EMSR518/492/422尚未执行；是否扩展需作者审阅这批结果后决定。']
report=O/'RESULTS_README.md';text=report.read_text(encoding='utf-8');assert '## 分区本身的覆盖量' not in text
report.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
receipt=json.loads((S/'completion.json').read_text());receipt.update(local_cache_audit='PASS',local_handoff='319 files MATCH;270 prediction caches',
    local_results=str(O),server_no_longer_needed=True,manuscript_changed=False,other_events_started=False,
    documented_utc=datetime.now(timezone.utc).isoformat())
(P/'execution_completion_20261017.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
p=P/'README.md';text=p.read_text(encoding='utf-8');p.write_text('> 2026-10-17执行更新：EMSR551正式空间补评价已完成，319文件已回收，本地缓存/计数审计通过，服务器不再需要。结果见 `../EMSR551_spatial_analysis_20261017_v1/RESULTS_README.md`。下文保留执行前准备记录；固定AOI结果继续复用，无其他事件推理。\n\n'+text,encoding='utf-8')
for name in ['analyze_EMSR551_spatial_20261017.py','receive_EMSR551_spatial_20261017.py','document_EMSR551_spatial_20261017.py']:
    d=O/'scripts';d.mkdir(exist_ok=True);shutil.copy2(R/'scripts'/name,d/name)
print(json.dumps(dict(status='RESULTS_DOCUMENTED',server_needed=False,manuscript_changed=False)))
