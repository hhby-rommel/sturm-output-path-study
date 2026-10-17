"""Local spatial cache audit and predefined summaries; no model evaluation."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import csv,json,statistics as st
import numpy as np

R=Path(__file__).resolve().parents[1]
S=R/'results/EMSR551_spatial_20261017_v1'
O=R/'results/EMSR551_spatial_analysis_20261017_v1'
assert not O.exists()
assert json.loads((S/'local_handoff_receipt.json').read_text())['status']=='HANDOFF_VERIFIED_LOCAL'
completion=json.loads((S/'completion.json').read_text());assert completion['status']=='EMSR551_SPATIAL_COMPLETE'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
regions=rows(S/'event_region_counts.csv');tile_rows=rows(S/'tile_region_counts.csv')
assert len(regions)==270 and len(tile_rows)==2430
idx={}
for q in regions:
    for k in ('seed','radius','TP','FP','FN','valid_pixels','reference_foreground','signed_imbalance'):q[k]=int(q[k])
    key=(q['system'],q['seed'],q['condition'],q['radius'],q['region'])
    assert key not in idx;idx[key]=q
systems=('main_only','decoder','fpn');views=('clean','degraded');seeds=range(2026,2031)
names=('boundary','off_boundary','excluded_neighbourhood');radii=(2,1,3)
with (S/'server_control/EMSR551_expected_five_seed_counts.csv').open(encoding='utf-8-sig') as f:
    expected={(q['system'],int(q['seed']),q['view']):q for q in csv.DictReader(f)}
tiles=sorted(p.name.removesuffix('.target.npy') for p in (S/'cache').glob('*.target.npy'))
assert len(tiles)==9
overall={};cache_checks=[]
for system in systems:
    for seed in seeds:
        for view in views:
            totals=dict(TP=0,FP=0,FN=0,reference_foreground=0,valid_pixels=0)
            for tile in tiles:
                target=np.load(S/'cache'/f'{tile}.target.npy',allow_pickle=False)
                original=np.load(S/'cache'/f'{tile}.original_valid.npy',allow_pickle=False)
                valid=original if view=='clean' else np.load(S/'cache'/f'{tile}.degraded_valid.npy',allow_pickle=False)
                pred=np.load(S/'cache'/f'{system}_{seed}.{view}.{tile}.prediction.npy',allow_pickle=False)
                assert target.shape==pred.shape==valid.shape==original.shape==(128,128)
                assert original.dtype==valid.dtype==np.dtype(bool) and not np.any(valid&~original)
                assert np.isin(pred,[0,1]).all()
                domain=valid & (target!=255)
                for k,mask in [('TP',(target==1)&(pred==1)),('FP',(target==0)&(pred==1)),('FN',(target==1)&(pred==0))]:totals[k]+=int((mask&domain).sum())
                totals['reference_foreground']+=int(((target==1)&domain).sum());totals['valid_pixels']+=int(domain.sum())
            old=expected[system,seed,view]
            assert all(totals[k]==int(old[k]) for k in ('TP','FP','FN'))
            for radius in radii:
                parts=[idx[system,seed,view,radius,n] for n in names]
                assert all(sum(q[k] for q in parts)==totals[k] for k in totals)
                assert sum(q['signed_imbalance'] for q in parts)==totals['FP']-totals['FN']
            ref=totals['reference_foreground'];assert ref==totals['TP']+totals['FN']>0
            bias=100*(totals['FP']-totals['FN'])/ref
            overall[system,seed,view]=dict(system=system,seed=seed,condition='perturbed' if view=='degraded' else view,
                **totals,signed_imbalance=totals['FP']-totals['FN'],F1=2*totals['TP']/(2*totals['TP']+totals['FP']+totals['FN']),
                signed_bias_percent=bias,absolute_extent_error_percent=abs(bias))
            cache_checks.append(dict(system=system,seed=seed,condition=view,tiles=9,cache_counts_equal=True,all_radii_reconciled=True))

def stats(values):return st.mean(values),st.stdev(values)
summary=[];region_summary=[];overall_paired=[];region_paired=[]
for view in views:
    condition='perturbed' if view=='degraded' else view
    for system in systems:
        rr=[overall[system,n,view] for n in seeds]
        q=dict(system=system,condition=condition,seeds=5)
        for metric in ('FP','FN','signed_imbalance','F1','signed_bias_percent','absolute_extent_error_percent'):
            q[metric+'_mean'],q[metric+'_sample_SD']=stats([r[metric] for r in rr])
        q['positive_bias_seeds']=sum(r['signed_imbalance']>0 for r in rr)
        summary.append(q)
        for radius in radii:
            for region in names:
                rr=[idx[system,n,view,radius,region] for n in seeds]
                q=dict(system=system,condition=condition,radius=radius,region=region,seeds=5)
                for metric in ('FP','FN','signed_imbalance','reference_foreground','valid_pixels'):
                    q[metric+'_mean'],q[metric+'_sample_SD']=stats([r[metric] for r in rr])
                for metric in ('FP','FN'):
                    shares=[100*idx[system,n,view,radius,region][metric]/overall[system,n,view][metric] for n in seeds if overall[system,n,view][metric]>0]
                    q[metric+'_share_percent_mean']=st.mean(shares) if shares else None
                    q[metric+'_share_defined_seeds']=len(shares)
                region_summary.append(q)
    for a,b in [('decoder','main_only'),('fpn','main_only'),('fpn','decoder')]:
        diffs=[]
        for n in seeds:
            x,y=overall[a,n,view],overall[b,n,view]
            assert x['reference_foreground']==y['reference_foreground']
            ref=x['reference_foreground'];d=dict(condition=condition,comparison=a+'-'+b,seed=n,
                delta_FP=x['FP']-y['FP'],delta_FN=x['FN']-y['FN'],
                delta_signed_imbalance=x['signed_imbalance']-y['signed_imbalance'],
                delta_signed_bias_pp=x['signed_bias_percent']-y['signed_bias_percent'],
                delta_absolute_extent_error_pp=x['absolute_extent_error_percent']-y['absolute_extent_error_percent'])
            diffs.append(d)
            for radius in radii:
                signed_sum=0
                for region in names:
                    x,y=idx[a,n,view,radius,region],idx[b,n,view,radius,region]
                    dfp,dfn=x['FP']-y['FP'],x['FN']-y['FN'];signed_sum+=dfp-dfn
                    region_paired.append(dict(condition=condition,comparison=a+'-'+b,seed=n,radius=radius,region=region,
                        delta_FP=dfp,delta_FN=dfn,delta_signed_imbalance=dfp-dfn,
                        delta_signed_bias_contribution_pp=100*(dfp-dfn)/ref))
                assert signed_sum==d['delta_signed_imbalance']
        q=dict(condition=condition,comparison=a+'-'+b,seeds=5,improved_extent_seeds=sum(d['delta_absolute_extent_error_pp']<0 for d in diffs))
        for metric in ('delta_FP','delta_FN','delta_signed_imbalance','delta_signed_bias_pp','delta_absolute_extent_error_pp'):
            q[metric+'_mean'],q[metric+'_paired_SD']=stats([d[metric] for d in diffs])
        overall_paired.append(q)

paired_summary=[]
for view in ('clean','perturbed'):
    for comparison in ('decoder-main_only','fpn-main_only','fpn-decoder'):
        for radius in radii:
            for region in names:
                rr=[q for q in region_paired if (q['condition'],q['comparison'],q['radius'],q['region'])==(view,comparison,radius,region)]
                assert len(rr)==5
                q=dict(condition=view,comparison=comparison,radius=radius,region=region,seeds=5)
                for metric in ('delta_FP','delta_FN','delta_signed_imbalance','delta_signed_bias_contribution_pp'):
                    q[metric+'_mean'],q[metric+'_paired_SD']=stats([r[metric] for r in rr])
                paired_summary.append(q)

O.mkdir()
def save(name,rr):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for name,rr in [('overall_per_seed.csv',list(overall.values())),('path_condition_summary.csv',summary),
                ('path_condition_region_summary.csv',region_summary),('paired_overall_changes.csv',overall_paired),
                ('paired_region_changes_per_seed.csv',region_paired),('paired_region_summary.csv',paired_summary),
                ('cache_count_audit.csv',cache_checks)]:save(name,rr)
lines=['# EMSR551 空间误差分解：五 seed / fixed12026', '',
    '已完成：15个正式checkpoint × clean/perturbed × 9个目标tiles。保持原两个batch上下文，共960个实际tile forwards；仅EMSR551输出和统计。',
    '270个目标prediction缓存完整；30组总体TP/FP/FN与既有固定输入结果完全一致。主边界半径2 pixels；1/3 pixels结果同时保存。', '',
    '## 总体事件结果', '',
    '| 条件 | 路径 | FP mean | FN mean | FP−FN mean | absolute extent error mean (%) |',
    '|---|---|---:|---:|---:|---:|']
for q in summary:lines.append(f"| {q['condition']} | {q['system']} | {q['FP_mean']:.1f} | {q['FN_mean']:.1f} | {q['signed_imbalance_mean']:+.1f} | {q['absolute_extent_error_percent_mean']:.2f} |")
lines+=['','## ±2 pixels 主分区', '',
    '份额先在每个seed内除以该seed总体FP或FN，再对五seed平均。boundary与off-boundary之外的tile-edge/original-invalid邻域单列，三者相加恢复总体。', '',
    '| 条件 | 路径 | 分区 | FP mean | FN mean | FP份额 mean (%) | FN份额 mean (%) |',
    '|---|---|---|---:|---:|---:|---:|']
for q in region_summary:
    if q['radius']==2:lines.append(f"| {q['condition']} | {q['system']} | {q['region']} | {q['FP_mean']:.1f} | {q['FN_mean']:.1f} | {q['FP_share_percent_mean']:.2f} | {q['FN_share_percent_mean']:.2f} |")
lines+=['', '## 相对 Native 的配对变化', '',
    '| 条件 | 比较 | ΔFP mean | ΔFN mean | Δ(FP−FN) mean | Δabsolute extent error mean (pp) | extent改善seeds |',
    '|---|---|---:|---:|---:|---:|---:|']
for q in overall_paired:
    if q['comparison'].endswith('-main_only'):lines.append(f"| {q['condition']} | {q['comparison']} | {q['delta_FP_mean']:+.1f} | {q['delta_FN_mean']:+.1f} | {q['delta_signed_imbalance_mean']:+.1f} | {q['delta_absolute_extent_error_pp_mean']:+.2f} | {q['improved_extent_seeds']}/5 |")
lines+=['','## 解释边界', '',
    '- off-boundary包含远离reference轮廓的水体内部和背景区域，不能一概称为interior。',
    '- 删除像素只改变评价有效域，不生成新的reference边界；tile边缘/原始无效邻域误差完整保留在excluded分区。',
    '- FP−FN及其配对变化可以跨分区相加；分区失衡的绝对值不能相加解释总体absolute extent error。',
    '- 总体absolute extent error按每个seed先取绝对值再平均；不能对五seed平均signed bias直接取绝对值。',
    '- 空间位置不证明物理原因。没有改变边界定义、阈值、checkpoint或输入流来追求更清楚的模式。',
    '- 未扩展EMSR518/492/422；未修改论文、训练模型、发布release或制作ZIP。',
    '- 本地全部文件已回收，后续分析无需服务器。']
(O/'RESULTS_README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(O/'completion.json').write_text(json.dumps(dict(status='SPATIAL_CACHE_AUDIT_AND_SUMMARY_COMPLETE',time_utc=datetime.now(timezone.utc).isoformat(),
    prediction_caches=270,model_conditions=30,baseline_counts_equal=True,all_radii_reconcile=True,
    radius_definitions_unchanged=True,training=False,inference=False,new_test_access=False,server_needed=False,
    source=str(S),other_events=False),indent=2),encoding='utf-8')
print(json.dumps(dict(status='SPATIAL_CACHE_AUDIT_AND_SUMMARY_COMPLETE',summary=str(O/'RESULTS_README.md'))))
