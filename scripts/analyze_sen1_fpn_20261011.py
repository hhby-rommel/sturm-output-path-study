# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
import json, pathlib, statistics, hashlib, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sen1_fpn_comparison_20261011_v1'
OUT.mkdir(exist_ok=True)
systems = {}
sources = {}
for system, base, prefix in [
    ('native', 'sen1_formal_collection_20261005_v1/records', 'main_only'),
    ('decoder', 'sen1_formal_collection_20261005_v1/records', 'decoder'),
    ('decoder_BN_frozen', 'sen1_same_bn_formal_20261008_v1', 'decoder_all_bn_frozen'),
    ('FPN', 'sen1_fpn_formal_20261011_v1', 'fpn')]:
    systems[system] = []
    for seed in [2026, 2027, 2028]:
        f = ROOT / 'results' / base / f'{prefix}_seed{seed}' / 'test_evaluation.json'
        d = json.loads(f.read_text(encoding='utf-8-sig'))
        assert d['task']['seed'] == seed
        sources[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        systems[system].append(d)
reference = systems['native'][0]
for ds in systems.values():
    for d in ds:
        assert d['test_manifest_sha256'] == reference['test_manifest_sha256']
        assert d['scope_chips'] == 90 and d['scope_windows'] == 1440
        for view in ['clean', 'degraded']:
            for key in ['input_sha256', 'valid_domain_sha256']:
                assert d['metrics'][view][key] == reference['metrics'][view][key]
summary = {}; differences = {}
lines = ['# Sen1 三种子输出路径与 BN 配方比较', '',
         '基于保存的一次性 test 计数；未训练、未推理、未读取 test 像素。90 chips / 1440 windows，输入与有效域 SHA 全部一致。', '',
         '| 系统 | Clean F1 均值±样本SD | 扰动 F1 均值±样本SD |', '|---|---:|---:|']
for name, ds in systems.items():
    summary[name] = {}
    for view in ['clean', 'degraded']:
        a = [d['metrics'][view]['F1'] for d in ds]
        summary[name][view] = dict(seeds=a, mean=statistics.mean(a), sample_sd=statistics.stdev(a))
    lines.append('| ' + name + ' | ' + ' | '.join(f"{summary[name][v]['mean']:.4f}±{summary[name][v]['sample_sd']:.4f}" for v in ['clean','degraded']) + ' |')
lines += ['', '| 配对差（pp） | Clean 均值±SD | 扰动均值±SD | 扰动逐种子差 |', '|---|---:|---:|---|']
for a,b in [('decoder','native'),('decoder_BN_frozen','native'),('FPN','native'),('FPN','decoder'),('decoder','decoder_BN_frozen')]:
    label=f'{a} − {b}'; differences[label]={}
    for view in ['clean','degraded']:
        x=[100*(u['metrics'][view]['F1']-v['metrics'][view]['F1']) for u,v in zip(systems[a],systems[b])]
        differences[label][view]=dict(seeds_pp=x,mean_pp=statistics.mean(x),sample_sd_pp=statistics.stdev(x))
    q=differences[label]
    lines.append('| '+label+' | '+' | '.join(f"{q[v]['mean_pp']:+.3f}±{q[v]['sample_sd_pp']:.3f}" for v in ['clean','degraded'])+' | '+', '.join(f'{z:+.3f}' for z in q['degraded']['seeds_pp'])+' |')
lines += ['', 'decoder_BN_frozen 为 decoder 全部 BN 冻结控制，并非原生输出控制。FPN 与正式 decoder 使用主体 BN 冻结、新输出路径 BN 训练策略；两者是输出路径实现比较。Sen1 划分非事件隔离。', '', '三种子差异用于描述方向和波动，不据此重新选权或调整预算。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(dict(summary=summary,paired=differences,sources=sources,input_domain_sha_gate='PASS'),indent=2),encoding='utf-8')
print('\n'.join(lines))
