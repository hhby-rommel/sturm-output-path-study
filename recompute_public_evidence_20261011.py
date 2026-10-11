# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recompute seed means and event pairing from the public saved-evidence bundle only."""
import json, pathlib, statistics, numpy as np

p = pathlib.Path(__file__).resolve().parent / 'evidence_20261011_v17.json'
bundle = json.loads(p.read_text(encoding='utf-8'))
r = {key.replace('\\', '/'):value for key,value in bundle['reports'].items()}
result = {'new_training':False, 'new_inference':False, 'new_test_access':False,
          'sen1':{}, 'STURM_event_pairing':{}}
sen = r['sen1_fpn_comparison/summary.json']
for system, views in sen['summary'].items():
    result['sen1'][system] = {}
    for view, row in views.items():
        a = row['seeds']
        mean, sd = statistics.mean(a), statistics.stdev(a)
        assert abs(mean-row['mean']) < 1e-12 and abs(sd-row['sample_sd']) < 1e-12
        result['sen1'][system][view] = {'mean':mean, 'sample_sd':sd}
for view, rows in r['val_test_events.json']['results'].items():
    a = {(q['split'], q['event']):q for q in rows['decoder']['event_differences']}
    b = {(q['split'], q['event']):q for q in rows['fpn']['event_differences']}
    keys = sorted(a)
    assert set(a) == set(b)
    diffs = {k:float(np.mean(np.array(b[k]['seed_differences_pp'])-np.array(a[k]['seed_differences_pp']))) for k in keys}
    result['STURM_event_pairing'][view] = {}
    for split in ['val','test','combined']:
        z = np.array([diffs[k] for k in keys if split=='combined' or k[0]==split])
        rng = np.random.default_rng(20261011)
        ci = np.quantile(z[rng.integers(0,len(z),(20000,len(z)))].mean(1),[.025,.975])
        saved = r['fpn_decoder_event_pairing.json'][view]['domains'][split]
        assert abs(float(z.mean())-saved['mean_pp']) < 1e-12
        assert np.allclose(ci,saved['bootstrap95_pp'],atol=1e-12,rtol=0)
        result['STURM_event_pairing'][view][split] = {'mean_pp':float(z.mean()),'positive':int((z>0).sum()),'negative':int((z<0).sum()),'bootstrap95_pp':ci.tolist()}
(p.parent/'recomputed_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS: seed means, sample SD and direct event pairing reproduced from saved JSON only.')
