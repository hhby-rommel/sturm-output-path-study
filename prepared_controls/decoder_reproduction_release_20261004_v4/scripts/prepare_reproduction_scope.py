# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Create disabled exact-scope templates; never starts a run or grants approval."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 tasks=[dict(arm=arm,seed=seed) for seed in (2026,2027,2028) for arm in ('no_skip_capacity','no_input','no_p1')]
 digest=sha(R/'entry_lock.json')
 records={'train_approval.json':dict(approved=False,action='train_val',lock_sha256=digest,tasks=tasks),
 'test_authorization.json':dict(approved=False,action='one_shot_test_after_each_task',lock_sha256=digest,tasks=tasks,test_entry_sha256=sha(R/'scripts/decoder_attribution_test_once_v1.py'))}
 for name,obj in records.items():(out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
 print('Disabled templates written; review and explicitly authorize train and test separately. No run started.')
if __name__=='__main__':main()
