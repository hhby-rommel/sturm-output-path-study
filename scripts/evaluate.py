# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Print an explicit reproduction command; run only with --execute."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true')
 p.add_argument('--allow-test',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
 command=[sys.executable,str(ROOT/c['entry']),*c.get('arguments',[])]
 if a.execute and a.dry_run:p.error('Choose execute or dry-run')
 if a.execute and c.get('may_access_test') and not a.allow_test:p.error('This historical entry includes test; --allow-test is required')
 print(json.dumps({'phase':c['phase'],'command':command,'execution_requested':a.execute,'may_access_test':c.get('may_access_test',False)},indent=2))
 if not a.execute:return
 for n in c.get('required_assets',[]):
  if not (ROOT/n).exists():p.error('Required external/generated asset missing: '+n)
 env=os.environ.copy();env['STURM_PROJECT_ROOT']=str(ROOT)
 subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
