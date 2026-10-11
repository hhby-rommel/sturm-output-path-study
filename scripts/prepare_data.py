# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restore original manifest/sampling metadata; no image or model access."""
from pathlib import Path
import argparse,gzip,json,hashlib
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
 items=json.loads((R/'compressed_metadata_manifest.json').read_text())
 print(json.dumps({'metadata_files':[x['restored'] for x in items],'write':a.write}))
 if not a.write:return
 for x in items:
  target=R/x['restored']
  if target.exists():raise FileExistsError(target)
  data=b''.join((R/part).read_bytes() for part in x['parts']) if 'parts' in x else (R/x['compressed']).read_bytes()
  raw=gzip.decompress(data)
  assert hashlib.sha256(raw).hexdigest()==x['original_sha256']
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
if __name__=='__main__':main()
