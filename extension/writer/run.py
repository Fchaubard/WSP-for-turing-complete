"""Collect fresh isolated first responses. Resume skips completed requests."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

HERE=Path(__file__).resolve().parent
def run(case,folder=HERE):
    base=folder/'trials'/case['id']
    reply=base.with_suffix('.reply.txt')
    meta=base.with_suffix('.meta.json')
    if reply.exists() and meta.exists(): return 'archived'
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='wsp-blind-') as tmp:
        raw=Path(tmp)/'out.txt'
        cmd=[os.environ.get('CODEX_APP_RUNTIME_CLI_PATH','codex'),'exec','--ephemeral',
             '--skip-git-repo-check','--ignore-user-config','-C',tmp,'-s','read-only','-m','gpt-6-astra',
             '-c','model_reasoning_effort="medium"','--color','never','--json','-o',str(raw),'-']
        result=subprocess.run(cmd,input=base.with_suffix('.prompt.txt').read_text(),
                              text=True,capture_output=True,timeout=float(os.environ.get("WSP_REQUEST_TIMEOUT","300")))
        events=[]
        for line in result.stdout.splitlines():
            try:events.append(json.loads(line))
            except ValueError: pass
        kinds=Counter(e.get('item',{}).get('type','unknown') for e in events if e.get('type')=='item.completed')
        if result.returncode or not raw.exists():
            errors=[e.get('message',e.get('error',{})) for e in events if e.get('type') in ('error','turn.failed')]
            raise RuntimeError(f'CLI returned {result.returncode}: {str(errors)[-400:]}')
        reply.write_text(raw.read_text())
        meta.write_text(json.dumps(dict(requested_model='gpt-6-astra',effort='medium',elapsed_seconds=round(time.monotonic()-start,2),
                          completed_items=kinds,usage=[e['usage'] for e in events if e.get('usage')]),indent=2)+'\n')
    return 'done'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--folder',type=Path,default=HERE)
    a=ap.parse_args(); trials=json.loads((a.folder/'manifest.json').read_text());random.Random(80370).shuffle(trials)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs={pool.submit(run,c,a.folder):c for c in trials}
        for f in as_completed(futs):
            c=futs[f]
            try: print(c['id'],c['task'],c['arm'],f.result(),flush=True)
            except Exception as exc: print(c['id'],'error',repr(exc),flush=True)
