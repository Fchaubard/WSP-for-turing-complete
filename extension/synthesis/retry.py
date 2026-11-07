"""Retry only requests with no returned message; preserve timeout counts separately."""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/'writer'))
from run import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');a=p.parse_args()
    trials=[r for r in json.loads((here/'manifest.json').read_text()) if r['id'] in a.ids]
    os.environ['WSP_REQUEST_TIMEOUT']='900'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs={pool.submit(run,r,here):r for r in trials}
        for f in concurrent.futures.as_completed(futs):
            print(futs[f]['id'],f.result(),flush=True)
