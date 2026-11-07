"""Collect at most one completed repair per predefined first trial; await outstanding first replies."""
import hashlib
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from feedback import HERE,SOURCE,selected,make,manifest

location=SOURCE.parent/"writer"/"run.py"
spec=importlib.util.spec_from_file_location("isolated_writer_runner",location)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

def repair(q):
    base=HERE/"trials"/q["id"]
    completed=base.with_suffix(".meta.json")
    failure=base.with_suffix(".failure.json")
    if completed.exists() or failure.exists():return "archived"
    t0=time.monotonic()
    while not (SOURCE/"trials"/(q["id"]+".meta.json")).exists():
        time.sleep(3)
    make(q)
    start=time.monotonic()
    try:
        return runner.run(q,HERE)
    except Exception as exc:
        result=dict(status="request_failed",exception=type(exc).__name__,
                    seconds=round(time.monotonic()-start,2))
        failure.write_text(json.dumps(result,indent=2)+"\n")
        return result

if __name__=="__main__":
    manifest()
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(repair,q):q for q in selected()}
        for done in as_completed(pending):
            q=pending[done]
            print(q["id"],q["task"],q["arm"],done.result(),flush=True)
