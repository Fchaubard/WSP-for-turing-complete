"""Score the public CORE-MATH worst-case sine inputs, downloaded separately."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from run import labeled,rows_for,save,reference,ulps,np
from compile import relu_sin_graph

URL=("https://raw.githubusercontent.com/llvm/llvm-project/"
     "1bfdad8e5c16f98c7a6aa531eb061fd2550ba169/libc/test/src/math/exhaustive/sin.wc")
DIGEST="d3ddc272c3fb23a4d07295cac7576d689e2ac3b99c1d4d7882f0939dafba9d96"
def run(payload):
    if hashlib.sha256(payload).hexdigest()!=DIGEST:
        raise ValueError("CORE-MATH source hash mismatch")
    values=set()
    for raw in payload.splitlines():
        try:
            x=float.fromhex(raw.decode().strip())
        except ValueError:
            continue
        if 0 < abs(x) <= 16: values.add(x)
    x=np.asarray(sorted(values))
    data=labeled({"core_domain":x,"core_shell":x[(np.abs(x)>3)&(np.abs(x)<=10)]},"sin")
    g=relu_sin_graph(-78)
    rows=[]
    rows_for(rows,"sin","relu_rotation_two_word",g,83,-78,data)
    save(rows,"hard_sin.csv")
    x,y,lib=data["core_domain"]
    assert np.array_equal(y,reference(x,"sin",220))
    pred=g.run(x,g.weights())
    failures=[dict(x=x[i].hex(),output=pred[i].hex(),mpfr=y[i].hex())
              for i in np.flatnonzero(pred!=y)]
    (Path(__file__).parent/"hard_sin_failures.json").write_text(json.dumps(failures,indent=2)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",type=Path)
    group.add_argument("--fetch",action="store_true")
    args=parser.parse_args()
    run(args.file.read_bytes() if args.file else urlopen(URL,timeout=120).read())
