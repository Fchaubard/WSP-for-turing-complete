"""Replay the frozen graph, including every in-domain signed CORE-MATH challenge."""
import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

from run import ROOT, URL, data, measures, reference
from infer import predict


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core-file',type=Path)
    args=ap.parse_args()
    payload=args.core_file.read_bytes() if args.core_file else urlopen(URL,timeout=90).read()
    with (ROOT/'results.csv').open() as f:
        rows={r['split']:r for r in csv.DictReader(f)
              if r['method']=='guard_three_word' and r['bits']=='84'}
    for name,x in data(payload).items():
        m=measures(predict(x),reference(x,53))
        assert all(m[k]==int(rows[name][k]) for k in ('n','exact','one_ulp','worst_ulp')),name
        print(name,m['exact'],'/',m['n'])


if __name__=='__main__':
    main()
