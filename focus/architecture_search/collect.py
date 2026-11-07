"""Freeze examples-only prompts, collect one independent graph each, then seal all replies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'extension/writer'))
from design import TASKS, SEEDS, SERIES, observed
spec=importlib.util.spec_from_file_location('existing_runner',ROOT/'extension/writer/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    cases=[]
    for task in TASKS:
        for seed in SEEDS:
            uid=hashlib.sha256(f'architecture-first-v1/{task}/{seed}'.encode()).hexdigest()[:12]
            x,y=observed(task,seed)
            mode='index' if task in SERIES else 'sequence' if task=='sum' else 'scalar' if x.ndim==1 else 'vector'
            domain=('The input is a nonnegative integer index at most 64.' if mode=='index' else
                    'The input is positive and at most 10.' if task=='selfpower' else
                    'Every input coordinate is a real value in [-10,10].')
            shape=('A sequence of real scalars maps to a sequence of the same length.' if mode=='sequence' else
                   f'One input has shape {list(x.shape[1:]) or "scalar"} and one output has shape {list(y.shape[1:]) or "scalar"}.')
            prompt=('You are designing a compact neural network from input/output examples. Use weight-space programming (WSP) to choose both architecture and numerical weights. '
                    'Use no tools, files, web, or external numerical solver. The MAIN goal is to infer the rule and extrapolate it accurately to unseen inputs throughout the deployment domain, including far beyond the displayed examples. '
                    'Among architectures that implement the inferred rule equally accurately, minimize stored coefficients, batch-one inference workspace and scalar arithmetic operations. '
                    'Favor a rule-based construction with a checkable certificate; fit quality only on displayed examples is insufficient. You may choose an MLP, RNN, CNN, transformer/attention, gated network, or another network representable in the graph language. '
                    'No architecture or initialization is prescribed. Do not report unsupported global optimality. '
                    f'Input mode is {mode}. {shape} {domain}\n'
                    +'Observed input/output pairs:\n'+json.dumps(list(zip(x.tolist(),y.tolist())),separators=(',',':'))+'\n'
                    +(HERE/'schema.txt').read_text()+'\nReturn only the JSON object. Your construction summary should explain the implementation for readers without giving hidden internal reasoning.\n')
            (HERE/'trials'/f'{uid}.prompt.txt').write_text(prompt)
            cases.append(dict(id=uid,task=task,seed=seed,arm='free_architecture_examples',mode=mode))
    (HERE/'manifest.json').write_text(json.dumps(cases,indent=1)+'\n')
    receipt={p.name:digest(p) for p in (HERE/'PROTOCOL.md',HERE/'schema.txt',HERE/'manifest.json')}
    receipt['prompts']={p.name:digest(p) for p in sorted((HERE/'trials').glob('*.prompt.txt'))}
    (HERE/'PRECOMMIT.json').write_text(json.dumps(receipt,indent=1)+'\n')

def one(case):
    try:runner.run(case,HERE);return case['id'],'complete'
    except Exception as exc:
        (HERE/'trials'/f"{case['id']}.error.json").write_text(json.dumps(dict(error=repr(exc),time_utc=time.time()))+'\n')
        return case['id'],'failed'

def collect():
    pre=json.loads((HERE/'PRECOMMIT.json').read_text())
    for n in ('PROTOCOL.md','schema.txt','manifest.json'):assert digest(HERE/n)==pre[n]
    for n,h in pre['prompts'].items():assert digest(HERE/'trials'/n)==h
    cases=json.loads((HERE/'manifest.json').read_text());assert len(cases)==30
    with ThreadPoolExecutor(max_workers=10) as pool:
        for f in as_completed([pool.submit(one,c) for c in cases]):print(*f.result(),flush=True)
    marker=dict(protocol=digest(HERE/'PROTOCOL.md'),precommit=digest(HERE/'PRECOMMIT.json'),
                trials={p.name:digest(p) for p in sorted((HERE/'trials').iterdir())})
    (HERE/'FROZEN.json').write_text(json.dumps(marker,indent=1)+'\n')
    print('FROZEN 30 independent proposals',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='collect':collect()
