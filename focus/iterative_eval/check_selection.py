"""Independent replay using raw JSON replies and only permitted feedback rows."""
import hashlib
import json

import numpy as np

from measure import COLLECT, ROOT, PROTOCOL, feedback, measures, observed, raw_prediction, sha

SCAFFOLD='7d9745e8e75b6f5000be9e94aba8eca1db83c647c03b1d8150e6d2babf814b95'


def same_vector(actual,recorded):
    stored=np.asarray(recorded,dtype=np.float64)
    return actual.shape==stored.shape and np.array_equal(actual.view(np.uint64),stored.view(np.uint64))


def parse_revision(blob,widths):
    if blob is None: return [None]*len(widths)
    try:
        text=blob.strip()
        if text.startswith(chr(96)*3): text=text.split('\n',1)[1].rsplit(chr(96)*3,1)[0]
        arrays=json.loads(text)
    except (IndexError,ValueError,TypeError):
        return [None]*len(widths)
    if not isinstance(arrays,list) or len(arrays)!=len(widths): return [None]*len(widths)
    out=[]
    for raw,slots in zip(arrays,widths):
        p=None
        if isinstance(raw,list) and len(raw)==slots and all(type(v) in (int,float) for v in raw):
            try:
                value=np.asarray(raw,dtype=float)
                if np.isfinite(value).all(): p=value
            except (OverflowError,TypeError,ValueError): pass
        out.append(p)
    return out


def summary(task,width,p,data,examples=False):
    out={}
    for name,(x,y) in data.items():
        pred=raw_prediction(task,width,p,x)
        m=measures(task,width,p,x,y)
        if not np.isfinite(pred).all() or not np.isfinite(m['split_loss']): return None
        out[name]=dict(loss=m['split_loss'],exact=m['exact_count'],n=m['n_examples'],rmse=m['raw_rmse'])
        if examples:
            with np.errstate(all='ignore'):
                err=(np.abs(pred-y)/np.maximum(1.,abs(y))).reshape(len(x),-1)
            matched=(pred==y).reshape(len(x),-1).all(axis=1)
            maxima=err.max(axis=1)
            # Python's stable sort makes the original row index the tie breaker.
            worst=sorted((j for j in range(len(x)) if not matched[j]),key=lambda j:-maxima[j])[:3]
            out[name]['examples']=[dict(input=x[j].tolist(),target=y[j].tolist(),
                prediction=pred[j].tolist(),maximum_scaled_error=float(maxima[j])) for j in worst]
    return out


def exact(m):
    return all(row['exact']==row['n'] for row in m.values())


def audit(marker,baseline):
    manifest={}
    for folder in ('objective','arithmetic'):
        for c in json.loads((ROOT/'focus'/folder/'manifest.json').read_text()):
            manifest[c['id']]={**c,'source':folder}
    assert set(marker['trajectories'])=={key+'.json' for key in manifest}
    assert marker['protocol_sha256']==PROTOCOL
    finals,attempts,stops,calls=[],[],[],[]
    for ident,case in manifest.items():
        path=COLLECT/'trajectories'/(ident+'.json')
        assert sha(path)==marker['trajectories'][path.name]
        saved=json.loads(path.read_text())
        assert all(saved[k]==v for k,v in case.items())
        assert saved['complete'] and saved['protocol_sha256']==PROTOCOL
        task,seed,arm=(case[k] for k in ('task','seed','arm'))
        data={'train':observed(task,seed),'dev':feedback(task,seed)}
        original_prompt=(ROOT/'focus'/case['source']/'trials'/(ident+'.prompt.txt')).read_text()
        state={}
        for width in case['sizes']:
            p=baseline[task,width,seed,arm]
            record=saved['states'][str(width)]
            initial=summary(task,width,p,data)
            assert same_vector(p,record['initial'])
            assert initial['train']==record['initial_train'] and initial['dev']==record['initial_dev']
            state[width]=dict(p=p,metrics=initial,stop='exact_train_dev' if exact(initial) else '',
                              streak=0,attempts=0,taken=0,naccepted=0)
        for iteration in (1,2,3):
            widths=[w for w in case['sizes'] if not state[w]['stop']]
            if not widths: break
            request=ident+'-r'+str(iteration); base=COLLECT/'trials'/request
            provenance=json.loads(base.with_suffix('.feedback.json').read_text())
            assert provenance['active']==widths and provenance['case']==ident and provenance['round']==iteration
            assert provenance['protocol_sha256']==PROTOCOL
            promptfile=base.with_suffix('.prompt.txt'); assert sha(promptfile)==provenance['prompt_sha256']
            lines=promptfile.read_text().splitlines(keepends=True)
            scaffold=''.join(lines[:-1])
            assert scaffold.count(original_prompt)==1
            scaffold=scaffold.replace(original_prompt,'<original-prompt>').replace('REVISION '+str(iteration)+'.','REVISION <round>.')
            assert hashlib.sha256(scaffold.encode()).hexdigest()==SCAFFOLD
            requested=json.loads(lines[-1]); assert requested==provenance['feedback']
            want=[dict(size=w,incumbent_weights=state[w]['p'].tolist(),
                       diagnostic=summary(task,w,state[w]['p'],data,True)) for w in widths]
            assert requested==want,('feedback rows or values differ',request)
            if base.with_suffix('.error.json').exists():
                terminal=json.loads(base.with_suffix('.error.json').read_text())
                response=None; usd={};elapsed=terminal['elapsed_seconds']; tool_events=''; outcome='error'
            else:
                meta=json.loads(base.with_suffix('.meta.json').read_text())
                assert meta['completed_items']=={'agent_message':1} and meta['requested_model']=='gpt-6-astra'
                assert meta['effort']=='medium'
                response=base.with_suffix('.reply.txt').read_text()
                elapsed=meta['elapsed_seconds']; usage=meta['usage']
                assert len(usage)==1,('expected one token-usage record',request)
                usd=usage[0];tool_events=0;outcome='completed'
            proposals=parse_revision(response,[len(state[w]['p']) for w in widths])
            digests={ending.lstrip('.').replace('.','_')+'_sha256':sha(base.with_suffix(ending))
                     if base.with_suffix(ending).exists() else ''
                     for ending in ('.prompt.txt','.feedback.json','.reply.txt','.meta.json','.error.json')}
            calls.append(dict(request=request,task=task,seed=seed,arm=arm,round=iteration,
                              active_sizes=len(widths),status=outcome,elapsed_seconds=elapsed,
                              recorded_tool_events=tool_events,**usd,**digests))
            for w,p in zip(widths,proposals):
                st=state[w]; record=saved['states'][str(w)]['history'][iteration-1]
                assert record['round']==iteration and record['request_id']==request
                assert (p is None and record['proposal'] is None) or (p is not None and same_vector(p,record['proposal']))
                old=st['metrics']; new=summary(task,w,p,data) if p is not None else None
                assert record['before_train']==old['train'] and record['before_dev']==old['dev']
                assert record['valid']==(new is not None)
                assert record['proposal_dev']==(new['dev'] if new else None)
                assert record['proposal_train']==(new['train'] if new else None)
                accepted=bool(new and (new['dev']['loss'],new['train']['loss'])<(old['dev']['loss'],old['train']['loss']))
                assert accepted==record['accepted']
                if accepted: st.update(p=p,metrics=new,streak=0,taken=iteration,naccepted=st['naccepted']+1)
                else: st['streak']+=1
                st['attempts']+=1
                chosen=st['metrics']
                assert chosen['train']==record['incumbent_train'] and chosen['dev']==record['incumbent_dev']
                st['stop']=('exact_train_dev' if exact(chosen) else 'two_nonimprovements'
                            if st['streak']==2 else 'round_limit' if iteration==3 else '')
                attempts.append(dict(task=task,size=w,seed=seed,arm=arm,round=iteration,
                                     request=request,valid=new is not None,accepted=accepted,
                                     proposal=p,selected=st['p']))
        for w,st in state.items():
            sr=saved['states'][str(w)]
            assert sr['stop']==st['stop'] and sr['stop']
            assert sr['attempts']==st['attempts']==len(sr['history'])
            assert sr['accepted_round']==st['taken'] and sr['reject_streak']==st['streak']
            assert same_vector(st['p'],sr['weights'])
            finals.append((task,w,seed,arm,st['p']))
            stops.append(dict(task=task,size=w,seed=seed,arm=arm,attempts=st['attempts'],
                              accepted=st['naccepted'],last_accepted_round=st['taken'],reason=st['stop']))
    assert len(finals)==174
    # Ensure no undisclosed terminal trials or extra prompts existed outside the 60 histories.
    trialdir=COLLECT/'trials'
    for suffix in ('.prompt.txt','.feedback.json'):
        assert {p.name[:-len(suffix)] for p in trialdir.glob('*'+suffix)}=={c['request'] for c in calls}
    return finals,attempts,stops,calls
