"""Same-solver refinement of all valid objective-matched Examples only responses."""
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'extension/optimizer'), str(ROOT / 'extension/writer'),
                str(ROOT / 'focus/arithmetic'), str(ROOT)]
from design import SMOOTH, SERIES, observed, inputs, predict, reference, initial
from compare import parse
from certificates import one_hinge
from error_metrics import rmse
import fixtures as arithmetic
from fit import residual_and_jacobian

spec = importlib.util.spec_from_file_location('archived_optimizer', ROOT / 'extension/optimizer/run.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
TOL = 1e-10


def save_csv(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fresh_proposals():
    registry = {}
    for source in ('objective', 'arithmetic'):
        folder = ROOT / 'focus' / source
        for case in json.loads((folder / 'manifest.json').read_text()):
            task, seed = case['task'], case['seed']
            path = folder / 'trials' / (case['id'] + '.reply.txt')
            digest, raw = '', None
            if path.exists():
                blob = path.read_bytes()
                digest = hashlib.sha256(blob).hexdigest()
                try:
                    raw = json.loads(blob) if source == 'arithmetic' else parse(blob.decode())
                    if not isinstance(raw, list) or len(raw) != len(case['sizes']):
                        raw = None
                except ValueError:
                    pass
            for index, size in enumerate(case['sizes']):
                count = (4 * size + 1) if source == 'arithmetic' else len(initial(task,size,seed))
                try:
                    p = np.asarray(raw[index], dtype=float)
                    if p.shape != (count,) or not np.isfinite(p).all():
                        p = None
                except (TypeError, ValueError, IndexError):
                    p = None
                status = 'valid' if p is not None else 'invalid' if path.exists() else 'missing'
                registry[task,size,seed,case['arm']] = (
                    p, dict(protocol=source, call=case['id'], reply_sha256=digest, status=status))
    return registry


def refine(start, residual, jacobian):
    nres = njac = accepted = 0

    def evaluate(p):
        nonlocal nres
        nres += 1
        return residual(p)

    def jac(p):
        nonlocal njac
        njac += 1
        return jacobian(p)

    p = start.copy()
    r = evaluate(p)
    cost = r @ r
    termination = 'initial zero'
    trf_nfev = trf_njev = 0
    if cost:
        fit = least_squares(evaluate, p, jac=jac, method='trf', max_nfev=250,
                            xtol=4e-15, ftol=4e-15, gtol=4e-15)
        trf_nfev, trf_njev = fit.nfev, fit.njev
        termination = fit.message
        fitted_cost = fit.fun @ fit.fun
        if fitted_cost <= cost:
            p, r, cost = fit.x, fit.fun, fitted_cost
        for _ in range(10):
            if cost == 0:
                break
            delta = np.linalg.lstsq(jac(p), -r, rcond=1e-12)[0]
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = p + alpha * delta
                if np.array_equal(candidate, p):
                    continue
                new = evaluate(candidate)
                new_cost = new @ new
                if new_cost < cost:
                    p, r, cost = candidate, new, new_cost
                    accepted += 1
                    break
            else:
                break
    return p, dict(trf_nfev=trf_nfev, trf_njev=trf_njev, total_residual_evaluations=nres,
                   total_jacobian_evaluations=njac, accepted_polish=accepted, termination=termination)


def prediction(task, size, p, x):
    if task in arithmetic.TASKS:
        return arithmetic.forward(size, p, x)[0]
    return predict(task, size, p, x)


def settings(task, size, x, y):
    if task in arithmetic.TASKS:
        return (lambda p: residual_and_jacobian(size, p, x, y)[0],
                lambda p: residual_and_jacobian(size, p, x, y)[1])
    scale = np.maximum(1, np.abs(y)).ravel()
    return (lambda p: (predict(task, size, p, x) - y).ravel() / scale,
            lambda p: original.jacobian(task, size, p, x) / scale[:,None])


def metrics(pred, y):
    equal = pred == y
    hits = equal if equal.ndim == 1 else equal.all(axis=1)
    return dict(exact_count=int(hits.sum()), exact=float(hits.mean()), n_examples=hits.size,
                n_scalars=y.size, rmse=rmse(pred, y),
                loss=float(.5 * np.mean(((pred-y)/np.maximum(1,np.abs(y)))**2)))


def main():
    registry = fresh_proposals()
    rows, status, cert_rows, pairs, weights = [], [], [], [], {}
    with np.load(ROOT / 'extension/optimizer/states.npz') as controls:
        for (task,size,seed,arm), (p0, evidence) in registry.items():
            if arm != 'examples':
                continue
            key = dict(task=task, size=size, seed=seed)
            record = dict(**key, **evidence)
            status.append(record)
            x, y = observed(task, seed)
            tests = inputs(task)
            dev = 'early' if task in SERIES else 'inner'
            splits = {'train':(x,y), 'dev':(tests[dev],reference(task, tests[dev])),
                      'outer':(tests['outer'],reference(task, tests['outer']))}
            refined = None
            if p0 is not None:
                residual, jacobian = settings(task, size, x, y)
                refined, diagnostics = refine(p0, residual, jacobian)
                record.update(diagnostics)
                weights[f'{task}_{size}_{seed}_examples'] = p0
                weights[f'{task}_{size}_{seed}_refined'] = refined
                group = {}
                for split,(xx,yy) in splits.items():
                    old = metrics(prediction(task,size,p0,xx), yy)
                    new = metrics(prediction(task,size,refined,xx), yy)
                    row = dict(**key,split=split)
                    for metric in ('exact_count', 'exact', 'rmse','loss'):
                        row['before_'+metric] = old[metric]
                        row['after_'+metric] = new[metric]
                        row['delta_'+metric] = new[metric] - old[metric]
                    row.update(n_examples=old['n_examples'],n_scalars=old['n_scalars'])
                    rows.append(row)
                    group[split] = row
                assert group['train']['delta_loss'] <= max(1e-25,1e-13*group['train']['before_loss'])
                pairs.append(dict(**key,
                                  train_delta_loss=group['train']['delta_loss'],
                                  outer_delta_loss=group['outer']['delta_loss'],
                                  outer_delta_rmse=group['outer']['delta_rmse'],
                                  outer_delta_exact=group['outer']['delta_exact']))
            if task in SMOOTH and size == 1:
                optimum, _, kind = one_hinge(x, y)
                formula, formula_evidence = registry[task,size,seed,'formula']
                methods = {'formula': (formula,formula_evidence['status']),
                           'examples':(p0,evidence['status']),
                           'adam_bfgs':(controls[f'{task}_1_{seed}_adam_bfgs'],'valid'),
                           'trf_xavier':(controls[f'{task}_1_{seed}_ls_xavier'],'valid'),
                           'refined_examples':(refined,evidence['status'])}
                for name,(theta,method_status) in methods.items():
                    row = dict(**key, method=name, status=method_status,
                               real_optimum=float(optimum), certificate=kind)
                    if theta is not None:
                        loss = metrics(prediction(task,size,theta,x),y)['loss']
                        gap = loss - float(optimum)
                        assert gap >= -1e-13
                        row.update(train_loss=loss, additive_gap=gap, within_1e_minus_10=abs(gap)<=TOL)
                    cert_rows.append(row)
    assert len(status) == 87 and len(cert_rows) == 12*5
    save_csv('status.csv',status)
    save_csv('before_after.csv',rows)
    save_csv('changes.csv',pairs)
    save_csv('certified_optima.csv',cert_rows)
    np.savez_compressed(HERE / 'weights.npz', **weights)
    summary = dict(planned=87, statuses=dict(Counter(r['status'] for r in status)),
                   change_counts={}, certified={})
    for name, tolerance in (('strict', 0), ('threshold_1e_minus_10', TOL)):
        summary['change_counts'][name] = dict(
            train_improved=sum(r['train_delta_loss'] < -tolerance for r in pairs),
            train_improved_outer_loss_worse=sum(r['train_delta_loss'] < -tolerance and r['outer_delta_loss'] > tolerance for r in pairs),
            train_improved_outer_rmse_worse=sum(r['train_delta_loss'] < -tolerance and r['outer_delta_rmse'] > tolerance for r in pairs),
            train_improved_outer_loss_better=sum(r['train_delta_loss'] < -tolerance and r['outer_delta_loss'] < -tolerance for r in pairs),
            outer_loss_better=sum(r['outer_delta_loss'] < -tolerance for r in pairs),
            outer_loss_worse=sum(r['outer_delta_loss'] > tolerance for r in pairs),
            perfect_before=sum(r['split']=='outer' and r['before_exact']==1 for r in rows),
            perfect_after=sum(r['split']=='outer' and r['after_exact']==1 for r in rows))
    summary['one_hinge_tradeoffs'] = {}
    for tolerance_name, tolerance in (('strict', 0), ('threshold_1e_minus_10', TOL)):
        subset = [r for r in pairs if r['task'] in SMOOTH and r['size'] == 1]
        summary['one_hinge_tradeoffs'][tolerance_name] = dict(
            valid=len(subset), train_improved=sum(r['train_delta_loss'] < -tolerance for r in subset),
            train_improved_outer_loss_worse=sum(r['train_delta_loss'] < -tolerance and
                r['outer_delta_loss'] > tolerance for r in subset),
            train_improved_outer_rmse_worse=sum(r['train_delta_loss'] < -tolerance and
                r['outer_delta_rmse'] > tolerance for r in subset))
    for method in ('formula','examples','adam_bfgs','trf_xavier','refined_examples'):
        selected = [r for r in cert_rows if r['method'] == method]
        summary['certified'][method] = dict(all=12, valid=sum(r['status']=='valid' for r in selected),
            within=sum(r.get('within_1e_minus_10', False) for r in selected))
    (HERE / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
