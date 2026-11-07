"""Generate the literal-edit prompts, then score the archived first responses."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from error_metrics import rmse


HERE = Path(__file__).resolve().parent
CASES = [(4, 2, 1.0, 1.0), (5, 0, 1.0, 1.0), (6, 4, 1.0, 1.0),
         (4, 1, -0.5, 2.0), (5, 3, 0.5, -1.0), (6, 5, -1.0, 0.5)]

def build(i):
    d, j, leak, gain = CASES[i]
    rng = np.random.default_rng(7731 + i)
    A = rng.uniform(-np.sqrt(3/d), np.sqrt(3/d), (d, d))
    b = rng.uniform(-np.sqrt(6/(d+1)), np.sqrt(6/(d+1)), d)
    return d, j, leak, gain, A, b

def prompt(i):
    d, j, leak, gain, A, b = build(i)
    prefix = (
        "You are a neural network weight writer. Return only a JSON array of literal "
        "[index, new_value] pairs. Do not use tools, code execution or follow-up questions. "
        "Edit as few scalar parameters as possible. Unlisted parameters keep their given values. "
        "All indices start at zero. The model has state initially zero and updates "
        "h = A @ h + b*x on each scalar input x. "
        f"The observed output is h[{j}]. It must equal s at every time for all real inputs, "
        f"where s starts at zero and updates s = ({leak})*s + ({gain})*x. "
        f"Parameters are A flattened in row-major order (indices 0 to {d*d-1}), "
        f"then b (indices {d*d} to {d*d+d-1}). Here are the initial arrays:\n"
    )
    return prefix + "A=" + json.dumps(A.tolist()) + "\nb=" + json.dumps(b.tolist())+"\n"

def score(i):
    d, j, leak, gain, A, b = build(i)
    vector = np.r_[A.ravel(), b]
    original = vector.copy()
    answer = (HERE/f"case{i+1}.reply.txt").read_text().strip()
    if answer.startswith("```"):
        answer = answer.split("\n",1)[1].rsplit("```",1)[0].strip()
    patches = json.loads(answer)
    seen = set()
    for k, v in patches:
        assert isinstance(k, int) and k not in seen and 0 <= k < vector.size and np.isfinite(v)
        vector[k] = v
        seen.add(k)
    A, b = vector[:d*d].reshape(d,d), vector[d*d:]
    targetrow = np.eye(d)[j] * leak
    row_witness = np.array_equal(A[j], targetrow) and b[j] == gain
    tests = {}
    for split, T in [('dev',8),('shell',64)]:
        rng = np.random.default_rng(9251+i)
        x = (rng.uniform(-3,3,(512,T)) if split=='dev' else
             rng.uniform(3,10,(512,T))*rng.choice([-1,1],(512,T)))
        h = np.zeros((len(x),d))
        s = np.zeros(len(x))
        ok = np.ones(len(x),bool)
        errors = []
        maxerr = 0.
        for u in x.T:
            h = h @ A.T + b*u[:,None]
            s = leak*s + gain*u
            errors.append(h[:,j] - s)
            ok &= h[:,j] == s
            maxerr = max(maxerr,float(np.max(np.abs(h[:,j]-s))))
        tests[split] = dict(complete_exact=int(ok.sum()), samples=len(x), outputs_per_stream=T,
                            rmse=rmse(errors, 0.), max_abs=maxerr)
    return dict(case=i+1,d=d,readout=j,leak=leak,gain=gain,edits=int(sum(original!=vector)),
                row_certificate=bool(row_witness),**tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','score'])
    cmd = parser.parse_args().action
    if cmd == 'generate':
        for i in range(len(CASES)):
            (HERE/f"case{i+1}.prompt.txt").write_text(prompt(i))
    else:
        scores = [score(i) for i in range(len(CASES))]
        (HERE/'scores.json').write_text(json.dumps(scores,indent=2)+'\n')
        print(json.dumps(scores))
