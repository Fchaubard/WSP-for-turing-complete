"""Compile a compensated dyadic-square circuit into ordinary dense ReLU layers."""
from benchmark import SEEDS, data, metrics, np, write_records, xavier

def layer(rng, inputs, outputs, bias=None):
    """Allocate a Xavier matrix, then write the specified affine map into it."""
    weights = xavier(rng,len(inputs),len(outputs))
    weights.fill(0)
    biases = np.zeros(len(outputs))
    for j,(name, terms) in enumerate(outputs.items()):
        for source,value in terms.items():
            weights[inputs.index(source),j] = value
        biases[j] = (bias or {}).get(name,0)
    return (weights,biases),list(outputs)

def compile_square(m, compensate=True, seed=SEEDS[0]):
    rng = np.random.default_rng(seed)
    layers = []
    names = ["x"]
    def emit(outputs,bias=None):
        nonlocal names
        written,names = layer(rng,names,outputs,bias)
        layers.append(written)
    def copy(*sources):
        return {x:{x:1} for x in sources}
    emit({"a":{"x":1/16},"b":{"x":-1/16}})
    emit({"t":{"a":1,"b":1},"q":{"a":1,"b":1},"p":{},"n":{}})
    for i in range(1,m+1):
        alpha = 4.**(-i)
        emit({**copy("t","q","p","n"),"h":{"t":1}},{"h":-0.5})
        emit({"t":{"t":2,"h":-4},**copy("q","p","n"),"u":{"t":2*alpha,"h":-4*alpha}})
        emit({**copy("t","q","p","n","u"),"s":{"q":1,"u":-1}})
        emit({**copy("t","s","p","n","u"),"d":{"q":1,"s":-1}})
        emit({**copy("t","s","p","n"),"pos":{"d":1,"u":-1},"neg":{"u":1,"d":-1}})
        emit({**copy("t"),"q":{"s":1},"p":{"p":1,"pos":1},"n":{"n":1,"neg":1}})
    emit({**copy("q"),"a":{"p":1,"n":-1},"b":{"n":1,"p":-1}})
    emit({"y":{"q":256,"a":256*int(compensate),"b":-256*int(compensate)}})
    return layers

def predict(layers,x):
    result = []
    for i in range(0,len(x),4096):
        a = x[i:i+4096,None]
        for w,b in layers[:-1]:
            a = np.maximum(0,a @ w+b)
        w,b = layers[-1]
        result.append((a@w+b).ravel())
    return np.concatenate(result)

def rounding_challenges():
    """Construct shell floats whose exact square lies just below a midpoint.

    For x = N times 2^(e-52) with N a 53-bit integer, let k be the number
    of low product bits dropped on rounding N*N. Roots of N^2 = 2^(k-1)-d
    modulo 2^k put the square d integer product units below a rounding tie.
    Lift roots one bit at a time, then keep normal inputs in the holdout shell.
    """
    values = set()
    for k in (52,53):
        for d in range(7,256,8):
            residue = 2**(k-1)-d
            roots = [1]
            for bit in range(1,k):
                modulus = 2**(bit+1)
                candidates = (z for v in roots for z in (v,v+2**bit))
                roots = [z for z in candidates if (z*z-residue)%modulus==0]
            for root in roots:
                for n in (root,root+2**k):
                    if n.bit_length()!=53 or (n*n).bit_length()!=53+k:
                        continue
                    for exponent in (1,2,3):
                        x = n*2.**(exponent-52)
                        if 3 < x < 10:
                            values.update([x,-x])
    return np.array(sorted(values))

def run():
    rows = []
    sets = data("square")
    # Independent fixed stress sample; both signs are represented equiprobably.
    rng = np.random.default_rng(9041831)
    stress = rng.uniform(3,10,131072)*rng.choice([-1.,1.],131072)
    tiny = 2.**np.arange(-500,1,dtype=float)
    for m in (16,24,27,30,35,40,48,55):
        for correct in (False, True):
            layers = compile_square(m,correct)
            params = sum(w.size+b.size for w,b in layers)
            nonzero = sum(np.count_nonzero(w)+np.count_nonzero(b) for w,b in layers)
            splits = dict(sets)
            if m in (35,48,55) and correct:
                splits.update(stress_shell=stress,powers_near_zero=tiny,midpoint_shell=rounding_challenges())
            for split,x in splits.items():
                pred = predict(layers,x)
                rows.append(dict(task="square",architecture="compensated_mlp",size=m,
                                 params=params,nonzero=nonzero,
                                 method="write_corrected" if correct else "correction_off",
                                 seed=SEEDS[0],split=split,**metrics(pred,x*x)))
        write_records(rows,"compensated.csv")
        print("finished compensated depth",m,flush=True)

if __name__ == "__main__":
    run()
