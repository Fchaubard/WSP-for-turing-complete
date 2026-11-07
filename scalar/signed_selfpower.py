"""An exhaustive, finite signed-integer self-power control using ReLU cardinal hats."""
from fractions import Fraction
from pathlib import Path
import csv
import numpy as np

def weights(rng, n_in, n_out):
    radius = np.sqrt(6/(n_in+n_out))
    return rng.uniform(-radius, radius, (n_in,n_out)), np.zeros(n_out)

def run():
    rng = np.random.default_rng(11)
    n = np.arange(-10,11)
    x = n.astype(float)
    central = np.abs(n) <= 3
    # Negative exponents use exact reciprocals. Fraction(0)**0 equals one.
    exact = [Fraction(int(i))**int(i) for i in n]
    reference = np.array([float(v) for v in exact])
    library = np.power(x,x)

    # Widths 23,21. The three adjacent hinges make the hat centered at n.
    w1,b1 = weights(rng,1,23)
    w2,b2 = weights(rng,23,21)
    readout,b3 = weights(rng,21,1)
    w1[:] = 1
    b1[:] = -np.arange(-11,12)
    w2.fill(0)
    indices = np.arange(21)
    for offset, coefficient in enumerate([1,-2,1]):
        w2[indices+offset,indices] = coefficient
    hidden = np.maximum(0,x[:,None]@w1+b1)
    features = np.maximum(0,hidden@w2+b2)
    assert np.array_equal(features,np.eye(21))
    assert sum(a.size for a in (w1,b1,w2,b2,readout,b3)) == 572
    rows = []
    for method in ("full_spec","central_min_norm","central_bp"):
        v = readout.copy()
        if method == "full_spec":
            v[:,0] = reference
        elif method == "central_min_norm":
            v[:,0] = np.linalg.lstsq(features[central],reference[central],rcond=None)[0]
        else:
            # With the same fixed hats as the least-squares control, backpropagate
            # half-summed square loss through the readout. The sampled rows of
            # the feature matrix are orthonormal, so unit learning rate is valid.
            for _ in range(2):
                prediction = (features[central] @ v + b3)[:,0]
                gradient = features[central].T @ (prediction-reference[central])
                v[:,0] -= gradient
        output = (features@v+b3)[:,0]
        for i,k in enumerate(n):
            split = "inner" if central[i] else "negative_shell" if k < 0 else "positive_shell"
            rows.append(dict(method=method,n=int(k),split=split,rational=str(exact[i]),
                             reference_hex=reference[i].hex(),numpy_hex=library[i].hex(),
                             output_hex=output[i].hex(),equals_reference=bool(output[i]==reference[i]),
                             equals_numpy=bool(output[i]==library[i])))
    path = Path(__file__).with_name("signed_selfpower.csv")
    with path.open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for method in ("full_spec","central_min_norm","central_bp"):
        for split in ("inner","negative_shell","positive_shell"):
            subset = [r for r in rows if r["method"]==method and r["split"]==split]
            print(method,split,len(subset),sum(r["equals_reference"] for r in subset),
                  sum(r["equals_numpy"] for r in subset))

if __name__ == "__main__":
    run()
