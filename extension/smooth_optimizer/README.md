# Full-graph Adam controls for positive self-power and tanh

Both graph structures and written parameter vectors are loaded from the same checkpoints used by the inference functions in the neighboring scalar and tanh directories. Only the explicit coefficient arrays are replaced or optimized. The graph structure, implicit operation signs and fixed numerical literals are known and shared. This study complements the identical 100-update sine control in the scalar directory.

The training set contains 1,024 uniformly drawn central coordinates per task (PCG64 seed 20261105), with 4,096 further central test coordinates. For positive self-power, the outer test contains 4,096 uniform values in [3,10]. For tanh, 4,096 values come from each signed shell side. The reference uses MPFR round-to-nearest-even at precision 53, validated against a separate 200-bit evaluation. Initialization is scalar Xavier-uniform on every stored slot. Each RNG subsequently selects minibatches of 32 with replacement. Training uses unclipped Adam on unscaled MSE for exactly 100 updates, as in the sine control.

The two JSON result files record the written scores, Xavier and terminal train/inner/outer losses, MAE, finite counts, exact counts and one-ULP counts. They also record moved slots, distinct slots that received any finite nonzero gradient, maximum absolute gradient and the first numerical failure, if present. The per-step files retain minibatch loss, gradient counts and magnitude. The compressed weight checkpoints preserve every initial and terminal vector. All six reported runs finished their full budgets; no numerical recovery was required. The written graphs both achieved zero loss on these inputs. Each fitted graph had zero exact or adjacent outer outputs.

```sh
python bench.py selfpower
python bench.py tanh
python replay.py
```

The recorded machine used Apple M5, macOS 26.0 arm64, Python 3.13.12, NumPy 2.5.4, gmpy2 2.3.2 and MPFR 4.2.2. The runner imports the original sine reverse-mode implementation. If a nonfinite loss, gradient or optimizer state occurs on another runtime, it records the first failing step, halts that fit, and reports the last accepted coefficients. It does not insert replacement values or restart the fit.
