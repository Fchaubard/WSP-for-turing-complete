# Full-batch gradient descent and limited-memory BFGS on 87 matched nonconvex settings

From the project root, under the published NumPy and SciPy versions, run

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python focus/lbfgs/experiment.py --workers 5
python focus/lbfgs/report.py
```

**Scope.** Every one of the same 87 nonconvex task/size/start combinations
is included. Sixty-nine use the original graphs and 18 use the no-skip,
two-input rectifier networks for addition and subtraction. The 21 convex
controls are kept outside this experiment. Each optimizer receives all 12
fixed central labeled examples, or the nine canonical sequence labels.
The objective is half the mean squared residual over every scalar output
after dividing each residual by max(1,|label|). Backpropagation computes the
gradient of that same full-batch objective. Each task uses the original
implementation of its reverse-mode gradient, prediction and data splits.
Scoring after fitting includes the exact fraction using whole examples and
raw RMSE using every scalar coordinate. For the autonomous sequences the
central development split is early unseen indices 9–32 and outer is 33–64.
The main running-sum outer split holds stream length fixed at six.

**Predeclared optimizer rules.** Full-batch gradient descent (GD) follows the
negative gradient with a scalar step size. There is no momentum, clipping or
per-parameter scaling. Each of at most 800 updates starts at rate 1. The rate
is halved, up to 80 times, until the finite candidate has strictly smaller
training loss and its decrease is at least 0.0001 times the rate times the
squared gradient norm (Armijo backtracking). Fitting stops if the largest
absolute gradient component is at most 1e-14, the step cannot change the
stored parameters, or all trial rates fail. This single rule applies across
tasks, including random recurrent transitions with very large gradients.

The limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer (L-BFGS) retains
ten pairs of parameter and gradient differences. We use SciPy's L-BFGS-B
implementation without any bounds. Runs allow at most 1,000 iterations,
20,000 joint loss/gradient evaluations inside the solver, and 40 line-search
trials per iteration. The largest absolute gradient stopping tolerance is
1e-14. Relative objective tolerance is zero (a step with no representable
decrease can still stop). A callback records each accepted iterate; the
retained checkpoint has the lowest finite training loss among the start,
accepted iterates and re-evaluated return point. Ties keep the earlier
checkpoint. Evaluation arrays are generated after fitting.

| Method key | Starting weights | Process |
| --- | --- | --- |
| `gd` | Xavier | GD only |
| `gd_lbfgs` | Final GD weights from the very same run | L-BFGS |
| `adam_lbfgs` | The 800-step Adam checkpoint from the matched Adam/full-BFGS schedule | L-BFGS |
| `lbfgs_xavier` | Xavier | L-BFGS |
| `adam800` | Xavier | Shared Adam control before second-stage optimization |
| `xavier` | Xavier | Initial control |

The 69 original Adam checkpoints are loaded as stored. For new arithmetic the
same 800 Adam updates are replayed. A full-BFGS replay from each reconstructed
Adam vector reproduced all 18 archived endpoints bit-for-bit. Starting weights,
task sizes, seeds and observed examples are shared between methods. The test
sets and both WSP conditions are unavailable to these optimizers.

**Files.** `scores.csv` has one row per method, task, size, seed and split
(1,566 in total), including exact counts and their example denominators,
the number of scalar coordinates, raw RMSE, and train/split weighted loss.
`weights.npz` contains all 522 checkpoint arrays under
`{task}_{size}_{seed}_{method}`. `diagnostics.csv` records actual iterations,
joint evaluations, stage selection and stopping reason for every trained arm;
`settings.json` gives exact software and solver settings. Each L-BFGS
`joint_evals` count includes the two explicit value/gradient checks of its
start and returned point. Its `solver_joint_evals` excludes those checks.
For an end-to-end count, add the work of each sequential method's shared warm-up:
171,024 joint evaluations across the 87 GD warm-ups or 69,600 across 87 Adam
warm-ups. The GD stage reuses the most recently evaluated gradient at each
accepted point; its count includes line-search evaluations and the initial
evaluation. Validation, post-fit reporting and held-out scoring are separate.

`summary.json` gives aggregate comparisons; `by_task.csv` preserves per-task medians of raw RMSE and
exact-match fractions on each split. `paired.csv` gives deltas relative to
each original primary arm for every task/size/seed without pooling errors
in different units. Win/loss counts use an additive 1e-10 tie tolerance.
`report.py` checks Xavier metrics against all 261 canonical rows and the
original 69 Adam checkpoints against their independent archived training
loss and exact fractions across all three splits.
