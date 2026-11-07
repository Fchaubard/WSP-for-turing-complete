# Supplementary refinement from objective-matched literal weights

The sampling frame, defined before fitting, consists of all 87 nonconvex settings
in the primary study (69 original graphs and 18 new arithmetic graphs). Use
every valid fresh first response to the *Examples only* request. The known-optimum
subset is fixed at the 12 scalar one-hinge fits: four functions by three starts.
Do not repair malformed proposals or substitute a different call. Each call's
three responses (two for running sum) share a prompt. A missing/invalid array
has no numerical endpoint. The extra least-squares run is a supplement to the
four primary methods; it is not counted as another literal response.

Run `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python focus/refinement/run.py`
from the project root in the published NumPy/SciPy environment after the
objective-matched replies have been collected. The script never sends requests
or reads prior, unmatched WSP replies. With an incomplete collection it emits
a partial status grid; rerunning after all replies arrive replaces the partial
files. All weights are binary64. Trust-region reflective least squares uses the
original residual Jacobian, the original relative objective/step/gradient
tolerances (4e-15), at most 250 residual evaluations inside the solver, and at
most ten further accepted Gauss–Newton updates with the original backtracking
schedule. Selection uses the displayed training labels alone. Fits that start
at zero training loss retain their weights.

For each split, report half the mean squared error scaled by max(1,|target|)
per output, ordinary RMSE in target units, and exact-equality counts and
fractions (whole outputs for lists). The change is the refined score minus the
literal score, so a negative loss change is an improvement. Primary descriptive
trade-off counts require a reduction exceeding 1e-10 in training loss and an
increase exceeding 1e-10 in the weighted outer loss. We separately report
changes in raw outer RMSE using the same absolute tolerance *in each target's
own units*, and strict zero-tolerance counts. The 12 known-optimum comparisons
use additive tolerance 1e-10 on the training loss and compare saved endpoints
against the separately computed exact real-arithmetic minimum. No comparisons
pool raw errors across different units.

The generated `before_after.csv` gives every split's denominators, exact
counts, raw RMSE, weighted loss and changes. `changes.csv` pairs train and
outer outcomes per setting. `certified_optima.csv` gives the independently
evaluated four primary endpoints plus the supplementary refined endpoint on
each certified objective. `status.csv` records protocol, fresh call ID,
response SHA-256, validity, solver reason and evaluation counts. Both before
and after weights are in `weights.npz`. To verify the solver itself, run
`python focus/refinement/verify.py` in the same single-thread environment.
All 87 replays starting from Xavier reproduced the primary trust-region
solver's archived final weights bit-for-bit.
