# Loss convexity on the original 90 matched settings

From the project root, run `python focus/nonconvex/audit.py` in the published
NumPy/SciPy environment. The script loads the original observations, labels,
graph and optimizer objective. It checks all 90 task/size/start settings without
consulting any fitted weights, held-out data or WSP responses.

**Inclusion.** `eligibility.csv` enumerates all 90 settings; `status` is
`nonconvex` for 69 and `convex` for 21. `witnesses.json` contains two full
weight vectors and the exact rational midpoint excess loss for each nonconvex
setting. `loss_left`, `loss_mid`, `loss_right` and `jensen_gap` are measured
with the original training code in binary64. The exact loss treats each
observed input and label as the exact dyadic value it represents, interprets
the graph over the reals, and divides each scalar residual by the larger of
one and the magnitude of its label before squaring. Both versions use half
the mean squared scaled residual. A positive gap disproves convexity; it says
nothing about how well a particular optimizer performed.

**Certificate.** Restrict the parameters to the path joining the listed two
points. The predictions have the form `baseline + gain × feature`. For the
four scalar hinge families, use the bias and readout of the first hinge to
produce a constant output. For driven sums, set the transition to zero and
use the first input and readout weights, so the feature is the current input.
For the two autonomous generators, use an identity transition on the first
coordinate, a free initial value there and its readout, making the feature
constant. For the four-wire sort graph, the baseline is the input itself.
The first block adds a multiple of the positive part of the first input wire
to that wire. Subsequent blocks have zero weights. Five, six and six of the
twelve sort inputs at the three seeds, respectively, strictly activate this
feature. Every other baseline is zero. Across families, unused weights are
set to zero.

Take the first factor and magnitude of the second to be (1,4) at one end and
(4,1) at the other. The two endpoint gains have magnitude 4, while the midpoint
gain has magnitude 25/4. Choose the common output-weight sign so that the
derivative of the squared loss along the signed gain at zero is nonnegative.
On every fixture the feature is nonzero at one or more training outputs, so
this loss is a strictly convex quadratic in the gain. It is strictly larger
at magnitude 25/4 than at 4. The audit computes that difference as an exact
fraction and independently checks it through the original optimizer and
prediction code. Finite-precision replay gives gaps from 0.87240228 to
16.01414.

**Excluded settings and dependencies.** All three original add/subtract
architectures and the memoryless sum graph are affine functions of their
unknown weights. The real training losses are fixed-design positive-weight
linear least squares. These 21 settings are retained in the grid. The 18
Fibonacci and triangular results span six architectures on a single canonical
trajectory per task, with different starting weights for each of three seeds.
The 69 runs represent 57 distinct training objectives after those duplicate
seed-level objectives are collapsed. Multiple widths also share observations
and the arrays in a model reply share a prompt. Additional arithmetic networks
are documented separately in `focus/arithmetic/`.
