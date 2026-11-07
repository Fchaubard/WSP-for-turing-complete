# Exact square capacity on a finite binary64 test

For the full lattice of 321 binary64 numbers from −10 through 10 in increments of 1/16, a one-hidden-layer residual ReLU model needs exactly **160 hinges (482 scalar slots)** to return their binary64 squares. This includes internal floating-point operations, subject to the magnitude and arithmetic assumptions below. Its minimum-width witness then misses **all 320 between-site midpoints** and all 4,096 fixed random outer-shell inputs. The midpoint set lies strictly inside the deployment interval. This isolates sample geometry from both target complexity and extrapolation distance.

## Statement

Let D be one of three sorted sets of binary64 inputs: n/16 for integer n from −160 to 160 inclusive; its inner restriction to |x| ≤ 3; or its outer restriction to 3 < |x| ≤ 10. Write N for its size (321, 97, or 224). For a residual scalar ReLU with w hinges, the maximum number of correct binary64 squared outputs on D is **min(N, 2(w+1))**. For w above the minimum required to get every output right, append zero-output hinges. The smallest sufficient width is ceil(N/2)−1. Resource accounting charges 3w+2 stored scalars, including all zero biases.

The hypothesis class uses binary64 weights and biases with magnitude at most 128, ReLU, an affine skip, and fixed-order binary64 arithmetic. Evaluate the skip as s = fl(fl(ax)+b), then process each hinge in ascending index order. Compute h = max(0,fl(fl(ux)+c)), multiply v*h with binary64 rounding, and add the result to s with separate rounding. Assume round-to-nearest ties-to-even with gradual underflow. The theorem allows arbitrary signs in hidden slopes and output weights. The compiler writes one sign of hidden slope to attain the upper bound. This is a sharp result for an explicitly capped class; arbitrary internal magnitudes and arithmetic reorderings fall outside the numerical proof.

## Why fewer hinges fail

The same stored coefficients evaluated over the reals define a continuous spline f with at most w+1 affine pieces. The following uniform error bound connects the machine network F to that real spline. Only widths w ≤ 160 need consideration for the lower bound.

1. With |x| ≤ 10 and each coefficient bounded by 128, both the first rounded product and its bias sum have magnitude below 2¹¹. Each operation rounds by less than 2⁻⁴⁰ in absolute value. Taking a ReLU is 1-Lipschitz. Thus the computed hinge differs from its real value by at most 2⁻³⁹. The same bound applies to the affine skip. These loose absolute bounds include subnormals.
2. The product of the computed hinge and its output coefficient has magnitude below 2¹⁸. Its rounding error is at most 2⁻³⁴. Including the propagated hidden error, each term differs from its real value by less than 2⁻³¹.
3. Each successive sum remains below 2²⁶ in magnitude, since the initialized sum is below 2¹¹ and there are at most 160 terms below 2¹⁸ with less than one unit of absolute rounding error per addition. Induction therefore bounds each addition's rounding error by 2⁻²⁶. In total |F−f| ≤ 2⁻³⁹ + 160(2⁻³¹ + 2⁻²⁶) < 2⁻¹⁸.

The correct binary64 square on this grid equals the real square: n²/256 is exactly representable. For any three distinct grid sites x₁<x₂<x₃, the secant of the square between the outer sites exceeds x₂² at x₂ by (x₂−x₁)(x₃−x₂) ≥ 1/256. If one affine piece of f passes within E of each target, linear interpolation of its endpoint residuals gives 2E ≥ 1/256. But E < 2⁻¹⁸ on any exact machine hit. Each affine piece can therefore account for at most two hits (assign a shared endpoint to either piece). This gives the stated upper bound on hits. It also supplies a quantitative failure witness below the full-fit threshold: by the pigeonhole principle, some affine piece contains at least three sites, and on at least one site the machine output error is at least 1/512 − 2⁻¹⁸.

## Why the bound is attainable

Select the first min(N,2(w+1)) ordered sites. Pair them consecutively and use the unique secant through each pair of target labels. For odd N, the final line uses the last two sites, sharing the penultimate site with the preceding pair. The slopes strictly increase. For disjoint adjacent pairs, each pair's secant extrapolates below the square on the other pair. Hence their intersection lies strictly between the pairs; for overlapping terminal pairs it is their shared site. Set a knot at each intersection. The piecewise secant network stores the first line in its skip and each subsequent slope increment v as v ReLU(x−t).

Within a uniform run, the knot between disjoint pairs is their gap midpoint. The two shells have even cardinality per side, so the only large gap joins two symmetric secants at t=0. All knots are multiples of 1/32, all slopes and increments multiples of 1/16, and all intercepts multiples of 1/256. Every stored value fits the coefficient cap (the largest absolute left intercept is 99.375). On the grid, every intermediate is a multiple of 1/512 or a coarser dyadic. Its magnitude stays below 2²⁶ by the same coarse range estimate; the scaled integer needs fewer than 53 bits. The separate operations are exact. At each selected x, the active secant returns x². This constructs the claimed maximum for every width. It also proves the unbounded-weight *real-arithmetic* variant by the simpler two-intersections argument. For arbitrary floating-point weights the cap or an alternative stability argument matters.

## Information-access control

The paired-secant routine accepts sampled (x,y) labels, without using the square formula, to form each line. A fit given the 97 central sites uses 48 hinges, fits that entire central grid, and misses all 224 outer sites. A fit given all 321 sites includes the evaluation labels and is a capacity witness.

A second control gives the solver the independent structural hypothesis “polynomial of degree at most two,” the deployment lattice, and just the three central labels (−1,1), (0,0), (1,1). Supervised gradient updates fit the nodal coefficients of the quadratic Lagrange basis. Its design matrix is the identity on these three input sites. Under exact arithmetic one full-batch step with unit learning rate on half-summed squared loss suffices. Two steps from five binary64 Xavier draws recover identical coefficients (1,0,1). Evaluating that inferred quadratic offline supplies the secant compiler's ordinates. The deployed ReLU weights are byte-identical to the full-grid witness and score 224/224 on outer sites. This is gradient fitting in a specified polynomial family followed by compilation. The architecture of an end-to-end trainable ReLU network offers a much larger hypothesis class. The result exposes exactly what extra information makes three inner labels sufficient.

## Results and replay

| Given sites | Minimum width | Dense slots | Own grid | Other grid | 4,096 random outer inputs |
|---|---:|---:|---:|---:|---:|
| All | 160 | 482 | 321/321 | 0/320 full-grid midpoints | 0 |
| Inner | 48 | 146 | 97/97 | 0/224 outer | 0 |
| Outer | 111 | 335 | 224/224 | 0/97 inner | 0 |

At width 159 the full-grid witness returns exactly 320/321 outputs (maximum error 0.0078125). The width-160 witness has maximum midpoint error 0.0029296875. All 322 grid/width combinations hit their theoretical maxima. The script uses only Python’s standard library. Running `python3 lattice.py` produces `lattice.json` containing the full sweep, initialized and final weight hashes at the thresholds, shell and midpoint transfers, and the three-label quadratic control. All indices, seeds, and evaluation conventions are in the script. Sample grids and midpoint probes were designed analytically. This result concerns a particular fixed architecture and bounded coefficients. Compare general memorization work and finite-precision expressivity for the broader settings. [Yun, Sra and Jadbabaie, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/dbea3d0e2a17c170c412c74273778159-Abstract.html) · [Hwang et al., ICML 2025](https://proceedings.mlr.press/v267/hwang25b.html)

## A lower bound for success on every bounded binary64 input

On the finer grid x=n/q, −10q ≤ n ≤ 10q, the same minimum-width formula holds for q=16,32,64,128. Replace the coarse per-addition bound above by its input-dependent version. For each computed hinge, |h| ≤ 1408 + 2⁻³⁹ and |fl(vh)| < 180225. The skip is initially smaller than 1410. At addition k the pre-rounded accumulator has magnitude at most 1410+180226k by induction: all values stay below 2²⁸ for w ≤ 1280, so each rounded addition loses less than one absolute unit. For a binary64 number rounded from a real value of this magnitude, absolute error is at most (1411+180226k)2⁻⁵³, also covering subnormals. The uniform bound on the whole computation is consequently

E(w) = 2⁻³⁹ + w 2⁻³¹ + 2⁻⁵³ [1411 w + 180226 w(w+1)/2].

At w=1280, E(w) ≈ 1.701 × 10⁻⁵ < 1/(2 × 128²) ≈ 3.052 × 10⁻⁵. The triple-of-sites argument therefore applies simultaneously for all w≤10q and q in the sweep. The upper construction uses knot, slope and intercept denominators dividing 2q, q and q² respectively. On sites and midpoints, its intermediates are dyadics with common denominator 2q² and magnitude below 2²⁸. Their scaled integer magnitudes remain below 2⁴³, so every separate operation is exact.

| q | Lattice sites | Minimum hinges | Dense slots | Correct one hinge below | Midpoints correct at threshold |
|---:|---:|---:|---:|---:|---:|
| 16 | 321 | 160 | 482 | 320/321 | 0/320 |
| 32 | 641 | 320 | 962 | 640/641 | 0/640 |
| 64 | 1,281 | 640 | 1,922 | 1,280/1,281 | 0/1,280 |
| 128 | 2,561 | 1,280 | 3,842 | 2,560/2,561 | 0/2,560 |

Every binary64-correct bounded-coefficient shallow square on the entire interval must also fit the 2,561-site subset. Consequently, **at least 1,280 hinges (3,842 scalar slots) are necessary** for this parameterization even with rounding at every operation. At 1,280 hinges the upper bound certifies only the lattice. The sweep output stores separately the analytic roundoff bound, required triple gap, entire lattice accuracy, and all midpoint results.
