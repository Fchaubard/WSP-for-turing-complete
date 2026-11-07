# Scalar weight-writing experiments

The four executables reproduce the CSVs in this directory. Run the commands from this directory in an environment with NumPy and SciPy. The recorded run used Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1 and binary64 on Apple M5 CPU. Floating-point reductions and optimizer trajectories can vary across BLAS libraries.

```sh
python benchmark.py all
python compensated.py
python selfpower.py
python signed_selfpower.py
```

## Samples, scoring and model information

Scalar inputs come from a fixed NumPy PCG64 stream seeded with 20261105. Every method receives the same 1,024 IID central training points. Independent successive draws give 4,096 in-range test points, 4,096 left-shell points and 4,096 right-shell points. Ranges are [-3,3], [-10,-3] and [3,10]. For two-input tasks the 8,192 shell examples have both coordinate magnitudes uniformly distributed from 3 to 10 and independent random signs. Test labels never enter gradient fitting or selection. We ran three training initializations (11, 22, 33). All weight matrices start Xavier-uniform with zero biases. Writers overwrite their allocated parameters, yielding the same program for any initialization. Affine vectors include all trainable biases except the two-parameter bias-free controls. The CSV distinguishes allocated slots from the number of nonzero written slots.

NumPy binary64 operations supply the reference. Exact is equality to that reference. ULP1 counts answers within one representable-number step on the signed IEEE-754 lattice, coalescing positive and negative zero. Scaled epsilon means absolute error at most 2^-52 times max(1,|target|). The CSV also gives mean absolute error, root mean squared error and maximum absolute error. Test accuracies are finite-set measurements.

Real x^x uses the positive branch with limit 1 at zero when defining the left endpoint of the spline. Interior x samples are uniform on [0,3] and the test shell is [3,10]. Negative intervals do not admit a continuous real x^x target on all real inputs. A separate signed-integer experiment exhausts the negative binary64 inputs in the shell for which the ordinary real power is defined. Dynamic range matters: 10^10 exceeds the lower endpoint value 3^3 by eight orders of magnitude.

## Shallow ReLU size sweep

The residual hinge network is f(x) = a x + b + sum_j v_j ReLU(w_j x + c_j), with 3H+2 slots. We tested H in {0,1,3,7,15,31,63}. All raw per-seed scores are in shallow.csv.

* The global writer compiles a continuous uniform-knot secant spline using formula values across the entire declared evaluation envelope. It knows the target function and the shell support, and for sine, tanh and self-power it queries knot ordinates in the shell. This measures representation with privileged specification access.
* The local writer queries only central knot ordinates. Both writers set w_j to 1, c_j to the negatives of the knots and v_j to differences of adjacent slopes. For square we subtract a constant minimax shift. Outside the covered interval the model extends the affine endpoint pieces.
* The fitted comparator optimizes all slots in the same architecture by analytic reverse-mode derivatives of MSE. It runs 500 minibatch Adam steps (batch 64, step 0.01, beta values 0.9 and 0.999, epsilon 1e-8), then full-batch L-BFGS-B with a 250-iteration cap, ftol 1e-16 and gtol 1e-12. The endpoint with lower training loss wins. Results characterize this finite budget.

**Sharp real-uniform square frontier.** A scalar H-hinge network has at most H+1 affine pieces. On an interval of length h the difference of the average endpoint values of x² and its midpoint value equals h²/4; the same difference is zero for every affine function. Hence its maximum error on the interval is at least h²/8. The longest piece has length at least L/(H+1) over an interval of length L. Equal-knot secant interpolation shifted down by L²/[8(H+1)²] attains that bound on every piece. On [-10,10] the minimax value is 50/(H+1)². The exact minimum width for real-uniform tolerance δ is max(0, ceil(sqrt(50/δ))-1). At δ = 2^-52 this equals 474,531,328 hinges. This is an absolute-error theorem in the shallow residual family, separate from any binary64 calculation or finite test set.

**Self-power knot design.** selfpower.csv compares uniform knots with knots that equidistribute a local curvature density. For f(x)=x^x, f''/f=(1+log x)^2+1/x. Approximate absolute error uses density sqrt(f''); relative error uses sqrt(f''/f). We integrate numerically in t=sqrt(x) to handle the integrable singularity. Each writer sees the same declared formula and [0,10] envelope. At H=63, relative density reduces maximum right-shell relative error from 0.03385 to 0.02042 and MAE from 9.55 million to 5.89 million. Scaling the relative-density readout to balance its largest positive and negative fractional errors across the declared envelope further reduces shell maximum relative error to 0.01010 (MAE 2.70 million). The balancing coefficient uses scalar root-finding within each segment to maximize the ratio of its secant to the known target. Absolute density reduces maximum absolute error from 260.37 million to 1.96 million but yields maximum relative error above 36,000. These measured improvements illustrate objective choice; the adaptive knots carry no global optimality claim.

## Deep ReLU square, numerical compensation and challenge sets

Set t=|x|/16, T(t)=2t-4ReLU(t-1/2) for t in [0,1]. The function 256[t - sum_{i=1}^m 4^-i T^i(t)] is the dyadic linear interpolant to x² for |x|<=16. Its maximum uniform error in real arithmetic on [-16,16] is 64*4^-m. The compiler knows this design envelope and uses no target samples. It implements a scaled version of the [dyadic squaring construction of Yarotsky](https://arxiv.org/abs/1610.01145).

The dense narrow MLP in deep.csv uses hidden widths 2 then 3 repeated m times, followed by scalar affine output. There are 12m+5 slots and 8m+4 written nonzeros. Its matched gradient baseline uses the same architecture and training protocol with 300 Adam steps and a 100-iteration L-BFGS cap. The m=30 writer scores 21.15% exact on the pooled shell, with maximum absolute error 9.95e-14.

The compensated circuit in compensated.py and compensated.csv implements the same interpolant using ordinary dense affine/ReLU layers, with a maximum hidden width of six, 203m+35 dense slots and 41m+14 nonzeros. Each subtraction of a nonnegative term u=4^-i T^i(t) from q computes s=q-u, d=q-s, e=d-u. Two ReLU channels split the signed residual e, and two nonnegative accumulators sum its positive and negative parts. Finally the circuit adds their difference to q and scales by 256. The subtraction uses the form of [Dekker's FastTwoSum transform](https://doi.org/10.1007/BF01397083). In real arithmetic e=0. This gives a same-shape numerical ablation: null exactly two final readout weights to omit the finite-precision correction. Both variants implement the identical real function. This comparison has no optimizer arm.

The uniform shell stress sample uses seed 9041831 and 131,072 magnitudes and signs independent of the benchmark samples. Two targeted checks expose distinct failures. The near-zero set comprises the 501 powers 2^-500,...,2^0. The rounding challenge comprises 304 signed shell inputs, 152 magnitudes. For a binary64 input x=N*2^(e-52), its exact mantissa product is N². We use elementary bit lifting to solve N² = 2^(k-1)-d modulo 2^k, where k (52 or 53) is the number of low bits discarded when rounding the product, and d ranges through 7,15,...,255. Admissible inputs have exact products just below rounding midpoints. The benchmark deliberately selects difficult examples.

| dyadic steps | dense slots | ID exact / 4,096 | shell exact / 8,192 | fresh shell / 131,072 | midpoint / 304 | near zero / 501 |
|---:|---:|---:|---:|---:|---:|---:|
| 35 | 7,140 | 4,047 | 8,192 | 131,072 | 0 | 32 |
| 48 | 9,779 | 4,096 | 8,192 | 131,072 | 40 | 45 |
| 55 | 11,200 | 4,096 | 8,192 | 131,072 | 304 | 52 |

For m=48, suppressing the two correction weights cuts ordinary shell exactness to 26.39%. The corrected network still returns 17.497465374439237 for the in-shell input 0x1.0bb639c98c0b5p+2, while the binary64 square equals 17.497465374439233. The discrepancy is one ULP. Higher m moves the first dyadic knot closer to zero. For sufficiently tiny x, the interpolant's first piece follows a line through the origin and its relative error grows.

## Positive controls and a train-only sine ablation

The 2-slot add and subtract controls in affine.csv fit a bias-free linear layer on 1,024 random central pairs. Ordinary full-batch gradient descent on half mean squared error takes 64 steps of size 1/lambda_max(X'X/n). All three starts recover the exact +/-1 coefficients and achieve perfect in-range and shell equality. The 3-slot affine version includes an intercept and uses ten Gauss-Newton iterations, leaving tiny residual biases in the fitted controls. Written weights score 100% for both architectures.

The 4-slot native tanh model y=v*tanh(wx+b)+c uses the same NumPy tanh operation as the reference. The literal program achieves exact equality, serving as an activation-matched control. Adam plus L-BFGS yields small residual errors. Ten additional ordinary Gauss-Newton refinements, each solving a full-batch residual-Jacobian least-squares system, recover 100% shell equality from all three initializations; central equality ranges from 89.09% to 93.24%. See native.csv.

In fold.csv a six-hinge first stage writes the triangular phase map arcsin(sin x) over [-10,10]. This uses the period and reflection law and receives no shell target ordinates. A second-stage hinge model with uniform knots in [-pi/2,pi/2] fits all linear readout coefficients by least squares on the same 1,024 central labels. The identity-phase control fits the same number of coefficients on central knots in [-3,3]. An end-to-end Adam+L-BFGS baseline trains all parameters in the exact same two-stage shape. At H=63 all three architectures have 211 slots. Pooled shell MAE is 4.746e-5 for written phase plus fitted readout, 3.264 for identity phase, and 3.169 for end-to-end training (median of three); none reach one ULP on those shell examples.

## Exhaustive signed-integer self-power control

Every finite negative binary64 input is a dyadic rational. For a noninteger its reduced denominator is even, so raising that negative base to itself has no conventional real value. The negative integers in the shell remain well-defined. The signed-integer experiment enumerates n=-10,...,10 with 0^0=1. It computes each reference as an exact rational raised to an integer power, rounded once to binary64. NumPy's binary64 power agrees on all 21 inputs on the recorded machine.

The two-hidden-layer ReLU network has widths 23 and 21 (572 dense affine slots). The first layer writes equally spaced hinges; the second takes their second differences to make cardinal hats. These features form an exact identity matrix on the integer grid. Writing the 21 rounded reference ordinates into the readout matches 7/7 central, 7/7 negative-shell and 7/7 positive-shell inputs. Fitting the readout by minimum-norm least squares on the central seven alone returns 7/7, 0/7 and 0/7, respectively. The latter convention sets all unconstrained coefficients to zero. The full writer stores the outer answers explicitly, so its exact result measures finite lookup capacity. signed_selfpower.csv records each exact rational and all binary64 outputs in hexadecimal.
