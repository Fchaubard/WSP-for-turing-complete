# Tanh from static affine operations and ReLUs

The deployed three-word binary64 graph matched MPFR on all 32,768 fresh outer-shell inputs and on all 60,574 signed CORE-MATH hard examples within [−10,10]. Native NumPy tanh matched 25,285/32,768 and 45,370/60,574 on those sets. Every remaining native error was one ULP on this machine. These are finite measurements.

## Inference and replay

```sh
python infer.py -3.3 0 9.5                     # NumPy only, frozen program
python run.py                                  # gmpy2 + NumPy, full sweep
python verify.py                               # regenerate independent MPFR labels, replay archive
python run.py --core-file /path/to/tanh.wc     # use a local corpus
```

The checkpoint is `program.npz`. `manifest.json` records its SHA-256, library versions, and the pinned corpus URL and hash. Inference rejects inputs outside [−10,10] and checks that every multiplication uses at least one input-independent wire. Only the offline coefficient compiler and scoring program require MPFR. The interpreter uses the arithmetic DAG implementation in the sibling scalar directory. Positive and negative zero count as numerically equal. The native measurements use NumPy 2.5.4 on the recorded Apple ARM binary64 environment.

## Installed computation

The writer receives the hyperbolic addition identity and the deployment envelope. It decomposes |x| into selected dyadic angles, maintaining numerator N and denominator D starting at 0 and 1. The selected angle δ=2ᵏ adds tD to N and tN to D using their old values, where t=expm1(2δ)/(2+expm1(2δ)) is written as a static coefficient from 256-bit arithmetic. In exact arithmetic the ratio after every step equals tanh of the sum of selected angles. There are no target-function calls on test inputs. For ReLU-controlled conditional copying we use R(z−128(1−b))−R(−z−128(1−b)). The state remains within that bound since the product of 1+t over all nonnegative angle exponents is at most 16, and over all negative exponents it is at most e. The graph applies compensated sums and static products to one, two, or three binary64 words.

To divide, the graph compares the current numerator remainder to 2⁻ʲD for successive bit positions j and conditionally subtracts that shifted denominator. In the direct-rounding variants it materializes the first 53 significant quotient bits and uses the next bit to add one final-place unit. The leading-one and guard positions are detected with affine/ReLU Boolean operations. A nonzero dyadic input cannot have an exact mathematical tanh at a binary midpoint, since that would make exp(2x) algebraic. Finite arithmetic can nevertheless select an incorrect bit, as the two-word counterexamples demonstrate. The sign is restored at the output. The graph copies x itself for |x|≤2⁻²⁷, where |x|−tanh|x|≤|x|³/3 is strictly below half the lower spacing. The score identifies the two signed zeros.

The deployed K=84 program uses input-bit exponents 3 down to −84 and 82 division digits, 525 stored coefficient words, 2,146 ReLUs, and 45,043 arithmetic/ReLU nodes including static arithmetic. The alternate expanded accounting charges 75,973 affine coefficient occurrences for dynamic affine operations, including fixed unit edges. The checked K=80 three-word graph achieves the same measured scores using 503 coefficient words and 2,068 ReLUs. This gives an upper bound within the tested architectures. K=79 still fails twice on the public central hard cases.

## Precision tests

Ordinary splits contain 4,096 uniform central, 4,096 negative-shell, and 4,096 positive-shell input floats; a fresh mixed-sign outer set adds 32,768. Midpoint probes invert a midpoint between successive binary64 tanh outputs with 240-bit MPFR atanh, then test the nearest binary64 input and its immediate predecessor/successor with both signs. This gives 4,608 central and 4,608 outer examples. The public hard set takes distinct in-envelope magnitudes from CORE-MATH's binary64 tanh file and includes their negatives. The file aggregates Testlibm, BACSEL and special cases; the signed subset contains 57,060 central and 3,514 outer inputs. An 87-point boundary set covers subnormals, small-branch transitions and outer endpoints. References evaluate MPFR tanh at 53-bit precision with ties to even, cross-checked against 224-bit evaluation converted to binary64. [CORE-MATH corpus documentation](https://core-math.gitlabpages.inria.fr/worst.html) · [Pinned source](https://raw.githubusercontent.com/mockingbirdnest/core-math/3ce26eb72b71ffe95784e42e280149a8c7e015d5/src/binary64/tanh/tanh.wc)

| Variant | Fresh outer exact / 32,768 | Generated midpoint exact / 9,216 | Public central hard exact / 57,060 | Public outer hard exact / 3,514 |
|---|---:|---:|---:|---:|
| Native NumPy tanh | 25,285 | 5,458 | 42,710 | 2,660 |
| Two-word, 120 summed digits | 32,768 | 9,216 | 55,896 | 3,512 |
| Two-word, K=84, direct guard | 32,768 | 9,216 | 57,024 | 3,512 |
| Three-word, K=84, direct guard | 32,768 | 9,216 | 57,060 | 3,514 |

For example, at −0x1.8449d79ba2c11p+2, the two-word direct-guard model outputs −0x1.fffe9762e2c2bp−1 while MPFR and the three-word model output −0x1.fffe9762e2c2cp−1. Every number in these comparisons is numerical binary64 equality. All reported hard-case misses of the two-word 120-digit or direct-guard variants are one ULP. `results.csv` retains the full sweeps and errors, `examples.json` contains hexadecimal counterexamples, and `counts.json` gives per-architecture sizes. A universal certificate for the final graph would additionally need to bound every compensated intermediate and every digit decision or exhaust the declared floating-point domain.
