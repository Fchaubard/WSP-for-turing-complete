# Full-domain binary64 square

This is an engineered extension of blind Astra trial 3354251e2. Its original reply remains frozen in the sibling synthesis directory. The trial first computes the exact product of a normalized integer significand in three limbs. We retain its first 2,492 expanded instructions. The replacement tail rescales normal results in a different order and computes subnormal outputs by selecting the exact product's rounding bit.

The 3,352-operation graph uses 949 ReLUs, 1,313 subtractions, 483 additions and 607 constant multiplications. Of these, 3,338 operations feed its output. The 1,489 literal-operand occurrences count literal uses in unary or binary operations, excluding the evaluator's dummy operand for ReLU. These are costs of this construction. The original used 2,543 operations and 732 ReLUs. The proof and counterexample to full-domain correctness of the original circuit appear in the paper appendix, extension_fullsquare.tex.

Run from the repository root with Python and NumPy:

    python extension/square_full/build.py
    python extension/square_full/verify.py

The generator checks the original reply's SHA-256, emits program.json and compiles with the original affine_relu evaluator. Verification compares output bits with an independent exact rational oracle on 79,106 distinct binary64 inputs, including both signs of zero. Machine details, program hash, counts and the original failure are in results.json. Independent exact-integer-square-root stress tests are in the sibling audit directory as square_certificate_review.py and square_certificate_review.json.

Evaluate your own decimal or exact hexadecimal inputs through the saved graph with NumPy as the sole third-party dependency:

    python extension/square_full/infer.py -3.25 -0.0 0x1.0000000000001p-508 10

The command prints the bit-exact binary64 input and prediction in hexadecimal and the prediction in decimal. The input domain is [-10,10].
