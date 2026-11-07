# Expanded ten-task comparison

| Directory | Reproducible evidence |
|---|---|
| writer | 60 fresh paired Astra requests, 180 complete vectors, 90 trained Xavier controls on identical arrays and central examples. table.csv has three-seed medians; scores.csv contains each split, size, start and arm. |
| optimizer | Candidate training objectives and stronger gradient and least-squares optimizers on all 90 matched networks, including proofs of optimal training loss for specific graphs. |
| convex | 2,016 measurements with fixed hidden features, including numerical global solutions. Full-envelope supervision is labeled separately. |
| recurrent | Two budgets for 102 settings of tanh RNNs and GRUs; scaled-cell control; independent saved-state replay and bounds on trained readouts. |
| square_full | A 3,352-operation engineered ReLU repair for correctly rounded squaring on every finite binary64 input in [-10,10]. The blind first-response certificate applies only to the shell. |
| smooth_optimizer | Actual reverse-mode training from three Xavier starts for each of the two deployed self-power and tanh graphs; sine is in scalar/train.py. |
| audit | Exact hinge thresholds for ordered binary64 evaluation on four square lattices, proofs and an audit of original claims. |
| scalar | Frozen affine/ReLU programs for sine and positive self-power, size sweeps, MPFR tests and explicit rounding failures. |
| tanh | Frozen affine/ReLU and native tanh evaluated against independent MPFR targets and hard inputs. |
| counters | Register-machine compiler, integer interpreter, six direct writer trials, nine gradient fits, matched-label structured controls. |
| synthesis | Directly generated neural circuits for three hard scalar targets with/without product gates, budgeted JSON evaluator, and six central-feedback revisions in repair/. |

Run offline from the project root:

    python extension/writer/compare.py score
    python extension/writer/report.py
    python extension/synthesis/experiment.py score
    python extension/counters/writer.py score
    python extension/counters/run.py
    python extension/audit/lattice.py
    python extension/recurrent/audit.py

Each directory documents further training, replay and external corpus requirements. Public replication using archived replies does not need a model account. Suites differ in denominators and numerical oracles; no cross-task pooled accuracy is claimed.

coverage.csv maps each requested task to its actual distribution, proven size statements and unresolved limits. The runnable fixed-width programs, directly written free-topology graphs and iterative engineered circuits answer different questions. To run the first-response square on your own inputs: python extension/synthesis/infer.py -9.5 3 4 10.

For the complete offline replay entry point, install the dependencies in extension/requirements.txt and run python extension/reproduce.py. Add --refit to retrain the main comparison and --hard-cases to fetch and verify the pinned public corpora. This rewrites derived results; use an extracted copy to preserve the frozen release. Separate same-graph optimizer ablations are documented in smooth_optimizer/. No new Astra responses are requested by this replay.
