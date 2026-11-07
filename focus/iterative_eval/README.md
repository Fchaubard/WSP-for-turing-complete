# Independent audit of feedback-based WSP

Run `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python focus/iterative_eval/evaluate.py`
from the project root in the study environment. The script requires the
collector's signed-off `focus/iterative/FROZEN.json`. The post-freeze audit
rechecks the original source hashes, the freeze hashes, all live-width choices,
all selections and all stopping rules. It reconstructs each exact feedback
body, including reported weights, scores and the actual worst inputs, labels,
predictions and tie order from train and development data alone. It checks
the model's reply shape, raw numbers, completion metadata and tool-event count.
The expected static prompt scaffold was inspected and hashed before scoring.
It is the same across calls after substituting the original prompt and round.
The original protocol and post-freeze confirmation sampling plan appear in
`focus/iterative/PROTOCOL.md` and `PLAN.md`, respectively.

The emitted `audit.json` is the completion marker for this independent
evaluation. Its checksums identify the exact CSV outputs. `endpoints.csv`
contains the two selected WSP conditions with the established endpoint
columns, plus the weighted loss on each split (522 rows).
`comparators.csv` rescored the two original WSP conditions and six gradient
methods on the identical datasets. `extended.csv` additionally identifies
the reused validation data, fresh confirmation sets and the overlapping
legacy 9–32 early sequence score. The ordinary `dev` rows always use
untouched examples (17–32 for Fibonacci and triangular).
`stage_scores.csv` includes round zero and every finite valid candidate and
post-decision incumbent on the displayed original splits. Invalid responses
have a decision but no numerical candidate endpoint. Confirmation results
are restricted to the final checkpoint of each method. `splits.csv`
publishes input/label fingerprints and denominators for each data split.
The collision check excludes the deliberately overlapping legacy early split.

`paired.csv` and `paired_summary.csv` compare the selected WSP endpoints
with their own round-zero arm and each gradient method. Loss and RMSE deltas
are iterative minus comparator. Negative values favor the iterative endpoint.
`by_task.csv` and `by_architecture.csv` retain original-output-unit RMSE,
exact fractions and weighted losses separately by target. Individual-seed
results and all exact numerators and denominators remain in the endpoint
CSVs. No aggregation of raw errors across function units is used.
`calls.csv`, `original_calls.csv`, `model_costs.csv`, `proposals.csv` and
`stopping.csv` separate new revision requests, the original reused requests,
compute across the two stages, per-size proposals, and per-size termination.
Summed request elapsed times include concurrent calls; cached input and
reasoning output tokens are subsets of the respective input and output
counts. No monetary or hardware cost is inferred from these data.

`cross_split_changes.csv` follows each arm and setting from its own round
zero to the selected endpoint across all available splits.
`certified_optima.csv` compares the 12 one-hinge training objectives with
their known weighted real-arithmetic minima using an additive 1e−10 tolerance.
