# Weight-Space Programming and Gradient Optimization on Nonconvex Algorithmic Tasks

The paper by Francois Chaubard and Astra (AI research collaborator) compares one-shot and iterative weight-space programming (WSP) with full-batch gradient descent using backpropagation, gradient descent followed by limited-memory BFGS, Adam followed by full BFGS, and trust-region least squares. Ten tasks give 29 nonconvex task–architecture pairs, each run from three original starts. Both WSP procedures are measured with examples alone and with the generating formula. A separate gradient control is fitted on the entire feedback pool. No method's endpoint is selected using test performance.

The current public manuscript is **output/pdf/weight-space-programming-public.pdf**. The dated pre-release edition is **output/pdf/weight-space-programming-2026-11-07.pdf**. The prior learning-curve edition remains **output/pdf/weight-space-programming-2026-11-06-learning-curves.pdf**. Its main text now includes the recorded training, development and untouched extrapolation curves, plus three full-domain function plots, alongside the original raw-error ranking. The large representative result table is preserved in the appendix. **output/iterative-wsp-training-curves.png** and **output/function-prediction-curves.png** are standalone previews; **output/iterative-wsp-learning-curves.csv** and **output/iterative-wsp-proposal-curves.csv** contain every plotted recorded state. **focus/learning_plots/** and **focus/main_visuals/** regenerate the figures offline. The preceding architecture edition is retained at **output/pdf/weight-space-programming-2026-11-06-architecture.pdf**. Figure 1 ranks all 14 complete fixed-network optimizer pipelines by the sum of ten task means of raw extrapolation RMSE. **output/all-optimizer-raw-rmse-ranking.csv** also includes the sum over all 87 settings, and **output/optimizer-rmse-task-contributions.csv** exposes each task's contribution. Section 5 reports 30 further graph-choice requests from examples. Their model descriptions, errors and measured execution costs are **output/architecture-choice-models.csv**, **output/architecture-choice-errors.csv** and **output/architecture-choice-costs.csv**. The full prompts and actual returned public explanations appear in the paper's appendix and in **focus/architecture_search/**. Earlier plots are restored in the same appendix. The prior illustrated edition is **output/pdf/weight-space-programming-2026-11-06-charts.pdf**. Open **output/table-3-task-charts.html** to select a task and see exact matches and raw errors side by side. **output/table-3-task-charts.pdf** contains the printable charts. The previous iterative edition is **output/pdf/weight-space-programming-2026-11-06-iterative.pdf**; the separately named one-shot PDF and table image preserve the earlier edition. **output/iterative-wsp-endpoints.csv** contains unrounded exact counts, raw root mean squared errors, and losses for individual starts. **output/iterative-wsp-by-size.csv** summarizes the three starts on the training, untouched central/early and outer sets. Integer indices 9–16 were used for feedback; the revised early test uses only 17–32. The original 9–32 diagnostic is labeled as overlapping. **output/optimizer-comparison.html** lets readers browse all sizes by task. The two original CSV files, **output/nonconvex-endpoints.csv** and **output/gradient-descent-and-lbfgs.csv**, still report the original early split. Historical convex controls are excluded from the primary totals.

To replay the iterative experiment offline in the published numerical environment:

1. Review **focus/iterative/PROTOCOL.md** and **focus/iterative_eval/PLAN.md**. The literal requests and replies, selection histories and global freeze are in **focus/iterative/**. Do not rerun the model to recreate an archived observation.
2. Run `python focus/iterative_eval/evaluate.py` to reconstruct the revisions and score the saved weights on untouched inputs. The previously recorded independent-test seed is reused on replay.
3. The optional full feedback-pool gradient control has saved weights in **focus/feedback_gradient/**. To refit, run `python focus/feedback_gradient/pooled.py fit`. To rescore those weights, run `python focus/feedback_gradient/pooled.py evaluate`.
4. Run `python focus/iterative_tables/report.py`, then `python focus/iterative_tables/task_charts.py`, then `python revision/iteration_release.py`, then `python paper/build.py`. The last command supplies a standalone LaTeX source ZIP. **output/research-bundle.zip** contains the code, weights, scores, prompts and replies.

To regenerate the architecture study from saved observations, run `python focus/architecture_search/evaluate.py`, `python focus/architecture_search/report.py`, `python focus/holistic_plots/build.py`, and `python revision/architecture_release.py`, then the manuscript and release commands below. **focus/architecture_search/README.md** documents graph evaluation and resource accounting. **focus/prompt_audit/** contains independent checks of proposals, costs, ranking totals and the iterative prompts. These replay commands require no new model calls.

To rebuild the illustrated learning-curve edition offline, run `python focus/learning_plots/build.py`, `python focus/main_visuals/build.py`, `python revision/learning_release.py`, and `python paper/build.py`. No new fitting or hosted-model calls are used.

## Read or run

- Open **output/pdf/weight-space-programming.pdf** for the manuscript. **output/preprint-source.zip** is the standalone LaTeX bundle, including figures and a resolved bibliography.
- Run **python demo.py verify** to exercise all ten compiled circuits, including the known one-ULP square failure. Inference needs only Python and NumPy. **DEMO.md** documents the commands and domain limits.
- Run **python verify.py** to check the archived headline results and reapply the literal Astra responses to the initialized networks.
- Read **literature/related_work.md** for the original X announcement, the release logs, the closest concurrent arithmetic study, and the sources that shaped the paper's claims.

See **extension/README.md** for the preliminary ten-task comparison. Its original prompts emphasized extrapolation; its optimizer checkpoints are reused only on identical problems. The current results are in **focus/endpoints/**. We distinguish the quality of training optimization from extrapolation at the same starting weights and central examples. Recurrent, smooth-function and branching programs have saved weights and independent references.

## What is certified

For a fixed coordinate readout in a generic dense linear RNN with zero initial state and no bias, the edit budget to realize a nonzero scalar impulse sequence of order r through horizon H is min(H, d+r), when state width d is at least r. The proof assumes exact real arithmetic. A generic initializer has a continuous joint density. Widths 1, 2, and 3 are respectively sufficient and necessary for an accumulator, homogeneous Fibonacci generator and homogeneous triangular generator. Exact sorting of four real numbers takes at least five ReLU gates and a five-gate DAG attains the bound. The separate bounded-binary64 comparator gives an upper bound of 65 ReLU executions to sort four binary64 values. Its machine guarantee requires the execution semantics stated in the appendix.

A directly returned 2,543-operation ReLU circuit is certified on the entire signed outer shell. Its separately engineered 3,352-operation repair is certified on all binary64 inputs in [-10,10], including subnormal outputs. The proof and executable graph are in extension/square_full; the size is an upper bound.

The shallow square has a sharp worst-case real-error bound of 50/(w+1)² on [-10,10] for a scalar residual network with w hinges. Exactness on random inputs is a much weaker requirement. The supplied 48-step square network scores 131,072/131,072 on a fresh outer-range sample, 40/304 on a midpoint challenge set, and 45/501 on small powers of two. The original shallow positive self-power and folded-sine circuits are approximations. The new deep WSP circuits with affine operations and rectified linear units in extension/scalar pass all 32,768 fresh shell samples per target against MPFR; both have recorded one-ULP failures. The signed self-power lookup has access to every integer ordinate. The paper identifies which solvers receive formulas, structural priors, or only central labels.

## Reproduce the experiments

The pinned environment uses Python 3.12, NumPy 2.5.4, SciPy 1.18.1, and Matplotlib 3.11.2. The original scalar jobs used Python 3.13.12 with the same package versions. The compute device was an Apple M5 with Accelerate BLAS. Exact-equality results near rounding boundaries depend on the numerical implementation.

From a fresh extraction run the following commands. The reproduction script regenerates the score files in that extraction. An authenticated model subscription is required only for new model calls.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py verify
python verify.py
bash reproduce.sh                   # initial experiments
pip install -r extension/requirements.txt
python extension/reproduce.py        # expanded offline replay
python revision/comparison_tables.py
python focus/nonconvex/audit.py
python focus/objective/experiment.py score
python focus/arithmetic/score.py
python focus/endpoints/analyze.py
python focus/report.py
python focus/refinement/run.py
python focus/lbfgs/experiment.py
python focus/lbfgs/report.py
python focus/lbfgs/paper_tables.py
python focus/coefficients/audit.py
python extension/optimizer_brief.py
python revision/export_metrics.py
python paper/build.py
python focus/export_table_image.py
```

The manuscript build uses pdflatex and BibTeX. The paper uses the official NeurIPS 2026 preprint style. The standalone source ZIP compiles with two runs of pdflatex on paper.tex. The code that writes new weights, the fixed templates supplied by the experimenters, and the Astra reply files are kept separate. For new gpt-6-astra proposals, run writer/run.sh with the case numbers in a fresh copy without the corresponding archived reply files. Independent reruns can generate different proposals. See the individual task directories for exact protocols.

| Directory | Contents |
| --- | --- |
| frontier | Exact-rational constructions for 480 finite prefixes and 75 global solutions, finite-precision stability experiments and the four-query gradient control |
| writer | Initial weights embedded in prompts, literal model replies, per-call model settings and aggregate usage, offline graders |
| scalar | Residual, compact deep, compensated, phase-folded and native-activation experiments, signed-integer self-power lookup, raw scores |
| sequence | Linear and tanh RNNs, signed integer clocks, sorter gates, robust operand routing, gradient fits and data-only synthesis |
| paper | Editable manuscript, supplementary material, plot generator, official conference style, offline build |

## Publication status

This research preprint and its supporting code, results, and analysis are publicly available in [WSP-for-turing-complete](https://github.com/Fchaubard/WSP-for-turing-complete). The repository does not represent a NeurIPS submission or acceptance. Independent replication on another numerical runtime remains useful.

The Python experiments and figures were developed with coding-agent assistance. Literal experiments preserve the original 21 CLI proposals, a 60-call matched comparison (180 arrays), branching-machine proposals, and directly synthesized circuits. Algorithmic compilers and subsequent stress-test-driven improvements are engineering work described in their own sections of the paper.
