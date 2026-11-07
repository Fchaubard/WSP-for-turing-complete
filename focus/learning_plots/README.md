# Iterative WSP training and development curves

These are descriptive plots of the **saved** experiment. They do not add fitting steps, hosted-model calls, or test-guided selection. Run `python focus/learning_plots/build.py` in the existing NumPy/Matplotlib environment to rebuild them.

## Plot meanings

- `learning_main.pdf` / `.png` and `learning_main.tex`: three panels of **median scaled RMSE**: training, development feedback, and extrapolation testing. Each of the two information conditions contributes all 87 task/size/start settings at every revision opportunity. Round 0 is the original one-shot reply. There are at most three revision opportunities, not convergence to an optimum. Lines connect recorded discrete summaries. Vertical axes are logarithmic with different ranges. Extrapolation was scored only after selection was frozen; these retrospective test scores never supplied revision feedback or selected weights. No confidence interval is drawn.
- `learning_tasks.pdf` / `.png`: ten panels of **mean raw RMSE within each task**, averaging tested sizes and starts. Means expose poor low-capacity recurrent networks whose contribution disappears in a zero median. This is the task aggregation used in the headline raw-error sum, although these panels display training/development rather than outer testing. Scales vary by task. All positive panels are logarithmic; the exactly correct running-sum panel uses a linear axis at zero.
- `learning_proposals.pdf` / `.png`: selected states and individual proposals in two illustrative examples-only trajectories (seed 164, square with four hidden units and sorting with four residual blocks). Open circles mark accepted proposals, crosses mark rejected proposals. The solid curve is the retained state, which stays unchanged on rejection. These examples illustrate the selector, not a random sample of effect sizes.
- `learning_appendix.tex`: provenance, numerical interpretation, per-task and proposal figure captions.

The original collector accepts a proposal when `(development loss, training loss)` improves lexicographically. Thus retained development loss cannot increase, but retained training loss can. Raw development RMSE also need not monotonically decline because the acceptance loss scales each scalar residual by `max(1,abs(target))` before aggregation. The main figure applies RMSE to those scaled residuals. It shows that lower development error need not transfer: examples-only outer medians are 0.0571, 0.221, 0.235 and 0.235; formula-and-examples outer medians are 0.0731, 0.00590, 0.227 and 0.227. The temporary formula-arm test improvement is descriptive and was not used for early stopping.

Development feedback is reused for both prompting and selection. **It is not the untouched central test.** In the original trajectory JSON it is named `dev`, but in the separate evaluator it is named `feedback_dev`; the evaluator's `dev` column denotes the untouched test. The plotting exports retain the JSON's `dev` name and document it here. Outer scores in the machine-readable CSV were evaluated after the original freeze and are never used to select or revise weights.

## Data and provenance

`selected_rounds.csv` records 2,088 selected split states: 174 trajectories × four opportunities × three splits (training, development feedback, outer). `source_round` records which proposal supplied the selected weights. `carried_after_stop` is true when a later opportunity repeats an already stopped state; it is not a new request. Thirty trajectories per condition are already exact on every training/development example at round 0 and remain so.

`proposal_rounds.csv` retains all 332 size-specific proposal decisions, each repeated for three splits; the one invalid proposal has empty numerical fields. `decisions.csv` records validity, acceptance and request identifiers once per proposal. `round_summary.csv` records descriptive medians and request/exact counts; `task_round_summary.csv` gives both raw mean and median for each task and condition.

`audit.json` identifies all 60 source interaction files by SHA-256, cross-checks retained training/development scores against the independent stage evaluator and all three final split states against the published endpoints, and verifies monotone development loss. All 332 proposal decisions and 217 acceptances are retained. Independent reproduction by a separate reviewer is under `focus/learning_review/`.

No value is replaced by a small positive constant to make logarithmic plotting possible. No smoothing, interpolation beyond connecting observed opportunities, statistical uncertainty claim, or extrapolation selection is introduced.
