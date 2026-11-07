# Independent review of the learning-curve edition

The final 88-page manuscript was checked after rebuilding with the corrected figure assets. The main text concludes on page 9. Main Figures 2 and 3 appear on pages 6 and 7; the all-task and proposal learning plots appear on pages 24 and 25. The revised plots render clearly without clipped axes, colliding titles, or footer overlap.

## Checks performed

- `audit.py` reconstructs 174 trajectories directly from the frozen interaction JSON. It reproduces every development-first acceptance decision, verifies nonincreasing selected development loss, and compares all 348 final training/development endpoints with the independently archived evaluator.
- `check_plot_data.py` checks all 2,088 selected split/round records (10,440 metric cells), 3,972 valid proposal metric cells, and 768 aggregate values against those reconstructed records and frozen stage scores. The invalid proposal remains present as three blank split records. The summary check includes all 240 newly added within-task raw-RMSE means.
- `check_curves.py` evaluates 9,009 plotted predictions directly from archived coefficient arrays, independently of the production prediction helper, and checks 3,003 target values and 36 observed training coordinates. The largest difference from scalar accumulation is 5.68e-14, consistent with floating-point reduction order.

## Interpretation and presentation

Round 0 denotes the original one-shot WSP response, not the random initialization. Later points denote at most three revision opportunities. Stopped trajectories retain their selected weights; they are not new model calls. The development set is reused for feedback and selection, so it is not described as an untouched test set. Training error can rise because the frozen selector prioritizes development loss.

The main learning plot uses medians of scaled RMSE over all 87 settings per information condition, including zero errors. The appendix task panels use means of raw RMSE over each task's sizes and starts; this different aggregation is explicit and exposes narrow recurrent networks that a zero median could hide. The exactly zero running-sum panel uses a linear axis. Other task panels use their stated logarithmic ranges. No uncertainty intervals or convergence guarantee are implied.

The function plots are three fixed illustrative cases at width 16 and seed 164. They are not average curves or evidence of the aggregate optimizer rank. All accepted and rejected proposal distinctions, raw data, errors, and captions were checked.

Initial presentation issues were corrected: footer and row-label overlap, clipped proposal-axis text, an ambiguous logarithmic-axis caption, and a zero-median annotation that had implied every setting was exact. No remaining material issue was found in the reviewed final pages.

Replay the checks with the project numerical environment:

```bash
python focus/learning_review/audit.py
python focus/learning_review/check_plot_data.py
python focus/learning_review/check_curves.py
```

## Scientific role of the main figures

The final impact review retained three distinct claims. Figure 1 compares fixed-network endpoint performance under the explicitly scale-sensitive sum of task RMSEs. Figure 2 now compares training fit, development feedback, and untouched extrapolation across revision opportunities; its new third panel exposes a temporary first-revision extrapolation improvement in the formula-and-examples condition that disappears in later selected states. Figure 3 illustrates how the resulting functions behave beyond the observed interval. Its examples are regression cases, not prevalence estimates.

The added Figure 2 outer panel uses existing frozen stage scores only. All eight plotted outer medians were checked, and the exact-case sets have intersection and union of 30 across all rounds and all three splits in each information condition. Lines connect recorded points without smoothing. No test score enters feedback, checkpoint selection, or a new experiment. The final three-panel asset renders clearly and its caption states the retrospective test protocol and different logarithmic ranges.

For the three Figure 3 iterative networks, every effective hinge lies inside the central interval: square in [-2.419, 2.419], sine in [-2.606, 2.606], and hyperbolic tangent in [-2.3, 2.3]. Their linear outer continuations therefore follow directly from the stored weights. This supports the stated mechanism in these cases without a claim that iterative WSP always worsens extrapolation.
