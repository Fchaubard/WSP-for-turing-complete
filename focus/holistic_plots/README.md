# Sum of raw extrapolation errors

Run `python focus/holistic_plots/build.py` in the project numerical environment. This build reads immutable endpoint scores and produces the main figure, a complete-range version and the restored prior plots. It never calls the model or changes a weight.

Each task contributes its mean raw RMSE over all tested sizes and three random starts. Adding the ten task means gives the plotted score. Running sum has six settings and the other tasks have nine. `ranking.csv` also records the literal sum of all 87 setting RMSEs. The ranking has all 14 procedures with a complete evaluation on that same grid, including four WSP information/feedback conditions, five training-only gradient endpoints, least squares, two additional-label gradient controls, WSP followed by least squares and adaptive moment estimation before curvature refinement. Original labels in the individual data identify the information and stage differences.

The score is sensitive to output magnitudes and large errors. A Fibonacci divergence produces a raw RMSE of approximately 7.399 × 10^233. The main figure retains its value and places an arrow where its bar exceeds 10^34. The appendix figure displays the full range. No large endpoint is dropped or capped in the numerical total. The iterative WSP scores are close at the displayed precision, so the top two direct labels use nine significant digits. Ordering always uses the full stored values.

`task_contributions.csv` contains each task's contribution. `per_setting.csv` contains all 1,218 scored endpoints. `audit.json` records the input checksums and the independent 270-digit decimal arithmetic check of the sums. Exact agreement at zero is evaluated in the original study tables.

The restoration of the earlier iterative comparison is a literal vector crop of the dated 58-page PDF, page 5. The crop is exactly the former chart, including its axis, legend and method labels. The other two restored figures copy the original vector artwork and preserve their captions with distinct references. Building the crop requires the existing pdflatex command.
