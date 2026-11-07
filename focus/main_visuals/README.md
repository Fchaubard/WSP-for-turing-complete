# Frozen function predictions

`build.py` reconstructs square, sine and hyperbolic tangent predictions from the saved examples-only one-shot WSP, accepted iterative WSP, and gradient-descent-then-L-BFGS weights. Each uses the largest tested scalar width (16, 50 coefficients) and first recorded random start (164). These are illustrative individual runs, not median curves or a claim about all settings.

The shaded interval is [-3,3], which supplies training and development inputs. Twelve plotted points are the original training examples. The curve grid spans [-10,10]; it is a visualization grid, not the empirical test set. Before writing the figure, the script replays all nine networks on the archived training, development-feedback, untouched central, and outer splits and requires raw RMSE and exact counts to match the published records (36 rows).

Run `python focus/main_visuals/build.py` in the project numerical environment. No model calls, fitting, or source-record modifications occur. `prediction_curves.csv` contains the actual grid and outputs. `observed_points.csv` contains the plotted training pairs. `verified_metrics.csv` contains the audited per-network metrics, and `audit.json` records source hashes and the selection rule. The vector PDF and high-resolution PNG are publication assets; `prediction_curves.tex` supplies the caption.
