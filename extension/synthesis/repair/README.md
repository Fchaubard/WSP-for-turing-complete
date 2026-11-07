# Central-only, single-revision circuit study

Six designated trials cross square, sine and nonnegative self-power with affine/ReLU and product/ReLU graphs using the original replicate label 9213. The first response remains a one-shot experiment. A new isolated request supplies it as data and asks GPT-6 Astra for one revised JSON program under the same 4,096-operation budget. This revision uses the existing runner and a 300-second wall-time limit. Reply metadata record elapsed time and completed item types.

The feedback generator recreates only the 4,096 central inputs (seed 131490), labels them with correctly rounded MPFR and provides the three worst ULP errors and exact/within-one counts. Nonfinite outputs precede finite ones; ties are resolved by input order. If an original is perfect, all three examples have zero error. If an original cannot compile, its diagnostic includes the error and the first three central labels. The prompt contains the original permitted-domain specification, including known range endpoints. No outer examples, scores, target values, failed examples or hints from our own compiled networks enter a repair prompt.

Scoring runs after a revision has been frozen. It independently reproduces the original 16,384-point outer shell and draws 32,768 fresh outer points with seed 640317. The signed tasks independently choose each sign with equal probability. Self-power uses the positive shell only. A second central holdout uses seed 385591. Scores from either evaluation script never enter prompt construction. These six pairs establish what happened under this variable-quality feedback; a perfect random central sample can supply no error signal.

- `trials/` contains exact repair prompts, diagnostic JSON, raw replies and request metadata.
- `manifest.json` records the designated original IDs. `status.json` records compile status, errors, byte hashes, expanded operation counts, time and tool events.
- `results.csv` contains per-stage/per-split exact counts, one-ULP counts and error metrics.
- `python feedback.py` reconstructs central-only prompts. `python score.py` and then `python report.py` replay paired results and regenerate its six table rows. The accompanying exposition is maintained in the paper source.
- Fresh collection is optional and uses `python collect.py`, the existing local Codex CLI and its account access. Existing replies and terminal request errors are left intact.
