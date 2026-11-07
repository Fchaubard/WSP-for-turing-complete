# Prompt and extrapolation audit

Run audit.py in this directory in the published numerical environment. This procedure only reads the already frozen records and writes audit artifacts in this directory.

prompts.csv records all 60 original and 161 iterative prompts, hashes and objectives. Every original includes a goal of exact prediction on unseen inputs across the supplied domain, subordinate to fit quality. Each revision contains the original prompt and a generalization reminder. Its operative selection criterion is development loss, then training loss.

square_geometry.csv computes all six width-16 square networks' active knot ranges, outer-knot counts, tail slopes, intercepts and real-function predictions at ±10, along with the standard evaluation scores. Every initial network had three or four outer knots. Every final network had zero. The analysis reads stored binary64 values as exact rational coefficients for the geometry. This calculation explains the changed piecewise-affine continuation. Causal attribution of the aggregate change to prompt wording requires an ablation.

diagnosis.tex gives the concise paper explanation. prompt_appendix.tex introduces the archived records and literal_trace.tex prints a complete original request, reply, first revision request and revision reply for the examples-only Fibonacci case at start 164. The four corresponding text files are exact copies. They were selected as a compact complete illustration, with no request content omitted.

The first-response and revision instructions requested numeric arrays and forbade explanations. Their final responses contain no explanatory reasoning text. Token metadata may count reasoning tokens without exposing their contents. The new architecture-selection study asks separately for a concise public construction summary.
