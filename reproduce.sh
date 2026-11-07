#!/usr/bin/env bash
# Run in an extracted copy to preserve the original score files for comparison.
set -euo pipefail
cd "$(dirname "$0")"
pybin="${PYTHON:-python3}"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for f in frontier/run.py frontier/gradient.py frontier/general_rational.py; do "$pybin" "$f"; done
for f in probe short_probe task_probe; do "$pybin" "writer/$f.py" score; done
"$pybin" scalar/benchmark.py all
for f in scalar/{compensated,selfpower,signed_selfpower}.py; do "$pybin" "$f"; done
for f in sequence/{run,synthesize,robust_sort,robust_control,gauge_control,tanh_recurrence,signed_indices}.py; do "$pybin" "$f"; done
"$pybin" paper/figures.py
"$pybin" verify.py
