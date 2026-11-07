#!/bin/bash
# Authenticated Codex CLI is required for fresh draws. Scoring archives works offline.
set -euo pipefail
cd "$(dirname "$0")"
writer_bin="${CODEX_APP_RUNTIME_CLI_PATH:-codex}"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
for case_id in "$@"; do
  prompt="$PWD/case${case_id}.prompt.txt"
  result="$PWD/case${case_id}.reply.txt"
  if test -e "$result"; then echo "Reply exists: $result" >&2; exit 2; fi
  "$writer_bin" exec --ephemeral --skip-git-repo-check --ignore-user-config -C "$scratch" -s read-only -m gpt-6-astra -c 'model_reasoning_effort="medium"' --color never --json -o "$result" - < "$prompt" > "$scratch/trace" 2> "$scratch/stderr" || { tail -4 "$scratch/stderr" >&2; tail -2 "$scratch/trace" >&2; exit 1; }
  python3 - "$scratch/trace" "$PWD/case${case_id}.meta.json" <<'PYMETA'
import json,sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
kinds=Counter(str(r.get('item',{}).get('type')) for r in rows if r.get('type')=='item.completed')
meta={'requested_model':'gpt-6-astra','reasoning_effort':'medium','completed_item_types':kinds,
      'usage':[r['usage'] for r in rows if r.get('usage')]}
open(sys.argv[2],'w').write(json.dumps(meta,indent=2)+'\n')
PYMETA
done
