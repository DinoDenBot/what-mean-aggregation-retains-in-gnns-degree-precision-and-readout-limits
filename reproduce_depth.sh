#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mode=${1:-check}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/depth-score-geometry/bundle" \
  python3 "$root/depth-score-geometry/bundle/test_canonical_leaf_depth.py"

if [ "$mode" = check ]; then
  exit 0
fi
if [ "$mode" != full ] || [ "$#" -ne 2 ]; then
  echo "usage: $0 check | full NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
output=$2
if [ -e "$output" ]; then
  echo "refusing to overwrite: $output" >&2
  exit 2
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/depth-score-geometry/bundle" \
  python3 "$root/depth-score-geometry/bundle/canonical_leaf_depth.py" \
  --config "$root/depth-score-geometry/EXPERIMENT_OFFER_V2.json" \
  --output "$output" --decimal-precision 80
python3 - "$output/metrics.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['scientific_success'], report
print('finite-depth calculation passed its validity and hypothesis checks')
PY
