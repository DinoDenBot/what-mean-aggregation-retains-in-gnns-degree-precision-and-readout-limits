#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

python3 theory-code/test_exact_poisson_ratio.py
python3 theory-code/test_smoothed_ratio_tv.py
python3 theory-code/gaussian_mean_fisher.py --self-test
sh reproduce_depth.sh check
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
root = Path('precision-intervention-experiment')
files = json.loads((root / 'FROZEN.json').read_text())['files']
for relative, digest in files.items():
    path = root / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
print(f'Frozen intervention inputs verified: {len(files)} files')
PY

uv run --python 3.12 --with numpy==2.1.2 \
  python finite-csbm-experiment/exact_ratio_fisher.py \
  --output "$temporary/finite-csbm.csv"
uv run --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 \
  python finite-csbm-experiment/bundle/test_bundle.py \
  --config finite-csbm-experiment/config.json \
  --output "$temporary/finite-tests.json"
uv run --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 \
  python inductive-csbm-transfer-experiment/bundle/test_bundle.py \
  --config inductive-csbm-transfer-experiment/config.json \
  --output "$temporary/transfer-tests.json"
uv run --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 \
  python precision-intervention-experiment/bundle/test_intervention.py

echo 'Reproduction code checks passed.'
