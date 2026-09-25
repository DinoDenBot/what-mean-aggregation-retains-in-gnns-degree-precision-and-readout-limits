#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"
OUTPUT="${2:-${SCRIPT_DIR}/reproduction-${MODE}}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "usage: $0 [smoke|full] [output-directory]" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"
python "${SCRIPT_DIR}/bundle/test_bundle.py" \
  --config "${SCRIPT_DIR}/config.json" \
  --output "${OUTPUT}/unit_test_report.json"
python "${SCRIPT_DIR}/bundle/run_experiment.py" \
  --config "${SCRIPT_DIR}/config.json" \
  --output "${OUTPUT}" \
  --mode "${MODE}"

python - "${OUTPUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = json.loads((root / "metrics.json").read_text())
if metrics.get("status") != "completed":
    raise SystemExit("reproduction did not complete")
lines = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name == "artifact_manifest.sha256":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root)}")
(root / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "completed", "mode": metrics["mode"], "verdict": metrics["decision"]["verdict"]}))
PY
