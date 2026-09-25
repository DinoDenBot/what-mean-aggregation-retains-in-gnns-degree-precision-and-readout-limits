#!/bin/bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
study="$root/inductive-csbm-transfer-experiment"
mode=${1:-smoke}
output=${2:?usage: reproduce_inductive_transfer.sh smoke|full NEW_OUTPUT_DIRECTORY}
python_bin=${PYTHON_BIN:-python3}
if [[ "$mode" != smoke && "$mode" != full ]]; then
  echo 'mode must be smoke or full' >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite: $output" >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" "$study/bundle/test_bundle.py" \
  --config "$study/config.json" --output "$output.unit_test_report.json"
arguments=(--config "$study/config.json" --output "$output" --mode "$mode")
if [[ "$mode" == full ]]; then
  arguments+=(--confirm-production inductive-csbm-transfer-v1)
fi
"$python_bin" "$study/bundle/run_experiment.py" "${arguments[@]}"
cp "$output.unit_test_report.json" "$output/unit_test_report.json"
"$python_bin" "$study/bundle/analyze_results.py" --results "$output"
"$python_bin" "$study/bundle/validate_results.py" \
  --results "$output" --unit-test-report "$output/unit_test_report.json"
echo "CSBM transfer $mode run completed: $output"
