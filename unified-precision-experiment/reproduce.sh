#!/usr/bin/env bash
set -Eeuo pipefail
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="${1:?usage: reproduce.sh new-output-directory}"
if [[ -e "${REPRO_DIR}" ]]; then
  echo "refusing to overwrite existing reproduction directory: ${REPRO_DIR}" >&2
  exit 2
fi
mkdir -p "${REPRO_DIR}"
cp "${STUDY_DIR}/run.py" "${STUDY_DIR}/SPEC.md" "${STUDY_DIR}/verify.py" "${REPRO_DIR}/"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "${REPRO_DIR}/run.py"
python "${REPRO_DIR}/verify.py"
