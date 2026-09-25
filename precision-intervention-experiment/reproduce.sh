#!/bin/sh
set -eu
cd "$(dirname "$0")"
output=${1:?Usage: sh reproduce.sh NEW_OUTPUT_DIRECTORY}
uv run --offline --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 python bundle/test_intervention.py
uv run --offline --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 python bundle/run.py --output "$output"
uv run --offline --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 python bundle/analyze.py --results "$output"
