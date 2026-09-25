# Frozen-model precision intervention

The protocol is in `SPEC.md`; executable settings are in `config.json`. The original synthetic-study source and 16 model checkpoints are staged in `bundle/` and `inputs/`. `FROZEN.json` identifies all scientific source and inputs before the new target graphs are generated. This experiment was authorized directly in the paper-review conversation and executed locally, outside Radia.

`bundle/test_intervention.py` checks native equivalence, degree protection, empty neighborhoods, output rounding, and noise scaling. The preflight smoke run uses a separate graph namespace and is only an interface check. Its outcomes do not enter the scientific result.

Reproduce after installing uv and the pinned dependencies (the original run used cached packages offline):

```sh
sh reproduce.sh results-reproduction
```

The runner refuses to overwrite an output directory, verifies the frozen manifest, and enforces a one-hour deadline. It uses four CPU threads. Python, NumPy, Torch, resource use, native-prediction checks, and weight verification are recorded in `execution.json`.

The raw output is `replicate_metrics.csv`. `graph_metrics.csv` averages noise realizations within each graph. `contrasts.csv` includes paired nested-bootstrap and block-only sensitivity intervals. `decision.json` applies the frozen primary criterion. `RESULT_MANIFEST.json` records the result-file hashes. Accuracy contrasts retain the arithmetic order used for loss contrasts; they must be interpreted as accuracy changes, with larger accuracy being better.

The primary intervention changes only the first-layer neighbor mean. Two secondary rounding arms change both layers. All degree features and self states remain intact. Finite dtype arms round completed aggregates; they do not emulate low-precision accumulation or count division. The learned features are continuous, so this study measures predictor sensitivity and degree protection without establishing the rational-denominator mechanism from the discrete theorem.
