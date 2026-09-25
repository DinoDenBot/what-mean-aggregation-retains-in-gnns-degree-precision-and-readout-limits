# What Mean Aggregation Retains in GNNs: Degree, Precision, and Readout Limits

Reproduction code for the numerical calculations and synthetic experiments
reported in the paper. The repository contains scripts, fixed configurations,
and the frozen model inputs needed for the intervention study. There are no
TeX sources or rendered manuscript files here.

## Start here

Install Python 3.12 and `uv`. From the repository root, run:

```sh
sh verify.sh
```

This checks the categorical and Gaussian calculations, the finite-depth code,
the finite-CSBM exact-ratio calculation, the frozen intervention inputs, and
the trained-model interfaces. It is a quick code check; it does not repeat the
full training experiments.

## Run the paper studies

Commands below use new output directories. Each study's `SPEC.md` or
`config.json` records its model, seeds, and analysis settings.

**Categorical precision and learned readouts (CPU).** The runner generates
`primary.csv`, `partitions.csv`, `probes.csv`, and `arithmetic.csv`, then checks
the calculations independently. The recorded environment was Python 3.12,
NumPy 2.5.2, SciPy 1.18.1, and PyTorch 2.14.0.

```sh
uv run --python 3.12 --with numpy==2.5.2 --with scipy==1.18.1 \
  --with torch==2.14.0 bash unified-precision-experiment/reproduce.sh \
  /tmp/unified-precision-run
```

**Quadratic readout rate (CPU).** This writes a `result/` directory under the
study, then verifies the binary cells with an independent moment calculation.

```sh
uv run --python 3.12 --with numpy==2.5.2 --with scipy==1.18.1 \
  python regular-readout-asymptotics/run.py
uv run --python 3.12 --with numpy==2.5.2 --with scipy==1.18.1 \
  python regular-readout-asymptotics/verify.py
```

**Finite-depth mean (CPU).** The script checks the code first, runs the full
finite-state calculation at 80-digit precision, and checks its validity and
hypothesis gates.

```sh
sh reproduce_depth.sh full /tmp/depth-run
```

**Finite-CSBM label-ratio bound (CPU).** The output CSV gives the four
graph-size and expected-degree cells in the paper.

```sh
uv run --python 3.12 --with numpy==2.1.2 \
  python finite-csbm-experiment/exact_ratio_fisher.py \
  --output /tmp/finite-csbm-exact.csv
```

**Trained CSBM studies (GPU recommended).** Use Python 3.12, NumPy 2.1.2,
and PyTorch 2.8.0 as specified in each `bundle/requirements.txt`. The first
command reproduces the finite-CSBM smooth-probe study; the second trains and
evaluates the two-layer CSBM transfer models. Both write detailed metrics and
analysis to their output directories.

```sh
PYTHON_BIN=python bash finite-csbm-experiment/reproduce.sh full /tmp/finite-csbm-run
PYTHON_BIN=python bash reproduce_inductive_transfer.sh full /tmp/csbm-transfer-run
```

**Precision and noise intervention (CPU).** This uses the included frozen
model checkpoints and baseline graph metrics. Warm the pinned `uv` cache first
because the frozen runner uses offline mode. Its output includes paired
contrasts and the original mixed decision.

```sh
uv run --python 3.12 --with torch==2.8.0 --with numpy==2.1.2 \
  python -c 'import torch, numpy'
sh precision-intervention-experiment/reproduce.sh /tmp/intervention-run
```

The exact-ratio, perturbed-ratio, and Gaussian-mean checks live in
`theory-code/` and are included in `sh verify.sh`.

## Known reproduction gap

The original generator and intermediate input for an earlier finite-precision
and coarsening figure were absent from the available artifact. Those figure
points cannot be regenerated from this repository. The unified precision
study reruns related arithmetic channels, but it is a separate calculation.
