# Frozen prospective protocol: inductive CSBM transfer

Status: preregistered and frozen before any production target evaluation.

Experiment ID: `inductive-csbm-transfer-v1`.

## Question and evidence boundary

The experiment asks whether retaining realized degree improves predictive
transfer when an ordinary two-layer GNN is trained for node classification on
several source graphs and then frozen before it sees independent target graphs.
It is a controlled inductive experiment, not a claim about arbitrary graph
families, deeper networks, real-world benchmarks, or optimization in general.

The primary comparison is mean aggregation versus the same mean architecture
with one added degree statistic. Sum and PNA provide architectural context but
do not enter the confirmatory decision. The design distinguishes:

1. an ordinary in-distribution advantage on fresh source-distribution graphs;
2. a target-distribution transfer advantage; and
3. an advantage that increases under shift.

No production result may be inspected before this protocol, the executable
bundle, its source manifest, and its archive digest are fixed.

## Contextual stochastic block model

Every graph has `n=4096` vertices and exactly balanced latent labels
`Y_i in {-1,+1}`, randomly permuted. Conditional on the labels, unordered
edges are mutually independent. For a graph with environment
`psi=(lambda,rho,mu)`, the within- and between-class edge probabilities are

\[
 p_{\rm same}=\frac{\lambda(1+\rho)}{n-2},\qquad
 p_{\rm cross}=\frac{\lambda(1-\rho)}{n}.
\]

This gives every vertex expected degree `lambda`. Node features have eight
coordinates:

\[
 X_i=\mu Y_i e_1+\epsilon_i,\qquad
 \epsilon_i\sim N(0,I_8).
\]

Only `X` and the graph are passed to a model. Labels are used solely as
supervised targets on source graphs and as held-out outcomes for metrics on
target graphs. Neighbor labels are never model inputs.

The source environment is

\[
 (\lambda_0,\rho_0,\mu_0)=(24,0.60,0.35).
\]

The graph is generated exactly from the finite Bernoulli SBM, using geometric
skip sampling over each within-class triangle and the cross-class rectangle.
No dense probability matrix or Poisson edge approximation is used.

## Frozen graph pools

There are eight independent training blocks with top-level seeds
`17, 29, 43, 71, 101, 131, 173, 211`. Within each block, all architectures
receive the same generated graphs:

- six independent source training graphs;
- two independent source validation graphs; and
- eight independent target graphs for every evaluation condition.

Source, validation, and target seeds use disjoint hash namespaces. Target
graphs are independent across conditions and training blocks. All four models
within a block are evaluated on the identical target graph realizations, so
model contrasts are paired. No target graph is generated until all 32 fitted
weights have been frozen and hashed.

Fresh graphs at the unchanged source environment form the `source` evaluation
condition. The six single-axis shifts are:

| Family | Low | High | Fixed coordinates |
|---|---:|---:|---|
| Mean degree | `lambda=12` | `lambda=48` | `rho=.60, mu=.35` |
| Homophily | `rho=.30` | `rho=.80` | `lambda=24, mu=.35` |
| Feature signal | `mu=.20` | `mu=.50` | `lambda=24, rho=.60` |

The descriptive combined grid is the Cartesian product

\[
 \lambda\in\{12,48\},\quad
 \rho\in\{0.30,0.80\},\quad
 \mu\in\{0.20,0.50\}.
\]

Homophily changes who connects to whom; feature signal changes the conditional
feature law. They are therefore distinct changes in the finite data-generating
process rather than rotations of a single Gaussian-location parameter.

## Architectures and training

All models are inductive two-layer message-passing classifiers. At each layer,
the update concatenates the current node state with the declared neighbor
aggregate, applies one affine map, SiLU, and dropout. A final affine head emits
one binary logit.

| Model | Neighbor aggregate | Hidden width | Role |
|---|---|---:|---|
| Mean | Arithmetic neighbor mean | 64 | Confirmatory normalized aggregator |
| Mean + degree | Mean plus frozen-standardized `log(1+d_i)` | 64 | Confirmatory degree restoration |
| Sum | Arithmetic neighbor sum | 64 | Cardinality-preserving context |
| PNA | Mean, standard deviation, maximum, and minimum crossed with identity, logarithmic amplification, and attenuation scalers | 24 | Degree/statistics-aware context |

The PNA width is reduced deterministically so its trainable parameter count is
close to the approximately 10,000-parameter primary models. The exact counts
are recorded before training. The mean and mean-plus-degree models must differ
by at most two percent in parameter count; PNA must be within twenty percent of
mean. These are validity gates, not fitted hyperparameters.

Degree standardization and PNA's average-log-degree normalization are computed
from the six source training graphs in a block and then frozen for validation
and every target condition. All methods use AdamW, learning rate `0.003`,
weight decay `1e-4`, full source-graph batches, gradient clipping at 5, and at
most 300 epochs. Early stopping uses only the two source validation graphs,
with a minimum of 50 epochs and patience 30. There is no target tuning,
condition-specific calibration, learning-rate search, or model selection.

## Metrics and estimands

Metrics are first reduced to a mean within each target graph, making the graph
rather than a node the resampling unit.

- Primary: binary test log loss in nats per node.
- Secondary: accuracy and Brier score.
- Diagnostic: all three metrics within fixed realized-degree bins
  `[0,7], [8,15], [16,23], [24,31], [32,47], [48,infinity)`; a graph-bin cell
  with fewer than 50 nodes is reported as missing.

For condition `c`, training block `s`, and target graph `g`, define the paired
log-loss advantage

\[
 A_{csg}=L_{\rm mean,csg}-L_{\rm mean+degree,csg}.
\]

Positive values favor degree retention. The shift-specific gain is

\[
 G_c=E[A_c]-E[A_{\rm source}].
\]

This difference separates an advantage that is already present on fresh
source graphs from an advantage that grows under distribution shift. Analogous
paired advantages are reported for Brier score. Accuracy uses
`accuracy(mean+degree)-accuracy(mean)` so positive values always favor the
degree model.

Uncertainty uses a nested paired bootstrap with 10,000 fixed-seed replicates:
resample the eight independent training blocks, then resample target graphs
within every selected block. Source and shifted target graphs are resampled
independently when estimating `G_c`. Nodes are never treated as independent
replicates. Ordinary two-sided 95% intervals are reported for every condition;
the four primary degree-shift tests use Bonferroni-protected one-sided 98.75%
lower bounds.

## Frozen scientific decision

The practical margins are `0.001` nat per node for target log-loss advantage
and `0.0005` nat per node for shift-specific gain.

Provided every validity gate passes:

1. **Degree improves transfer** if the simultaneous lower bound for `A_c`
   exceeds `0.001` at both `lambda=12` and `lambda=48`.
2. **Shift-amplified transfer** additionally requires the simultaneous lower
   bound for `G_c` to exceed `0.0005` for at least one of those two shifts.
3. **Probability-only support** is recorded if degree improves transfer and the
   simultaneous accuracy-gain intervals for both degree shifts lie entirely
   inside `[-0.002,0.002]` while the log-loss criterion passes.
4. **Practically little benefit** is recorded if the simultaneous log-loss
   advantage intervals for both degree shifts lie inside `[-0.001,0.001]`.
5. **Harm under degree shift** is recorded if the simultaneous upper bound for
   `A_c` is below `-0.001` for either degree shift.
6. Every other valid pattern is mixed or inconclusive.

The single-axis homophily and feature-signal conditions are prespecified
secondary analyses with within-family simultaneous log-loss intervals for
target advantage and shift gain. The combined grid, sum, PNA, accuracy outside
the equivalence rule, Brier score, and degree-bin breakdowns are descriptive. A
favorable contextual comparator does not substitute for the
mean-versus-mean-plus-degree decision.

## Validity gates

- All unit tests pass, including exact balance, seed reproducibility,
  undirected simple-graph construction, sampler-frequency checks, permutation
  invariance, aggregator identities, label-input exclusion, and one full-size
  CUDA forward/backward step for every architecture on the production image.
- The source, validation, and target seed namespaces are disjoint.
- Every graph is simple, undirected, exactly label-balanced, and finite.
- Every realized mean degree is within ten percent of its declared expectation.
- Every declared model/graph/condition metric is present and finite.
- Model parameter-count gates pass.
- Every fit reaches 50 epochs, produces a finite validation checkpoint, and has
  no nonfinite gradients or losses.
- The global weights manifest exists before target generation, and its hashes
  are unchanged after evaluation.
- All architectures have exactly matched target graph IDs within a block and
  condition.
- Accuracy and Brier values lie in `[0,1]`; log loss is nonnegative.
- Production analysis uses exactly 10,000 bootstrap replicates and the frozen
  margins above.

A failed validity gate makes scientific status `invalid`, never supported or
contradicted. Successful execution alone does not imply a favorable result.

## Execution and evidence boundary

The full run uses only generated inputs. The recommended target is one Secure
Cloud NVIDIA GeForce RTX 4090 under the pinned official PyTorch 2.8 image. The
scientific process has no network access or credentials. A durable outer
wrapper records status, PID, log, progress, environment, frozen weights,
authoritative outputs, and SHA-256 manifests. The pod is stopped only after
the retrieved results match the pod-side checksum manifest.

Development smoke output is operational only. It uses smaller graphs, fewer
conditions, three epochs, and cannot enter the paper or change this protocol.
