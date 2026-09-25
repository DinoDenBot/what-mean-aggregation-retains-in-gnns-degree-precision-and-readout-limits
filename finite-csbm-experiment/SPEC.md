# Preregistered finite-CSBM one-hop validation

Status: protocol frozen before any production run.

## Question

Does the rate-versus-content Fisher separation observed for Poisson stars
persist for an exact finite-`n` rooted contextual stochastic block model
(CSBM), where neighborhood counts are binomial and the environmental tangent
also contains graph homophily?

This is a one-hop validation, not a depth theorem.  Conditional on a balanced
labeling, the incident edges of a fixed root can be sampled exactly without
materializing graph edges that no one-hop representation queries.  A unit
test compares this root-row sampler with explicit finite adjacency sampling.

## Finite graph experiment

Take an even number `n` of vertices with balanced labels in `{+1,-1}` and fix
a root of label `+1`.  There are

\[
M_+=n/2-1,\qquad M_-=n/2
\]

possible same- and cross-class neighbors.  Conditional on the labels, edges
are independent with

\[
q_+(\lambda,\rho)=\frac{\lambda(1+\rho)}{2M_+},\qquad
q_-(\lambda,\rho)=\frac{\lambda(1-\rho)}{2M_-}.
\]

Thus the root has exactly expected degree `lambda`, with expected same/cross
counts `lambda(1+rho)/2` and `lambda(1-rho)/2`.  Neighbor contexts obey

\[
X_j\mid Y_j=s\sim N(\mu s,1).
\]

The fixed, parameter-independent message map is

\[
x_j=(1,Y_j,X_j,Y_jX_j).
\]

The final interaction is included equally for all architectures and is the
output of a valid local message map.  It lets a cardinality-preserving sum
represent the complete local score, isolating aggregation from message-map
misspecification.

The environmental parameter is

\[
\psi=(\lambda,\rho,\mu),
\]

with fiducial `rho=0.5` and `mu=0.75`.  For observed same/cross counts
`N_s`, the complete local scores are

\[
S_\lambda=\sum_{s\in\{+1,-1\}}
\frac{N_s-M_sq_s}{\lambda(1-q_s)},
\]

\[
S_\rho=\sum_{s\in\{+1,-1\}}
\frac{s(N_s-M_sq_s)}{(1+s\rho)(1-q_s)},
\qquad
S_\mu=\sum_{j\in N(v)}(Y_jX_j-\mu).
\]

The analytic raw Fisher matrix uses the Bernoulli formula

\[
F_{ab}^{\rm edge}=\sum_s
M_s\frac{\partial_aq_s\partial_bq_s}{q_s(1-q_s)},
\qquad F_{\mu\mu}=\lambda,
\]

with zero edge/mark cross block.  Scores are whitened by its symmetric
inverse square root.  Canonical efficiencies for `lambda`, `rho`, and `mu`
are evaluated as Rayleigh quotients in the corresponding raw parameter
directions; the generalized eigenvalues are also reported.

## Architectures

The confirmatory comparison is deliberately small:

| Module | Role |
|---|---|
| mean | normalized-loss target |
| sum | sufficient positive control |
| mean + degree | sufficient restoration control |
| single-head GAT | exploratory normalized attention |

Every module is trained at the fiducial environment to predict the whitened
complete local score, then frozen.  The immediate aggregate `A` is primary;
the common post-aggregation hidden state `H` is secondary.  GAT carries no
confirmatory rate-loss threshold.

## Conditions and independent pools

- Graph sizes: `n in {512, 2048}`.
- Expected degrees: `lambda in {16, 64}`.
- Seeds: `17, 29, 43`.
- Architecture train/validation: `20,000 / 5,000` independent roots.
- Nuisance fit/validation: `16,000 / 4,000` independent roots.
- Fisher evaluation: `25,000` independent roots.
- Bootstrap: `1,000` paired root-neighborhood resamples per fixed seed.

Sample identities for architecture training, architecture validation,
nuisance fitting, nuisance validation, and Fisher evaluation are disjoint.
The nuisance network standardizes its inputs using its fit pool only.

## Estimator

For observation `T` and whitened raw score `Z`, fit
`m_T(T)=E[Z|T]` on the independent nuisance pool.  On the Fisher pool use

\[
\widehat E_T=\frac1m\sum_i
\{\widehat m_iZ_i^\top+Z_i\widehat m_i^\top
-\widehat m_i\widehat m_i^\top\}.
\]

Regression error produces a PSD downward bias conditional on the fitted
nuisance function.  Report unprojected matrices and eigenvalues; tiny
negative values are diagnostic rather than silently clipped.

## Primary condition and frozen decision rule

The primary condition is `n=512, lambda=64`, assessed separately for every
declared seed with 95% paired percentile intervals.

The scientific result is **supported** only if all validity gates pass and:

1. mean's rate upper bound is below `0.20`;
2. mean's homophily and context lower bounds both exceed `0.65`;
3. for sum and mean+degree, every canonical-direction lower bound exceeds
   `0.75`;
4. the paired rate advantage of each sufficient control over mean has lower
   bound above `0.40`.

It is **contradicted** after valid estimation if mean's rate lower bound
exceeds `0.50`, or if either sufficient control has a rate upper bound below
`0.50`.  Other valid outcomes are inconclusive.  GAT is exploratory.

## Validity gates

- All unit tests pass, including analytic score/Fisher checks and the
  explicit-adjacency root-row equivalence test.
- Raw whitened-score covariance operator error is at most `0.04`.
- Analytic sufficient-oracle directional error is at most `0.05`.
- Learned-versus-analytic sufficient-oracle directional error is at most
  `0.07`.
- Every unprojected generalized eigenvalue at the primary condition is at
  least `-0.06`.
- The maximum estimated violation eigenvalue of `E_H <= E_A` at the primary
  condition is at most `0.06`.
- All declared sample-ID ranges are disjoint.

## Evidence boundary

A successful result would show that the tangent-specific split survives an
exact finite-CSBM one-hop law with binomial edge counts, homophily, and
contextual marks.  It would not prove recursive behavior, latent-label
behavior, finite-graph generalization risk, or sparse-depth Fisher dynamics.

