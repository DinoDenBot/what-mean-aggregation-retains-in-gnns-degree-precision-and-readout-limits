# Regular-readout asymptotics diagnostic

Direct user-authorized local follow-up to the unified precision audit. This is
an exploratory deterministic population calculation, not a preregistered
confirmation.

## Questions

1. Does the degree information recovered by the centered quadratic statistic
   of the standardized categorical mean approach the predicted
   `(K-1)/(2 lambda)` law?
2. For binary categorical means, do dyadic partition curves organize around
   the conjectured resolution coordinate `bits - 2 log2(lambda)`?

## Model and estimands

For the primary binary calculation, independent counts have laws
`C_b ~ Poisson(lambda p_b)`, with `lambda` in
`{32,64,128,256,512,1024}` and `p_1` in `{.2,.35,.5}`. The observation is the
exact ratio `R=C_1/N`, with a separate empty symbol. The normalized degree
score is `Z=(N-lambda)/sqrt(lambda)`, recentered under the truncated population.

The standardized mean fluctuation is
`X=sqrt(lambda)*(R-p_1)/sqrt(p_1*(1-p_1))` on nonempty observations and zero on
the empty observation. The quadratic decoder uses features `(1,X^2)`. Its
population score-projection lower bound is compared with `1/(2 lambda)`.
Canonical information is computed by grouping counts by their reduced integer
ratio. Dyadic partitions use
`floor(2^b R)` with `b=2 log2(lambda)+delta` and
`delta in {-4,-2,0,2}`, clipped to nonnegative integer resolution.

For a three-category check, use `p=(.2,.3,.5)` and
`lambda in {16,32,64,128}`. Whiten the two contrast coordinates of the mean,
use `(1,||X||^2)`, and compare its degree lower bound with `1/lambda`, the
prediction `(K-1)/(2 lambda)`.

## Validity and interpretation

Enumerate a rectangular Poisson population with each marginal omitted upper
tail at most `1e-14/K`; normalize and recenter scores on the retained law.
Require raw normalized degree information within `1e-9` of one, all lower
bounds within the canonical information up to `1e-9`, and agreement of the
quadratic bound with an independent one-dimensional Poisson/binomial-moment
calculation within `1e-10` for the binary cells. Rerun selected endpoint cells
at tail `1e-16` and require differences below `1e-8`.

The quadratic calculation tests a derived asymptotic prediction; it does not
upper-bound all smooth readouts. Partition results are a descriptive test of a
resolution conjecture and do not establish a sharp universal threshold,
finite-sample learnability, or a neural-network complexity lower bound.

Resources: one local CPU process, expected below 10 minutes and 4 GiB RAM. No
external data, credentials, network, or accelerator.
