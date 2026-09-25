# Unified precision and decoder diagnostic

Direct user-authorized local follow-up, designed after the previous manuscript
results. This is an exploratory diagnostic, not a preregistered confirmation or
a Radia-controller job. Preserve all declared cells and training seeds.

Question: on the same observation law, can canonical information separate loss
caused by precision from the failure of a fitted smooth score decoder? Does
retaining the entire rounded proportion vector change finite-precision results?

Primary model: independent Poisson counts with lambda=256, p=(.35,.65).
Observe scalar R=N1/N, with a distinguished empty symbol. Precision channels
return floor(2^b R)/2^b for b=4,6,8,10,12,16,24, plus exact R. The value R=1
has its own endpoint bin. All channels are frozen when differentiating the law.
Compute canonical normalized Fisher matrices by conditional-score grouping.
For each output, compute population histogram score projections at
2,4,6,8,10,12,16,24 bits, with a separate empty bin. These are nested measurable
coarsenings of the observed output, not additional information from counts.

Fit three independent 64-64 SiLU score regressors per channel (seeds 17,29,43).
Each seed shares independent 24,000/6,000 train/validation count draws across
channels. Inputs are the scalar output centered at p and scaled by the nominal
sampling standard deviation, plus an empty indicator. Targets are the two raw
Fisher-normalized scores. Adam, learning rate .001, batches of 1024, 800 updates,
validation every 50 updates; select minimum validation MSE. No hyperparameter
search or target selection. Report all seeds and median/range across seeds.
Evaluate the variational score-projection objective 2 E[m Z^T]-E[m m^T]
(symmetrized for a matrix) by population enumeration, independently of training.
This is an achieved lower bound and may be negative; do not clip or treat it as
an upper bound. Ranges describe training variability, not confidence coverage.

Arithmetic sensitivity: lambda=64,256,1024 with K=2 and p1=.1,.35,.5;
lambda=16,64 with K=3 and p=(.2,.3,.5) or (.05,.15,.8).
For float32, float16, and emulated bfloat16, compare output-only rounding with
casting each count and the total before division and rounding the result.
For every format/pipeline compare the first K-1 coordinates to all K rounded
coordinates; compute exact-rational full information as a reference. bfloat16
uses round-to-nearest ties-to-even through float32, matching the earlier emulator.
These are declared arithmetic channels, not claims about arbitrary hardware.

Validity: Poisson rectangular truncation with each marginal upper tail at most
1e-14/K, positive mass grouping, raw whitened Fisher residual <1e-9, score mean
<1e-10, PSD contraction and nested-partition inequalities to 1e-8. Check format
rounding against Torch; finite-difference grouped likelihood scores at selected
cells; cross-check K=2 arithmetic results against the existing figure values.
Rerun primary and selected sensitivity canonical calculations with tail 1e-16
and require efficiency differences <1e-8. Hold grouping channels fixed under
finite differences. Independently check partition information by the missing-
score-variance identity and exact-state probe MSE. Training is only a diagnostic;
a failure to exhibit a probe gap is reported, not repaired by selecting new seeds.

Interpretation: precision erasure is indicated by contraction of canonical
information. Decoder failure is indicated by a gap below canonical information
on the same channel; recovery by finer partitions supplies a constructive
population certificate, not a claim about finite-data sample complexity.
The intended pattern is canonical degree retention below .1 at 4 bits and above
.5 at 16 bits, with an achieved smooth-probe gap above .1 at high precision.
These descriptive reference values are not statistical significance tests.
Composition retention and all adverse arithmetic controls remain reportable.

Resources: one local CPU process, four Torch threads, expected <30 minutes,
hard two-hour wall limit, <8 GiB RAM and <500 MiB output. Generated synthetic
data only; no external compute, credentials, or network during execution.
Deliver code, environment versions, config/source hashes, checkpoints, complete
CSV results, validation JSON, and a readable report.
