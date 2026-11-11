# Prior-predictive epidemic dynamics

The compartment structure follows the public
[Stan boarding-school case study](https://mc-stan.org/learn-stan/case-studies/boarding_school_case_study.html).
The example uses an explicit synthetic initial state and no observed series.
The model was written for this documentation. Its positive rates have stated
lognormal priors; the build draws one scenario with seed 1, solves the SIR
system and checks conservation, nonnegativity and susceptible monotonicity.

Run `julia --project=docs docs/make.jl` to reproduce the calculation and figure.
No fit runs and no draws or figures are committed. The Reproduce footer links
a build-generated receipt containing the source hashes and numerical residual.
