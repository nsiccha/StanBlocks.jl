# Conditional school effects

The input is the eight-schools example from Rubin (1981), reproduced in the
[Stan eight-schools example](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started#example-1-eight-schools).
The documentation deliberately conditions on a stated population mean and
standard deviation. It does not present that conditional calculation as a fit
of the population distribution. Normal conjugacy gives all shown moments and
intervals during the docs build; no fit cache or sampler is required.

Run `julia --project=docs docs/make.jl` to execute `model.jl`, verify the density
and regenerate the figure. The page's Reproduce footer prints the source hash
and the numerical-check receipt from that build.
