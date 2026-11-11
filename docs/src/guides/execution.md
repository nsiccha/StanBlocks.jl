# Execution

Compile a bound model, evaluate its density and execute descriptor operations
through BridgeStan.

## Compilation

`stan_instantiate` returns a problem implementing `LogDensityProblems`. Its
coordinate vector is unconstrained; bounded or structured parameters are
transformed by Stan. Use the reported dimension and do not substitute a
vector of constrained parameters.

```@example execution
using StanBlocks, LogDensityProblems
model = @slic (; y=[0.2, -0.1, 0.4]) begin
    mu ~ normal(0, 1)
    y ~ normal(mu, 1)
end
problem = stan_instantiate(model; nan_on_error=false)
theta = zeros(LogDensityProblems.dimension(problem))
value, gradient = LogDensityProblems.logdensity_and_gradient(problem, theta)
@assert isfinite(value) && all(isfinite, gradient)
show(stdout, MIME"text/plain"(), (dimension=length(theta), log_density=value, gradient=gradient)); println()
```

This call evaluates one parameter point. It does not sample a posterior.
Connect the problem to an inference method that accepts this interface, then
check that method's diagnostics before using its draws.

Native compilation needs a C++ toolchain and BridgeStan. The first compilation
can take longer than subsequent calls because generated libraries are cached
by their source and build inputs. Build errors propagate; use
`nan_on_error=false` when diagnosing invalid density evaluations.

## Operations

The descriptor is generated from the traced program. Its operation entries
identify the required inputs and outputs; an operation named `fit` constructs
the density problem and does not run a sampler.

```@example execution
d = stan_descriptor(model; name=:measurement)
show(stdout, MIME"text/plain"(), [(name=op.name, inputs=op.inputs, outputs=op.outputs) for op in d.operations]); println()
```

At the parameter point above, a predictive draw is a conditional simulation.
The pointwise log-likelihood entries score each supplied observation at that
same point.

```@example execution
prediction = stan_execute(d, :predict; problem, draws=theta, seed=23)
likelihood = stan_execute(d, :pointwise_loglik; problem, draws=theta, seed=23)
show(stdout, MIME"text/plain"(), (prediction=prediction, log_likelihood=likelihood)); println()
```

A posterior predictive calculation repeats this operation across a posterior
representation. The [golf example](../examples/golf.md) instead integrates a
one-dimensional posterior deterministically and stores its checked summary.
No fit runs during the documentation build.

## Data and interfaces

`stan_data_json` writes the data the emitted program reads, including derived
sizes. `stan_interface` describes the encoding and output layout for a consumer
outside Julia. Generate both alongside the Stan source so their identities
remain consistent.

```@example execution
println(stan_data_json(model))
```

A Python application can consume the program and its data through BridgeStan.
BridgeStan is the supported runtime; CmdStan's standalone generated-quantities
workflow does not provide the same transformed-parameter replay contract.

For ragged outputs, use descriptor `segments`; for container shapes, use
`dims`; for evaluated bounds, use `constraint_values`. These fields describe
the bound model and can change when inputs are rebound. A source hash alone
does not identify the data or a posterior fit.

## Diagnostics

`StanBlocksError` records the phase of a failure. `diagnostic` extracts a
structured location and classification when available. A successful
`stanc_check` establishes acceptance of generated Stan syntax and types; a
finite density check additionally establishes evaluation at the checked point.
Neither result replaces inference diagnostics or predictive assessment.

The [golf example](../examples/golf.md) records both numerical agreement and
grid refinement, and shows the predictive comparison those checks support.
