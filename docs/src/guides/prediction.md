# Prediction

Distinguish a fitted model, a prior simulation and prediction using an existing
fit. The distinction follows from which observations and inputs are supplied.

## Conditioning

Binding observations keeps their density contributions. The descriptor lists
the operations available for that particular bound model.

```@example prediction
using StanBlocks
const measurement = @slic begin
    mu ~ normal(0, 1)
    y ~ normal(mu, 1; n=n)
end
observed = measurement(; n=3, y=[0.2, -0.1, 0.4])
show(stdout, MIME"text/plain"(), [o.name for o in stan_descriptor(observed).operations]); println()
```

Predictive and pointwise-likelihood outputs are tagged in the descriptor.
Read those tags instead of relying on a suffix convention, since an authored
variable can use the same suffix.

```@example prediction
show(stdout, MIME"text/plain"(), [(name=o.name, role=o.generative, observation=o.source) for o in stan_descriptor(observed).outputs]); println()
```

## Prior prediction

Omitting the response removes the likelihood. Parameters that no longer affect
an observation are drawn from their priors in dependency order, together with
the unobserved response. This is prior simulation, not posterior prediction.

```@example prediction
prior = measurement(; n=3)
println(stan_code(prior))
```

The generated program has no inference target. A proper prior and the required
random-number generators are necessary for this operation. A flat prior is not
a generative distribution, even when a posterior using it could be proper.

## Posterior prediction

For posterior prediction, retain a fitted problem and its unconstrained draws.
`stan_execute(descriptor, :predict; problem, draws, seed)` evaluates the model's
predictive outputs for those draws. The execution guide shows a complete call
with an explicitly supplied parameter point; a point is not itself a posterior
sample. A scientific result must say where its draws or other posterior
representation came from.

Changing inputs may require a new compiled problem when the program changes.
Use compatible parameter names and shapes when replaying draws, and preserve
population-level uncertainty instead of substituting point estimates unless
that conditional analysis is intentional.

## Held-out inputs

`StanBlocks.stan.maybecv` marks an entire input for dependency-based replay.
The mark propagates to quantities that depend on it. It does not construct
row-level folds or retain an unmarked subset of an input automatically.

```@example prediction
const grouped = @slic begin
    mu ~ normal(0, 1)
    tau ~ normal(0, 1; lower=0)
    J = maximum(group)
    alpha ~ normal(mu, tau; n=J)
    y ~ normal(alpha[group], 1)
end
marked = grouped(; y=[0.2, -0.1, 0.4],
    group=StanBlocks.stan.maybecv(:group, [1, 1, 2]))
show(stdout, MIME"text/plain"(), [(name=i.name, held_out=i.held_out) for i in stan_descriptor(marked).inputs if !i.derived && !i.inlined]); println()
```

Marking all group labels removes every response likelihood reached through
those labels. It does not fit the population distribution to “the remaining
groups.” To assess a new group, fit the training subset first, then use that
fit's population draws to generate the new group's effect and observations.
The replay program's lack of a likelihood is expected in that second stage.

This mechanism is useful for separating training and predictive calculations,
but does not choose the statistical validation design. Define the sampling
unit, training split and scoring rule before using it. The
[partial-pooling example](../examples/pooling.md) states explicitly which
population quantities are fixed and which uncertainty its result includes.
