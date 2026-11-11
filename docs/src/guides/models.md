# Models

Define a model, bind observations and inspect the roles of its inputs and outputs.

## Definition and binding

A model declaration records probability statements and deterministic relations.
Calling it with keyword arguments supplies data. The following declaration
becomes a likelihood for the supplied observations, while `mu` remains unknown.

```@example models
using StanBlocks
const measurement = @slic begin
    mu ~ normal(0, 1)
    y ~ normal(mu, 1)
end
posterior = measurement(; y=[0.2, -0.1, 0.4])
descriptor = stan_descriptor(posterior)
show(stdout, MIME"text/plain"(), [(name=i.name, observed=i.observed) for i in descriptor.inputs if !i.derived && !i.inlined]); println()
```

Data binding is by name. Rebinding produces another model value; it does not
edit the original declaration. Supply covariates and constants through these
arguments as well. Ordinary numeric variables in the surrounding Julia module
are not implicit model inputs.

## Probability statements

The left side of `~` is an observation when it has bound data and a latent
quantity otherwise. A deterministic assignment introduces a named value, not
another sampling step. The descriptor reports the resulting output roles.

```@example models
derived = Base.merge(measurement, quote
    centered = mu - 0.5
end)(; y=[0.2, -0.1, 0.4])
show(stdout, MIME"text/plain"(), [(name=o.name, block=o.kind, meaning=o.generative) for o in stan_descriptor(derived).outputs]); println()
```

A prior whose variable cannot affect any observation is generated from its
random-number generator rather than retained as an inference parameter.
[Prediction](prediction.md) explains this behavior when observations are
omitted or marked for replay.

The top-level body accepts probability statements, whole-value assignments,
submodel calls and a return value. Independent and sequential loops use
[`@plate` and `@scan`](arrays.md). Put other conditionals, loops and local
mutation inside a [Stan function](functions.md).

## Constraints

Distribution support and explicit bounds determine parameter constraints.
A bound on a probability statement also truncates that probability density;
it is not merely an instruction to a numerical optimizer. Here the scale has
a half-normal prior, and the evaluated lower bound is visible in the descriptor.

```@example models
scaled = @slic (; y=[0.2, -0.1, 0.4]) begin
    mu ~ normal(0, 1)
    sigma ~ normal(0, 1; lower=0)
    y ~ normal(mu, sigma)
end
sigma_output = only(o for o in stan_descriptor(scaled).outputs if o.name == :sigma)
show(stdout, MIME"text/plain"(), (name=sigma_output.name, type=sigma_output.type, bounds=sigma_output.constraint_values)); println()
```

Use `restricted` only when the intended model needs a bounded kernel without
its truncation normalizer. That distinction matters when a bound or the mass
inside it depends on another unknown quantity. A parameter with `flat()` or a
bare typed declaration has no proper prior; it needs an identifiable likelihood
and an explicit modeling justification.

Annotations such as `beta::vector[k]` or `L::cholesky_factor_corr[k]` specify a
Stan carrier when inference from the probability statement is insufficient.
They belong to the model language; they do not require Julia types with those
names to exist. [Arrays and loops](arrays.md) shows executable container forms.

## Inspection

`stan_code` returns the complete program the compiler emits. Inspect it before
choosing an inference method. `stanc_check` checks Stan's types and syntax;
it does not prove that a distribution has the intended support or that a fit
is informative.

```@example models
code = stan_code(scaled)
check = stanc_check(code)
@assert check.ok check.output
println(code)
```

The [golf example](../examples/golf.md) checks a bounded scale parameter's
Stan density against an independently evaluated likelihood before interpreting
its posterior.
