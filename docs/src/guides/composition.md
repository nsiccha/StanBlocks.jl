# Composition

Reuse a probabilistic component inside another model and change one assumption
without maintaining another complete program.

## Submodels

An anonymous `@slic` block receives inputs by keyword and can return a value.
At a call site, `~` includes the component's parameter and density statements;
the returned value becomes the caller's left-hand side.

```@example composition
using StanBlocks
const standardized_effect = @slic begin
    z ~ normal(0, 1)
    return location + scale * z
end
const regression = @slic begin
    slope ~ standardized_effect(; location=0.0, scale=1.0)
    y ~ normal(slope * x, 1)
end
model = regression(; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.1, 1.2])
show(stdout, MIME"text/plain"(), [(name=o.name, block=o.kind) for o in stan_descriptor(model).outputs]); println()
```

The returned slope is a deterministic transformation of the component's
standard-normal parameter. Its internal parameter receives a namespace from
the receiving name, so using the component twice introduces two sets of latent
quantities. Shared inputs are passed explicitly at both call sites.

A named submodel can instead declare positional inputs. Its body still defines
a probabilistic component, rather than a deterministic Stan function.

```@example composition
@slic offset_effect(location::real) = begin
    deviation ~ normal(0, 1)
    return location + deviation
end
positional = @slic (; y=[0.2, -0.1]) begin
    mu ~ offset_effect(0.5)
    y ~ normal(mu, 1)
end
println(stan_code(positional))
```

Use a module-level `const` binding for a reusable component that should retain
its identity. Definitions in another module retain their defining scope; import
the component into the caller and supply its inputs through the call.

## Replacement

`Base.merge` with a quoted statement replaces the declaration with the same
left-hand-side name. Here the observation family changes while the slope's
component and prior remain shared.

```@example composition
robust_regression = Base.merge(regression, quote
    y ~ student_t(4, slope * x, 1)
end)
robust = robust_regression(; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.1, 1.2])
println(stan_code(robust))
```

This is a structural change to the model. It differs from binding observations,
which retains the probability statement and uses the supplied value in its
density. Replacing a latent declaration with a fixed value instead removes
that declaration from the model.

```@example composition
fixed = Base.merge(regression, (; slope=0.8))(; x=[-1.0, 0.0, 1.0], y=[-0.8, 0.1, 1.2])
show(stdout, MIME"text/plain"(), [o.name for o in stan_descriptor(fixed).outputs if o.kind == :parameter]); println()
```

The empty parameter list is the intended result: the fixed slope no longer
introduces its component's standard-normal parameter. A fixed value remains
an input to the observation density.

Composition does not choose a parameterization or establish identifiability.
Compare the statistical assumptions and their predictive consequences, rather
than treating two generated programs as evidence of equivalent inference.
[Partial pooling](../examples/pooling.md) uses one school component across
independent cells and verifies the resulting joint density.
