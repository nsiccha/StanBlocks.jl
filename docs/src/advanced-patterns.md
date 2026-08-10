# Advanced composition patterns

The original Quarto notebooks explored model families rather than isolated
models. Their lasting contribution is the architecture: function-valued
specialization, reusable sub-models, post-hoc variants, and explicit custom
transform responsibilities. The current syntax and limits are summarized here;
the [Feature atlas](feature-atlas.md) remains the exhaustive language reference.

## Link-specialized model families

The PCR sensitivity models from
[`pcr-sensitivity-vs-time`](https://github.com/bob-carpenter/pcr-sensitivity-vs-time)
form a two-by-five family: two link functions crossed with heterogeneous,
first-order random-walk, second-order random-walk, regression, and mixture
latent curves.

Function-valued arguments let one model definition specialize at trace time:

```julia
import StanBlocks.stan: logit

@deffun begin
    response_probability(::typeof(logit), eta::vector[n])::vector[n] = inv_logit(eta)
    response_probability(::typeof(log), eta::vector[n])::vector[n]   = exp(eta)

    upper_intercept(::typeof(logit))::real = positive_infinity()
    upper_intercept(::typeof(log))::real   = 0.0
end

regression_curve = @slic begin
    alpha ~ normal(0, 0.5; upper=upper_intercept(link_f))
    beta  ~ normal(0, 0.5; upper=0)
    return alpha + beta * time
end

linked_regression = @slic begin
    eta ~ regression_curve(; link_f, time)
    probability = response_probability(link_f, eta)
    detected ~ bernoulli(probability)
end

logit_model = linked_regression(; link_f=logit, time, detected)
log_model   = linked_regression(; link_f=log,   time, detected)
```

The function object is a compile-time token: StanBlocks dispatches to the
matching method and emits an ordinary specialized Stan function. It never
passes a runtime Julia function into Stan. Variadic helpers can factor the
pointwise likelihood and predictive-draw companions shared by the family; see
[variadic and function-typed dispatch](feature-atlas.md#Variadic-and-function-typed-dispatch).

## Crowdsourcing variants with `Base.merge`

The models in
[`crowdsource-computo-bayes`](https://github.com/seongwoohan/crowdsource-computo-bayes)
compare a full item/rater model with many restrictions: fixed item effects,
shared rater accuracy, and scalar rather than rater-specific parameters. Define
the full model once, name each replaceable component, and create variants by
splicing assignments or sampling statements:

```julia
full = @slic begin
    prevalence ~ beta(2, 2)
    sensitivity ~ normal(1, 2; n=n_raters)
    specificity ~ normal(2, 2; n=n_raters)
    difficulty  ~ normal(0, 1; n=n_items)
    discrimination ~ lognormal(0, 0.25; n=n_items)
    slip ~ beta(2, 2; n=n_items)

    rating ~ crowd_rating(
        item, rater, prevalence,
        sensitivity, specificity,
        difficulty, discrimination, slip,
    )
end

no_slip = Base.merge(full, quote
    slip = rep_vector(0.0, n_items)
end)

no_slip_unit_discrimination = Base.merge(no_slip, quote
    discrimination = rep_vector(1.0, n_items)
end)

shared_accuracy = Base.merge(no_slip_unit_discrimination, quote
    accuracy ~ normal(1, 2; lower=0)
    sensitivity = rep_vector(accuracy, n_raters)
    specificity = rep_vector(accuracy, n_raters)
end)
```

`Base.merge` returns a new model and leaves its input unchanged. A statement
with an existing bare LHS name replaces that component; a new name appends.
Chaining named quote blocks is covered by the package's executable
[`crowdsource` regression test](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/test/items.jl).
The `crowd_rating_lpmf`/`crowd_rating_lpmfs`/`crowd_rating_rng` family is a
separate `@deffun` concern, keeping deterministic loops out of `@slic`.

## Custom constraints and transforms

Prefer Stan's native constrained types when they express the parameter:

```julia
simplex_model = @slic begin
    weights::simplex[K] ~ dirichlet(alpha)
    outcome ~ normal(dot_product(weights, component_mean), sigma)
end
```

This gives Stan ownership of the unconstrained dimension, transform, and
Jacobian. It also works for `ordered`, `positive_ordered`, covariance/correlation
matrices, and Cholesky-factor types; see the
[ragged/native constraint table](authoring.md#Ragged-data-and-container-views).

A genuinely custom transform has three responsibilities that must agree:

1. determine the dimension and support of the free parameter;
2. map the free value to the constrained value used by the model;
3. add the exact change-of-variables adjustment, together with any prior on the
   constrained value.

The legacy disk notebook usefully illustrated the separation by returning a
named tuple from a constraining helper:

```julia
@deffun disk_from_free(z::vector[n]) = begin
    radius = inv_logit(z[1])
    angle = 6.283185307179586 * inv_logit(z[2])
    return (;
        radius,
        angle,
        x=radius * cos(angle),
        y=radius * sin(angle),
    )
end
```

This helper is intended for a two-element `z`; the symbolic `n` keeps the input
shape available to the generated function signature.

That helper alone is **not** a uniform-disk distribution: a correct registered
`*_lpdf` must include the transform's Jacobian. The original notebook explicitly
left that derivation as a placeholder, so it is not reproduced as runnable
sampling code here.

The companion simplex exploration implemented ALR, ILR, expanded-softmax, and
several stick-breaking transformations based on
[`bob-carpenter/transforms`](https://github.com/bob-carpenter/transforms). Those
are valuable research references, but a custom implementation should be
validated against the native `simplex` path before use. Register a complete
density family through [`@lpxf` and `@lhs`](index.md#@lpxf-and-@lhs-—-opt-in-dispatch-hooks),
then check both `stanc` and BridgeStan log-density/gradient behavior.

## PosteriorDB implementations

The old `implementations.qmd` page only embedded the package extension. The
maintained source is
[`ext/PosteriorDBExt.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/ext/PosteriorDBExt.jl),
which contains the actual model catalog and is exercised by the test suite.
With `PosteriorDB.jl` loaded, an implemented posterior can be translated through
the extension:

```julia
using StanBlocks, PosteriorDB

db = PosteriorDB.database()
posterior = PosteriorDB.posterior(db, posterior_name)
model = StanBlocks.slic_implementation(posterior)

println(stan_code(model))
stanc_check(model)
```

The catalog includes regression, hierarchical radon, time-series, mixture, and
other reference posteriors. Keeping these definitions in the extension—rather
than copying them into documentation—means the examples compiled in tests are
the examples readers inspect.

## Choosing the maintained source

| Question | Maintained source |
|---|---|
| What syntax is supported now? | [Home guide](index.md) and [Feature atlas](feature-atlas.md) |
| How do applied models fit together? | [Worked model patterns](worked-examples.md) |
| Which small snippets compile? | [Sandbox Gallery](gallery.md) |
| Which reference posterior definitions are tested? | [`PosteriorDBExt.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/ext/PosteriorDBExt.jl) |
| Where is the large real-world consumer? | [BayesianRegressionModels catalog](https://nsiccha.github.io/BayesianRegressionModels.jl/) |
