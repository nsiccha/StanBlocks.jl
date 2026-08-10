# Worked model patterns

This page collects the durable modeling ideas from the original StanBlocks
case-study notebooks. The examples are deliberately small: each one highlights
how to structure a model, while [`stan_code`](index.md#Inspection-and-compilation)
remains the source of the generated Stan program.

For a larger executable catalog, see the [Sandbox Gallery](gallery.md), the
[PosteriorDB implementations](advanced-patterns.md#PosteriorDB-implementations),
and the [BayesianRegressionModels catalog](https://nsiccha.github.io/BayesianRegressionModels.jl/).

## Hierarchical regression: radon

The progression from complete pooling to county-specific intercepts is a compact
example of changing the statistical structure without changing the observation
model. This partially pooled form corresponds to the central model in the
[Stan radon case study](https://mc-stan.org/learn-stan/case-studies/radon_cmdstanpy_plotnine.html):

```julia
radon_partial_pooling = @slic begin
    mu_alpha    ~ normal(0, 10)
    sigma_alpha ~ std_normal(; lower=0)
    alpha       ~ normal(mu_alpha, sigma_alpha; n=n_counties)
    beta        ~ normal(0, 10)
    sigma_y     ~ std_normal(; lower=0)

    log_radon ~ normal(
        alpha[county_idx] + beta * to_vector(floor_measure),
        sigma_y,
    )
end

posterior = radon_partial_pooling(;
    log_radon,
    floor_measure,
    county_idx,
    n_counties,
)
```

Complete pooling replaces `alpha[county_idx]` with one scalar `alpha`. No
pooling keeps the vector but gives its entries independent broad priors. The
package's tested [`PosteriorDBExt.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/ext/PosteriorDBExt.jl)
contains those variants plus centered and non-centered hierarchical forms.

## Composing a mechanistic golf model

The [Stan golf case study](https://mc-stan.org/learn-stan/case-studies/golf.html)
starts with logistic regression and then models the geometry of a successful
putt. In StanBlocks, the angle and distance mechanisms can be independent
sub-models:

```julia
angle_probability = @slic begin
    threshold_angle = asin((cup_radius - ball_radius) ./ distance)
    sigma_angle ~ std_normal(; lower=0)
    return 2 * Phi(threshold_angle / sigma_angle) - 1
end

distance_probability = @slic begin
    sigma_distance ~ std_normal(; lower=0)
    return (
        Phi((tolerance - overshoot) ./ ((distance + overshoot) * sigma_distance))
        - Phi(-overshoot ./ ((distance + overshoot) * sigma_distance))
    )
end

golf = @slic begin
    p_angle    ~ angle_probability(; cup_radius, ball_radius, distance)
    p_distance ~ distance_probability(; tolerance, overshoot, distance)
    made ~ binomial(attempts, p_angle .* p_distance)
end
```

The sub-model parameters are namespaced (`p_angle_sigma_angle` and
`p_distance_sigma_distance` in the emitted program), so the pieces can be
reused or replaced independently. A logistic baseline is only three lines:

```julia
golf_logistic = @slic begin
    intercept
    slope
    made ~ binomial_logit(attempts, intercept + slope * distance)
end
```

## Reusing a basis-function Gaussian process

The [motorcycle case study](https://users.aalto.fi/~ave/casestudies/Motorcycle/motorcycle.html#GP_model_with_Hilbert_basis_functions)
uses a Hilbert-space Gaussian-process approximation. Precompute the basis and
frequency terms in Julia, then keep the probabilistic basis weights in a
sub-model:

```julia
basis_gp = @slic begin
    lengthscale ~ lognormal(0, 1)
    marginal_sd ~ lognormal(0, 1)
    weight      ~ std_normal(; n=n_basis)

    spectral_scale = marginal_sd * exp(
        -0.25 * square(lengthscale) * frequency_sq
    )
    return basis * (spectral_scale .* weight)
end

motorcycle_heteroskedastic = @slic begin
    mean_curve      ~ basis_gp(; basis, frequency_sq, n_basis)
    log_scale_curve ~ basis_gp(; basis, frequency_sq, n_basis)
    acceleration ~ normal(mean_curve, exp(log_scale_curve))
end
```

The same component supplies two independently parameterized curves. A
homoskedastic model replaces `log_scale_curve` with one scalar scale parameter.

## Scientific case-study map

Several original notebooks were useful as a map of what a blockless model DSL
should express, but were incomplete sketches rather than reproducible examples.
Their references and durable modeling lessons are retained here without
presenting placeholder code as a working model.

| Case study | StanBlocks pattern | Original source |
|---|---|---|
| Boarding-school disease transmission | ODE solution feeding a negative-binomial observation model; a reporting-rate parameter turns latent incidence into reported cases | [Stan case study](https://mc-stan.org/learn-stan/case-studies/boarding_school_case_study.html) |
| Planetary motion | ODE state assembled from inferred initial position/momentum and integrated at observation times | [Stan case study](https://mc-stan.org/learn-stan/case-studies/planetary_motion.html) |
| Multiple species-site occupancy | Nested site/species likelihood terms and posterior predictive occupancy summaries belong in `@deffun` helpers and generated quantities | [Stan case study](https://mc-stan.org/learn-stan/case-studies/dorazio-royle-occupancy.html) |
| Soil carbon | A latent ODE trajectory plus measurement-error layer; the richer variant adds latent true enrichment before the observed mean | [Stan case study](https://mc-stan.org/learn-stan/case-studies/soil-knit.html) |

Use the current `ode_rk45`/`ode_rk45_tol` family rather than the legacy
`integrate_ode_rk45` spelling. Deterministic recurrence and likelihood helpers
belong in `@deffun`; the flat `@slic` body then wires their outputs to priors and
observations. The [Feature atlas](feature-atlas.md#Closures) shows the current
closure-based ODE form, including captured parameters.

## Inspecting and checking an example

The old Quarto notebooks printed every generated program into tabsets. The
maintained workflow is more direct:

```julia
model = radon_partial_pooling(;
    log_radon,
    floor_measure,
    county_idx,
    n_counties,
)

println(stan_code(model))
transpiles(model)       # tracing/code-generation check
stanc_check(model)      # Stan parser and semantic check
```

For a semantic check, instantiate through BridgeStan and compare the log density
and gradient at representative unconstrained points. See the
[end-to-end workflow](index.md#Quick-Start) and the
[current checking boundary](authoring.md#Current-high-level-limits).
