# StanBlocks.jl feature atlas

*StanCon 2026 companion: author source → corresponding Stan emission*

[View the StanCon 2026 slides](https://nsiccha.github.io/StanBlocks.jl/stancon-2026/)

This is the long-form companion to the talk. It inventories the **current
author-visible language surface** reviewed against StanBlocks commit `e6f607d`
(2026-08-04). The presentation files themselves are newer; the compiler source
covered here is unchanged by those documentation commits.

Every StanBlocks authoring example is followed by its corresponding Stan
emission. Small qualitative examples isolate the exact lowering under
discussion; the end-to-end comparison panels reproduce the complete
`stan_code(model)` result, including helper functions for vectorised pointwise
densities, sized RNGs, ragged carriers, lifted closures, and distribution
combinators. Descriptor and execution snippets are Julia API calls rather than
transpilation examples and are labelled separately.

This atlas covers language and workflow features, not every registered Stan
builtin. Several hundred Stan functions and distribution signatures are
registered; the source of truth for those is
[`src/slic_stan/builtin.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/src/slic_stan/builtin.jl).

## Overview

The StanCon 2026 "quality-of-life improvements" from the opening slide, in that
order — each with the one detail worth remembering and a link to its worked
example:

1. **Activity analysis** — A backward likelihood-reachability pass marks every binding that can affect an observed quantity, which is what lets the compiler choose `transformed data` vs `transformed parameters` placement and move prior-only computations into `generated quantities`. → [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints)
2. **Submodels** — Reusable model fragments (anonymous `@slic (…)` blocks and named `@slic f(…)=…` functions) compose into larger models with automatic parameter namespacing (e.g. `eta_beta`). → [Composition](#Composition-and-model-families)
3. **User-defined types** — `@usertype` lets author-defined structured types (currently NamedTuples and ragged containers) flow through tracing, data binding, and Stan emission as first-class shapes. → [Structured data](#Structured-data-and-compiler-owned-plates)
4. **Higher-order user-defined functions** — Functions (distributions included) can take and return other functions, so combinators like `weighted`/`truncated`/`censored` and custom HOFs are authored once and reused. → [Function signatures](#Function-signatures,-defaults,-keywords,-varargs,-and-higher-order-functions)
5. **Automatic but optional type & shape inference** — The compiler infers scalar types, array shapes, and constraints from data and usage, but you can always pin them explicitly (typed LHS, sized signatures). → [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints)
6. **(Ragged) plates** — `plate(…) do` is a compiler-owned sampling loop that promotes fresh per-cell variables to outer storage and supports ragged (uneven-length) cells without hand-written index bookkeeping. → [Structured data](#Structured-data-and-compiler-owned-plates)
7. **Metaprogramming** — Caller-side macros expand before tracing, and `@inline` helpers, trailing-`!` mutation, `@stan_assert`, and `return_type_of` queries let you compute model structure at transpile time. → [Expansion tools](#Expansion,-inlining,-assertions,-and-return-type-queries)
8. **Julia-style multiple dispatch** — Named submodels and helpers dispatch on positional argument types (variadic and function-typed included), so one name resolves to different fragments by the shapes it is called with. → [Typed dispatch](#Named-submodels:-typed-positional-dispatch)
9. **…and more** — Generated observation outputs (pointwise log-likelihood, predictive draws, missing-outcome imputation), custom distribution triads, fused GLMs, post-hoc `Base.merge` variants and cross-validation taint, executable descriptors, and the scientific-computing surface (ODEs, Torsten, GP, `reduce_sum`) — plus the honest current boundaries, in the sections below the capability walkthrough.

## Feature map

| Area | Current surface | Detailed MWE |
|---|---|---|
| Model declarations | `@slic`, `~`, `=`, typed LHS, bare flat-prior parameters | [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints) |
| Analysis | block placement, type/shape/constraint inference, likelihood activity | [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints) |
| Composition | anonymous and named submodels, typed dispatch, `Base.merge`, data rebinding | [Composition](#Composition-and-model-families) |
| Structured data/models | `RaggedVector`, ragged constraints, `EachCol`/`EachRow`, fancy indexing, `plate` | [Structured models](#Structured-data-and-compiler-owned-plates) |
| Julia-like functions | closures, HOFs, varargs, positional defaults, required/optional kwargs | [Function signatures](#Function-signatures,-defaults,-keywords,-varargs,-and-higher-order-functions) |
| Distribution abstraction | custom `_lpdf`/`_lpdfs`/`_rng`, `@lpxf`, `@lhs`, weighted/truncated/censored/interval | [Distributions](#Custom-and-higher-order-distributions) |
| Metaprogramming | caller macros, `@inline`, trailing `!`, `@stan_assert`, `return_type_of` | [Expansion tools](#Expansion,-inlining,-assertions,-and-return-type-queries) |
| Deterministic helpers | `@deffun`, loops, branches, mutation, comprehensions, iteration | [`@deffun`](#Deterministic-programs-with-@deffun) |
| Observation workflow | pointwise log likelihoods, predictive draws, missing-outcome imputation | [Generated outputs](#Generated-observation-outputs) |
| Model variants | post-hoc overrides and lower-level cross-validation taint | [Variants](#Post-hoc-variants-and-cross-validation-taint) |
| Reflection/execution | descriptors, definition closure, derived operations, BridgeStan | [Descriptor](#Executable-model-descriptors) |
| Scientific models | ODE solvers, Torsten signatures, GP covariance, `reduce_sum`, fused GLMs | [Scientific surface](#Scientific-computing-surface) |

## One model: block placement, types, shapes, and constraints

### StanBlocks source

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
using StanBlocks

m = @slic (; x = [-1.0, 0.0, 1.0], y = [0.2, -0.1, 0.7]) begin
    mx = mean(x)

    alpha ~ normal(0, 2)
    beta  ~ normal(0, 1)
    sigma ~ exponential(1)

    mu = alpha + beta * (x - mx)
    y ~ normal(mu, sigma)
end
""", :m)
```

```@raw html
</div>
```

What happened:

- Julia vectors `x` and `y` established Stan vector types and generated their
  data dimensions `x_n` and `y_n`.
- `mx` depends only on data, so it moved to `transformed data`.
- The prior family inferred `sigma`'s lower bound.
- `mu` depends on parameters and feeds the likelihood, so it lives in
  `transformed parameters`.
- The observed `y` generated pointwise log-likelihood and posterior-predictive
  twins automatically.

### Typed LHS and a prior-free parameter

Use explicit types when the prior call does not determine the desired container,
or when the type itself carries Stan semantics:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0.1, -0.2, 0.4]

typed_model = @slic (; X, y, k = size(X, 2)) begin
    beta::vector[k]                 # improper-flat sampler parameter
    sigma::real ~ exponential(1)
    y ~ normal(X * beta, sigma)
end
""", :typed_model)
```

```@raw html
</div>
```

A bare typed declaration is allowed only in a model/submodel and means a
parameter with no density statement. Inside `@deffun`, the same syntax declares
a local that the function must fill. Native constrained types use the same LHS
surface, for example
`L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)` when `L` participates in a
downstream likelihood.

## Composition and model families

### Anonymous submodels: inputs by name

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0.1, -0.2, 0.4]

population_effect = @slic begin
    k = dims(X)[2]
    beta ~ normal(0, 1; n = k)
    return X * beta
end

m = @slic (; X, y) begin
    eta ~ population_effect(; X)
    y ~ normal(eta, 1)
end
""", :m)
```

```@raw html
</div>
```

Submodel parameters are namespaced under the receiving LHS. Anonymous
submodels receive inputs through kwargs or enclosing scope; they deliberately
have no positional calling form.

### Named submodels: typed positional dispatch

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
x = [-1.0, 0.0, 1.0]
y = [0.1, -0.2, 0.4]

@slic random_slope(x::vector[n]) = begin
    beta ~ normal(0, 1)
    return beta * x
end

@slic offset(x::vector[n]) = begin
    alpha ~ normal(0, 2)
    return alpha + x
end

m = @slic (; x, y) begin
    eta1 ~ random_slope(x)
    eta2 ~ offset(eta1)
    y ~ normal(eta2, 1)
end
""", :m)
```

```@raw html
</div>
```

Typed arguments select Julia methods on the traced Stan center type. Multiple
definitions of `@slic f(::real)` and `@slic f(::int)` are distinct methods;
untyped positional arguments match any traced value.

### Post-hoc variants with `Base.merge`

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
base = @slic begin
    beta ~ normal(0, 1)
    y ~ normal(beta * x, 1)
end

wide = Base.merge(base, quote
    beta ~ normal(0, 5)
end)

wide_model = wide(; x=[-1.0, 0.0, 1.0], y=[0.1, -0.2, 0.4])

fixed_model = Base.merge(base, (; beta=0.25))(
    ; x=[-1.0, 0.0, 1.0], y=[0.1, -0.2, 0.4],
)
""", :wide_model)
```

```@raw html
</div>
```

The merged statement matches by the bare LHS name and replaces the original.
Typed and untyped versions of the same LHS still match. Additional names append
new statements. A merged `NamedTuple`, such as `(; beta=0.25)`, instead removes
the matching statement and stores the supplied value as data: this fixes the
name without retaining its prior as a likelihood. Statement replacements and
fixed bindings may be combined in one `Base.merge` call. Ordinary model kwargs
only bind data and do not remove statements. A merged `SlicModel` value can also
be interpolated into the callee position of generated AST.

## Structured data and compiler-owned plates

### `plate`: an independent-cell loop that may introduce parameters

StanBlocks source:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
plate_model = @slic (; y = randn(4), mu0 = 0.5) begin
    sigma ~ exponential(1)

    theta ~ plate(y; outer = 4) do yi
        t ~ normal(mu0, 1)
        yi ~ normal(t, sigma)
        t
    end
end
""", :plate_model)
```

```@raw html
</div>
```

Positional inputs are sliced; lexical captures are shared; fresh cell bindings
are promoted to outer storage; the trailing expression becomes the cell output.
Fixed vector cells become columns of a matrix. Additional outer axes add Stan
array prefixes. Cells must remain independent: `plate` is not a scan.

### Vector-valued plate cells

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
vector_plate_model = @slic (; n_groups = 8, k = 3) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)
    tau::vector[k] ~ normal(0, 1; lower=0)

    b::vector[k] ~ plate(; outer=n_groups) do g
        z::vector[k] ~ std_normal()
        diag_pre_multiply(tau, L) * z
    end
end
""", :vector_plate_model)
```

```@raw html
</div>
```

Both `z` and `b` have logical shape `matrix[k,n_groups]`; the per-group vectors
occupy columns.

### Ragged data

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
groups = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]

ragged_model = @slic (; groups, g=2, y=0.0) begin
    first = groups[1]
    selected = groups[g]
    ng = length(groups)
    y ~ normal(sum(first) + sum(selected) + ng, 1)
end
""", :ragged_model)
```

```@raw html
</div>
```

The nested Julia vectors become a nominal carrier with flat `mem` and inclusive
group `ends`. `groups[g]` lowers to a Stan helper that reconstructs the requested
slice; `length(groups)` counts groups, not scalar elements.

### Ragged constrained parameters

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
ragged_constraint_model = @slic (; Ks=[2, 3, 4], y=0.3) begin
    p::simplex[Ks] ~ flat()
    y ~ normal(sum(p[1]), 0.1)
end
""", :ragged_constraint_model)
```

```@raw html
</div>
```

Stan cannot declare a runtime-ragged pile of simplices. StanBlocks emits a flat
unconstrained parameter, then a compiler-owned per-group constrain/Jacobian loop
and a logical ragged view. The same route covers `ordered`, `positive_ordered`,
`cholesky_factor_corr`, and square `cholesky_factor_cov` groups. This feature
requires BridgeStan 2.9 / Stan 2.39 or newer.

### Dense views and indexing

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun col_sums(X::matrix[m,k]) = [sum(c) for c in EachCol(X)]

views_model = @slic (; X=[1.0 2.0; 3.0 4.0], y=[0.0, 0.0]) begin
    sums = col_sums(X)
    y ~ normal(sums, 1)
end
""", :views_model)
```

```@raw html
</div>
```

| Source | Stan lowering |
|---|---|
| `EachCol(X)[j]` | `col(X,j)` |
| `EachRow(X)[i]` | `row(X,i)` |
| `length(EachCol(X))` | `cols(X)` |
| `length(EachRow(X))` | `rows(X)` |
| `alpha[county_idx]` | vectorised integer-array indexing |
| `X[i,:,:]`, `X[:,:,k]`, `L[i,:,:]` | resolved multi-colon slice |

## Function signatures, defaults, keywords, varargs, and higher-order functions

### Positional defaults and keyword arguments

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun affine(x::real, a::real = 2.0)::real = a * x + 1.0
@deffun scale(x::vector[n]; factor, bias)::vector[n] = factor * x + bias
@deffun shift(x::vector[n]; a, b = 2.0)::vector[n] = a * x + b

signature_model = @slic (; x = [1.0, 2.0], y = [0.0, 0.0]) begin
    a1 = affine(3.0)             # default a=2
    a2 = affine(3.0, 4.0)
    s = scale(x; factor=0.5, bias=0.0) # required kwargs
    z = shift(s; a=1.5)          # optional b defaults to 2
    y ~ normal(z + a1 + a2, 1)
end
""", :signature_model)
```

```@raw html
</div>
```

Defaults are resolved at the call site; Stan receives a specialised call with
ordinary positional arguments. Omitting a required keyword errors during
tracing rather than emitting an under-specified function call.

### Variadic and function-typed dispatch

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
import StanBlocks.stan: logit

@deffun begin
    my_bernoulli_lpmf(y, ::typeof(logit), eta) =
        bernoulli_logit_lpmf(y, eta)

    my_bernoulli_lpmf(y, ::typeof(log), eta) =
        bernoulli_lpmf(y, exp(eta))

    my_bernoulli_lpmfs(y::int, args...) =
        my_bernoulli_lpmf(y, args...)

    apply_twice(f, x::real)::real = f(f(x))
end

dispatch_model = @slic (; y=1, eta=0.2, obs=0.0) begin
    ll = my_bernoulli_lpmfs(y, logit, eta)
    z = apply_twice(exp, eta)
    obs ~ normal(ll + z, 1)
end
""", :dispatch_model)
```

```@raw html
</div>
```

Function values are compile-time tokens. StanBlocks selects and emits the
specialised method; no runtime function object reaches Stan.

### Closures

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
ts = [0.5, 1.0]
yobs = 0.6

closure_model = @slic (; ts, yobs) begin
    lambda ~ exponential(1)
    y = ode_rk45(
        (t, state) -> -lambda * state,
        [1.0], 0.0, to_array_1d(ts),
    )
    yobs ~ normal(y[1][1], 0.05)
end
""", :closure_model)
```

```@raw html
</div>
```

The closure is lifted into a generated Stan function. Captured data and
parameters become explicit trailing arguments, and likelihood activity follows
through those captures so `lambda` stays a live parameter.

## Custom and higher-order distributions

### Custom distribution triad

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun begin
    @lhs @lpxf robust_lpdf(y::real, mu::real, sigma::real) =
        student_t_lpdf(y, 4, mu, sigma)

    robust_lpdfs(y::real, mu::real, sigma::real)::real =
        robust_lpdf(y, mu, sigma)

    robust_rng(mu::real, sigma::real)::real =
        student_t_rng(4, mu, sigma)
end

custom_model = @slic (; y = 0.2) begin
    mu ~ normal(0, 2)
    sigma ~ exponential(1)
    y ~ robust(mu, sigma)
end
""", :custom_model)
```

```@raw html
</div>
```

| Companion | Used for |
|---|---|
| `robust_lpdf` | joint sampling statement |
| `robust_lpdfs` | pointwise generated log likelihood |
| `robust_rng` | posterior-predictive draw |
| `@lpxf` | density/pointwise/RNG name registration |
| `@lhs` | base-level LHS type inference for `y ~ robust(...)` |

This MWE deliberately declares a scalar observation. A vector observation needs
a matching `robust_lpdf(y::vector[n], ...)` and pointwise companion. A ragged
observation additionally needs the sized-token RNG form
`robust_rng(vector[n], args...)::vector[n]`.

### Distribution higher-order functions

Each combinator takes a base distribution token and specializes a complete Stan
distribution family. Keeping the examples separate makes the generated density,
pointwise companion, predictive RNG, and transitive helper closure visible for
each semantic choice. At regular width, switch between the StanBlocks and
generated-Stan tabs; use **Compare side by side** for the full-width view.

#### Weighted observations

Use a data weight to scale an observation's log-density contribution without
changing the posterior-predictive distribution.

Qualitatively:

```julia
y ~ weighted(normal, weight, mu, sigma)
```

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
using StanBlocks

weighted_model = @slic (; y=0.1, w=2.0) begin
    mu ~ normal(0.0, 1.0)
    y ~ weighted(normal, w, mu, 1.0)
end
""", :weighted_model)
```

```@raw html
</div>
```

The compiler specializes the `normal` token into `weighted_normal_lpdf`, its
pointwise `weighted_normal_lpdfs` companion, and `weighted_normal_rng`. The RNG
deliberately ignores the weight: weighting changes evidence, not the data-generating
process.

#### Truncated observations

Truncation conditions a latent draw to lie inside the supplied bounds. Its
normalizing constant belongs in the density, and prediction must draw from the
same conditioned distribution.

Qualitatively:

```julia
y ~ truncated(normal, mu, sigma; lower=lo, upper=hi)
```

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
using StanBlocks

truncated_model = @slic (; y=0.2, lo=-1.0, hi=1.0) begin
    mu ~ normal(0.0, 1.0)
    y ~ truncated(normal, mu, 1.0; lower=lo, upper=hi)
end
""", :truncated_model)
```

```@raw html
</div>
```

The density rejects invalid bounds, assigns negative infinity outside them, and
subtracts a stable two-sided normalizing term inside. The predictive companion
uses bounded rejection sampling and fails explicitly after 100,000 attempts.

#### Censored observations

Censoring records a threshold atom when a latent draw falls outside the bounds.
The density therefore uses tail mass at an endpoint and ordinary density in the
interior; prediction clamps an unconstrained draw.

Qualitatively:

```julia
y ~ censored(normal, mu, sigma; lower=lo, upper=hi)
```

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
using StanBlocks

censored_model = @slic (; y=0.3, lo=-1.0, hi=1.0) begin
    mu ~ normal(0.0, 1.0)
    y ~ censored(normal, mu, 1.0; lower=lo, upper=hi)
end
""", :censored_model)
```

```@raw html
</div>
```

The emitted family exposes both stable normal tail helpers. Lower and upper
threshold observations contribute the matching tail mass; interior observations
use `normal_lpdf`; generated draws are clamped to the observed support.

#### Interval-censored observations

An interval-censored value contributes evidence that a latent draw fell in
`(lo, hi]`. The stored scalar is only a carrier for the observation statement;
the interval endpoints determine the likelihood.

Qualitatively:

```julia
y ~ interval_censored(normal, lo, hi, mu, sigma)
```

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
using StanBlocks

interval_model = @slic (; y=0.4, lo=-1.0, hi=1.0) begin
    mu ~ normal(0.0, 1.0)
    y ~ interval_censored(normal, lo, hi, mu, 1.0)
end
""", :interval_model)
```

```@raw html
</div>
```

The density is the stable log difference between the endpoint CDFs. Prediction
returns the uncoarsened latent draw; consumers can apply their own interval
reporting convention rather than losing information in the compiler.

### Fused GLMs

Each fused likelihood remains a native Stan GLM in the model block. StanBlocks
generates pointwise-density and predictive-RNG companions around it.

#### Normal identity-link GLM

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0.1, -0.2, 0.4]

normal_glm_model = @slic (; X, y, k=size(X, 2)) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1; n=k)
    sigma ~ exponential(1)
    y ~ normal_id_glm(X, alpha, beta, sigma)
end
""", :normal_glm_model)
```

```@raw html
</div>
```

#### Bernoulli logit-link GLM

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0, 1, 1]

bernoulli_glm_model = @slic (; X, y, k=size(X, 2)) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1; n=k)
    y ~ bernoulli_logit_glm(X, alpha, beta)
end
""", :bernoulli_glm_model)
```

```@raw html
</div>
```

#### Poisson log-link GLM

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0, 1, 2]

poisson_glm_model = @slic (; X, y, k=size(X, 2)) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1; n=k)
    y ~ poisson_log_glm(X, alpha, beta)
end
""", :poisson_glm_model)
```

```@raw html
</div>
```

#### Negative-binomial log-link GLM

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
y = [0, 1, 2]

negbin_glm_model = @slic (; X, y, k=size(X, 2)) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1; n=k)
    phi ~ exponential(1)
    y ~ neg_binomial_2_log_glm(X, alpha, beta, phi)
end
""", :negbin_glm_model)
```

```@raw html
</div>
```

The model block retains the fused Stan primitive. Where Stan has no matching
fused RNG, StanBlocks expands only the predictive draw to the base-family RNG
at `alpha + X * beta`; the likelihood remains fused.

## Expansion, inlining, assertions, and return-type queries

### Caller macros expand before tracing

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
macro center(x)
    :($x - mean($x))
end

x = [-1.0, 0.0, 1.0]
y = [0.1, -0.2, 0.4]

macro_model = @slic (; x, y) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1)
    y ~ normal(alpha + beta * @center(x), 1)
end
""", :macro_model)
```

```@raw html
</div>
```

`@views`, `@.`, `@inbounds`, and user-defined Julia macros use the same route.
They expand in the caller module; StanBlocks traces the expanded syntax.

### Inline helpers and caller-scope mutation

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun @inline scale(x::vector[n], s::real)::vector[n] = x * s

@deffun set_first!(buf::vector[n])::vector[n] = begin
    buf[1] = 42.0
    buf
end

@deffun mutate_scaled(x::vector[n])::vector[n] = begin
    buf::vector[n] = scale(x, 2.0)
    set_first!(buf)
end

inline_model = @slic (; x=[1.0, 2.0], y=0.0) begin
    changed = mutate_scaled(x)
    y ~ normal(sum(changed), 1)
end
""", :inline_model)
```

```@raw html
</div>
```

Calls expand at the call site rather than producing a Stan `functions` entry.
Locals receive hygienic per-call names. A trailing `!` is the Julia-convention
spelling for the same inline route and makes caller-buffer mutation expressible.
At a `compile_slic_bundle` boundary, a macro-free UDF metadata entry with
`markers=(:stanonly, :inline)` lowers to this same path.

### Runtime assertions

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun safe_log(x::real)::real = begin
    @stan_assert x > 0 "safe_log: x must be positive"
    log(x)
end

assertion_model = @slic (; x=1.0, y=0.0) begin
    logged = safe_log(x)
    y ~ normal(logged, 1)
end
""", :assertion_model)
```

```@raw html
</div>
```

For `compile_slic_bundle`, the macro-free equivalent is an `assertions` record
such as `((; condition=:(x > 0), message="safe_log: x must be positive"),)` on
the UDF metadata entry. The compiler prepends the validated record to the
bodyful definition using the same `@stan_assert` lowering.

### Transpile-time return type queries

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun element_type(x::real)::real = x

@deffun @stanonly copy_vec(x::vector[n]) = begin
    out::return_type_of(element_type, x[1])[n]
    for i in 1:n
        out[i] = x[i]
    end
    out
end

return_type_model = @slic (; x=[1.0, 2.0], y=[0.0, 0.0]) begin
    copied = copy_vec(x)
    y ~ normal(copied, 1)
end
""", :return_type_model)
```

```@raw html
</div>
```

`return_type_of` exposes the same registered inference table used by the
transpiler. Computed `typeof(f(x[1]))[n]` annotations provide a related
higher-order form: a real-valued `f` produces a `vector`, while an integer-valued
`f` produces `array[] int`.

### Opt-in Julia and Stan emission

`@deffun` definitions are Stan-only by default. Eligible deterministic,
bodyful definitions annotated with `@juliacompat` also install one Julia
method. That supports ordinary unit tests of shared deterministic helpers:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun @juliacompat affine(x::real, a::real = 2.0)::real = a * x + 1.0

affine(3.0) == 7.0

dual_model = @slic (; y=0.0) begin
    y ~ normal(affine(3.0), 1)
end
""", :dual_model)
```

```@raw html
</div>
```

Probability, RNG, ODE, and parallel builtins are outside the bounded Julia
target. Their own `_lpdf`/`_rng`-family definitions remain Stan-only even when
annotated. `@stanonly` can document an intentionally Stan-only helper or opt
one member out of a surrounding `@juliacompat` group.

## Data binding and mock shapes

`@slic` captures a model body before it needs the final dataset. Bind minimal
mock values to establish the types and shapes, then rebind real data later:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
base = @slic (; y = [1], x = [0.0]) begin
    alpha ~ normal(0, 1)
    y ~ bernoulli_logit(alpha + x)
end

posterior = base(; y = [1, 0, 1], x = [-1.0, 0.0, 1.0])
""", :posterior)
```

```@raw html
</div>
```

| Julia value | Inferred Stan carrier |
|---|---|
| `1` / `1.0` | `int` / `real` |
| `[1, 2]` | `array[2] int` |
| `[1.0, 2.0]` | `vector[2]` |
| `Matrix{Float64}` | `matrix[m,n]` (transposed during Stan data preparation) |
| `Vector{Vector{<:Real}}` | `RaggedVector` carrier, not a dense 2-D array |

Calling a traced `StanModel` with new data reuses the trace and replaces only
the values. Changing a structure that affects tracing—such as whether an
outcome contains missing entries—requires a new trace.

## Generated observation outputs

For an ordinary top-level dense observation, qualitatively:

```julia
y ~ normal(mu, sigma)
```

The complete minimal model makes the generated helper definitions and both
derived outputs visible:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
dense_output_model = @slic (; y=[0.1, -0.2, 0.4]) begin
    mu ~ normal(0, 1)
    sigma ~ exponential(1)
    y ~ normal(mu, sigma)
end
""", :dense_output_model)
```

```@raw html
</div>
```

The current shape contract is:

| Observation form | Predictive output | Log-likelihood output |
|---|---|---|
| Dense, top-level | observation-shaped draw | elementwise |
| Dense, inside `plate` | filled per cell | not synthesized |
| Ragged, top-level | flat draw + descriptor `segments` | one aggregate per group |
| Ragged compiler-owned slice inside `plate` | flat draw + `segments` | one aggregate per group |

Consumers should use descriptor fields `generative` and `source`, not parse the
`_gen` or `_likelihood` suffixes.

### Missing continuous outcomes

Author source:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
y = Union{Missing,Float64}[1.0, missing, 3.0]

m = @slic (; y) begin
    mu ~ normal(0, 2)
    sigma ~ exponential(1)
    y ~ normal(mu, sigma)
end
""", :m)
```

```@raw html
</div>
```

The usual imputation does not enlarge the HMC parameter vector: missing values
are posterior-predictive draws unless the completed outcome feeds another live
likelihood. Missing predictors, discrete missing outcomes, and inherently joint
outcomes such as `multi_normal` require an explicit model.

## Deterministic programs with `@deffun`

`@slic` is a flat declaration language. `@deffun` is where deterministic
control flow, mutation, and local allocation live. Type and shape annotations
on UDF arguments, returns, and ordinary assigned locals are optional; inference
specialises them from the call site and RHS. Annotate only for dispatch, named
dimension locals, fresh uninitialised storage, or an otherwise ambiguous result.

### Value iteration and accumulation

StanBlocks source:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun centered_sum(x::vector[n])::real = begin
    xbar = mean(x)
    acc = 0.0
    for xi in x
        acc += xi - xbar
    end
    acc
end

centered_model = @slic (; x=[1.0, 2.0, 4.0], y=0.0) begin
    centered = centered_sum(x)
    y ~ normal(centered, 1)
end
""", :centered_model)
```

```@raw html
</div>
```

### Supported iteration forms

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun squares(x::vector[n]) = [xi * xi for xi in x]

@deffun weighted_sum(x::vector[n])::real = begin
    acc = 0.0
    for (i, xi) in enumerate(x)
        acc += i * xi
    end
    acc
end

@deffun products(a::vector[n], b::vector[n]) =
    [ai * bi for (ai, bi) in zip(a, b)]

@deffun outer(x::vector[n], y::vector[m]) =
    [x[i] * y[j] for i in 1:n, j in 1:m]

iteration_model = @slic (
    ; x=[1.0, 2.0], a=[2.0, 3.0], b=[4.0, 5.0], y=0.0,
) begin
    sq = squares(x)
    ws = weighted_sum(x)
    ps = products(a, b)
    op = outer(x, b)
    y ~ normal(sum(sq) + ws + sum(ps) + sum(to_vector(op)), 1)
end
""", :iteration_model)
```

```@raw html
</div>
```

Emission patterns:

| Source form | Stan lowering |
|---|---|
| `for xi in x` | index loop plus `xi = x[i]` |
| `enumerate(x)` | 1-based index and element bindings |
| `zip(a,b,...)` | loop to the shortest container |
| one-axis real comprehension | allocated `vector` plus fill loop |
| two-axis real comprehension | allocated `matrix` plus nested fill loops |
| integer-valued comprehension | `array[] int`, preserving index usability |

Nested `if`/`else`, `for`, `while`, `=`, `+=`, `-=`, `*=`, `.=` and indexed
assignment work in `@deffun`. `elseif` chains, filtered/stepped generators,
flattened ragged comprehensions, and 3-D+ comprehensions reject explicitly.

## Post-hoc variants and cross-validation taint

`Base.merge` handles structural variants. Data kwargs rebind values. A lower-level
cross-validation marker can additionally taint a held-out input:

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
person = [1, 2, 1, 2]

model = @slic (; person, y=[0.1, -0.2, 0.3, 0.0]) begin
    n_person = maximum(person)
    alpha ~ normal(0, 1; n=n_person)
    y ~ normal(alpha[person], 1)
end

held_out = model(;
    person=StanBlocks.stan.maybecv(:person, [1, 2, 1, 2]),
)
""", :held_out)
```

```@raw html
</div>
```

Activity propagates from the marked input. Likelihood terms reached by the mark
move out of the fit; affected latent variables are redrawn in generated
quantities; unaffected population parameters stay fitted. The descriptor reports
the resulting `held_out` state and removes `:fit` if no live likelihood remains.

This is compiler machinery rather than a complete user-facing CV workflow. BRM
and other consumers decide how units, folds, posterior draws, and result objects
are presented.

## Executable model descriptors

This is Julia descriptor API, not a transpilation example:

```julia
d = stan_descriptor(m; name=:demo)
```

For the core regression MWE, the current descriptor reports:

```text
inputs:
  y_n  derived=true
  y    observed=true
  x_n  derived=true
  x    observed=false

outputs:
  alpha          parameter / posterior
  beta           parameter / posterior
  sigma          parameter / posterior
  y_likelihood   generated_quantity / pointwise_loglik / source=y
  y_gen          generated_quantity / draw / source=y

operations:
  transpile · instantiate · fit · predict · pointwise_loglik
```

The descriptor also publishes the ordered inventory of emitted Stan function
definitions, exact signatures, source spans, and dependency links. Consumers can
select an exact definition and its transitive closure without parsing Stan text.

The execution calls below consume that descriptor and likewise emit no new Stan
program:

```julia
problem = stan_execute(d, :fit)
pred = stan_execute(d, :predict; problem, draws=theta, seed=123)
ll = stan_execute(d, :pointwise_loglik; problem, draws=theta, seed=123)
```

Operations are derived and fail closed. For example, `:fit` exists only if at
least one parameter and one live likelihood reach the model block.

## Scientific computing surface

### ODE solvers

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(@__MODULE__, raw"""
@deffun pk_rhs(t::real, y::vector[ny], ke::real)::vector[ny] = begin
    dy::vector[ny]
    dy[1] = -ke * y[1]
    dy
end

ts = [0.5, 1.0]
conc = [0.8, 0.6]

ode_model = @slic (; ts, conc) begin
    ke ~ lognormal(0, 0.5)
    sigma ~ exponential(1)
    pred = to_vector(
        ode_rk45(pk_rhs, [1.0], 0.0, to_array_1d(ts), ke)[:, 1]
    )
    conc ~ lognormal(log(pred), sigma)
end
""", :ode_model)
```

```@raw html
</div>
```

`ode_rk45` is the default and `ode_bdf` the stiff escalation. Solver results are
Stan arrays of state vectors, so extract one state with `[:,k]` and convert it
back to a vector for a vector observation.

### Torsten

The builtin signature layer includes the Torsten analytical
`pmx_solve_onecpt`/`pmx_solve_twocpt` family. With a Torsten-enabled BridgeStan
build, these calls have been exercised end to end through compilation and a
gradient-correct log density. They remain an environment-dependent extension of
the ordinary BridgeStan route, not a bundled StanBlocks compiler.

### Other registered scientific primitives

- `reduce_sum`, `reduce_sum_static`, and `simple_reduce_sum`;
- Gaussian-process covariance functions, including multidimensional inputs;
- Stan ODE variants and tolerance forms;
- matrix decompositions, solves, eigenvalues, and covariance helpers;
- fused GLMs and hundreds of scalar/vector/matrix probability signatures.

## The compute path

```text
@slic model + @deffun helpers
        │
        ├── stan_model / stan_code ──→ inspectable Stan source
        │
        ├── stanc_check / compiles ──→ syntax + Stan semantic checks
        │
        └── stan_instantiate ────────→ BridgeStan-backed StanProblem
                                          │
                                          └─ LogDensityProblems value + gradient
```

`transpiles(model)` checks only tracing/code generation. A meaningful compiler
change still needs `stanc` and, where semantics matter, a BridgeStan
log-density/gradient comparison. The generated Stan may also be compiled and
sampled through the usual CmdStan workflow.

## Honest current boundaries

| Requested construct | Current answer |
|---|---|
| Ordinary `for`/`if` in `@slic` | No: put deterministic control flow in `@deffun` |
| A loop that introduces independent parameters | Use compiler-owned `plate` |
| A scan where cell `i` consumes cell `i-1` | No: write a deterministic recurrence in `@deffun` |
| Matrix-valued plate cell | No: shared matrix outside, scalar/vector cell result |
| General 3-D+ Julia container | No |
| Filtered/stepped/ragged comprehension | No; write an explicit supported loop |
| Missing continuous outcome vector | Automatic |
| Missing predictor or discrete outcome | Explicit model required |
| Ragged continuous observation | Yes, with groupwise log likelihood |
| Ragged integer observation/predictive draw | No carrier yet |
| Arbitrary existing Julia function | No auto-transpilation; register through `@deffun`/builtins |
| Direct `target +=` in a UDF | No; express a distribution through `_lpdf`/`@lpxf` |
| Full Julia runtime parity for every Stan builtin | No; deterministic `@deffun` subset only |

## What changed since the submitted abstract

| Abstract wording | StanCon 2026 status |
|---|---|
| closures were “potentially planned” | shipped, including captured ODE parameters/data |
| keyword/default arguments were “potentially planned” | shipped: required/optional kwargs and positional defaults |
| Julia-style metaprogramming was “potentially planned” | caller macros expand before tracing; inline helpers and assertions shipped |
| composable submodels | expanded with named positional/typed submodel functions and first-class merged callees |
| custom struct-like types / named tuples | tuple/named-tuple results and nominal compiler carriers support structured lowering |
| automated predictive/log-likelihood output | broadened to descriptors, missing outcomes, plates, and ragged-group semantics |
| PKPD motivation | now includes numerical ODE and Torsten signatures, with SbPMX as a higher-level PKPD consumer |

## Primary references

- Maria I. Gorinova, Andrew D. Gordon & Charles Sutton,
  [*Probabilistic Programming with Densities in SlicStan: Efficient, Flexible and Deterministic*](https://doi.org/10.1145/3290348),
  POPL 2019.
- [SlicStan public repository](https://github.com/mgorinova/SlicStan), including
  its explicit description as a blockless Stan-like language and its research-code caveat.
- [StanBlocks.jl](https://github.com/nsiccha/StanBlocks.jl),
  [current authoring support](https://nsiccha.github.io/StanBlocks.jl/dev/authoring),
  and [API reference](https://nsiccha.github.io/StanBlocks.jl/dev/api).
- [BayesianRegressionModels.jl](https://github.com/nsiccha/BayesianRegressionModels.jl)
  and the [worked model catalogue](https://nsiccha.github.io/BayesianRegressionModels.jl/).

[View the StanCon 2026 slides](https://nsiccha.github.io/StanBlocks.jl/stancon-2026/)
