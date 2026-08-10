# StanBlocks.jl feature atlas

*StanCon 2026 companion: author source → essential Stan emission*

[View the StanCon 2026 slides](https://nsiccha.github.io/StanBlocks.jl/stancon-2026/)

This is the long-form companion to the talk. It inventories the **current
author-visible language surface** reviewed against StanBlocks commit `e6f607d`
(2026-08-04). The presentation files themselves are newer; the compiler source
covered here is unchanged by those documentation commits.

The Stan snippets show the **essential emission**: declarations, statements,
loops, and generated quantities that explain the transformation. StanBlocks may
also emit helper functions for vectorised pointwise densities, sized RNGs,
ragged carriers, lifted closures, or distribution combinators. Use
`stan_code(model)` for the complete program produced by a particular checkout.

This atlas covers language and workflow features, not every registered Stan
builtin. Several hundred Stan functions and distribution signatures are
registered; the source of truth for those is
[`src/slic_stan/builtin.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/src/slic_stan/builtin.jl).

## Feature map

| Area | Current surface | Detailed MWE |
|---|---|---|
| Model declarations | `@slic`, `~`, `=`, typed LHS, bare flat-prior parameters | [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints) |
| Analysis | block placement, type/shape/constraint inference, likelihood activity | [Core model](#One-model:-block-placement,-types,-shapes,-and-constraints) |
| Composition | anonymous and named submodels, typed dispatch, `Base.merge`, data rebinding | [Composition](#Composition-and-model-families) |
| Deterministic helpers | `@deffun`, loops, branches, mutation, comprehensions, iteration | [`@deffun`](#Deterministic-programs-with-@deffun) |
| Julia-like functions | closures, HOFs, varargs, positional defaults, required/optional kwargs | [Function signatures](#Function-signatures,-defaults,-keywords,-varargs,-and-higher-order-functions) |
| Metaprogramming | caller macros, `@inline`, trailing `!`, `@stan_assert`, `return_type_of` | [Expansion tools](#Expansion,-inlining,-assertions,-and-return-type-queries) |
| Observation workflow | pointwise log likelihoods, predictive draws, missing-outcome imputation | [Generated outputs](#Generated-observation-outputs) |
| Distribution abstraction | custom `_lpdf`/`_lpdfs`/`_rng`, `@lpxf`, `@lhs`, weighted/truncated/censored/interval | [Distributions](#Custom-and-higher-order-distributions) |
| Structured data/models | `RaggedVector`, ragged constraints, `EachCol`/`EachRow`, fancy indexing, `plate` | [Structured models](#Structured-data-and-compiler-owned-plates) |
| Scientific models | ODE solvers, Torsten signatures, GP covariance, `reduce_sum`, fused GLMs | [Scientific surface](#Scientific-computing-surface) |
| Reflection/execution | descriptors, definition closure, derived operations, BridgeStan | [Descriptor](#Executable-model-descriptors) |
| Model variants | post-hoc overrides and lower-level cross-validation taint | [Variants](#Post-hoc-variants-and-cross-validation-taint) |

## One model: block placement, types, shapes, and constraints

### StanBlocks source

```julia
using StanBlocks

m = @slic (; x = [-1.0, 0.0, 1.0], y = [0.2, -0.1, 0.7]) begin
    mx = mean(x)

    alpha ~ normal(0, 2)
    beta  ~ normal(0, 1)
    sigma ~ exponential(1)

    mu = alpha + beta * (x - mx)
    y ~ normal(mu, sigma)
end
```

### Essential Stan emission

```stan
data {
  int x_n;
  vector[x_n] x;
  int y_n;
  vector[y_n] y;
}
transformed data {
  real mx = mean(x);
}
parameters {
  real alpha;
  real beta;
  real<lower=0.0> sigma;
}
transformed parameters {
  vector[x_n] mu = alpha + beta * (x - mx);
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[y_n] y_likelihood = normal_lpdfs(y, mu, sigma);
  vector[y_n] y_gen = normal_vector_rng(y_n, mu, sigma);
}
```

What happened:

1. Julia vectors `x` and `y` established Stan vector types and generated their
   data dimensions `x_n` and `y_n`.
2. `mx` depends only on data, so it moved to `transformed data`.
3. The prior family inferred `sigma`'s lower bound.
4. `mu` depends on parameters and feeds the likelihood, so it lives in
   `transformed parameters`.
5. The observed `y` generated pointwise log-likelihood and posterior-predictive
   twins automatically.

### Typed LHS and a prior-free parameter

Use explicit types when the prior call does not determine the desired container,
or when the type itself carries Stan semantics:

```julia
@slic (; X, y, k = size(X, 2)) begin
    beta::vector[k]                 # improper-flat sampler parameter
    sigma::real ~ exponential(1)
    y ~ normal(X * beta, sigma)
end
```

Essential emission:

```stan
parameters {
  vector[k] beta;
  real<lower=0.0> sigma;
}
model {
  sigma ~ exponential(1);
  y ~ normal(X * beta, sigma);
}
```

A bare typed declaration is allowed only in a model/submodel and means a
parameter with no density statement. Inside `@deffun`, the same syntax declares
a local that the function must fill. Native constrained types use the same LHS
surface, for example
`L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)` when `L` participates in a
downstream likelihood.

## Data binding and mock shapes

`@slic` captures a model body before it needs the final dataset. Bind minimal
mock values to establish the types and shapes, then rebind real data later:

```julia
base = @slic (; y = [1], t = [1.0]) begin
    n = dims(y)[1]
    y ~ my_bernoulli(link_f, theta)
end

posterior = base(; y = real_y, t = real_t, link_f = logit)
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

For an ordinary top-level dense observation:

```julia
y ~ normal(mu, sigma)
```

StanBlocks adds:

```stan
generated quantities {
  vector[y_n] y_likelihood = normal_lpdfs(y, mu, sigma);
  vector[y_n] y_gen = normal_vector_rng(y_n, mu, sigma);
}
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

```julia
y = Union{Missing,Float64}[1.0, missing, 3.0]

m = @slic (; y) begin
    mu ~ normal(0, 2)
    sigma ~ exponential(1)
    y ~ normal(mu, sigma)
end
```

Essential emission:

```stan
data {
  vector[y_obs_n] y_obs;
  array[y_ii_obs_n] int y_ii_obs;
  array[y_ii_mis_n] int y_ii_mis;
}
parameters {
  real mu;
  real<lower=0.0> sigma;
}
model {
  mu ~ normal(0, 2);
  sigma ~ exponential(1);
  y_obs ~ normal(mu, sigma);
}
generated quantities {
  vector[y_ii_mis_n] y_mis = normal_vector_rng(y_ii_mis_n, mu, sigma);
  vector[y_obs_n] y_obs_likelihood = normal_lpdfs(y_obs, mu, sigma);
  vector[y_obs_n] y_obs_gen = normal_vector_rng(y_obs_n, mu, sigma);
  vector[y_ii_obs_n + y_ii_mis_n] y =
      merge_missing(y_obs, y_mis, y_ii_obs, y_ii_mis);
}
```

The usual imputation does not enlarge the HMC parameter vector: missing values
are posterior-predictive draws unless the completed outcome feeds another live
likelihood. Missing predictors, discrete missing outcomes, and inherently joint
outcomes such as `multi_normal` require an explicit model.

## Composition and model families

### Anonymous submodels: inputs by name

```julia
population_effect = @slic begin
    k = dims(X)[2]
    beta ~ normal(0, 1; n = k)
    return X * beta
end

m = @slic (; X, y) begin
    eta ~ population_effect(; X)
    y ~ normal(eta, 1)
end
```

Essential emission:

```stan
parameters {
  vector[k] eta_beta;
}
transformed parameters {
  vector[y_n] eta = X * eta_beta;
}
model {
  eta_beta ~ normal(0, 1);
  y ~ normal(eta, 1);
}
```

Submodel parameters are namespaced under the receiving LHS. Anonymous
submodels receive inputs through kwargs or enclosing scope; they deliberately
have no positional calling form.

### Named submodels: typed positional dispatch

```julia
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
```

Typed arguments select Julia methods on the traced Stan center type. Multiple
definitions of `@slic f(::real)` and `@slic f(::int)` are distinct methods;
untyped positional arguments match any traced value.

### Post-hoc variants with `Base.merge`

```julia
base = @slic begin
    beta ~ normal(0, 1)
    y ~ normal(beta * x, 1)
end

wide = Base.merge(base, quote
    beta ~ normal(0, 5)
end)
```

The merged statement matches by the bare LHS name and replaces the original.
Typed and untyped versions of the same LHS still match. Additional names append
new statements. A merged `SlicModel` value can also be interpolated into the
callee position of generated AST. The crowdsourcing family in
[Advanced composition patterns](advanced-patterns.md#Crowdsourcing-variants-with-Base.merge)
shows chained assignment and sampling-statement variants in context.

## Deterministic programs with `@deffun`

`@slic` is a flat declaration language. `@deffun` is where deterministic
control flow, mutation, and local allocation live.

### Value iteration and accumulation

StanBlocks source:

```julia
@deffun centered_sum(x::vector[n])::real = begin
    xbar = mean(x)
    acc = 0.0
    for xi in x
        acc += xi - xbar
    end
    acc
end
```

Exact essential Stan function:

```stan
real centered_sum(vector x) {
  real xbar = mean(x);
  real acc = 0.0;
  for (value_index__vi_1 in 1:num_elements(x)) {
    real xi = x[value_index__vi_1];
    acc += xi - xbar;
  }
  return acc;
}
```

### Supported iteration forms

```julia
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

## Function signatures, defaults, keywords, varargs, and higher-order functions

### Positional defaults and keyword arguments

```julia
@deffun affine(x::real, a::real = 2.0; b = 1.0)::real = a * x + b
@deffun scale(x::vector[n]; factor)::vector[n] = factor * x
@deffun shift(x::vector[n]; a, b = 2.0)::vector[n] = a * x .+ b
```

```julia
affine(3.0)                    # default a=2, b=1
affine(3.0, 4.0; b=2.0)
scale(x; factor=0.5)           # required kwarg
shift(x; a=1.5)                # optional b defaults to 2
```

Defaults are resolved at the call site; Stan receives a specialised call with
ordinary positional arguments. Omitting a required keyword errors during
tracing rather than emitting an under-specified function call.

### Variadic and function-typed dispatch

```julia
@deffun begin
    my_bernoulli_lpmf(y, ::typeof(logit), eta) =
        bernoulli_logit_lpmf(y, eta)

    my_bernoulli_lpmf(y, ::typeof(log), eta) =
        bernoulli_lpmf(y, exp(eta))

    my_bernoulli_lpmfs(y::int, args...) =
        my_bernoulli_lpmf(y, args...)

    apply_twice(f, x::real)::real = f(f(x))
end
```

Function values are compile-time tokens. StanBlocks selects and emits the
specialised method; no runtime function object reaches Stan.

### Closures

```julia
@slic (; ts, yobs) begin
    lambda ~ exponential(1)
    y = ode_rk45(
        (t, state) -> -lambda * state,
        [1.0], 0.0, to_array_1d(ts),
    )
    yobs ~ normal(y[1][1], 0.05)
end
```

The closure is lifted into a generated Stan function. Captured data and
parameters become explicit trailing arguments, and likelihood activity follows
through those captures so `lambda` stays a live parameter.

## Expansion, inlining, assertions, and return-type queries

### Caller macros expand before tracing

```julia
macro center(x)
    :($x - mean($x))
end

@slic (; x, y) begin
    alpha ~ normal(0, 1)
    beta ~ normal(0, 1)
    y ~ normal(alpha + beta * @center(x), 1)
end
```

`@views`, `@.`, `@inbounds`, and user-defined Julia macros use the same route.
They expand in the caller module; StanBlocks traces the expanded syntax.

### Inline helpers and caller-scope mutation

```julia
@deffun @inline scale(x::vector[n], s::real)::vector[n] = x * s

@deffun set_first!(buf::vector[n])::vector[n] = begin
    buf[1] = 42.0
    buf
end
```

Calls expand at the call site rather than producing a Stan `functions` entry.
Locals receive hygienic per-call names. A trailing `!` is the Julia-convention
spelling for the same inline route and makes caller-buffer mutation expressible.

### Runtime assertions

```julia
@deffun safe_log(x::real)::real = begin
    @stan_assert x > 0 "safe_log: x must be positive"
    log(x)
end
```

```stan
if (!(x > 0)) reject("safe_log: x must be positive");
```

### Transpile-time return type queries

```julia
@deffun element_type(x::real)::real = x

@deffun copy_vec(x::vector[n]) = begin
    out::return_type_of(element_type, x[1])[n]
    for i in 1:n
        out[i] = x[i]
    end
    out
end
```

`return_type_of` exposes the same registered inference table used by the
transpiler. Computed `typeof(f(x[1]))[n]` annotations provide a related
higher-order form: a real-valued `f` produces a `vector`, while an integer-valued
`f` produces `array[] int`.

### Dual Julia and Stan emission

Eligible deterministic, bodyful `@deffun` definitions also install one Julia
method. That supports ordinary unit tests of shared deterministic helpers:

```julia
@deffun affine(x::real, a::real = 2.0; b = 1.0)::real = a * x + b

affine(3.0) == 7.0
```

Probability, RNG, ODE, and parallel builtins are outside the bounded Julia
target. Their own `_lpdf`/`_rng`-family definitions skip Julia emission
automatically; use `@stanonly` for another intentionally Stan-only helper.

## Custom and higher-order distributions

### Custom distribution triad

```julia
@deffun begin
    @lhs @lpxf robust_lpdf(y::real, mu::real, sigma::real) =
        student_t_lpdf(y, 4, mu, sigma)

    robust_lpdfs(y::real, mu::real, sigma::real)::real =
        robust_lpdf(y, mu, sigma)

    robust_rng(mu::real, sigma::real)::real =
        student_t_rng(4, mu, sigma)
end

@slic (; y = 0.2) begin
    mu ~ normal(0, 2)
    sigma ~ exponential(1)
    y ~ robust(mu, sigma)
end
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

```julia
@slic (; y1, y2, y3, y4, w, lo, hi) begin
    mu ~ normal(0, 1)

    y1 ~ weighted(normal, w, mu, 1)
    y2 ~ truncated(normal, mu, 1; lower=lo, upper=hi)
    y3 ~ censored(normal, mu, 1; lower=lo, upper=hi)
    y4 ~ interval_censored(normal, lo, hi, mu, 1)
end
```

Essential semantic lowering:

| Author form | Density | Predictive draw |
|---|---|---|
| `weighted` | multiply base log density by data weight | unweighted base RNG |
| `truncated` | base density minus truncation normalizer | rejection sample inside bounds |
| `censored` | tail mass at threshold atoms; density in interior | clamp base RNG |
| `interval_censored` | `log(CDF(hi)-CDF(lo))` for `(lo,hi]` evidence | uncoarsened base RNG |

For the concrete censored-normal MWE, the model block currently emits:

```stan
model {
  mu ~ normal(0, 1);
  y ~ clamping_normal(lo, hi, mu, 1);
}
generated quantities {
  vector[y_n] y_likelihood = clamping_normal_lpdfs(y, lo, hi, mu, 1);
  vector[y_n] y_gen = clamping_vector_normal_rng(y_n, lo, hi, mu, 1);
}
```

The generated helper uses a stable log-CDF below `lo`, a stable log-CCDF above
`hi`, the base density in the interior, and a clamped base draw for prediction.

### Fused GLMs

```julia
y ~ normal_id_glm(X, alpha, beta, sigma)
y ~ bernoulli_logit_glm(X, alpha, beta)
y ~ poisson_log_glm(X, alpha, beta)
y ~ neg_binomial_2_log_glm(X, alpha, beta, phi)
```

The model block retains the fused Stan primitive. Where Stan has no matching
fused RNG, StanBlocks expands only the predictive draw to the base-family RNG
at `alpha + X * beta`; the likelihood remains fused.

## Structured data and compiler-owned plates

### `plate`: an independent-cell loop that may introduce parameters

StanBlocks source:

```julia
@slic (; y = randn(4), mu0 = 0.5) begin
    sigma ~ exponential(1)

    theta ~ plate(y; outer = 4) do yi
        t ~ normal(mu0, 1)
        yi ~ normal(t, sigma)
        t
    end
end
```

Essential exact emission:

```stan
parameters {
  real<lower=0.0> sigma;
  vector[4] theta_t;
}
transformed parameters {
  vector[4] theta;
  for (plate_i__pl_1 in 1:4) {
    theta[plate_i__pl_1] = theta_t[plate_i__pl_1];
  }
}
model {
  sigma ~ exponential(1);
  for (plate_i__pl_1 in 1:4) {
    theta_t[plate_i__pl_1] ~ normal(mu0, 1);
    y[plate_i__pl_1] ~ normal(theta_t[plate_i__pl_1], sigma);
  }
}
generated quantities {
  vector[y_n] y_gen;
  for (plate_i__pl_1 in 1:4) {
    y_gen[plate_i__pl_1] = normal_rng(theta_t[plate_i__pl_1], sigma);
  }
}
```

Positional inputs are sliced; lexical captures are shared; fresh cell bindings
are promoted to outer storage; the trailing expression becomes the cell output.
Fixed vector cells become columns of a matrix. Additional outer axes add Stan
array prefixes. Cells must remain independent: `plate` is not a scan.

### Vector-valued plate cells

```julia
@slic (; n_groups = 8, k = 3) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)
    tau::vector[k] ~ normal(0, 1; lower=0)

    b::vector[k] ~ plate(; outer=n_groups) do g
        z::vector[k] ~ std_normal()
        diag_pre_multiply(tau, L) * z
    end
end
```

Both `z` and `b` have logical shape `matrix[k,n_groups]`; the per-group vectors
occupy columns.

### Ragged data

```julia
groups = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]

@slic (; groups, g=2, y=0.0) begin
    first = groups[1]
    selected = groups[g]
    ng = length(groups)
    y ~ normal(sum(first) + sum(selected) + ng, 1)
end
```

The nested Julia vectors become a nominal carrier with flat `mem` and inclusive
group `ends`. `groups[g]` lowers to a Stan helper that reconstructs the requested
slice; `length(groups)` counts groups, not scalar elements.

### Ragged constrained parameters

```julia
@slic (; Ks=[2, 3, 4], y=0.3) begin
    p::simplex[Ks] ~ flat()
    y ~ normal(sum(p[1]), 0.1)
end
```

Stan cannot declare a runtime-ragged pile of simplices. StanBlocks emits a flat
unconstrained parameter, then a compiler-owned per-group constrain/Jacobian loop
and a logical ragged view. The same route covers `ordered`, `positive_ordered`,
`cholesky_factor_corr`, and square `cholesky_factor_cov` groups. This feature
requires BridgeStan 2.9 / Stan 2.39 or newer.

### Dense views and indexing

```julia
cols = EachCol(X)
rows = EachRow(X)

@deffun col_sums(X::matrix[m,k]) = [sum(c) for c in EachCol(X)]
```

| Source | Stan lowering |
|---|---|
| `EachCol(X)[j]` | `col(X,j)` |
| `EachRow(X)[i]` | `row(X,i)` |
| `length(EachCol(X))` | `cols(X)` |
| `length(EachRow(X))` | `rows(X)` |
| `alpha[county_idx]` | vectorised integer-array indexing |
| `X[i,:,:]`, `X[:,:,k]`, `L[i,:,:]` | resolved multi-colon slice |

## Post-hoc variants and cross-validation taint

`Base.merge` handles structural variants. Data kwargs rebind values. A lower-level
cross-validation marker can additionally taint a held-out input:

```julia
held_out = model(; person = StanBlocks.stan.maybecv(:person, person_new))
```

Activity propagates from the marked input. Likelihood terms reached by the mark
move out of the fit; affected latent variables are redrawn in generated
quantities; unaffected population parameters stay fitted. The descriptor reports
the resulting `held_out` state and removes `:fit` if no live likelihood remains.

This is compiler machinery rather than a complete user-facing CV workflow. BRM
and other consumers decide how units, folds, posterior draws, and result objects
are presented.

## Executable model descriptors

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

```julia
problem = stan_execute(d, :fit)
pred = stan_execute(d, :predict; problem, draws=theta, seed=123)
ll = stan_execute(d, :pointwise_loglik; problem, draws=theta, seed=123)
```

Operations are derived and fail closed. For example, `:fit` exists only if at
least one parameter and one live likelihood reach the model block.

## Scientific computing surface

### ODE solvers

```julia
@deffun pk_rhs(t::real, y::vector[ny], ke::real)::vector[ny] = begin
    dy::vector[ny]
    dy[1] = -ke * y[1]
    dy
end

@slic (; ts, conc) begin
    ke ~ lognormal(0, 0.5)
    sigma ~ exponential(1)
    pred = to_vector(
        ode_rk45(pk_rhs, [1.0], 0.0, to_array_1d(ts), ke)[:, 1]
    )
    conc ~ lognormal(log(pred), sigma)
end
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
