# StanBlocks.jl

StanBlocks.jl is a Julia → Stan transpiler. You write a probabilistic model in (a restricted subset of) Julia syntax, and StanBlocks generates the equivalent compilable [Stan](https://mc-stan.org/) program — including automatic block placement (`data`/`transformed_data`/`parameters`/`transformed_parameters`/`model`/`generated_quantities`), automatic type/shape/constraint inference, automatic posterior pointwise likelihoods and predictive draws, and (limited) higher-order user-defined functions.

::: warning Caveats
The transpiler is intentionally restricted: `@slic` model bodies are straight-line declarative blocks (no `for`/`while`/`if`/`&&`/`||`/comprehensions — put those in [`@deffun`](#user-defined-functions-deffun) bodies) and not all Julia constructs map to Stan. See [Caveats](#caveats) below.
:::

## Quick Start

```julia
using StanBlocks
import StanLogDensityProblems, JSON

# Model definition (no data bound yet)
earn_height_model = @slic begin
    beta  ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn  ~ normal(beta[1] + beta[2] * to_vector(height), sigma)
end

# Bind data
earn_height_posterior = earn_height_model(; earn, height)

# Inspect the generated Stan code
println(stan_code(earn_height_posterior))

# Compile and instantiate (needs StanLogDensityProblems.jl + JSON.jl)
earn_height_problem = stan_instantiate(earn_height_posterior)
```

`stan_instantiate` returns a `LogDensityProblems`-compatible posterior:

```julia
using LogDensityProblems
LogDensityProblems.dimension(earn_height_problem)               # → number of unconstrained parameters
LogDensityProblems.logdensity(earn_height_problem, theta)       # log target
LogDensityProblems.logdensity_and_gradient(earn_height_problem, theta)
```

## Defining models with `@slic`

`@slic` captures a `begin … end` block as an unevaluated AST plus a defining module:

```julia
m = @slic begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end
```

Two-arg form binds data inline (handy for one-shot models):

```julia
m = @slic (;y=randn(20)) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end
```

Calling a `SlicModel` (or a traced `StanModel`) with kwargs **binds or replaces** its data:

```julia
m1 = m(;y=randn(50))     # rebind to fresh data
m2 = stan_model(m)(;y=randn(100))  # update data on a pre-traced model (cheaper)
```

For each `Vector` data kwarg `x`, an `Int` `x_n = length(x)` is added automatically; for each `Matrix`, both `x_m` and `x_n` are added (so you can refer to those names inside the model). `Vector{<:AbstractVector{<:Real}}` data is automatically [encoded as a ragged vector](#ragged-vectors).

### Allowed statements in a `@slic` body

A `@slic` body is a flat sequence of:

| Form              | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `lhs ~ rhs(args…)` | Sample / observe (registers `lhs` as a parameter or observation) |
| `lhs = expr`      | Deterministic assignment                                         |
| `lhs::T = expr`   | Typed-LHS sugar — see below                                      |
| `lhs ~ submodel(;…)` | Embed a [sub-model](#sub-models)                              |

Control flow (`for`, `while`, `if`, `&&`, `||`, ternary, comprehensions) is **not** allowed at model level — move it into an `@deffun` body. The transpiler emits an explicit error pointing this out.

### Activity analysis

StanBlocks decides for each variable which Stan block it belongs to:

| Block                       | Description                                                       |
|-----------------------------|-------------------------------------------------------------------|
| `data`                      | Passed in from Julia (observed quantities)                        |
| `transformed_data`          | Computed once from data                                           |
| `parameters`                | Sampled by HMC — these set the unconstrained dimension            |
| `transformed_parameters`    | Deterministic functions of parameters, evaluated each gradient step |
| `model`                     | Likelihood / target contributions                                 |
| `generated_quantities`      | Computed per sample, do not enter the likelihood                  |

Shapes that come from data are exposed automatically as integer `data` (e.g. `n`, `m`).

### Constraints

Constraints are passed as keyword arguments to the prior and turn into Stan-level constraint declarations:

```julia
@slic (;y) begin
    sigma ~ std_normal(;lower=0.)         # sigma in [0, ∞)
    theta ~ uniform(0., 1.)               # bounds inferred from arguments
    rho   ~ beta(2., 2.)                  # [0,1] bounds inferred automatically
    mu    ~ normal(0., 1.; n=length(y))   # vector parameter, size from data
    y     ~ normal(mu, sigma)
end
```

Many priors auto-add their natural bounds (so you don't repeat yourself):

| Distribution                                                                                | Inferred constraint    |
|---------------------------------------------------------------------------------------------|------------------------|
| `beta`, `beta_proportion`                                                                   | `lower=0, upper=1`     |
| `von_mises`                                                                                 | `lower=0, upper=2π`    |
| `uniform(a, b)`                                                                             | `lower=a, upper=b`     |
| `lognormal`, `chi_square`, `inv_chi_square`, `scaled_inv_chi_square`, `exponential`, `gamma`, `inv_gamma`, `weibull`, `frechet`, `rayleigh`, `loglogistic`, `pareto`, `pareto_type_2` | `lower=0`              |

You can additionally pass `n=…` or `m=…, n=…` on any prior to make the parameter a vector or matrix (with the size taken from any constant or any data-qualified expression — see [`maybe_lazy_size`](#dynamic-shape-handling)).

## User-defined functions: `@deffun`

`@deffun` registers a Stan-compatible function with type-annotated arguments. The body is real Julia — it may use `for`/`while`/`if`/comprehensions/etc. — and is also transpiled to Stan.

```julia
@deffun garch11_lpdf(y::vector[T], mu::real, alpha0::real, alpha1::real, beta1::real)::real = begin
    sigma2 = alpha0
    rv = 0.
    for t in 1:T
        rv += normal_lpdf(y[t], mu, sqrt(sigma2))
        sigma2 = alpha0 + alpha1 * square(y[t] - mu) + beta1 * sigma2
    end
    return rv
end
```

Then in a `@slic` block:

```julia
@slic (;y=randn(100)) begin
    mu     ~ std_normal()
    alpha0 ~ std_normal(;lower=0.)
    alpha1 ~ uniform(0., 1.)
    beta1  ~ uniform(0., 1.)
    y      ~ garch11(mu, alpha0, alpha1, beta1)
end
```

### Type/shape annotations

Annotations follow the pattern `name::type[size_args…]`. Size arguments introduced this way are **available as locals inside the function body**:

```julia
@deffun mean_real_vector(x::vector[n])::real = sum(x) / n
@deffun mean_matrix(A::matrix[m, n])::real = sum(A) / (m * n)
```

Underscored names mean "I take this argument but ignore the value, just use its shape":

```julia
@deffun pad_zero(_::vector[n])::vector[n+1] = append_row(rep_vector(0., n), 0.)
```

For functions ending in `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf`:

- the return type is automatically `real`
- companion `_lpdfs`/`_lpmfs`/`_lcdfs`/`_lccdfs` (pointwise) and `_rng` (predictive) stubs are generated automatically and used by [posterior pointwise likelihood / predictive generation](#posterior-pointwise-likelihood-and-predictive-draws)

```julia
@deffun my_normal_lpdf(y, mu, sigma) = normal_lpdf(y, mu, sigma)
@deffun my_normal_rng(mu, sigma)::real = normal_rng(mu, sigma)
```

### Variadic and higher-order

`args...` is supported, including dispatching on the **type of a function argument** (the `::typeof(g)` pattern):

```julia
@deffun begin
    simple_lpdf(y, x) = 0.
    simple_rng(x)     = 0.

    my_lpdf(y, args...) = reject(1)              # default
    my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)

    fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)
end

@slic (;obs=0.) begin
    loc ~ std_normal()
    obs ~ fof(simple, loc)            # dispatched to simple_lpdf
end
```

### Opt-in dispatch hooks

- [`@lpxf`](api#StanBlocks.@lpxf) — register the three SLIC dispatch hooks (`lpxf_expr`, `rng_expr`, `likelihood_expr`) for one or more `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf` symbols. The companion `_rng` and `_lpdfs` (or `_lpmfs`/etc.) names must already exist.
- [`@lhs`](api#StanBlocks.@lhs) — used **inside** a `@deffun` block, registers a method for base-level LHS inference (so `lhs ~ foo(args…)` works for that method). Without it, only the `_lpdf`-keyed tracetype is registered (so the method dispatches when called explicitly, but not via `~`). Compose with `@lpxf` in either order.

```julia
@deffun begin
    @lhs @lpxf my_normal_lpdf(y::real, mu::real, sigma::real) = normal_lpdf(y, mu, sigma)
end
```

### `@defsig`

`@defsig` is the lower-level signature-only macro used for bulk definitions (mostly internal — see `src/slic_stan/builtin.jl`). End users should usually prefer `@deffun`.

## Sub-models

A `SlicModel` defined elsewhere can be embedded as if it were a distribution:

```julia
prior = @slic begin
    mu  ~ std_normal()
    tau ~ std_normal(;lower=0.)
    return mu, tau          # what the sub-model "produces"
end

hierarchical = @slic (;y) begin
    (mu, tau) = prior()
    theta     ~ normal(mu, tau; n=length(y))
    y         ~ normal(theta, 1.)
end
```

Sub-models can also be sampled from (`x ~ submodel(;…)`); the sub-model's parameters are namespaced under `x_*` in the generated Stan code.

You can also append/replace statements on a model after the fact:

```julia
extended = base_model(:( y_pred = normal_rng(mu, sigma) ))
```

## Posterior pointwise likelihood and predictive draws

For each `~` statement, StanBlocks automatically emits the corresponding pointwise log-likelihood and a predictive RNG draw into `generated_quantities`, using the `_lpdfs`/`_lpmfs`/`_rng` companions. So given:

```julia
m = @slic (;y=randn(10)) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end
```

…the generated Stan exposes `log_lik_y[i]` and `y_pred[i]` per draw, ready for PSIS-LOO / posterior predictive checks. See [`stan_code(model)`](api#StanBlocks.stan_code) to inspect.

## Built-in Stan functions

Several hundred built-in Stan functions and distributions are pre-registered with their type/shape signatures (see [`src/slic_stan/builtin.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/dev/src/slic_stan/builtin.jl)). Highlights:

- **Vector/matrix construction**: `rep_vector`, `rep_matrix`, `rep_array`, `linspaced_vector`, `linspaced_array`, `to_vector`, `to_row_vector`, `to_matrix`, `to_array_1d`, `diag_matrix`, `one_hot_vector`, `identity_matrix`
- **Append / reshape**: `append_row`, `append_col`, `append_array`, `reshape`
- **Linear algebra**: `dot_product`, `rows_dot_product`, `cumulative_sum`, `mdivide_left_tri_low`, `mdivide_right_tri_low`, `diag_pre_multiply`, `diag_post_multiply`, `cholesky_decompose`, `quad_form`, `crossprod`, `inverse`, `determinant`, `eigenvalues_sym`, …
- **Introspection**: `dims(X)`, `rows(X)`, `cols(X)`, `num_elements(X)`
- **Scalar math**: `inv_logit`, `logit`, `log1m`, `log1p_exp`, `log_inv_logit`, `log1m_exp`, `log_sum_exp`, `Phi`, `Phi_approx`, `square`, `lgamma`
- **Reductions / parallelism**: `reduce_sum`, `reduce_sum_static`, `simple_reduce_sum`
- **Random draws** (in `generated_quantities`): `normal_rng`, `binomial_rng`, `dirichlet_rng`, `multi_normal_rng`, `multi_normal_cholesky_rng`, …
- **ODE integrators**: `ode_rk45`, `ode_rk45_tol`, `ode_ckrk`, `ode_adams`, `ode_bdf` and their `_tol` variants
- **GP covariances**: `gp_exp_quad_cov`, `gp_exponential_cov`, `gp_matern32_cov`, `gp_matern52_cov`, `gp_periodic_cov`, `gp_dot_prod_cov`
- **Distributions**: `normal`, `student_t`, `cauchy`, `lognormal`, `gamma`, `beta`, `exponential`, `uniform`, `bernoulli`, `bernoulli_logit`, `binomial`, `binomial_logit`, `neg_binomial_2`, `multi_normal`, `multi_normal_cholesky`, `dirichlet`, `lkj_corr`, `lkj_corr_cholesky`, `wishart`, … (each with `_lpdf`/`_lpmf`, `_lpdfs`/`_lpmfs` and `_rng` variants where appropriate)

A few short examples (all from the test suite, which doubles as an executable spec):

```julia
# rep_vector → vector
@slic (;n=5, obs=randn(5)) begin
    v   = rep_vector(0., n)
    obs ~ normal(v, 1.)
end

# rows_dot_product → vector[m]
@slic (;m=3, n=4, obs=randn(3)) begin
    A   = rep_matrix(1., m, n)
    obs ~ normal(rows_dot_product(A, A), 1.)
end

# multi_normal with a covariance matrix
@slic (;obs=randn(3)) begin
    mu  ~ std_normal(;n=3)
    cov = diag_matrix(rep_vector(1., 3))
    obs ~ multi_normal(mu, cov)
end

# Predictive draw in generated_quantities
@slic (;n=5, obs=randn(5)) begin
    mu    ~ std_normal(;n=n)
    obs   ~ normal(mu, 1.)
    y_rep = vector_std_normal_rng(n)
end
```

### Dynamic shape handling

When the size of a value depends on parameters, StanBlocks can fall back to runtime queries (`dims(X)[i]`) instead of a static expression. Two helpers govern this:

- `maybe_lazy_size` — keeps a size expression if it is structurally simple **or** fully data-qualified; otherwise replaces it with `dims(var)[i]`
- `fold_shape_query` — constant-folds `dims(X)[i]` to `X`'s known size when fully data-qualified, so shape-derived sizes still land in `transformed_data`

You generally don't call these — they're applied automatically when type/shape inference would otherwise stall.

### Ragged vectors

A `Vector{<:AbstractVector{<:Real}}` data kwarg is automatically converted by `to_ragged` to a `(; mem, ends)` named tuple — `mem` is the concatenated memory, `ends` are inclusive 1-based end offsets per subvector. Use `ragged_n`, `ragged_total`, `ragged_start`, `ragged_end`, `ragged_length` to access subvectors:

```julia
@slic (;y=[randn(3), randn(5), randn(2)]) begin
    mu ~ std_normal(;n=ragged_n(y))
    for_each_ragged_obs(y, mu)   # define this in a @deffun, since loops aren't allowed in @slic
end
```

## Inspection and compilation

| API                                              | Use                                                |
|--------------------------------------------------|----------------------------------------------------|
| [`stan_code(model)`](api#StanBlocks.stan_code)   | Returns the generated Stan source as a `String`    |
| [`stan_model(slic)`](api#StanBlocks.stan_model)  | Performs the trace once, returns a [`StanModel`](api#StanBlocks.StanModel) you can re-data via `model(; new_kwargs…)` |
| [`stan_data(model)`](api#StanBlocks.stan_data)   | Returns the data dictionary that will be passed to BridgeStan |
| [`stan_instantiate(model)`](api#StanBlocks.stan_instantiate) | Compiles via BridgeStan and returns a `StanLogDensityProblems.StanProblem` |

A `StanModel` returned by `stan_model` is **cheap to re-data**: `model(; y=new_y)` returns a new model that reuses the trace.

## Errors

Anything that goes wrong during transpilation, compilation, or evaluation is wrapped in a [`StanBlocksError`](api#StanBlocks.StanBlocksError). The display includes:

- the pipeline `phase` (`:transpile`, `:compile`, `:evaluate`)
- a short `context` string
- the underlying cause and (when applicable) the stack of `@slic`/`@deffun` expressions being processed at the time

## Caveats

### Constant terms in log-density

Stan's `~` statement [drops constants](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement) that don't depend on parameters. StanBlocks does **not** drop constants — it always emits the full log-density. The absolute value will differ from a hand-written Stan model, but the posterior geometry is identical and sampling is unaffected.

### Limited Julia coverage

Only Julia constructs that map to Stan are supported. Arbitrary Julia functions cannot be transpiled — define them via `@deffun`.

### No control flow at model level

`for`/`while`/`if`/`&&`/`||`/ternary/comprehensions are not allowed in `@slic` bodies. Move that logic into an `@deffun` body or use a vectorised form.

### Dotted operators

Don't use `.+`/`.*` in `@slic` bodies — Stan uses `+`/`*` for scalar–vector arithmetic and StanBlocks follows that convention.

### Name resolution

`@slic` captures `__module__`, so `@deffun` functions defined in the same module (including a package extension) are found automatically. The lookup order is: model scope → `builtin` → defining module → `Main`.

## Legacy Julia backend

The original `@stan`/`@parameters`/`@transformed_parameters`/`@generated_quantities`/`@bsum` macros (the pre-Stan, pure-Julia backend) are still exported but **deprecated**. New code should use `@slic` + `@deffun`.

## Real-world example: BayesianRegressionModels.jl

The largest production consumer of `@slic` / `@deffun` is [`BayesianRegressionModels.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl) (BRM) — a `brms`/`bambi`-style regression DSL that lowers a formula like

```julia
log_odds_bin ~ 1 + c2 + (1 | g1)
bin_succ     ~ BinomialLogit(bin_n, log_odds_bin)
```

into a fully-traced Stan model via StanBlocks. If you want to see how `@slic` and `@deffun` are used in anger — typed-LHS sugar, sub-models composed across multiple regressions, `@lhs @lpxf` UDFs, ragged data, control-flow-in-`@deffun`, etc. — read `src/sbimpl.jl` and the catalog of ~50 worked examples under `web-macro/examples/`.

A few patterns worth lifting:

```julia
# Sub-model returning a value, used as `~` rhs from another model
popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
    return X * beta_pop
end

# Typed-LHS sampling routed to a UDF via `@lhs @lpxf`
StanBlocks.@deffun begin
    @lhs @lpxf multi_lkj_corr_cholesky_lpdf(L::cholesky_factor_corr[m, n], x::real)::real = begin
        rv = 0.
        for i in 1:m
            rv += lkj_corr_cholesky_lpdf(L[i, :, :], x)::real
        end
        rv
    end
end

ranef_correlated_by = StanBlocks.@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(; lower=0.)
    z::vector[n_strata, n_terms]   ~ multi_std_normal()
    b = stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
    return rows_dot_product(Z, b[group_idx, :])
end

# Loop hoisted into a @deffun helper because @slic forbids control flow
StanBlocks.@deffun begin
    ar1_recurse(phi::real, epsilon::vector[n], n::int)::vector[n] = begin
        u = rep_vector(0., n)
        u[1] = epsilon[1]
        for t in 2:n
            u[t] = phi * u[t-1] + epsilon[t]
        end
        u
    end
end

_sb_ar1 = StanBlocks.@slic begin
    n_obs = num_elements(time)
    phi_raw ~ std_normal()
    phi     = tanh(phi_raw)
    epsilon ~ std_normal(; n=n_obs)
    return ar1_recurse(phi, epsilon, n_obs)
end
```

The deployed catalog lives at <https://nsiccha.github.io/BayesianRegressionModels.jl/> — each entry is a small formula, the resulting `@slic` body, and the generated Stan code, side by side. For coverage of distributions, link functions, interactions, splines, AR processes, measurement-error covariates, hurdle / zero-inflated likelihoods, multi-membership / stratified random effects, etc., this is the most thorough working bibliography of the macro surface.

## See also

- [API Reference](api) — full list of exported functions and macros
- [Case Studies](https://nsiccha.github.io/StanBlocks.jl/slic/) — golf, radon, crowdsourcing, ISBA PCR, and more
- [BayesianRegressionModels.jl](https://github.com/nsiccha/BayesianRegressionModels.jl) ([catalog](https://nsiccha.github.io/BayesianRegressionModels.jl/)) — the canonical large-scale `@slic` / `@deffun` consumer
- [Stan Documentation](https://mc-stan.org/docs/stan-users-guide/)
- [BridgeStan](https://github.com/roualdes/bridgestan) — Stan ↔ Julia bridge used for compilation
- [StanLogDensityProblems.jl](https://github.com/sethaxen/StanLogDensityProblems.jl) — `LogDensityProblems` interface around BridgeStan
- [PosteriorDB.jl](https://github.com/sethaxen/PosteriorDB.jl) — reference posteriors used for testing
