# StanBlocks.jl

A Julia → Stan transpiler. You write a probabilistic model in (a restricted subset of) Julia syntax; StanBlocks generates the corresponding compilable [Stan](https://mc-stan.org/) program. Out of the box you get:

- automatic **block placement** — variables are routed to `data` / `transformed_data` / `parameters` / `transformed_parameters` / `model` / `generated_quantities` based on what they depend on
- automatic **type, shape and constraint inference** — including for user-defined functions
- automatic **posterior pointwise log-likelihoods** and **predictive draws** for every `~` statement
- (limited) **higher-order user-defined functions** — `map`-, `reduce_sum`-, and `broadcast`-style patterns
- full **Stan toolchain** — `stan_code(model)` to inspect, `stan_instantiate(model)` to compile via [BridgeStan](https://github.com/roualdes/bridgestan), then sample/score with anything that consumes [`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl)

::: warning Restricted syntax
A `@slic` body is a **flat declarative block**: `~` and `=` statements only. No `for`/`while`/`if`/`&&`/`||`/ternary/comprehension at the model level — put those in [`@deffun`](#user-defined-functions-with-deffun) bodies. The transpiler errors with a pointer if you forget. See [Caveats](#caveats).
:::

## Quick Start

```julia
using StanBlocks
import StanLogDensityProblems, JSON, LogDensityProblems

# 1. Define the model (no data bound yet — just an AST + defining module)
earn_height_model = @slic begin
    beta  ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn  ~ normal(beta[1] + beta[2] * to_vector(height), sigma)
end

# 2. Bind data — `;earn, height` is shorthand for `;earn=earn, height=height`
earn_height_posterior = earn_height_model(; earn, height)

# 3. Inspect the generated Stan code
println(stan_code(earn_height_posterior))

# 4. Compile (BridgeStan + JSON in scope) and use as a LogDensityProblem
problem = stan_instantiate(earn_height_posterior)
LogDensityProblems.dimension(problem)                       # # of unconstrained params
LogDensityProblems.logdensity(problem, theta)               # log target
LogDensityProblems.logdensity_and_gradient(problem, theta)  # log target + ∇
```

## Defining models with `@slic`

`@slic` captures a `begin … end` block as an unevaluated AST plus a defining module. Two forms:

```julia
# AST only — bind data later
m = @slic begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end

# AST + data inline (handy for one-shot models)
m = @slic (;y=randn(20)) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end
```

### Allowed statements

A `@slic` body is a flat sequence of:

| Form                       | Meaning                                                          |
|----------------------------|------------------------------------------------------------------|
| `lhs ~ rhs(args…)`         | Sample / observe (registers `lhs` as a parameter or observation) |
| `lhs::T[size…] ~ rhs(args…)` | Same, but with explicit type and shape — see [Typed-LHS](#typed-lhs) |
| `lhs::T[size…]`            | Declare a flat-prior parameter with explicit type and shape       |
| `lhs = expr`               | Deterministic assignment                                         |
| `lhs::T[size…] = expr`     | Typed-LHS assignment                                             |
| `lhs ~ submodel(;…)`       | Embed a [sub-model](#sub-models) (its parameters land under `lhs_*`) |
| `return expr`              | (Sub-models only) declare what value the sub-model produces      |

Control flow at the model level is **not** allowed — push it into a [`@deffun`](#user-defined-functions-with-deffun).

### Typed-LHS

Stan needs an explicit type and shape for any non-scalar parameter. When inference can't figure it out, declare it on the LHS:

```julia
@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(;lower=0.)
    z::vector[n_strata, n_terms]   ~ multi_std_normal()
end
```

This works for both `~` (sampling) and `=` (assignment). The size arguments inside `[…]` may reference any data-qualified name.

A bare typed declaration is also a parameter declaration with no prior:

```julia
@slic (;y=randn(20), X=randn(20, 3)) begin
    beta::vector[3]                 # flat-prior parameter
    y ~ normal(X * beta, 1.0)
end
```

This form is model/sub-model syntax: the declaration goes directly into Stan's
`parameters` block and contributes no density statement. A bare declaration inside
an `@deffun` remains an ordinary function-local declaration. Use `~ flat(...)` when
the distribution call must carry `lower`, `upper`, `offset`, or `multiplier` kwargs;
the bare form has no RHS from which to obtain those constraints.

Model-body indexed assignment remains unavailable to user code: do not follow a
bare declaration with `beta[i] = ...`. Compiler-owned inline/plate lowering may
use certified indexed fills internally; those are reclassified as transformed
data, transformed parameters, or generated quantities from their right-hand
sides rather than emitted as parameters. These internal fills follow Stan's
ordinary partial-initialization semantics: the compiler-owned producer is
responsible for covering every element it later reads.

### Data binding

Calling a `SlicModel` (or a traced `StanModel`) with kwargs **binds or replaces** its data:

```julia
m1 = m(;y=randn(50))                # SlicModel call: re-bind data
sm = stan_model(m)                  # trace once (expensive)
m2 = sm(;y=randn(100))              # StanModel call: re-data only (cheap)
```

For each `Vector` data kwarg `x`, an `Int` `x_n = length(x)` is added automatically; for each `Matrix`, both `x_m` and `x_n` are added. So you can refer to those names inside the model without ceremony. `Vector{<:AbstractVector{<:Real}}` data is automatically [encoded as a ragged vector](#ragged-vectors).

You can also append/replace statements on a model after the fact by passing AST snippets:

```julia
extended = base_model(:( y_pred = normal_rng(mu, sigma) ))
```

If a passed snippet has the same LHS as an existing statement, it replaces that statement; otherwise it's appended.

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

Shapes derived from data (the auto-added `_n`, `_m` integers, the result of [`dims(X)[i]`](#dynamic-shape-handling), etc.) land in `transformed_data` automatically.

### Constraints

Constraints are passed as keyword arguments to the prior and lower to Stan-level constraint declarations:

```julia
@slic (;y) begin
    sigma ~ std_normal(;lower=0.)         # σ ∈ [0, ∞)
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

You can additionally pass `n=…` or `m=…, n=…` on any prior to make the parameter a vector or matrix. The size arguments may reference constants or any data-qualified expression — see [Dynamic shape handling](#dynamic-shape-handling) for the runtime fallback.

## User-defined functions with `@deffun`

`@deffun` registers a Stan-compatible function with type-annotated arguments. Its body is real Julia and may use `for`/`while`/`if`. It also supports bounded one-dimensional comprehensions of the form `[scalar_expr for i in lo:hi]`; these lower to a typed local plus a Stan `for` loop. Multiple or nested generators, filters, arbitrary iterables, stepped ranges, and non-scalar elements are rejected explicitly. The body is transpiled to Stan.

UDF bodies must not contain `~` sampling statements or `target +=` increments — UDFs cannot introduce parameters or directly manipulate the log density. The macro errors at expansion time if either is found.

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

Multiple definitions in one block:

```julia
@deffun begin
    mean_real_vector(x::vector[n])::real = sum(x) / n
    mean_matrix(A::matrix[m, n])::real   = sum(A) / (m * n)
end
```

### Type/shape annotations

Annotations follow the pattern `name::type[size_args…]`. Size arguments introduced this way are **available as locals inside the function body**:

```julia
@deffun mean_real_vector(x::vector[n])::real = sum(x) / n
@deffun mean_matrix(A::matrix[m, n])::real   = sum(A) / (m * n)
```

Underscored names mean "I take this argument but ignore the value, just use its shape":

```julia
@deffun pad_zero(_::vector[n])::vector[n+1] = append_row(rep_vector(0., n), 0.)
```

### Transpile-time return-type queries

`return_type_of(f, args...)` exposes the same inference table the transpiler
uses for calls to non-inline `@deffun` functions and `@defsig`-registered
callables. With representative Julia values it returns a printable `StanType`:

```julia
@deffun identity_vec(x::vector[n])::vector[n] = x

return_type_of(identity_vec, [1.0, 2.0, 3.0])  # vector[3]
```

Inside an `@deffun` body the query is a compile-time type token, so it can drive
a computed declaration without emitting a Stan call:

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

The public contract currently covers scalar and sized-container results.
Inline functions, closures, `@slic` sub-models, arbitrary Julia functions, and
tuple/user-defined-type results need a full call-site trace and fail with an
explicit error rather than returning `anything`.
Within a UDF, write already-bound symbolic output dimensions explicitly as
`return_type_of(f, x)[n]`; this keeps the declaration tied to the surrounding
signature's shape names.

### `_lpdf` / `_lpmf` / `_lcdf` / `_lccdf` companions

For functions whose name ends in one of `_lpdf` / `_lpmf` / `_lcdf` / `_lccdf`:

- the return type is automatically `real`
- companion `_lpdfs` / `_lpmfs` / `_lcdfs` / `_lccdfs` (pointwise) and `_rng` (predictive) stubs are generated automatically and used by [posterior pointwise likelihood / predictive generation](#posterior-pointwise-likelihood-and-predictive-draws)

```julia
@deffun my_normal_lpdf(y, mu, sigma) = normal_lpdf(y, mu, sigma)
@deffun my_normal_rng(mu, sigma)::real = normal_rng(mu, sigma)
```

### Variadic and higher-order functions

`args...` is supported, including dispatching on the **type of a function argument** (the `::typeof(g)` pattern):

```julia
@deffun begin
    simple_lpdf(y, x) = 0.
    simple_rng(x)     = 0.

    my_lpdf(y, args...) = reject(1)               # default branch
    my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)

    fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)   # passes f through
end

@slic (;obs=0.) begin
    loc ~ std_normal()
    obs ~ fof(simple, loc)         # dispatched to simple_lpdf via my_lpdf
end
```

### `@lpxf` and `@lhs` — opt-in dispatch hooks

For UDFs that you want to use via `~` (or whose pointwise/predictive companions need to be discoverable), opt in with these markers:

- **[`@lpxf`](api#StanBlocks.@lpxf)** — register the three SLIC dispatch hooks (`lpxf_expr`, `rng_expr`, `likelihood_expr`) for one or more `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf` symbols. The companion `_rng` and `_lpdfs` (or `_lpmfs`/etc.) names must already exist.
- **[`@lhs`](api#StanBlocks.@lhs)** — used **inside** a `@deffun` block, registers a method for base-level LHS inference (so `lhs ~ foo(args…)` works for that method). Without it, only the `_lpdf`-keyed tracetype is registered (so the method dispatches when called explicitly, but not via `~`). Compose with `@lpxf` in either order.

```julia
@deffun begin
    @lhs @lpxf my_normal_lpdf(y::real, mu::real, sigma::real) = normal_lpdf(y, mu, sigma)
end
```

### `@inline` UDFs (and the trailing `!` synonym)

Annotate a `@deffun` with `@inline` — or give the function a Julia-convention
trailing `!` in its name — to expand every call **at the call site** instead
of emitting a Stan function. Use this for small structured helpers and for
mutation patterns.

```julia
@deffun @inline scale(x::vector[n], s::real)::vector[n] = x * s

@deffun set_first!(buf::vector[n])::vector[n] = begin
    buf[1] = 42.
    return buf
end
```

The trailing `!` is purely a Julia naming convention (no special mutation
flag); the only actionable thing for SLIC is "inline this UDF." Mutation
falls out naturally — substituting the call-site argument means subsequent
assignments (`buf .= …`, `buf[i] = …`, `buf += …`) act on the caller's
variable. Any post-substitution assignment errors surface via SLIC's normal
rules.

What inline UDFs support:

- **Single-expression and multi-statement bodies** — pre-statements hoist
  into the enclosing block; locals get a per-callsite `__il_<n>` suffix so
  they don't collide with the caller's vars or with sibling expansions.
- **Varargs** — `args...` splats through to the inner call: `@inline
  wrap(args...)::real = three(args...)`.
- **Higher-order arguments** — function values flow through normally:
  `@inline twice(f, x)::real = f(f(x))`.
- **Calls inside non-inline UDFs** — the inlined body lands in the
  enclosing function's body; per-block scoping keeps `for`/`while`/`if`
  branches from leaking hoisted statements outside their scope.

Restrictions:

- Inline UDFs **cannot** introduce parameters or directly manipulate the
  log density (no `~`, no `target +=` — same rule as any UDF).
- `@inline` cannot combine with `@lhs` / `@lpxf` (no triad registration
  for inlined log-density UDFs).
- Recursion is not detected (a self-recursive inline UDF would loop
  forever during tracing).
- Early `return` inside control flow is not specially handled — the body's
  final-position expression is the call's value.

### Julia macros inside `@slic` and `@deffun` bodies

Standard Julia macros (`@views`, `@.`, `@inbounds`, user-defined macros) are
expanded against the calling module before tracing, so they work
transparently. SLIC-internal markers (`@doc`, `@lpxf`, `@lhs`, `@inline`) are
preserved verbatim and handled structurally by the parser.

```julia
macro center(x); :($x - mean($x)); end

@slic (;y=randn(20), x=randn(20)) begin
    alpha ~ std_normal()
    beta  ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y ~ normal(alpha + beta * @center(x), sigma)
end
```

### `@defsig`

`@defsig` is the lower-level signature-only macro used for bulk built-in registrations (see `src/slic_stan/builtin.jl`). End users should usually prefer `@deffun`.

## Sub-models

A `SlicModel` defined elsewhere can be embedded as if it were a distribution. Use `~ submodel(;…)` — assignment (`=`) doesn't work for sub-models, only for `StanExpr`-valued RHSs:

```julia
# Re-usable population effects sub-model
popefs = @slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(;n=n_covariates)
    return X * beta_pop                  # what the sub-model "produces"
end

# `eta ~ popefs(;X)` traces the sub-model's body in a fresh sub-scope.
# `eta` becomes a (transformed) parameter holding the sub-model's return
# value; the sub-model's free parameters land under `eta_*` in the
# generated Stan (`eta_beta_pop` here), so multiple instantiations don't
# collide.
@slic (;y, X) begin
    eta ~ popefs(;X)
    y   ~ normal(eta, 1.)
end
```

## Posterior pointwise likelihood and predictive draws

For each `~` statement, StanBlocks automatically emits the corresponding pointwise log-likelihood and a predictive RNG draw into `generated_quantities`, using the `_lpdfs`/`_lpmfs`/`_rng` companions:

```julia
m = @slic (;y=randn(10)) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(mu, sigma)
end
```

The generated Stan exposes per-draw `log_lik_y[i]` and `y_pred[i]` — ready for PSIS-LOO / posterior predictive checks. Inspect via `stan_code(m)`.

For your own UDFs, the same machinery activates as soon as you provide the `_lpdfs` and `_rng` companions (manually or via `@lpxf`).

## Built-in Stan functions

Several hundred built-in Stan functions and distributions are pre-registered with their type/shape signatures. The full list lives in [`src/slic_stan/builtin.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/dev/src/slic_stan/builtin.jl). Highlights:

- **Vector / matrix construction**: `rep_vector`, `rep_matrix`, `rep_array`, `linspaced_vector`, `linspaced_array`, `to_vector`, `to_row_vector`, `to_matrix`, `to_array_1d`, `diag_matrix`, `one_hot_vector`, `identity_matrix`
- **Append / reshape**: `append_row`, `append_col`, `append_array`, `reshape`
- **Linear algebra**: `dot_product`, `rows_dot_product`, `cumulative_sum`, `mdivide_left_tri_low`, `mdivide_right_tri_low`, `diag_pre_multiply`, `diag_post_multiply`, `cholesky_decompose`, `quad_form`, `crossprod`, `inverse`, `determinant`, `eigenvalues_sym`, …
- **Introspection**: `dims(X)`, `rows(X)`, `cols(X)`, `num_elements(X)`
- **Scalar math**: `inv_logit`, `logit`, `log1m`, `log1p_exp`, `log_inv_logit`, `log1m_exp`, `log_sum_exp`, `Phi`, `Phi_approx`, `square`, `lgamma`
- **Reductions / parallelism**: `reduce_sum`, `reduce_sum_static`, `simple_reduce_sum`
- **Random draws** (in `generated_quantities`): `normal_rng`, `binomial_rng`, `dirichlet_rng`, `multi_normal_rng`, `multi_normal_cholesky_rng`, …
- **ODE integrators**: `ode_rk45`, `ode_rk45_tol`, `ode_ckrk`, `ode_adams`, `ode_bdf` and their `_tol` variants
- **GP covariances**: `gp_exp_quad_cov`, `gp_exponential_cov`, `gp_matern32_cov`, `gp_matern52_cov`, `gp_periodic_cov`, `gp_dot_prod_cov`
- **Distributions**: `normal`, `student_t`, `cauchy`, `lognormal`, `gamma`, `beta`, `exponential`, `uniform`, `bernoulli`, `bernoulli_logit`, `binomial`, `binomial_logit`, `neg_binomial_2`, `multi_normal`, `multi_normal_cholesky`, `dirichlet`, `lkj_corr`, `lkj_corr_cholesky`, `wishart`, … (each with `_lpdf`/`_lpmf`, `_lpdfs`/`_lpmfs` and `_rng` variants where appropriate)

A few short examples (drawn from the test suite, which doubles as an executable spec):

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

## Cookbook

A few common patterns, end-to-end:

### Linear regression

```julia
@slic (;y, x) begin
    alpha ~ flat()
    beta  ~ flat()
    sigma ~ flat(;lower=0.)
    y     ~ normal(alpha + beta * to_vector(x), sigma)
end
```

### Hierarchical (normal–normal)

```julia
@slic (;y) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    theta ~ normal(mu, sigma; n=length(y))
    y     ~ normal(theta, 1.)
end
```

### Logistic regression

```julia
@slic (;y, X) begin
    alpha ~ std_normal()
    beta  ~ std_normal(;n=cols(X))
    y     ~ bernoulli_logit(alpha + X * beta)
end
```

### Time series with a `@deffun` recurrence

```julia
@deffun ar1_recurse(phi::real, epsilon::vector[n], n::int)::vector[n] = begin
    u = rep_vector(0., n)
    u[1] = epsilon[1]
    for t in 2:n
        u[t] = phi * u[t-1] + epsilon[t]
    end
    u
end

@slic (;y, time) begin
    n_obs = num_elements(time)
    phi_raw ~ std_normal()
    phi     = tanh(phi_raw)                  # phi ∈ (-1, 1)
    epsilon ~ std_normal(;n=n_obs)
    y       ~ normal(ar1_recurse(phi, epsilon, n_obs), 1.)
end
```

### LKJ-Cholesky correlation prior

```julia
@slic (;y) begin
    L::cholesky_factor_corr[2] ~ lkj_corr_cholesky(1.)
    tau ~ std_normal(;n=2, lower=0.)
    Sigma_chol = diag_pre_multiply(tau, L)
    mu  ~ std_normal(;n=2)
    for_i_in_eachcol(y, mu, Sigma_chol)      # define this loop in @deffun
end
```

## Inspection and compilation

| API                                                          | Use                                                                                  |
|--------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`stan_code(model)`](api#StanBlocks.stan_code)               | Returns the generated Stan source as a `String`                                      |
| [`stan_model(slic)`](api#StanBlocks.stan_model)              | Performs the trace once, returns a [`StanModel`](api#StanBlocks.StanModel) you can re-data via `model(; new_kwargs…)` |
| `StanBlocks.stan_data(model)`                                | The data dictionary that will be passed to BridgeStan                                |
| [`stan_instantiate(model)`](api#StanBlocks.stan_instantiate) | Compiles via BridgeStan and returns a `StanLogDensityProblems.StanProblem`           |

A `StanModel` returned by `stan_model` is **cheap to re-data**: `model(; y=new_y)` returns a new model that reuses the trace.

`stan_instantiate` accepts a few kwargs:

| kwarg            | Default                       | Meaning                                                                               |
|------------------|-------------------------------|---------------------------------------------------------------------------------------|
| `path=…`         | `tmp/<hash>.stan`             | Where to write the `.stan` file. Hash-based by default, so identical code is cached.  |
| `nan_on_error`   | `true`                        | Make BridgeStan return `NaN` instead of throwing on evaluation failures               |
| `make_args`      | `["STAN_THREADS=true"]`       | Extra arguments forwarded to Stan's `make`                                            |
| `warn`           | `false`                       | Forwarded to BridgeStan                                                               |

## Errors

Anything that goes wrong during transpilation, compilation, or evaluation is wrapped in a [`StanBlocksError`](api#StanBlocks.StanBlocksError):

- `phase::Symbol` — `:transpile`, `:compile`, or `:evaluate`
- `context::String` — short description (e.g. `"model: eight_schools"`)
- `cause` — the underlying exception (and, when applicable, a backtrace and a stack of `@slic`/`@deffun` expressions being processed at the time)

The default `showerror` prints the cause **and** the SLIC expression stack — so you see both the Julia traceback and the model-level location of the failure.

## Advanced topics

### Dynamic shape handling

When the size of a value depends on parameters, StanBlocks falls back to runtime queries (`dims(X)[i]`) instead of a static expression. Two helpers govern this:

- `maybe_lazy_size` — keeps a size expression if it is structurally simple **or** fully data-qualified; otherwise replaces it with `dims(var)[i]`
- `fold_shape_query` — constant-folds `dims(X)[i]` to `X`'s known size when fully data-qualified, so shape-derived sizes still land in `transformed_data`

You generally don't call these — they're applied automatically when type/shape inference would otherwise stall.

### Ragged vectors

A `Vector{<:AbstractVector{<:Real}}` data kwarg is automatically converted by `to_ragged` to a `(; mem, ends)` named tuple — `mem` is the concatenated memory, `ends` are inclusive 1-based end offsets per subvector. Use `ragged_n`, `ragged_total`, `ragged_start`, `ragged_end`, `ragged_length` to access subvectors:

```julia
@deffun reduce_ragged_lpdf(rag, n_groups::int)::real = begin
    rv = 0.
    for g in 1:n_groups
        s, e = ragged_start(rag, g), ragged_end(rag, g)
        rv += normal_lpdf(rag.mem[s:e], 0., 1.)
    end
    rv
end

@slic (;y=[randn(3), randn(5), randn(2)]) begin
    n_groups = ragged_n(y)
    y ~ reduce_ragged(n_groups)     # `y` itself (the (mem, ends) ntup) is used as the LHS
end
```

`ragged_start(x, i)` and `ragged_end(x, i)` take the whole `(; mem, ends)` named tuple as their first argument — not `ends` separately. See `src/slic_stan/builtin.jl` for the full set.

## Caveats

### Constant terms in log-density

Stan's `~` statement [drops constants](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement) that don't depend on parameters. StanBlocks does **not** drop constants — it always emits the full log-density. The absolute value will differ from a hand-written Stan model, but the posterior geometry is identical and sampling is unaffected.

### Limited Julia coverage

Only Julia constructs that map to Stan are supported. Arbitrary Julia functions cannot be transpiled — define them via `@deffun`.

### No control flow at model level

`for`/`while`/`if`/`&&`/`||`/ternary/comprehensions are not allowed in `@slic` bodies. Move that logic into a `@deffun` body or use a vectorised form. The transpiler emits an explicit error pointing you at the offending statement.

### Dotted operators

Don't use `.+`/`.*` in `@slic` bodies — Stan uses `+`/`*` for scalar–vector arithmetic and StanBlocks follows that convention.

### Name resolution

`@slic` captures `__module__`, so `@deffun` functions defined in the same module (including a package extension) are found automatically. The lookup order is: model scope → built-in registry → defining module → `Main`.

## Real-world example: BayesianRegressionModels.jl

The largest production consumer of `@slic` / `@deffun` is [`BayesianRegressionModels.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl) (BRM) — a `brms`/`bambi`-style regression DSL that lowers a formula like

```julia
log_odds_bin ~ 1 + c2 + (1 | g1)
bin_succ     ~ BinomialLogit(bin_n, log_odds_bin)
```

into a fully-traced Stan model via StanBlocks. If you want to see how `@slic` and `@deffun` are used in anger — typed-LHS sugar, sub-models composed across multiple regressions, `@lhs @lpxf` UDFs, ragged data, control-flow-in-`@deffun`, etc. — read [`src/sbimpl.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/src/sbimpl.jl) and the catalog of ~50 worked examples under [`web-macro/examples/`](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/main/web-macro/examples).

A few patterns worth lifting:

```julia
# Sub-model returning a value, used as `~` rhs from another model
popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(;n=n_covariates)
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
    tau::vector[n_strata, n_terms] ~ multi_std_normal(;lower=0.)
    z::vector[n_strata, n_terms]   ~ multi_std_normal()
    b = stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
    return rows_dot_product(Z, b[group_idx, :])
end
```

The deployed catalog lives at <https://nsiccha.github.io/BayesianRegressionModels.jl/> — each entry pairs a small formula, the resulting `@slic` body, and the generated Stan code side by side. For coverage of distributions, link functions, interactions, splines, AR processes, measurement-error covariates, hurdle / zero-inflated likelihoods, multi-membership / stratified random effects, etc., this is the most thorough working bibliography of the `@slic` macro surface.

## See also

- [API Reference](api) — full list of exported functions and macros
- [Case Studies](https://nsiccha.github.io/StanBlocks.jl/slic/) — golf, radon, crowdsourcing, ISBA PCR, and more
- [BayesianRegressionModels.jl](https://github.com/nsiccha/BayesianRegressionModels.jl) ([catalog](https://nsiccha.github.io/BayesianRegressionModels.jl/)) — the canonical large-scale `@slic` / `@deffun` consumer
- [Stan Documentation](https://mc-stan.org/docs/stan-users-guide/)
- [BridgeStan](https://github.com/roualdes/bridgestan) — Stan ↔ Julia bridge used for compilation
- [StanLogDensityProblems.jl](https://github.com/sethaxen/StanLogDensityProblems.jl) — `LogDensityProblems` interface around BridgeStan
- [PosteriorDB.jl](https://github.com/sethaxen/PosteriorDB.jl) — reference posteriors used for testing
