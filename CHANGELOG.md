# Changelog

All notable changes to StanBlocks.jl are documented here. This project follows
[semantic versioning](https://semver.org/) (pre-1.0: the minor version is the
breaking digit).

## v0.2.6

- **A `truncated` draw succeeds whatever mass its range holds.** A predictive
  or prior draw through `truncated(...)` (or a family's `lower=`/`upper=`)
  still redraws the base family until a draw lands inside the bounds, but after
  100 misses it now inverts the base CDF on the range: an integer draw is the
  exact quantile, a real one is bisected to float resolution inside the
  family's support. It used to redraw up to 100000 times and then fail with
  `truncated: rejection sampler exceeded 100000 draws`, so a range holding less
  than about 1e-5 of the base law's mass (a count class far above the drawn
  mean, `normal(0, 1)` beyond 6σ) could not be drawn. Draws from
  `neg_binomial`, `neg_binomial_2` and `binomial` read the upper tail through
  the incomplete-beta symmetry, since Stan Math's CCDFs of these families are
  `-Inf` below about 1e-16 of mass. A range whose mass is not representable in
  floating point rejects with a message saying so. Seeded draws from a range
  holding less than about 5% of the mass can differ from earlier versions.

## v0.2.5

- **Stan's `lchoose` is a builtin.** `lchoose(n, k)`, the log of the binomial
  coefficient, traces in a `@slic` or `@deffun` body, in the forms `lbeta`
  takes: two ints or reals, a scalar with a vector, row vector, matrix or real
  array, or two containers of one kind. It used to fail with `Could not find
  lchoose in model, builtin, … or Main!`. A `@juliacompat` helper gets its Julia
  method too, with Stan Math's values, domain error and NaN rule.

## v0.2.4

- **A custom family's sized draw types an untyped observation.** For a family
  whose draw is a whole vector or matrix and whose only predictive companion is
  the sized-token form (`foo_rng(vector[n], mu::vector[n], …)::vector[n]`), an
  untyped `y ~ foo(mu, …)` is now a `vector[n]` whenever the family's other
  arguments carry `n`, with no `@lhs` on the density. This is the shape of a
  forecast's values-free `maybecv(:y)` observation (per cell in a `plate` or
  `@plate for`), of the response a prior program omits, and of a latent
  parameter of the family. Such a `y` used to be a scalar: its re-draw asked
  for a scalar `foo_rng`, and a vector function of the draw failed to type. A
  size that only the token carries still leaves `y` a scalar, and that error
  now says how to give it a shape.

## v0.2.3

- **A sub-model's own data is scoped to its call, like its parameters.** Data
  a sub-model binds itself — a value fixed with `Base.merge(sub, (; k = v))`,
  bound with `sub(; k = v)` before the call, or given to `@slic (; k = v)` — is
  now declared under the receiving LHS (`hx_k` for `hx ~ sub(…)`; nested calls
  compose, `o_w_k`). It used to be declared as plain `k`, so two calls binding
  different values shared one data variable and silently read the first value
  (two HSGP terms fixed to length scales 0.3 and 0.9 both used 0.3). Data the
  caller passes by keyword (`sub(; k = caller_k)`) keeps the caller's name. The
  data names a program declares for such a sub-model change accordingly, in the
  Stan code, `stan_data`, the descriptor's inputs and re-binding
  (`stan_model(m)(; hx_k = …)`). Two bindings that still declare one data name
  with different values are refused, naming both values.

## v0.2.2

- **A RaggedVector selects groups and sums them as Julia does.** `rv[idx]`
  with an integer array, a range or a comparison mask is the groups `idx` as a
  RaggedVector, for ragged data, a plate result, an arithmetic result and a
  `@deffun` argument (a `RaggedMatrix` selects matrices the same way); it used
  to fail to trace. `sum(rv)` is `rv[1] + rv[2] + …` (groups of equal length,
  checked by Stan) and `mean(rv)` is `sum(rv) / length(rv)`; both used to
  transpile to an undeclared `sum_RaggedVector` / `mean_RaggedVector` that
  stanc rejected. `sum(rv.mem)` is the total of every entry. `sd(rv)`, which
  emitted an undeclared `sd_RaggedVector` the same way, is refused when the
  program is built, naming `sd(rv[g])` and `sd(rv.mem)`.
- **A computed `@plate for` / `@scan` local is a loop local, not an output.**
  A bare `tmp = …` (or `tmp::T = …`, or an `=` binding of a `laplace_marginal`
  do block) in an annotated loop body that reads a parameter is now a Stan local
  declared inside the loop of each block that reads it, as the annotated-loop
  scoping decision specified, instead of a per-cell `transformed parameters` /
  `generated quantities` array. Its `loop_<i>_<name>` output, and for a ragged
  or varying-size value its `__pl_len`/`__pl_end` sizing data, are gone from
  the draws and the descriptor; write `tmp[i] = …` to keep a value. A local
  computed from data alone stays in `transformed data`; `~` locals, indexed
  arrays and every `plate(...) do` cell binding are stored as before. A typed
  `=` local now also works inside a nested annotated loop.
- **Arrays of rank three or more are data.** A real `Array{<:Real,N}` data
  kwarg with `N ≥ 3` is declared as an array of matrices over its last two
  dimensions (`array[A_d1] matrix[A_d2, A_d3] A;`) and an integer one as
  `array[A_d1, A_d2, A_d3] int A;`, with the sizes `A_d1`, … added as data.
  `A[i, j, k]` reads the same element as in Julia, and `A[i]` is the `i`th
  matrix. It used to fail with "Do not know how to handle `stan_type(A, …)`".
- **A vector of matrices is ragged matrix data.** A `Vector{<:AbstractMatrix{<:Real}}`
  data kwarg (one design matrix per group, say) ingests as a `RaggedMatrix`,
  as a vector of vectors ingests as a `RaggedVector`: `X[g]` is the `g`th matrix
  and `length(X)` the number of matrices, in the model body, in a plate cell, as
  a plate positional, and passed to a submodel or a `@deffun`. Stan receives one
  tuple of the column-major values, ends, rows and columns. It used to fail with
  "Do not know how to handle `stan_type(X, …)`".
- **Tuple inputs and outputs in the model descriptor.** `ModelInput` and
  `ModelOutput` gain a trailing `fields`: for a Stan tuple entry (a ragged
  vector, a table, `NamedTuple` or `Tuple` data, a record a `@deffun` returns)
  one row per component in tuple order with its `type`, `size`, nested
  `fields` and `value` (inputs) or `dims` (outputs); `(;)` otherwise. Component
  sizes such as a ragged `y`'s `y_mem_n`/`y_ends_n` or a table's `df_nrow` are
  now `derived`, so `required_inputs` no longer asks for them. Code that
  constructs these structs positionally must pass the new field.
- **`NamedTuple`/`Tuple` data with a ragged field encodes.** Its value now holds
  the field's ingested `(mem, ends)` form, as a top-level ragged input does;
  before, `prepare_for_stan` failed on the raw vector of vectors.

## v0.2.1 — additive transpiler features since v0.2.0

**Non-breaking** — additive to the `v0.2.0` transpiler API.

- **Observation / likelihood submodels — `data ~ submodel(...)`.** A submodel
  can now be embedded with observed data on the LHS: it `return`s its single
  internal sampling statement (the *observation slot*) that the outer data binds
  to, so the submodel emits its own `obs ~ family(...)` likelihood and carries
  its own observation-model parameters. Each observation stream becomes one
  removable line (`some_data ~ some_stream(; state...)`) with no `Base.merge`.
  The existing kwarg form (`nominal ~ submodel(; data, ...)`) is unchanged, and
  the resulting log-density is byte-identical to the hand-written `Base.merge`
  composition.
- **Collected-type `plate` LHS — `rv::matrix[K,N] ~ plate(...)`.** A `plate`
  result may be declared with its collected type; the per-cell result form is
  now rejected with a clear error. A typed-LHS `~` distribution additionally
  checks support agreement and errors on a mismatch.
- **Location-first sized-token `_rng` overloads.** Generated-quantities `_rng`
  overloads for the location-first 3-arg families (`skew_normal`,
  `exp_mod_normal`, `skew_double_exponential`, `pareto_type_2`, and `student_t`)
  now accept a vector leading location, fixing GQ draws for regression-style
  likelihoods such as `y ~ skew_double_exponential(mu_vec, sigma_vec, tau)`.
- **Prior-only programs lower to `generated quantities`.** A likelihood-free
  (`data`-less) `@slic` program now lowers every parameter, plate and fill to a
  `fixed_param` generated-quantities draw instead of running NUTS over the
  prior. `ordered` / `positive_ordered` priors deliberately stay sampled (no
  family `_rng` yields a sorted vector), and an improper `flat()` prior that
  would be re-drawn now errors clearly at trace time.
- **Explicit-`N` `multinomial` family.** A native `multinomial` family taking an
  explicit trial count `N`, composable with custom count-composition densities.
- **Complete `multi_normal_cholesky` likelihood triad.** Density, pointwise
  log-likelihood (`_lpdfs`), and `_rng` for `multi_normal_cholesky`.
- **`to_int` real→int conversion primitive.** Stan's `to_int` is registered as a
  native builtin (scalar `(real,) => int`, array `(real[n],) => int[n]`). Because
  StanBlocks places deterministic functions of data in `transformed data` (where
  the argument is data-qualified), a data-side `to_int(round(x))` satisfies
  Stan's data-qualifier contract with no annotation.
- **`csr_matrix_times_vector` sparse-CSR product.**
  `csr_matrix_times_vector(m, n, w, v, u, b) :: vector[m]` — the load-bearing
  sparse `A*b` primitive for compressed-row processes (e.g. spline `Rt`).
- **`row_vector` arithmetic + `append_col`.** The vector/matrix operator family
  is mirrored onto `row_vector` (unary minus, scalar scaling, `±` same-shape,
  `row_vector * matrix`), and `append_col` gains the column-wise
  `row_vector ++ row_vector => row_vector` concatenation plus scalar
  prepend/append forms.
- **Tuple `p[i]` element access.** Indexing a positional tuple returned by a
  `@deffun` with `p[i]` now agrees with `getfield`: it emits `p.N` and resolves
  the element type from the tuple (previously it emitted an invalid bracket and
  typed the element `anything`).
- **int/real 2-D array row-slice `y[i, :]`.** Row-slicing an `int[m,n]` /
  `real[m,n]` array is now typed (mirroring the existing `matrix` rows), so an
  int-array row-slice assignment inside a custom `_rng` transpiles; the
  `tracetype`-error formatter was hardened alongside.
- **Clearer `@deffun` errors for dimensionless container args.** A bare,
  dimensionless container argument (`I::matrix`, `w::vector`) in a `@deffun`
  signature is now rejected at signature parse with a message naming the
  argument and the required `matrix[m, n]` / `vector[n]` form, instead of
  surfacing later as an opaque `::anything` tracetype error.

## v0.2.0 — StanBlocks is now a Julia→Stan transpiler

**Breaking.** StanBlocks has been rebuilt around a single purpose: a Julia
frontend that transpiles one model definition to Stan source and, via
BridgeStan, to a differentiable log-density. The previous, `Distributions`-based
collection of Julia log-density implementations of `posteriordb` models (the
`v0.1.x` line) has been **removed**. Code that relied on the old
Julia-implementation API will not work on `v0.2.0`; pin `StanBlocks = "0.1"` to
stay on the previous package.

The transpiler surface (see the
[README](README.md) and the
[feature atlas](https://nsiccha.github.io/StanBlocks.jl/dev/feature-atlas)):

- **Activity analysis & inference** — automatic block placement; inferred types,
  shapes, and constraints for model bodies and user-defined functions.
- **Composition** — anonymous and named typed-positional sub-models, post-hoc
  `Base.merge` variants, and first-class cross-validation (`cv`) with correct
  density-taint handling.
- **Structured data & `plate`** — ragged data, ragged constrained parameters,
  `EachCol` / `EachRow` views, and compiler-owned independent-cell `plate` loops
  (scalar, fixed-vector, selected ragged/constrained cells).
- **Functions** — defaults, keyword arguments, varargs, higher-order functions,
  Julia-style multiple dispatch, automatic shape extraction, `@deffun`,
  `@inline` UDFs with caller-scope mutation, `@stan_assert`, `return_type_of`,
  and `@juliacompat` / `@stanonly`.
- **Closures** — lifted into generated Stan functions with captured data and
  parameters as explicit trailing arguments; likelihood activity follows the
  captures (ODE-friendly).
- **Distributions** — author-your-own triad (`_lpdf` / `_lpdfs` / `_rng`,
  `@lhs` / `@lpxf`); `weighted`, `truncated`, `censored`, `interval_censored`;
  fused GLM families.
- **Scientific computing** — ODE solvers, Torsten-style pharmacometrics
  signatures, Gaussian-process helpers, and `reduce_sum`.
- **Generated quantities** — automatic pointwise log-likelihood and predictive
  draws; automatic imputation of partly-missing continuous outcomes.
- **Reflection & ergonomics** — executable model descriptors, user-defined types
  (`@usertype`), transparent expansion of user Julia macros inside model bodies,
  and approximate Blue-style formatting.

The dependency set changed accordingly: `BridgeStan`, `StanLogDensityProblems`,
`Tables`, `JSON`, and `LogDensityProblems` are now the runtime deps; `PosteriorDB`
is a weak dependency (the `PosteriorDBExt` extension).

## v0.1.0 – v0.1.5

The original `Distributions`-based Julia log-density implementations of
`posteriordb` models, with the Stan transpiler present only as experimental
extensions. See the git history for details.
