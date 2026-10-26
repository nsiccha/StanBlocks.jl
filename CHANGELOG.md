# Changelog

All notable changes to StanBlocks.jl are documented here. This project follows
[semantic versioning](https://semver.org/) (pre-1.0: the minor version is the
breaking digit).

## Unreleased

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
