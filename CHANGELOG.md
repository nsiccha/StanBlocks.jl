# Changelog

All notable changes to StanBlocks.jl are documented here. This project follows
[semantic versioning](https://semver.org/) (pre-1.0: the minor version is the
breaking digit).

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
