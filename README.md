# StanBlocks.jl

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/StanBlocks.jl/dev/)
[![CI](https://github.com/nsiccha/StanBlocks.jl/actions/workflows/test.yml/badge.svg)](https://github.com/nsiccha/StanBlocks.jl/actions/workflows/test.yml)

**A Julia frontend for writing and composing Stan models.** Write a model once
in Julia syntax; StanBlocks transpiles it to Stan source *and* — through
[BridgeStan](https://github.com/roualdes/bridgestan) — hands you a
differentiable log-density you can sample or optimise directly. It is a
deliberately modest promise: a *frontend*, not a replacement for Stan. The
generated Stan code is part of the product, meant to be read.

```julia
using StanBlocks
import PosteriorDB, StanLogDensityProblems, JSON

# Data from PosteriorDB
pdb = PosteriorDB.database()
post = PosteriorDB.posterior(pdb, "earnings-earn_height")
(;earn, height) = (;Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(post)))])...)

# The whole model — no `data`/`parameters`/`model` blocks to write by hand
earn_height_model = @slic begin
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end

earn_height_posterior = earn_height_model(; earn, height)  # bind data
println(stan_code(earn_height_posterior))                  # read the Stan
earn_height_problem = stan_instantiate(earn_height_posterior)  # compile + differentiate
```

`stanc` never sees a block you wrote: **activity analysis** decides what is
`data`, `transformed data`, `parameters`, `transformed parameters`, `model`, or
`generated quantities`, and **type/shape/constraint inference** fills in every
declaration — for model bodies *and* user-defined functions (arguments, body,
and return type alike).

## What it does

StanBlocks is one small compiler with a wide surface. The
[feature atlas](https://nsiccha.github.io/StanBlocks.jl/dev/feature-atlas) is the
canonical works / does-not-work matrix; the highlights:

**Activity analysis & inference** — automatic block placement; inferred types,
shapes, and constraints; a single dependency graph evaluated at several
execution times (data, transformed data, parameters, …, generated quantities).

**Composition — the reason for a frontend** — anonymous sub-models, named
typed-positional sub-models, and post-hoc `Base.merge` variants that graft new
statements onto an existing model. Cross-validation is a first-class post-hoc
variant (the `cv` flag), with correct density-taint handling.

**Structured data & compiler-owned `plate` loops** — first-class ragged data and
ragged *constrained* parameters, `EachCol` / `EachRow` dense views, and `plate`
loops over independent per-cell parameters (scalar, fixed-vector, and selected
ragged/constrained cells).

**Functions** — positional defaults, keyword arguments, varargs, and
higher-order functions (`map`, `broadcasted`, `sum`, …); `à-la-Julia` multiple
dispatch (on base type + dimensionality, on dimensionality alone, or on a
higher-order function argument via `f(::typeof(g)) = …`); automatic shape
extraction (`f(x::vector[n]) = …` makes `n` available in the body); `@deffun`
for deterministic control-flow programs; `@inline` UDFs with caller-scope
mutation; `@stan_assert`; and `return_type_of(f, args...)` for transpile-time
type/shape queries.

**Closures** — see [below](#closures); a headline capability, especially for
scientific models.

**Distributions** — author your own distribution triad (`_lpdf` / `_lpdfs` /
`_rng`, opted into `~` sampling with `@lhs`/`@lpxf`); distribution
higher-order functions `weighted`, `truncated`, `censored`, and
`interval_censored`; and fused GLM families (`normal_id`, `bernoulli_logit`,
`poisson_log`, `neg_binomial_2_log`).

**Scientific-computing surface** — ODE solvers (`ode_rk45`, `ode_bdf`, …),
[Torsten](https://metrumresearchgroup.github.io/Torsten/)-style pharmacometrics
signatures, Gaussian-process helpers, and `reduce_sum` for within-chain
parallelism.

**Generated quantities, for free** — automatic pointwise log-likelihood and
posterior-predictive draws, plus automatic imputation of partly-missing
*continuous* outcomes (Turing-style).

**Reflection & ergonomics** — executable model descriptors with derived
fit / predict / log-likelihood operations; user-defined types (`@usertype`,
NamedTuples, ragged carriers); user-written Julia macros expand transparently
inside `@slic` / `@deffun` bodies (`@views`, `@.`, `@inbounds`, …); and
approximate automatic formatting à la
[Blue](https://github.com/JuliaDiff/BlueStyle).

## Closures

Closures ship, and they are what make scientific models — ODEs above all —
read like ordinary Julia:

```julia
closure_model = @slic (; ts, yobs) begin
    lambda ~ exponential(1)
    y = ode_rk45(
        (t, state) -> -lambda * state,
        [1.0], 0.0, to_array_1d(ts),
    )
    yobs ~ normal(y[1][1], 0.05)
end
```

The closure `(t, state) -> -lambda * state` is **lifted into a generated Stan
function**. Captured data and parameters become explicit trailing arguments, and
likelihood activity follows *through* those captures — so `lambda` stays a live
parameter of the model even though it is only referenced from inside the closure
handed to the solver. No runtime function object ever reaches Stan.

## How it works

StanBlocks descends from [SlicStan](https://github.com/mgorinova/SlicStan): the
same destination (clean Stan), a different route. One traced model is lowered by
four compiler verbs — **`capture`**, **`forward!`**, **`backward!`**, and
**`distribute!`** — into a single dependency graph, then emitted (via
`Base.show`) at each execution time. Downstream, the compiled log-density is
differentiated with [Mooncake](https://github.com/chalk-lab/Mooncake.jl).

Almost anything expressible in Julia can be transpiled to Stan. Whether you
*should* is a separate question: unless Stan is meaningfully faster than
Julia + Mooncake for your model, staying in Julia keeps many advantages.

See the [user guide](https://nsiccha.github.io/StanBlocks.jl/dev/) for `@deffun`,
`@inline` UDFs, sub-models, posterior pointwise likelihood / predictive draws,
the full built-in catalog, and a regression-DSL-scale worked example.

## Honest current boundaries

The boundary is intentional and explicit: `@slic` is flat declarative code,
deterministic control flow belongs in `@deffun`, and `plate` is an
independent-cell parameter loop rather than a general scan. Not supported:

* ordinary `for` / `if` / `while` / `&&` / `||` / ternary / comprehensions in a
  `@slic` body — move them into a `@deffun`, or use a vectorised form;
* a scan whose cell `i` consumes cell `i-1`;
* matrix-valued `plate` cells and cell-to-cell dependencies;
* general 3-D-and-higher Julia containers;
* missing *predictors* or *discrete* outcomes; ragged *integer* observations;
* `target +=` in a UDF, and top-level control flow / mutability;
* auto-transpiling Julia functions not defined via `@deffun`;
* full Julia-runtime parity for Stan's probability / RNG / ODE / parallel
  built-ins.

See the [authoring support guide](https://nsiccha.github.io/StanBlocks.jl/dev/authoring)
for the full works / does-not-work matrix.

`@deffun` definitions are Stan-only by default. Add `@juliacompat` to an
eligible deterministic, bodyful bare-symbol definition to also install one
callable Julia method from the user-facing signature, while keeping the same
Stan lowering; `@stanonly` is the explicit opt-out marker. Signature-only /
type-token glue, qualified extensions of existing functions, and
probability / RNG / ODE-family definitions (`*_lpdf`, `*_lpmf`, `*_rng`,
`ode_*`, …) remain Stan-only.

# Caveats

## Constant terms in the log density

Stan's `~` statement
[drops constants](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement).
StanBlocks does **not** — it always emits the full log density. The absolute
value differs from a hand-written Stan model, but the posterior geometry is
identical and sampling is unaffected.

## No control flow at the model level

`for` / `while` / `if` / `&&` / `||` / ternary / comprehensions are not allowed
in `@slic` bodies. Move that logic into a `@deffun` body, or use a vectorised
form.
