# Grey-seal integrated population model

This is the capstone case study: a full **integrated population model** (IPM)
for the Baltic grey seal, ported from the
[`n-kall/sealIPM`](https://github.com/n-kall/sealIPM) Stan program. An IPM fits
one latent population process to *several* independent data streams at once —
here **eight** of them, from aerial pup counts to hunting-bag totals to
reproductive-tract signs — so the shared demographic parameters are informed by
everything observed about the population.

It exercises nearly the whole StanBlocks surface in one model: a **year-recursive
state process** (a scan), a **numerical ODE** solved inside that scan, a custom
**Dirichlet-multinomial fate allocation**, **six** user-defined distribution
families, **two** observation-stream submodels, and **named-tuple** state
carriers. Because the pieces are many and interdependent, the model is assembled
from a small library of cards with
[`compile_slic_bundle`](@ref) rather than one inline `@slic` block — 31
`@deffun` helper cards, two anonymous observation-stream submodels, and one
parent body. Everything below is evaluated at documentation-build time, so the
displayed Julia is the exact source that produced the ~52 KB Stan program beside
it. The arrays are a tiny build fixture (3 age classes × 2 sexes × 3 years), not
real survey data.

!!! note "Companion port in BayesianRegressionModels.jl"
    The **same** grey-seal IPM is also ported in BRM, onto its StanBlocks
    backend:
    [Grey-seal IPM](https://nsiccha.github.io/BayesianRegressionModels.jl/dev/seal-ipm).
    Both port the identical [`n-kall/sealIPM`](https://github.com/n-kall/sealIPM)
    source verbatim through `compile_slic_bundle`.

## The shape of the model

The parent body is short and readable — it declares the demographic and
observation parameters, computes a handful of transformed quantities, runs the
state process **once**, and then attaches each observation stream as a single
line.

- **The state process is a scan.** `run_state_process` walks the years forward:
  each year updates the density-dependent birth rate, the age/sex population
  composition, the Sweden/Finland hunting pressure (whose expected catch is an
  `ode_rk45_tol` integral of a hunting-hazard ODE, `dH_dt`), and a stochastic
  split of every demographic class into *survived / bycaught / hunted-SE /
  hunted-FI* via `multinomial_allocation`. A recurrence like this cannot live in
  `@slic` (no control flow) — it is a `@deffun`, called loop-free from the body.
- **It returns a named tuple of state carriers.** `run_state_process` returns
  `(; birth_rate, pregnancy_rate, population_total, hunted_sweden, …,
  reproductive_probs)`, and the body reads them **by name** —
  `state.population_total`, `state.bycatch_expected`. (StanBlocks named tuples
  are authoring sugar that lower to Stan's positional tuple access; see the
  wastewater study's tuple note.)
- **Eight observation streams, each one removable line.** Every stream is a
  Form-A observation (`data ~ family(...)` or `data ~ submodel(...)`): the real
  observed vector is on the left, so deleting the line drops that stream (and a
  submodel node drops its own parameters too). Two streams that carry their own
  parameters — the aerial count's over-dispersion, the bycatch composition's
  selectivity bias — are **submodels**; the other six call **custom families**
  directly.
- **Six custom distribution families.** `aerial_count` (negative-binomial),
  `harvest_bags` (normal with a shared CV), `hunting_comp` / `bycatch_comp` /
  `reproductive_signs` (multinomial compositions), and `pregnancy` (binomial)
  are each defined as a `@deffun` `_lpmf`/`_lpmfs`/`_rng` triad (the density card
  carries `@lhs @lpxf`), so the model gets posterior-predictive draws and
  pointwise log-likelihoods for every stream for free.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Grey-seal IPM">
```

```@eval
Main.FeatureAtlasDocs.grey_seal_comparison(; show_udfs = [
    "run_state_process",     # the year-recursive state scan
    "dH_dt",                 # hunting-hazard ODE RHS (ode_rk45_tol)
    "multinomial_allocation", # Dirichlet-approx stochastic fate split
    "aerial_count_lpmf",     # one custom density family (the @lhs @lpxf triad head)
])
```

```@raw html
</div>
```

The Julia panes above are, in order: the **parent `@slic` body**, then four
representative `@deffun` cards — the state-process scan, the ODE right-hand side,
the fate-allocation helper, and one custom-family density head. The full library
(all 31 UDF cards + the two observation-stream submodels) is vendored verbatim in
[`docs/grey_seal_ipm.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/docs/grey_seal_ipm.jl);
they all appear in the generated Stan's `functions {}` block. The Stan pane is
the complete emitted program.

## What this exercises

- **`compile_slic_bundle`** — a multi-source workspace (31 UDF cards + two
  anonymous submodels + a parent body) assembled and traced in one call, the
  natural shape for a model too large for a single inline block.
- **A year-recursive scan** (`run_state_process`) in a `@deffun`, with
  **named-tuple state carriers** read by field (`state.population_total`).
- **A numerical ODE inside the scan** — `ode_rk45_tol` over a `@deffun`
  hunting-hazard right-hand side, per demographic class per year.
- **A custom Dirichlet-multinomial fate allocation** (`multinomial_allocation`)
  using `digamma` / `trigamma` / `cholesky_decompose` / `softmax`.
- **Six user-defined distribution families** (`_lpmf`/`_lpmfs`/`_rng` triads with
  `@lhs @lpxf`) driving negative-binomial, normal, binomial, and multinomial
  observation likelihoods — each with automatic posterior-predictive and
  pointwise-log-likelihood twins.
- **Observation submodels** (`data ~ submodel(...)`) and direct-family
  observations (`data ~ family(...)`), eight streams in total, each added or
  dropped as one line.
- A full executable **descriptor** — the assembled model offers
  `fit` / `predict` / `pointwise_loglik`.
