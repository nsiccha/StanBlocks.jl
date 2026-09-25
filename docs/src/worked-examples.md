# Worked examples

These pages turn the former Quarto notebooks into a guided tour of model
authoring with StanBlocks. Runnable examples are evaluated during the
documentation build: the Julia source shown on a page is the exact source that
produced the complete Stan program shown with it.

## How to read the executable examples

Every generated example uses the same presentation as the
[feature atlas](feature-atlas.md):

1. read the model in the **StanBlocks** tab;
2. switch to **Generated Stan** to inspect the complete emitted program; or
3. choose **Compare side by side** for a wide modal with both versions.

Family pages evaluate one source block and label every resulting Stan program.
Nothing asks the reader to run Julia in order to see the output. The prose
before each block explains the statistical progression and calls out the DSL
features that make it possible.

## Model families

- [Golf models](examples/golf-models.md) progresses from logistic regression to
  geometry-based putting models, residual variation, and estimated physical
  tolerances. It highlights `Base.merge`, inferred declarations, and ordinary
  Julia preprocessing.
- [PCR sensitivity versus time](examples/isba-2024.md) builds a 5×2 family from
  five latent-time structures and two link functions. It highlights
  `@deffun`, higher-order dispatch, varargs, custom likelihoods, and recursive
  model-family construction.
- [Crowdsourced ratings](examples/crowdsource.md) builds a latent-truth/rater
  model and 18 restrictions of it. It highlights custom likelihood families,
  generated checks, and post-hoc component replacement.
- [Reusable constraints](examples/constraints.md) covers a disk transform and
  ten simplex transforms. It highlights named-tuple returns, function-valued
  arguments, custom parameter-introducing distributions, and Jacobian terms.

## Case studies

- [Golf putting](examples/case-studies/golf.md) — composition through reusable
  probability submodels.
- [Motorcycle data](examples/case-studies/motorcycle.md) — a Hilbert-space GP
  component reused for both the mean and log scale.
- [Multilevel radon regression](examples/case-studies/radon.md) — complete, no,
  and partial pooling plus the CV marker.
- [Planetary motion](examples/case-studies/planets.md) — the original forward
  simulator, `k`-only inverse problem, and full unknown-star ODE model.
- [Disease transmission](examples/case-studies/school.md) — an SIR system with
  prevalence, incidence, and under-reported-incidence observation models.
- [Multiple species-site occupancy](examples/case-studies/species.md) — the
  original discrete-state-marginalized occupancy likelihood and generated
  abundance quantities.
- [Soil carbon](examples/case-studies/soil.md) — a two-pool feedback ODE reused
  by direct-residual and latent measurement-error observation models.
- [Wastewater renewal model](examples/case-studies/wastewater.md) — the CDC
  `ww-inference-model` as a composable modeling ladder: two observation
  submodels (admissions + wastewater), a renewal core with a swappable Rt
  process (RW / diff-AR(1) / sparse-CSR spline) and shedding kernel, and a
  multi-subpopulation capstone with tuple-returning renewal and per-site effects.
- [EpiSewer composable library](examples/case-studies/episewer.md) — the R
  `EpiSewer` monolith and `EpiSewer.jl` components realized as one StanBlocks
  library: a renewal core with swappable Rt process (RW / sparse-CSR spline) and
  observation family (LOD-censored concentration / digital-PCR counts via
  `to_int`), assembled by `Base.merge`.
- [Monster pharmacokinetics](examples/monster.md) — a four-compartment PBPK
  model expressed both directly in StanBlocks and through BRM's named
  subject-level formula predictors and group-local kernel.

## Maintained implementation references

- [PosteriorDB implementations](examples/posteriordb-implementations.md)
  inventories the 77 current optional-extension models and explains how the
  catalogue is organized without duplicating its maintained source file.

## Design and historical material

- [The original `@slic` design overview](examples/slic-overview.md)
- [Simplex transform experiments](examples/simplex-transforms.md)

These historical pages are curated records rather than raw dumps: each says
what the experiment was trying to demonstrate, which parts correspond to
current StanBlocks features, and where the original exploration intentionally
stopped.
