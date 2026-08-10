# Multiple species--site occupancy

This page reproduces Bob Carpenter's [Dorazio--Royle multiple-species occupancy
case study](https://mc-stan.org/learn-stan/case-studies/dorazio-royle-occupancy.html)
from its [authoritative `stan-dev/example-models`
notebook](https://github.com/stan-dev/example-models/blob/master/knitr/dorazio-royle-occupancy/dorazio-et-al-knitr.Rmd).
The original StanBlocks page retained only an empty data tuple and a partial
model body. The executable model below restores the likelihood helpers,
hierarchical priors, discrete-state marginalisation, and generated quantities
from the source program.

## Statistical model

For species ``i`` and site ``j``, ``x_{ij}`` is the number of detections in
``K`` visits. Three latent binary states explain a zero:

- ``w_i`` says whether species ``i`` belongs to the regional community, with
  ``w_i \sim \operatorname{Bernoulli}(\Omega)``;
- ``z_{ij}`` says whether an available species occupies site ``j``, with
  logit-scale occurrence ``\operatorname{logit}(\psi_i)``;
- conditional on occupancy, detections follow
  ``\operatorname{Binomial}(K,\theta_i)``.

A positive count proves site occupancy. A zero must instead sum the occupied
but undetected and unoccupied possibilities with `log_sum_exp`. Species never
detected anywhere require one more marginalisation: they may be unavailable
from the region, or available but unobserved at every site. These sums remove
all discrete states from the HMC parameter vector exactly as in the original
Stan model.

Species-specific occurrence and detection logits are correlated random
effects:

```math
(u_i,v_i) \sim \mathcal N(0,\Sigma),\qquad
\operatorname{logit}(\psi_i)=\alpha+u_i,\qquad
\operatorname{logit}(\theta_i)=\beta+v_i.
```

The two marginal scales have half-Cauchy priors; their correlation has the
same transformed `beta(2,2)` prior as the source. `Omega ~ beta(2,2)` regularises
the fraction of a finite superpopulation of size `S` that is regionally
available.

## What changes in the StanBlocks spelling

- Model-level loops, branches, and direct `target +=` are deliberately absent
  from `@slic`. The exact joint marginal likelihood is therefore a custom
  `@lhs @lpxf` distribution whose typed `@deffun` body contains the source
  program's loops and `if` branch. This changes the organisation, not the log
  density.
- The readable `detection_table` is flattened row by row before tracing. That
  gives the custom distribution an `int[n]` observation and lets its sized
  `_rng` companion generate a complete replicated detection table. The table's
  statistical indexing remains ``(i-1)J+j``.
- The original fixes the correlated-effect dimension at two and constrains its
  count/dimension inputs in `data`. Here those relationships are symbolic data
  sizes so the displayed source remains reusable, while `@stan_assert` checks
  the same two-effect, table-shape, count-range, and superpopulation invariants
  before the likelihood is evaluated.
- The source parameter `beta` is named `detection_intercept` here so calls to
  the beta distribution stay visually unambiguous. Generated Stan otherwise
  preserves the same prior and linear predictor.
- The source samples `rho_uv` directly and writes `(rho_uv + 1) / 2 ~ beta(2,2)`.
  StanBlocks samples `rho_uv_unit` on ``[0,1]`` and transforms it to
  `rho_uv = 2 * rho_uv_unit - 1`. The omitted Jacobian is the constant `log(2)`,
  so the normalized posterior is unchanged.
- `E_N` is the model-based expectation ``S\Omega``; `E_N_2` samples the
  posterior availability of every never-detected species; and `sim_uv` plus
  its two logits describe a new species. All are emitted as generated
  quantities because the corresponding `_rng` calls do not feed a likelihood.

The mock table below establishes types and dimensions only. Rebind the model
to the full 28-species, 20-site, 18-visit data from the source notebook for the
published analysis.

Use the tabs to inspect the exact Julia source evaluated by this documentation
build and the complete generated Stan program. **Compare side by side** opens
the feature-atlas modal.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:SpeciesOccupancyCaseStudy), raw"""
using StanBlocks

@deffun @stanonly begin
    occupancy_covariance(
        scales::vector[n_effects], correlation::real,
    )::matrix[n_effects, n_effects] = begin
        @stan_assert n_effects == 2 "occupancy_covariance requires two effects"
        covariance::matrix[n_effects, n_effects]
        covariance[1, 1] = square(scales[1])
        covariance[2, 2] = square(scales[2])
        covariance[1, 2] = scales[1] * scales[2] * correlation
        covariance[2, 1] = covariance[1, 2]
        covariance
    end

    occupancy_logit(
        uv::vector[n_species, n_effects], column::int, intercept::real,
    )::vector[n_species] = begin
        result::vector[n_species]
        for i in 1:n_species
            result[i] = uv[i, column] + intercept
        end
        result
    end

    occupancy_lp_observed(
        detection::int, visits::int,
        logit_psi::real, logit_theta::real,
    )::real =
        log_inv_logit(logit_psi) +
        binomial_logit_lpmf(detection, visits, logit_theta)

    occupancy_lp_unobserved(
        visits::int, logit_psi::real, logit_theta::real,
    )::real =
        log_sum_exp(
            occupancy_lp_observed(0, visits, logit_psi, logit_theta),
            log_inv_logit(-logit_psi),
        )

    occupancy_lp_never_observed(
        n_sites::int, visits::int,
        logit_psi::real, logit_theta::real, Omega::real,
    )::real = begin
        lp_unavailable = bernoulli_lpmf(0, Omega)
        lp_available =
            bernoulli_lpmf(1, Omega) +
            n_sites * occupancy_lp_unobserved(
                visits, logit_psi, logit_theta,
            )
        log_sum_exp(lp_unavailable, lp_available)
    end

    @lhs @lpxf community_occupancy_lpmf(
        detections::int[n_cells],
        n_observed::int, n_sites::int, visits::int,
        superpopulation::int,
        logit_psi::vector[superpopulation],
        logit_theta::vector[superpopulation], Omega::real,
    )::real = begin
        @stan_assert n_cells == n_observed * n_sites "detections must be n_observed by n_sites"
        @stan_assert visits >= 1 "visits must be positive"
        @stan_assert n_observed >= 1 "n_observed must be positive"
        @stan_assert superpopulation >= n_observed "superpopulation must include observed species"
        lp = 0.0
        for i in 1:n_observed
            lp += bernoulli_lpmf(1, Omega)
            for j in 1:n_sites
                detection = detections[(i - 1) * n_sites + j]
                @stan_assert detection >= 0 "detections must be nonnegative"
                @stan_assert detection <= visits "detections cannot exceed visits"
                if detection > 0
                    lp += occupancy_lp_observed(
                        detection, visits, logit_psi[i], logit_theta[i],
                    )
                else
                    lp += occupancy_lp_unobserved(
                        visits, logit_psi[i], logit_theta[i],
                    )
                end
            end
        end
        for i in (n_observed + 1):superpopulation
            lp += occupancy_lp_never_observed(
                n_sites, visits, logit_psi[i], logit_theta[i], Omega,
            )
        end
        lp
    end

    community_occupancy_lpmfs(detections::int[n_cells], args...) =
        community_occupancy_lpmf(detections, args...)

    community_occupancy_rng(
        int[n_cells],
        n_observed::int, n_sites::int, visits::int,
        superpopulation::int,
        logit_psi::vector[superpopulation],
        logit_theta::vector[superpopulation], Omega::real,
    )::int[n_cells] = begin
        replicated::int[n_cells]
        for i in 1:n_observed
            for j in 1:n_sites
                index = (i - 1) * n_sites + j
                if bernoulli_logit_rng(logit_psi[i]) == 1
                    replicated[index] = binomial_rng(
                        visits, inv_logit(logit_theta[i]),
                    )
                else
                    replicated[index] = 0
                end
            end
        end
        replicated
    end

    occupancy_species_count_rng(
        n_observed::int, n_sites::int, visits::int,
        superpopulation::int,
        logit_psi::vector[superpopulation],
        logit_theta::vector[superpopulation], Omega::real,
    )::int = begin
        species_count = n_observed
        for i in (n_observed + 1):superpopulation
            lp_unavailable = bernoulli_lpmf(0, Omega)
            lp_available =
                bernoulli_lpmf(1, Omega) +
                n_sites * occupancy_lp_unobserved(
                    visits, logit_psi[i], logit_theta[i],
                )
            probability_available = exp(
                lp_available - log_sum_exp(lp_unavailable, lp_available),
            )
            species_count += bernoulli_rng(probability_available)
        end
        species_count
    end
end

detection_table = [
    1 0 2
    0 1 0
]
detections = vec(permutedims(detection_table))
n_observed = size(detection_table, 1)
n_sites = size(detection_table, 2)
visits = 4
superpopulation = 4
n_effects = 2

species_occupancy = @slic (;
    detections, n_observed, n_sites, visits, superpopulation, n_effects,
) begin
    alpha ~ cauchy(0.0, 2.5)
    detection_intercept ~ cauchy(0.0, 2.5)
    sigma_uv::vector[n_effects] ~ cauchy(0.0, 2.5; lower=0.0)
    rho_uv_unit ~ beta(2.0, 2.0)
    rho_uv = 2.0 * rho_uv_unit - 1.0
    covariance = occupancy_covariance(sigma_uv, rho_uv)
    uv::vector[superpopulation, n_effects] ~
        multi_normal(rep_vector(0.0, n_effects), covariance)
    Omega ~ beta(2.0, 2.0)

    logit_psi = occupancy_logit(uv, 1, alpha)
    logit_theta = occupancy_logit(uv, 2, detection_intercept)
    detections ~ community_occupancy(
        n_observed, n_sites, visits, superpopulation,
        logit_psi, logit_theta, Omega,
    )

    E_N = superpopulation * Omega
    E_N_2 = occupancy_species_count_rng(
        n_observed, n_sites, visits, superpopulation,
        logit_psi, logit_theta, Omega,
    )
    sim_uv = multi_normal_rng(rep_vector(0.0, n_effects), covariance)
    logit_psi_sim = alpha + sim_uv[1]
    logit_theta_sim = detection_intercept + sim_uv[2]
end
""", :species_occupancy)
```

```@raw html
</div>
```

The source case study goes on to fit the full dataset and reconstruct the
posterior distribution of total species richness. This page keeps that model
and its predictive quantities intact, but does not present numerical results
from the tiny documentation data as though they were scientific estimates.
