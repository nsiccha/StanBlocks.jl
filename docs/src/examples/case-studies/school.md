# Disease transmission at a boarding school

This case study follows Léo Grinsztajn and collaborators' [boarding-school
influenza case study](https://mc-stan.org/learn-stan/case-studies/boarding_school_case_study.html).
The [authoritative source
notebook](https://github.com/stan-dev/example-models/blob/master/knitr/disease_transmission/boarding_school_case_study.Rmd)
fits a susceptible--infectious--removed (SIR) system to daily counts of pupils
in bed. The first model below reproduces that Stan program. The other two
complete the observation-model variants that the original StanBlocks notebook
had only sketched: infections acquired during each day, then reported
infections when only a fraction of cases are observed.

All three models use the same latent epidemic. If ``S(t)``, ``I(t)``, and
``R(t)`` are the three compartments and ``N`` is the school population, then

```math
\frac{dS}{dt}=-\beta IS/N,\qquad
\frac{dI}{dt}=\beta IS/N-\gamma I,\qquad
\frac{dR}{dt}=\gamma I.
```

Here ``\beta`` is the infection rate, ``\gamma`` is the removal rate,
``R_0=\beta/\gamma``, and ``1/\gamma`` is the mean infectious period. A
negative-binomial observation model allows more day-to-day variation than a
Poisson model; `phi_inv` is given an exponential prior and inverted to obtain
Stan's `neg_binomial_2` precision ``\phi``.

## The three observation models

### Infectious prevalence

The official case study treats the number of pupils in bed on day ``t`` as a
noisy measurement of ``I(t)``. This is the right interpretation when “in bed”
is a snapshot of current infectious prevalence. The posterior-predictive count
and pointwise log likelihood are generated automatically from the observed
`cases ~ neg_binomial_2(...)` statement.

### Daily incidence

If the recorded count instead means *new* infections during a day, its latent
mean is the susceptible decrement ``S(t-1)-S(t)``. The second model changes
only that observation statement. It therefore has one fewer observation than
the prevalence model: a decrement needs two adjacent solver states. The
incidence variants bind that shorter count vector directly, so their generated
predictive arrays have the same length and every element is initialized.

### Reported daily incidence

The third model estimates a reporting fraction ``p_{reported}`` with a
`beta(1,2)` prior and scales incidence by it. This separates infections from
observed reports, though the two can be weakly identified without external
information about reporting.

## StanBlocks implementation

- `school_sir_rhs` is a typed `@deffun`: loops and mutation belong in helper
  functions, while the `@slic` body remains a flat probabilistic declaration.
- Current Stan uses the variadic `ode_rk45` interface. The legacy `theta`,
  `x_r`, and `x_i` packing arrays from the source Stan program are unnecessary;
  `infection_rate`, `gamma`, and `population` are ordinary trailing arguments.
- The source parameter `beta` is named `infection_rate`, leaving the builtin
  `beta(...)` distribution unshadowed when the reporting variant declares
  `p_reported`.
- The three `@slic` definitions deliberately keep the complete model visible.
  Their only statistical difference is the final observation statement (and
  `p_reported` in the third); the shared ODE helper keeps the differential
  equations themselves in one place.
- The small Julia values establish types and shapes for documentation
  transpilation. They are not the boarding-school data used for inference;
  rebind the resulting model to the complete case-study data before fitting.

Use the tabs to inspect the exact Julia source evaluated by this documentation
build and each complete generated Stan program. **Compare side by side** opens
the feature-atlas modal.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main.FeatureAtlasDocs.example_module(:BoardingSchoolCaseStudy), raw"""
using StanBlocks

@deffun @stanonly school_sir_rhs(
    t::real, state::vector[n_state],
    infection_rate::real, removal_rate::real, population::real,
)::vector[n_state] = begin
    susceptible = state[1]
    infectious = state[2]

    derivative::vector[n_state]
    derivative[1] = -infection_rate * infectious * susceptible / population
    derivative[2] =
        infection_rate * infectious * susceptible / population -
        removal_rate * infectious
    derivative[3] = removal_rate * infectious
    derivative
end

school_data = (;
    y0=[762.0, 1.0, 0.0],
    t0=0.0,
    times=[1.0, 2.0, 3.0, 4.0],
    population=763.0,
    cases=[3, 8, 15, 20],
    n_days=4,
)
school_incidence_data = Base.merge(
    school_data,
    (; cases=school_data.cases[1:(school_data.n_days - 1)]),
)

school_prevalence = @slic school_data begin
    infection_rate::real ~ normal(2.0, 1.0; lower=0.0)
    gamma::real ~ normal(0.4, 0.5; lower=0.0)
    phi_inv::real ~ exponential(5.0; lower=0.0)

    R0 = infection_rate / gamma
    recovery_time = 1.0 / gamma
    phi = 1.0 / phi_inv
    trajectory = ode_rk45(
        school_sir_rhs, y0, t0, to_array_1d(times),
        infection_rate, gamma, population,
    )
    prevalence = to_vector(trajectory[:, 2])
    susceptible = to_vector(trajectory[:, 1])
    incidence =
        susceptible[1:(n_days - 1)] - susceptible[2:n_days]

    cases ~ neg_binomial_2(prevalence, phi)
end

school_incidence = @slic school_incidence_data begin
    infection_rate::real ~ normal(2.0, 1.0; lower=0.0)
    gamma::real ~ normal(0.4, 0.5; lower=0.0)
    phi_inv::real ~ exponential(5.0; lower=0.0)

    R0 = infection_rate / gamma
    recovery_time = 1.0 / gamma
    phi = 1.0 / phi_inv
    trajectory = ode_rk45(
        school_sir_rhs, y0, t0, to_array_1d(times),
        infection_rate, gamma, population,
    )
    susceptible = to_vector(trajectory[:, 1])
    incidence =
        susceptible[1:(n_days - 1)] - susceptible[2:n_days]

    cases ~ neg_binomial_2(incidence, phi)
end

school_reported_incidence = @slic school_incidence_data begin
    infection_rate::real ~ normal(2.0, 1.0; lower=0.0)
    gamma::real ~ normal(0.4, 0.5; lower=0.0)
    phi_inv::real ~ exponential(5.0; lower=0.0)
    p_reported ~ beta(1.0, 2.0)

    R0 = infection_rate / gamma
    recovery_time = 1.0 / gamma
    phi = 1.0 / phi_inv
    trajectory = ode_rk45(
        school_sir_rhs, y0, t0, to_array_1d(times),
        infection_rate, gamma, population,
    )
    susceptible = to_vector(trajectory[:, 1])
    incidence =
        susceptible[1:(n_days - 1)] - susceptible[2:n_days]

    cases ~ neg_binomial_2(incidence * p_reported, phi)
end

school_models = (;
    infectious_prevalence=school_prevalence,
    daily_incidence=school_incidence,
    reported_daily_incidence=school_reported_incidence,
)
""", :school_models)
```

```@raw html
</div>
```

The original article continues through fitting, diagnostics, prior-predictive
simulation, and posterior-predictive checks. This page reproduces the model
definitions rather than claiming those numerical analyses for the tiny shape
data above.
