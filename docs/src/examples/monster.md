# Monster pharmacokinetic model

The Monster model is a hierarchical physiologically based pharmacokinetic
(PBPK) model for the uptake and elimination of tetrachloroethylene.  Its name
comes from A. C. Monster, first author of the experimental study that motivated
the later Bayesian analysis.  The authoritative implementation material is
Niko Siccha's [`nsiccha/monster`](https://github.com/nsiccha/monster)
repository.  In particular,
[`flexible_monster.stan`](https://github.com/nsiccha/monster/blob/a362738efd9c83525a2edb6db3e3646fba1a293f/stan/flexible_monster.stan)
contains the model used for the reported fits, while the later
[`ragged.stan`](https://github.com/nsiccha/monster/blob/a362738efd9c83525a2edb6db3e3646fba1a293f/stan/ragged.stan)
and
[`basis.stan`](https://github.com/nsiccha/monster/blob/a362738efd9c83525a2edb6db3e3646fba1a293f/stan/basis.stan)
make the experiment indexing and reusable physiology especially clear.

This page uses those programs as design sources rather than attempting a
line-for-line translation.  The source repository includes alternative
population parameterisations, centred/non-centred switches, prior-only and
incremental-likelihood branches, a custom Strang-splitting integrator, and a
BDF solver.  The examples below retain the scientific core, choose one clear
hierarchy, and port the source's Strang-splitting path.  That makes the
StanBlocks and BRM versions small enough to compare without presenting
implementation switches as parts of the biological model.

## From physiology to four state equations

Each subject has four tissue concentration states.  Blood flow carries material
between those tissues; inhalation adds a source during the exposure phase, and
a saturating Michaelis--Menten term removes material in the final compartment.
For tissue ``j``, the source implementations write the dynamics as

```math
\frac{dC_j}{dt}
= FVP_j\left(\sum_k FFPF_k C_k + C_{\mathrm{exposure}} - C_j\right)
+ \mathbb{1}[j=4]\frac{-V_{MI}C_4}{K_{MI}+C_4}.
```

After the first measurement time—240 minutes in these experiments—the
inhalation source is removed and the same system describes washout.  The model
does not observe the four tissue states directly.  It derives venous and
exhaled concentrations from the solved trajectory and compares both with
measurements on a lognormal scale.

## The fifteen individual parameters

The source model's fifteen positive quantities are not anonymous coefficients:

| Positions | Names | Role |
| --- | --- | --- |
| 1 | `VPR` | pulmonary-to-venous flow ratio |
| 2–5 | `Fwp`, `Fpp`, `Ff`, `Fl` | tissue fractions of venous flow |
| 6–8 | `Vwp`, `Vpp`, `Vl` | lean-tissue volume fractions; fat volume is measured separately |
| 9 | `Pba` | blood/air partition coefficient |
| 10–13 | `Pwp`, `Ppp`, `Pf`, `Pl` | tissue/blood partition coefficients |
| 14–15 | `VMI`, `KMI` | Michaelis--Menten elimination parameters |

The four flow fractions are normalised with `softmax`.  The first two lean
volume fractions share the non-fat/non-liver mass left after transforming
`Vl`.  Everything else is exponentiated from an unconstrained log-scale
quantity.  This preserves the physically meaningful support without copying
the source repository's optional constraint switches.

The three measured subject covariates are lean body mass, fat-mass fraction,
and pulmonary volume flow.  They determine absolute compartment volumes and
flows inside the simulator; they are not additional fitted coefficients.

## Direct StanBlocks model

The direct version mirrors the handwritten Stan organization:

- `monster_lambert_w0_exp` and `monster_exact_mm_step` implement the stable
  closed-form Michaelis--Menten update from the source;
- `monster_experiment` transforms one subject's parameters, alternates those
  nonlinear half-steps with exact matrix-exponential transport steps, and
  derives the two observables while carrying the trajectory and checkpoint
  state through the source's nested `while` loops;
- `monster_subject` evaluates both exposure experiments and flattens their
  outputs in a documented experiment → output → time order;
- a subject `plate` supplies the non-centred 15-vector and collects one
  prediction vector per person;
- the two elements of `sigma` remain distinct venous and exhaled noise scales.

The documentation fixture uses the first two people, four representative time
points, and both exposure concentrations from the source data.  It is a compile
fixture, not a replacement scientific fit.  Open the comparison to see the
exact Julia evaluated during this documentation build and the complete Stan
program it generated.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated direct StanBlocks model">
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MonsterCaseStudy), raw"""
using StanBlocks

@deffun @stanonly begin
    monster_lambert_w0_exp(earg::real)::real = begin
        if is_nan(earg)
            return earg
        end
        if earg > 700
            return lambert_w0(exp(700.0)) + (earg - 700.0) *
                lambert_w0(exp(700.0)) / (lambert_w0(exp(700.0)) + 1)
        end
        if earg < -40
            return lambert_w0(exp(-40.0)) * exp(earg + 40.0)
        end
        return lambert_w0(exp(earg))
    end

    monster_exact_mm_step(
        dt::real, concentration::real, vmax::real, km::real,
    )::real = begin
        minimum_concentration = 1e-12
        if concentration <= minimum_concentration
            return minimum_concentration
        end
        if km == 0
            return concentration - dt * vmax
        end
        earg = (dt * vmax + concentration) / km + log(concentration / km)
        return km * monster_lambert_w0_exp(earg)
    end

    monster_log_interpolate(
        fraction::real, left::vector[n_state], right::vector[n_state],
    )::vector[n_state] = begin
        minimum_concentration = 1e-12
        return exp(
            (1 - fraction) * log(minimum_concentration + left) +
            fraction * log(minimum_concentration + right)
        )
    end

    monster_experiment(
        times::vector[n_time], exposure::real,
        raw_params::vector[n_param], measured::vector[n_measured],
        n_state::int, n_output::int, n_substeps::int,
    )::matrix[n_time, n_output] = begin
        minimum_concentration = 1e-12
        lean_body_mass = measured[1]
        fat_mass_fraction = measured[2]
        pulmonary_flow = measured[3]

        body_mass = lean_body_mass / (1 - fat_mass_fraction)
        fat_volume = fat_mass_fraction * body_mass / 0.92
        alveolar_flow = 0.7 * pulmonary_flow

        vpr = exp(raw_params[1])
        unit_tissue_flow = softmax(raw_params[2:5])
        liver_fraction = 0.837 * inv_logit(raw_params[8])
        first_two_volumes = (0.837 - liver_fraction) * softmax(raw_params[6:7])
        tissue_volume::vector[n_state]
        tissue_volume[1:2] = lean_body_mass * first_two_volumes
        tissue_volume[3] = fat_volume
        tissue_volume[4] = lean_body_mass * liver_fraction
        pba = exp(raw_params[9])
        partition = exp(raw_params[10:13])
        effective_volume = tissue_volume .* partition
        venous_flow = alveolar_flow / vpr
        tissue_flow = unit_tissue_flow * venous_flow
        flow_over_volume = tissue_flow ./ effective_volume
        pulmonary_flow_total = venous_flow + alveolar_flow / pba
        flow_fraction = tissue_flow / pulmonary_flow_total
        exposure_source = alveolar_flow * exposure / pulmonary_flow_total
        vmax = -(lean_body_mass^0.7) * exp(raw_params[14]) / effective_volume[n_state]
        km = exp(raw_params[15]) / effective_volume[n_state]

        transport = add_diag(
            flow_over_volume * flow_fraction', -flow_over_volume,
        )
        source_equilibrium = exposure_source * (transport \ flow_over_volume)
        dt = times[1] / n_substeps
        transition = matrix_exp(dt * transport)
        transition_source = transition * source_equilibrium - source_equilibrium

        concentration = rep_vector(minimum_concentration, n_state)
        last_concentration = concentration
        states::vector[n_time, n_state]
        last_time = 0.0
        next_time = 0.0
        time_index = 1
        next_checkpoint = times[time_index]
        while time_index <= n_time
            next_time = last_time + dt
            concentration[n_state] = monster_exact_mm_step(
                dt / 2, concentration[n_state], vmax, km,
            )
            if time_index == 1
                concentration = transition * concentration + transition_source
            else
                concentration = transition * concentration
            end
            concentration[n_state] = monster_exact_mm_step(
                dt / 2, concentration[n_state], vmax, km,
            )

            while next_time >= next_checkpoint
                states[time_index] = monster_log_interpolate(
                    (next_checkpoint - last_time) / dt,
                    last_concentration, concentration,
                )
                if time_index == 1
                    concentration = states[time_index]
                    next_time = times[time_index]
                end
                time_index = time_index + 1
                if time_index <= n_time
                    next_checkpoint = times[time_index]
                else
                    break
                end
            end
            last_time = next_time
            last_concentration = concentration
        end

        prediction::matrix[n_time, n_output]
        for time in 1:n_time
            venous = dot_product(unit_tissue_flow, states[time])
            inhaled = time == 1 ? exposure : 0.0
            alveolar = (inhaled + venous) / (vpr + pba)
            exhaled = 0.7 * alveolar + 0.3 * inhaled
            prediction[time, 1] = minimum_concentration + venous
            prediction[time, 2] = minimum_concentration + exhaled
        end
        prediction
    end

    monster_subject(
        times::vector[n_time], exposures::vector[n_experiment],
        raw_params::vector[n_param], measured::vector[n_measured],
        n_state::int, n_output::int, n_substeps::int,
        n_subject_observation::int,
    )::vector[n_subject_observation] = begin
        prediction::vector[n_subject_observation]
        index = 1
        for experiment in 1:n_experiment
            experiment_prediction::matrix[n_time, n_output] = monster_experiment(
                times, exposures[experiment], raw_params, measured,
                n_state, n_output, n_substeps,
            )
            for output in 1:n_output
                for time in 1:n_time
                    prediction[index] = experiment_prediction[time, output]
                    index += 1
                end
            end
        end
        prediction
    end

    monster_observation_scale(
        sigma::vector[n_output], n_time::int,
        n_experiment::int, n_subject_observation::int,
    )::vector[n_subject_observation] = begin
        scale::vector[n_subject_observation]
        index = 1
        for experiment in 1:n_experiment
            for output in 1:n_output
                for time in 1:n_time
                    scale[index] = sigma[output]
                    index += 1
                end
            end
        end
        scale
    end
end

n_person = 2
n_state = 4
n_output = 2
n_param = 15
times = [240.0, 360.0, 1320.0, 2760.0]
n_substeps = 240
exposures = [0.488, 0.976]
measured_params = [62.0 0.114 7.6; 71.0 0.134 11.6]
n_subject_observation = length(times) * length(exposures) * n_output

person_1 = vcat(
    [2.8, 0.92, 0.17, 0.082], [0.34, 0.033, 0.0063, 0.00345],
    [5.7, 1.76, 0.36, 0.147], [0.632, 0.058, 0.0129, 0.0052],
)
person_2 = vcat(
    [3.0, 1.2, 0.15, 0.066], [0.345, 0.049, 0.0063, 0.0027],
    [8.8, 2.9, 0.36, 0.19], [0.699, 0.075, 0.0114, 0.0067],
)
observed = hcat(person_1, person_2)

reference = [
    1.6, 0.48, 0.2, 0.07, 0.25,
    0.28, 0.56, 0.033,
    12.0, 4.8, 1.6, 125.0, 4.8, 0.042, 16.0,
]
raw_prior_location = log.(reference)
raw_prior_location[8] = log(reference[8] / (0.837 - reference[8]))

monster_direct = @slic begin
    population_raw_location::vector[n_param] ~ normal(raw_prior_location, 0.35)
    population_raw_scale::vector[n_param] ~ normal(0, 0.30; lower=0)
    prediction::matrix[n_subject_observation, n_person] ~ plate(
        EachRow(measured_params); outer=n_person,
    ) do measured
        z::vector[n_param] ~ std_normal()
        monster_subject(
            times, exposures,
            population_raw_location + population_raw_scale .* z,
            to_vector(measured), n_state, n_output, n_substeps,
            n_subject_observation,
        )
    end
    sigma::vector[n_output] ~ normal(0, 0.25; lower=0)
    observation_scale = monster_observation_scale(
        sigma, length(times), length(exposures), n_subject_observation,
    )
    observed_flat = to_vector(observed)
    observed_flat ~ lognormal(
        log(to_vector(prediction)),
        to_vector(rep_matrix(observation_scale, n_person)),
    )
end

monster_direct_posterior = monster_direct(;
    n_person, n_state, n_output, n_param, n_substeps,
    times, exposures, measured_params, n_subject_observation,
    observed, raw_prior_location,
)
""", :monster_direct_posterior)
```

```@raw html
</div>
```

## BRM version

The BRM/SBBRMI version deliberately reuses `monster_experiment` rather than
reimplementing the physiology.  The change is in how subject variation is
declared:

- each biological quantity is a named linear predictor, so population
  locations can be inspected and given priors by name;
- `(1 | subject)` adds an independent subject deviation to each raw-scale
  quantity, replacing the direct model's hand-written `z` vector and scale
  multiplication;
- `kernel(...)` gathers one row per subject, runs the complete PBPK simulator,
  and owns both ragged time-course likelihoods;
- venous and exhaled measurements remain separate outputs.  They are not
  interleaved behind a compartment mask, because separate observation columns
  make their meaning and their posterior-predictive outputs explicit.

The direct example gives all fifteen subject deviations a learned half-normal
population scale.  BRM's plain random-intercept blocks use its standard scale
prior instead.  This is an intentional prior-level difference, not a change to
the state equations or observation means.

The fixture below again contains two subjects, two exposure experiments, and
four observation times.  `monster_subject_output` is only a layout adapter: it
selects one of the two outputs while preserving experiment → time order for
the ragged BRM columns.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated BRM/SBBRMI model">
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MonsterCaseStudy), raw"""
using BayesianRegressionModels, Distributions

@deffun @stanonly begin
    monster_raw_params(
        vpr::real, fwp::real, fpp::real, ff::real, fl::real,
        vwp::real, vpp::real, vl::real,
        pba::real, pwp::real, ppp::real, pf::real, pl::real,
        vmi::real, kmi::real, n_param::int,
    )::vector[n_param] = begin
        raw_params::vector[n_param]
        raw_params[1] = vpr
        raw_params[2] = fwp
        raw_params[3] = fpp
        raw_params[4] = ff
        raw_params[5] = fl
        raw_params[6] = vwp
        raw_params[7] = vpp
        raw_params[8] = vl
        raw_params[9] = pba
        raw_params[10] = pwp
        raw_params[11] = ppp
        raw_params[12] = pf
        raw_params[13] = pl
        raw_params[14] = vmi
        raw_params[15] = kmi
        raw_params
    end

    monster_subject_output(
        times::vector[n_time], exposures::vector[n_experiment],
        raw_params::vector[n_param], measured::vector[n_measured],
        output::int, n_state::int, n_output::int,
        n_substeps::int,
        n_subject_output::int,
    )::vector[n_subject_output] = begin
        prediction::vector[n_subject_output]
        index = 1
        for experiment in 1:n_experiment
            experiment_prediction::matrix[n_time, n_output] = monster_experiment(
                times, exposures[experiment], raw_params, measured,
                n_state, n_output, n_substeps,
            )
            for time in 1:n_time
                prediction[index] = experiment_prediction[time, output]
                index += 1
            end
        end
        prediction
    end
end

venous_person_1 = vcat(person_1[1:4], person_1[9:12])
venous_person_2 = vcat(person_2[1:4], person_2[9:12])
exhaled_person_1 = vcat(person_1[5:8], person_1[13:16])
exhaled_person_2 = vcat(person_2[5:8], person_2[13:16])
n_subject_output = length(times) * length(exposures)

monster_schedule = (;
    subject = ["person-1", "person-2"],
    times = [copy(times), copy(times)],
    exposures = [copy(exposures), copy(exposures)],
    measured = [collect(measured_params[1, :]), collect(measured_params[2, :])],
    venous_y = [venous_person_1, venous_person_2],
    exhaled_y = [exhaled_person_1, exhaled_person_2],
    n_state = fill(n_state, n_person),
    n_output = fill(n_output, n_person),
    n_param = fill(n_param, n_person),
    n_substeps = fill(n_substeps, n_person),
    n_subject_output = fill(n_subject_output, n_person),
)

monster_brm = @brm monster_schedule begin
    sigma_venous ~ Exponential(0.25)
    sigma_exhaled ~ Exponential(0.25)

    log_VPR ~ 1 + (1 | subject)
    raw_Fwp ~ 1 + (1 | subject)
    raw_Fpp ~ 1 + (1 | subject)
    raw_Ff ~ 1 + (1 | subject)
    raw_Fl ~ 1 + (1 | subject)
    raw_Vwp ~ 1 + (1 | subject)
    raw_Vpp ~ 1 + (1 | subject)
    raw_Vl ~ 1 + (1 | subject)
    log_Pba ~ 1 + (1 | subject)
    log_Pwp ~ 1 + (1 | subject)
    log_Ppp ~ 1 + (1 | subject)
    log_Pf ~ 1 + (1 | subject)
    log_Pl ~ 1 + (1 | subject)
    log_VMI ~ 1 + (1 | subject)
    log_KMI ~ 1 + (1 | subject)

    effect(log_VPR, Intercept) ~ Normal(log(1.6), 0.35)
    effect(raw_Fwp, Intercept) ~ Normal(log(0.48), 0.35)
    effect(raw_Fpp, Intercept) ~ Normal(log(0.20), 0.35)
    effect(raw_Ff, Intercept) ~ Normal(log(0.07), 0.35)
    effect(raw_Fl, Intercept) ~ Normal(log(0.25), 0.35)
    effect(raw_Vwp, Intercept) ~ Normal(log(0.28), 0.35)
    effect(raw_Vpp, Intercept) ~ Normal(log(0.56), 0.35)
    effect(raw_Vl, Intercept) ~ Normal(log(0.033 / (0.837 - 0.033)), 0.35)
    effect(log_Pba, Intercept) ~ Normal(log(12.0), 0.35)
    effect(log_Pwp, Intercept) ~ Normal(log(4.8), 0.35)
    effect(log_Ppp, Intercept) ~ Normal(log(1.6), 0.35)
    effect(log_Pf, Intercept) ~ Normal(log(125.0), 0.35)
    effect(log_Pl, Intercept) ~ Normal(log(4.8), 0.35)
    effect(log_VMI, Intercept) ~ Normal(log(0.042), 0.35)
    effect(log_KMI, Intercept) ~ Normal(log(16.0), 0.35)

    venous_prediction ~ kernel(
        times, exposures, measured, venous_y, exhaled_y,
        n_state, n_output, n_param, n_substeps, n_subject_output,
        log_VPR, raw_Fwp, raw_Fpp, raw_Ff, raw_Fl,
        raw_Vwp, raw_Vpp, raw_Vl,
        log_Pba, log_Pwp, log_Ppp, log_Pf, log_Pl, log_VMI, log_KMI,
    ) do ts, experiment_exposures, measured_values, venous_observed,
         exhaled_observed, state_count, output_count, parameter_count,
         substeps, subject_output_count, vpr, fwp, fpp, ff, fl, vwp, vpp, vl,
         pba, pwp, ppp, pf, pl, vmi, kmi
        raw_params = monster_raw_params(
            vpr, fwp, fpp, ff, fl, vwp, vpp, vl,
            pba, pwp, ppp, pf, pl, vmi, kmi, parameter_count,
        )

        venous = monster_subject_output(
            ts, experiment_exposures, raw_params, measured_values,
            1, state_count, output_count, substeps, subject_output_count,
        )
        exhaled = monster_subject_output(
            ts, experiment_exposures, raw_params, measured_values,
            2, state_count, output_count, substeps, subject_output_count,
        )
        venous_observed ~ lognormal(log(venous), sigma_venous)
        exhaled_observed ~ lognormal(log(exhaled), sigma_exhaled)
        venous
    end
end

monster_brm_backend = SBBRMI(monster_brm; mod=@__MODULE__)
monster_brm_posterior = monster_brm_backend.model
""", :monster_brm_posterior)
```

```@raw html
</div>
```

## What the two interfaces reveal

Both generated programs run the same exposure/washout simulator and derive the
same venous and exhaled means.  Its Strang split applies half of the nonlinear
Michaelis--Menten update, advances the linear tissue transport exactly with
`matrix_exp`, then applies the other nonlinear half.  The stable
`lambert_w0(exp(x))` helper follows the source's separate ordinary, very large,
and very small argument branches.  As in the repository's fitted-model path,
240 substeps over the 240-minute exposure establish a one-minute grid, and that
same step size continues through washout.  Whenever a step crosses an
observation checkpoint, `monster_log_interpolate` geometrically interpolates
the two positive concentration vectors.  At the first checkpoint it also
anchors the trajectory exactly at the exposure/washout boundary.  This is the
source program's global-grid and off-grid-checkpoint algorithm, not a separate
per-observation approximation.

The direct model makes the non-centred vector
algebra compact and exposes complete control over its shared scale prior.  The
BRM model spends more lines naming the fifteen quantities, but those names are
then stable addresses for priors, posterior summaries, and future covariates.
For example, adding a measured effect of body size to a parameter is a formula
change to that one predictor rather than a rewrite of the PBPK kernel.

The source repository's BDF alternative, likelihood-increment switches, and
alternative centred hierarchy remain valuable performance and workflow
experiments.  They are intentionally outside this executable port.  The
essential model is no longer a historical stub: both interfaces express the
four-state mechanism, the source-inspired closed-form/Strang simulator, two
experiments per subject, all fifteen individual parameters, and both
observation channels as build-checked Stan programs.
