# Soil carbon modeling

This page reproduces the two statistical models in Bob Carpenter's
[soil-carbon case study](https://mc-stan.org/learn-stan/case-studies/soil-knit.html).
Both models describe carbon moving between two soil pools and leaving the
system as evolved CO2.  They differ only in whether the reported CO2 means are
treated as direct observations of that process or as noisy summaries of latent
sample-level values.

The short time series below is a build fixture rather than the case study's
experimental data.  The process equations, initial condition, priors, and two
observation models are preserved.  Open either comparison to see the complete
generated Stan program next to the exact Julia source evaluated by this docs
build; readers do not have to run the model to reveal the Stan.

## The two-pool process model

Let ``C_1(t)`` and ``C_2(t)`` denote carbon in the two soil pools.  Carbon
leaves pool 1 at rate ``k_1 C_1`` and pool 2 at rate ``k_2 C_2``, while
fractions ``\alpha_{21}`` and ``\alpha_{12}`` feed those losses back into
the other pool:

```math
\frac{dC_1}{dt} = -k_1 C_1 + \alpha_{12}k_2C_2,
\qquad
\frac{dC_2}{dt} = \alpha_{21}k_1C_1 - k_2C_2.
```

The mixing fraction ``\gamma`` divides the initial total carbon between the
pools.  At each observation time, cumulative evolved CO2 is the initial total
minus the carbon still present:

```math
C_1(0)=\gamma C_0,\qquad
C_2(0)=(1-\gamma)C_0,\qquad
\widehat{\mathrm{CO2}}(t)=C_0-C_1(t)-C_2(t).
```

In StanBlocks, the ODE right-hand side is a typed, Stan-only `@deffun`.
`ode_rk45` uses Stan's current variadic interface, so the physical parameters
are passed directly instead of being packed into the legacy `theta`, `x_r`,
and `x_i` arrays.  The solver returns a time-by-state matrix; selecting its
two columns and converting them with `to_vector` gives the process prediction.

The reusable `soil_dynamics` submodel owns both the positive rate/feedback
parameters and the ODE solve.  A parent model writes
`eCO2_hat ~ soil_dynamics(; ...)`: this is SLIC composition, not a probability
statement.  It inlines the submodel hygienically and binds its returned
prediction to `eCO2_hat`.

## Direct residual model

The first model compares the reported mean at each time directly with the ODE
prediction.  Its positive `sigma` consequently absorbs both measurement
uncertainty and process/model discrepancy.  Call-time keyword arguments become
Stan data, while names first introduced on the left of `~` become parameters
or transformed quantities.  StanBlocks infers their shapes from the ODE result
and likelihood, and emits the declared lower bounds into Stan.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:SoilCaseStudy), raw"""
using StanBlocks

@deffun @stanonly begin
    two_pool_feedback(
        t::real, carbon::vector[ny],
        k1::real, k2::real, alpha21::real, alpha12::real,
    )::vector[ny] = begin
        dcarbon::vector[ny]
        dcarbon[1] = -k1 * carbon[1] + alpha12 * k2 * carbon[2]
        dcarbon[2] = alpha21 * k1 * carbon[1] - k2 * carbon[2]
        dcarbon
    end
end

soil_data = (;
    totalC_t0=7.7,
    t0=0.0,
    ts=[1.0, 2.0, 3.0, 4.0],
    eCO2mean=[0.3, 0.8, 1.5, 2.1],
)
soil_measurement_data = Base.merge(
    soil_data,
    (; eCO2sd=[0.10, 0.10, 0.15, 0.20]),
)

soil_dynamics = @slic begin
    k1 ~ std_normal(; lower=0)
    k2 ~ std_normal(; lower=0)
    alpha21 ~ std_normal(; lower=0)
    alpha12 ~ std_normal(; lower=0)
    gamma ~ beta(10, 1)

    carbon0 = append_row(
        rep_vector(gamma * totalC_t0, 1),
        (1 - gamma) * totalC_t0,
    )
    trajectory = ode_rk45(
        two_pool_feedback, carbon0, t0, to_array_1d(ts),
        k1, k2, alpha21, alpha12,
    )
    return totalC_t0 -
        to_vector(trajectory[:, 1]) -
        to_vector(trajectory[:, 2])
end

soil_measurement = @slic begin
    eCO2_hat ~ soil_dynamics(; totalC_t0, t0, ts)
    sigma ~ cauchy(0, 1; lower=0)
    eCO2mean ~ normal(eCO2_hat, sigma)
end
soil_measurement_posterior = soil_measurement(; soil_data...)
""", :soil_measurement_posterior)
```

```@raw html
</div>
```

## Separate measurement error

The extension keeps the process model unchanged and inserts a positive latent
`eCO2` value at every observation time.  The process residual scale `sigma`
describes variation around the ODE trajectory; the known `eCO2sd` values then
describe uncertainty in the reported means.  This is the distinction the
direct model cannot make.

`Base.merge` constructs the second data set from the common process data plus
the measurement standard deviations.  The latent vector's length is tied
explicitly to `ts`, and `lower = 0` preserves the original model's physical
support.  Most importantly, the model reuses the same `soil_dynamics`
submodel rather than copying the rates, priors, initial condition, and ODE.
The generated-Stan view shows the full inlined program despite that source-level
composition.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:SoilCaseStudy), raw"""
soil_latent = @slic begin
    eCO2_hat ~ soil_dynamics(; totalC_t0, t0, ts)
    sigma ~ cauchy(0, 1; lower=0)
    eCO2 ~ normal(eCO2_hat, sigma; n=length(ts), lower=0)
    eCO2mean ~ normal(eCO2, eCO2sd)
end
soil_latent_posterior = soil_latent(; soil_measurement_data...)
""", :soil_latent_posterior)
```

```@raw html
</div>
```

These pages reproduce the model definitions rather than the original case
study's fitting and plotting workflow.  The source case study remains the place
to compare posterior predictive trajectories and diagnose how the two error
models allocate uncertainty.
