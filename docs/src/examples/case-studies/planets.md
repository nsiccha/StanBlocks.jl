# Planetary motion

This page reproduces the three Stan programs from Charles C. Margossian and
Andrew Gelman's [planetary-motion case study](https://mc-stan.org/learn-stan/case-studies/planetary_motion.html):
the forward simulator, the one-parameter inverse problem, and the full model
with unknown initial conditions and star position. The authoritative source is
the [`stan-dev/example-models` directory](https://github.com/stan-dev/example-models/tree/master/knitr/planetary_motion/model).

The statistical progression is the important part of the case study. Start
with a known orbit and simulate noisy positions; then infer only the
gravitational interaction ``k``; finally unfix the initial position, initial
momentum, and star position. Keeping the three models together makes it clear
which additional posterior modes enter as parameters are released.

## Model sequence

### 1. Forward simulation (`planetary_motion_sim.stan`)

The planet starts at ``q_0=(1,0)`` with momentum ``p_0=(0,1)``. The planetary
mass and gravitational interaction are fixed, so the ODE trajectory is completely
determined. The only random quantities of interest are noisy ``x`` and ``y``
position measurements.

### 2. Infer the gravitational interaction (`planetary_motion.stan`)

The initial state and star position remain fixed, but ``k`` receives a
half-normal prior. This is the deliberately simplified inverse problem used to
diagnose the case study's multimodality. It retains the original BDF solver and
its explicit tolerances.

### 3. Infer the full system (`planetary_motion_star.stan`)

The final model estimates ``k``, both coordinates of ``q_0`` and ``p_0``, and
the star position. As in the original, the second momentum coordinate is
positive through a lognormal prior and ``k`` has the physically informed,
tight `normal(1, 0.001)` prior.

## What changes in the StanBlocks spelling

- The two Hamiltonian right-hand sides are ordinary typed `@deffun`
  definitions. `ode_rk45_tol` and `ode_bdf_tol` use Stan's current variadic ODE
  interface, so the legacy `theta`, `x_r`, and empty `x_i` packing arrays
  disappear; physical arguments are passed by name and position.
- The initial state and observation arrays are ordinary Julia data bound with
  `model(; data...)`; StanBlocks emits the required Stan data and transformed
  data declarations after tracing their use. In this documentation fixture,
  constants that the original writes in `transformed data` are supplied as
  Julia values with the same fixed shapes; the statistical model is unchanged,
  although the generated data block is not textually identical.
- In the simulation model, fresh `qx` and `qy` draws affect no likelihood.
  StanBlocks' activity analysis therefore emits them as generated-quantity RNG
  draws automatically. The original Stan file needed an unrelated dummy
  parameter only to make the program sampleable.
- The inverse models use the same coordinate-wise normal likelihood as the
  originals. Writing `q_obs[:, 1] ~ ...` and `q_obs[:, 2] ~ ...` lets the
  compiler generate the matching predictive quantities as well.

Use the tabs to switch between the exact Julia source evaluated by this docs
build and all three complete generated Stan programs. **Compare side by side**
opens the same material in the feature-atlas modal.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main.FeatureAtlasDocs.example_module(:PlanetaryMotionCaseStudy), raw"""
using StanBlocks

@deffun @stanonly begin
    planetary_origin_rhs(t::real, state::vector[ny], m::real, k::real)::vector[ny] = begin
        q1 = state[1]
        q2 = state[2]
        r_cube = (q1^2 + q2^2)^1.5
        dstate::vector[ny]
        dstate[1] = state[3] / m
        dstate[2] = state[4] / m
        dstate[3] = -k * q1 / r_cube
        dstate[4] = -k * q2 / r_cube
        dstate
    end

    planetary_star_rhs(
        t::real, state::vector[ny], k::real,
        star_x::real, star_y::real, m::real,
    )::vector[ny] = begin
        q1 = state[1] - star_x
        q2 = state[2] - star_y
        r_cube = (q1^2 + q2^2)^1.5
        dstate::vector[ny]
        dstate[1] = state[3] / m
        dstate[2] = state[4] / m
        dstate[3] = -k * q1 / r_cube
        dstate[4] = -k * q2 / r_cube
        dstate
    end
end

n = 4
times = collect(0.1:0.1:0.4)
q_obs = zeros(n, 2)
initial_state = [1.0, 0.0, 0.0, 1.0]
m = 1.0
sigma_x = sigma_y = sigma = 0.01

planetary_sim = @slic (; times, initial_state, m, sigma_x, sigma_y) begin
    trajectory = ode_rk45(
        planetary_origin_rhs, initial_state, 0.0, to_array_1d(times), m, 1.0,
    )
    qx ~ normal(to_vector(trajectory[:, 1]), sigma_x)
    qy ~ normal(to_vector(trajectory[:, 2]), sigma_y)
end

planetary_k = @slic (; times, initial_state, m, q_obs, sigma_x, sigma_y) begin
    k ~ normal(0, 1; lower=0)
    trajectory = ode_bdf_tol(
        planetary_origin_rhs, initial_state, 0.0, to_array_1d(times),
        1e-6, 1e-6, 1000, m, k,
    )
    q_obs[:, 1] ~ normal(to_vector(trajectory[:, 1]), sigma_x)
    q_obs[:, 2] ~ normal(to_vector(trajectory[:, 2]), sigma_y)
end

planetary_star = @slic (; times, m, q_obs, sigma) begin
    k ~ normal(1, 0.001; lower=0)
    q0 ~ normal(0, 1; n=2)
    p01 ~ normal(0, 1)
    p02 ~ lognormal(0, 1)
    star ~ normal(0, 0.5; n=2)

    p0 = append_row(rep_vector(p01, 1), p02)
    y0 = append_row(q0, p0)
    trajectory = ode_rk45_tol(
        planetary_star_rhs, y0, 0.0, to_array_1d(times),
        1e-6, 1e-6, 1000, k, star[1], star[2], m,
    )
    q_obs[:, 1] ~ normal(to_vector(trajectory[:, 1]), sigma)
    q_obs[:, 2] ~ normal(to_vector(trajectory[:, 2]), sigma)
end

planetary_models = (;
    simulation=planetary_sim,
    infer_k=planetary_k,
    infer_initial_state_and_star=planetary_star,
)
""", :planetary_models)
```

```@raw html
</div>
```

The page reproduces the model definitions, not the long MCMC and plotting
workflow around them. The original case study remains the right source for the
multichain diagnostics, initialisation experiments, and posterior-predictive
plots that motivate this sequence.
