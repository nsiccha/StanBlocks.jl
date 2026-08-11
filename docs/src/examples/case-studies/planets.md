# Planetary motion

This page reproduces the three models from Charles C. Margossian and
Andrew Gelman's [planetary-motion case study](https://mc-stan.org/learn-stan/case-studies/planetary_motion.html):
the forward simulator, the one-parameter inverse problem, and the full model
with unknown initial conditions and star position. The authoritative source is
the [`stan-dev/example-models` directory](https://github.com/stan-dev/example-models/tree/master/knitr/planetary_motion/model).

The statistical progression is the important part of the case study. Start
with a known orbit and simulate noisy positions; then infer only the
gravitational interaction ``k``; finally unfix the initial position, initial
momentum, and star position. Keeping the three models together makes it clear
which additional posterior modes enter as parameters are released. Unlike the
three standalone upstream Stan files, the StanBlocks version defines the
orbital dynamics and observation model once. Each case then supplies known
values through explicit fixed bindings and replaces only the statements that
really differ.

## Model sequence

### 1. Forward simulation (`planetary_motion_sim.stan`)

The planet starts at ``q_0=(1,0)`` with momentum ``p_0=(0,1)``. The planetary
mass and gravitational interaction are fixed, so the ODE trajectory is completely
determined. The only random quantities of interest are noisy ``x`` and ``y``
position measurements.

### 2. Infer the gravitational interaction (`planetary_motion.stan`)

The initial state and star position remain fixed, but ``k`` receives a
half-normal prior. This is the deliberately simplified inverse problem used to
diagnose the case study's multimodality. As in the upstream file, this variant
uses the tolerance-controlled BDF solver.

### 3. Infer the full system (`planetary_motion_star.stan`)

The final model estimates ``k``, both coordinates of ``q_0`` and ``p_0``, and
the star position. As in the original, the second momentum coordinate is
positive through a lognormal prior and ``k`` has the physically informed,
tight `normal(1, 0.001)` prior.

## What changes in the StanBlocks spelling

- One `planetary_rhs` accepts the star coordinates explicitly. Its argument,
  return, and local types are inferred from the call and assignments: type and
  shape annotations in `@deffun` are optional. Fixing the star at the origin
  gives the first two models; estimating it gives the third.
- `planetary` is the only `@slic` definition, including every prior, the shared
  dynamics, and the observations. `Base.merge(planetary, (; q0=fixed_q0, ...))`
  explicitly fixes selected names: it removes their matching sampling or
  assignment statements and stores the supplied values as model data. The
  mixed form `Base.merge(model, quote ... end, (; x=value, ...))` can change a
  solver or prior and fix values in one operation. No intermediate template is
  needed.
- Ordinary `planetary(; q0=fixed_q0)` still means data binding only: it leaves
  an existing `q0 ~ ...` statement in place as a likelihood contribution. The
  explicit `Base.merge` form is therefore what distinguishes fixing a quantity
  from conditioning on it.
- The shared `ode_rk45_tol` call uses Stan's current variadic tolerance
  interface, so the legacy `theta`, `x_r`, and empty `x_i` packing arrays
  disappear.
- The common model names its coordinate observations `qx` and `qy`. Binding
  those names as data produces the two inverse-model likelihoods. Leaving them
  unbound in the simulator makes the same statements fresh draws, which
  StanBlocks moves to generated quantities. No indexed sampling LHS is needed.
- Constants that the originals write in `transformed data` are supplied as
  Julia values with the same fixed shapes. The statistical models are
  unchanged, although the generated data blocks are not textually identical.

Use the tabs to switch between the exact Julia source evaluated by this docs
build and all three complete generated Stan programs. **Compare side by side**
opens the same material in the feature-atlas modal.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main.FeatureAtlasDocs.example_module(:PlanetaryMotionCaseStudy), raw"""
using StanBlocks

@deffun @stanonly planetary_rhs(t, state, k, star_x, star_y, m) = begin
    q1 = state[1] - star_x
    q2 = state[2] - star_y
    r_cube = (q1^2 + q2^2)^1.5
    dstate = state
    dstate[1] = state[3] / m
    dstate[2] = state[4] / m
    dstate[3] = -k * q1 / r_cube
    dstate[4] = -k * q2 / r_cube
    dstate
end

n = 4
times = collect(0.1:0.1:0.4)
qx_obs = zeros(n)
qy_obs = zeros(n)
fixed_q0 = [1.0, 0.0]
fixed_p01 = 0.0
fixed_p02 = 1.0
fixed_star = [0.0, 0.0]
fixed_k = 1.0
m = 1.0
sigma_x = sigma_y = sigma = 0.01

planetary = @slic begin
    k ~ normal(1, 0.001; lower=0)
    q0 ~ normal(0, 1; n=2)
    p01 ~ normal(0, 1)
    p02 ~ lognormal(0, 1)
    star ~ normal(0, 0.5; n=2)

    p0 = append_row(rep_vector(p01, 1), p02)
    initial_state = append_row(q0, p0)
    trajectory = ode_rk45_tol(
        planetary_rhs, initial_state, 0.0, to_array_1d(times),
        1e-6, 1e-6, 1000, k, star[1], star[2], m,
    )
    qx ~ normal(to_vector(trajectory[:, 1]), sigma_x)
    qy ~ normal(to_vector(trajectory[:, 2]), sigma_y)
end

planetary_sim = Base.merge(planetary, quote
    trajectory = ode_rk45(
        planetary_rhs, initial_state, 0.0, to_array_1d(times),
        k, star[1], star[2], m,
    )
end, (;
    q0=fixed_q0, p01=fixed_p01, p02=fixed_p02,
    star=fixed_star, k=fixed_k,
))(; times, m, sigma_x, sigma_y)

planetary_k = Base.merge(planetary, quote
    k ~ normal(0, 1; lower=0)
    trajectory = ode_bdf_tol(
        planetary_rhs, initial_state, 0.0, to_array_1d(times),
        1e-6, 1e-6, 1000, k, star[1], star[2], m,
    )
end, (;
    q0=fixed_q0, p01=fixed_p01, p02=fixed_p02,
    star=fixed_star,
))(;
    times, m, sigma_x, sigma_y, qx=qx_obs, qy=qy_obs,
)

planetary_star = planetary(;
    times,
    m, sigma_x=sigma, sigma_y=sigma, qx=qx_obs, qy=qy_obs,
)

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
