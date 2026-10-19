# Activity analysis: one model, three roles

You write **one** `@slic` model. Depending on **what data you bind — and how — that
same source becomes** a prior-predictive simulator, a posterior fit, or a
cross-validation / population-prediction model. You never write three separate
models, and there is no mode flag: the roles fall out of a static analysis of the
traced body.

::: tip What this page is
[Data binding](index.md#data-binding) and the [Activity analysis](index.md#activity-analysis)
block-placement table show *where* each variable lands. This page explains the rule
underneath — how binding, omitting, or marking data reshapes the whole program.
:::

## The two passes

Two passes over the traced body do all the work:

1. **Likelihood reachability.** StanBlocks walks the body and marks every variable
   that flows into an observation's `~`. A parameter that has a prior but reaches
   **no** likelihood is dead as a fit target, so it moves to `generated_quantities`
   and is re-drawn from its RNG instead of being sampled. Reachability is computed
   as if *every* observation were bound, so it is a property of the model, not of
   the particular data you pass.

2. **Cross-validation taint.** `StanBlocks.stan.maybecv(:name, value)` marks a bound
   input as *held out*. The mark is **contagious** — it propagates through every
   expression the marked input reaches. A parameter tainted this way relocates to
   `generated_quantities` (re-drawn from its prior); a tainted observation is
   **dropped from the model block** — its likelihood is held out — while its
   predictive and pointwise-log-likelihood companions still emit. Untainted
   parameters are left fitted.

## One model

```julia
m = @slic begin
    mu    ~ std_normal()
    tau   ~ std_normal(; lower = 0.)
    J     = maximum(subject)
    alpha ~ normal(mu, tau; n = J)
    sigma ~ std_normal(; lower = 0.)
    y     ~ normal(alpha[subject], sigma)
end
```

The three sections below are the **actual generated Stan** for that one model under
three different data bindings.

### 1. Bind the outcome → posterior

```julia
m(; subject, y)
```

Every parameter is sampled and the observation contributes a likelihood; StanBlocks
also emits the automatic pointwise log-likelihood and predictive draw.

```stan
parameters {
    real mu;
    real<lower=0.0> tau;
    vector[J] alpha;
    real<lower=0.0> sigma;
}
model {
    mu ~ std_normal();
    tau ~ std_normal();
    alpha ~ normal(mu, tau);
    sigma ~ std_normal();
    y ~ normal(alpha[subject], sigma);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, alpha[subject], sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, alpha[subject], sigma);
}
```

Available operations: `:transpile`, `:instantiate`, `:fit`, `:predict`,
`:pointwise_loglik`.

### 2. Omit the outcome → prior predictive

```julia
m(; subject)          # no `y`
```

With no observation bound there is no likelihood, so *every* parameter is dead as a
fit target. The `parameters` and `model` blocks are **empty**, and the entire model
— parameters and the outcome `y` alike — forward-simulates in `generated_quantities`,
each prior as an `_rng` draw in dependency order, the transforms after them.

```stan
parameters {
}
model {
}
generated quantities {
    real mu = std_normal_rng();
    real tau = lower_conditioning_normal_rng(0.0, 0.0, 1.0);
    vector[J] alpha = normal_vector_rng(J, mu, tau);
    real sigma = lower_conditioning_normal_rng(0.0, 0.0, 1.0);
    array[subject_n] real y = normal_rng(alpha[subject], sigma);
}
```

A prior's own `lower=` / `upper=` bound is a truncation of it: `tau ~ std_normal(;
lower = 0.)` is a half-normal, so its draw goes through the same rejection sampler
the `truncated(...)` distribution combinator uses, never a bare `std_normal_rng()`
(which would be negative half the time). Bounds the family already implies
(`exponential` ⇒ `lower = 0`) need no truncation and draw natively.

This holds for the whole program. What relocates is decided by likelihood
reachability — any parameter no likelihood reaches, however deep in a transform
chain — never by whether a parameter is a "leaf". So a `plate` (its fresh
per-cell samples, collected result and compiler-owned loop), an inlined helper's
element fills, and every transformed-parameter chain feeding them lower to
`generated quantities` together. The compiled program therefore has
`LogDensityProblems.dimension(prob) == 0` and is Stan's `fixed_param` case — draw
exact prior samples in milliseconds with an empty parameter vector, no
adaptation:

```julia
prob = stan_instantiate(m(; subject))
draw = BridgeStan.param_constrain(prob.model, Float64[]; include_tp = true, include_gq = true,
                                  rng = BridgeStan.StanRNG(prob.model, seed))
```

Every prior that is re-drawn needs its family's `_rng` companion for that shape
(a custom `@lpxf foo_lpdf` family ships `foo_rng`, sized-token overload
included); a missing one is a trace-time error naming the symbol, the family and
the signature to add, and an improper `flat()` prior — nothing to draw from — is
an error too. Two prior shapes deliberately stay *sampled* parameters instead,
because no exact draw exists yet: an `ordered` / `positive_ordered` prior (no
family rng yields a sorted vector) and a ragged constrained parameter with an
informative prior (its per-group constrain step has no ragged rng). Everything
prior-only around them still lowers to `generated quantities`; such a program
merely keeps `dimension > 0`.

Available operations shrink to `:transpile`, `:instantiate` — there is nothing to
fit, only a prior to simulate.

### 3. Mark the group index → cross-validation

```julia
m(; subject = StanBlocks.stan.maybecv(:subject, subject), y)
```

Marking `subject` taints `J = maximum(subject)`, therefore `alpha` (whose size is
`J`), therefore the `y ~` term (it reads `alpha[subject]`). So `alpha` leaves the
`parameters` block and is **re-drawn from its prior** in `generated_quantities`, and
the `y` likelihood is **dropped** from the model. `mu`, `tau`, and `sigma` are not
tainted, so they stay parameters.

```stan
parameters {
    real mu;
    real<lower=0.0> tau;
    real<lower=0.0> sigma;
}
model {
    mu ~ std_normal();
    tau ~ std_normal();
    sigma ~ std_normal();
}
generated quantities {
    vector[J] alpha = normal_vector_rng(J, mu, tau);
    vector[y_n] y_likelihood = normal_lpdfs(y, alpha[subject], sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, alpha[subject], sigma);
}
```

Here `subject` is the only data and everything flows from it, so the **whole**
dataset is held out: no likelihood term remains, so `:fit` is **not** offered
(operations are `:transpile`, `:instantiate`, `:predict`, `:pointwise_loglik`).

Use this program with the draws of an earlier fit. `mu`, `tau`, and `sigma` stay
parameters so that the fitted draws can be supplied for them by unconstrained
parameter name; `alpha` is then re-drawn for new groups from the fitted population.
That is population prediction. Sampled on its own, this program gives the retained
parameters only their priors. Mark *one* observation input while another stays
bound, and sampling the program keeps the shared population parameters fitted by
the retained likelihood: that is leave-group-out cross-validation.

### Marking a response is not omitting it

The cross-validation mark and an omitted outcome are two different transformations.
For

```julia
m2 = @slic begin
    mu      ~ std_normal()
    sigma_y ~ std_normal(; lower = 0.)
    sigma_z ~ std_normal(; lower = 0.)
    y       ~ normal(mu, sigma_y)
    z       ~ normal(mu, sigma_z)
end
```

- `m2(; y, z = StanBlocks.stan.maybecv(:z, z))` drops the `z ~` term from the model
  block and keeps its `z_gen` and `z_likelihood` companions. Nothing that `z` reads is
  tainted, so `mu`, `sigma_y`, and `sigma_z` all stay parameters, and an earlier fit's
  draws supply them. Sampled on its own, this program gives `sigma_z` only its prior.
- `m2(; y)` does not condition on `z` at all. Everything that no remaining likelihood
  reaches moves to `generated_quantities`, as in section 2: `sigma_z` is re-drawn from
  its prior and `z` is simulated, while `mu` and `sigma_y` stay fitted through `y`.
  A producer that needs the `z_gen` name declares `z` as an observation
  (`StanBlocks.SlicModel(body, data, mod, (:z,))`).

To fit a model without conditioning on a response, omit the response. To predict a
held-out response from an earlier fit, mark it.

### Holding out some elements of a response

`maybecv(:z, z)` holds out every element of `z`. To fit on some elements and score
the others (a forecast window, or a block of test rows), mark only those elements:

```julia
m2(; y, z = StanBlocks.stan.maybecv(:z, z; held_out = t .> cutoff))
```

- The model block keeps the likelihood of the unmarked elements only:
  `z ~ heldout_normal(z_held_out, mu, sigma_z)`. A marked element's density is
  skipped, not multiplied by zero, and for a family whose density is elementwise in
  its arguments it is never evaluated at all. That covers the scalar built-in
  families (`normal`, `lognormal`, `gamma`, `poisson`, `binomial`, …), and
  `weighted`, `truncated`, `censored` and `interval_censored` over one.
  So a marked value outside the family's support cannot turn the target into `NaN`,
  including where Stan raises an error instead of returning `-Inf` (`lognormal` or
  `poisson` below 0, `gamma` at 0, `beta` outside [0, 1], `binomial` above its
  trials).
- A family with a shared or row-structured argument (`categorical(p)`,
  `categorical_logit`, `ordered_logistic`, the fused GLMs, a custom `@lpxf` family)
  is scored whole and its marked entries are dropped, so its marked values must lie
  where its density accepts them.
- No cv taint is set, so every parameter stays a parameter and the unmarked elements
  fit it.
- Every element gets its `z_gen` draw, and a marked element's `z_likelihood` entry is
  its log predictive density. A ragged response keeps one `z_likelihood` entry per
  group. Where Stan would raise an error for a value, that entry is `-Inf`, scored
  without calling the family, so a marked element outside the support cannot fail
  the generated-quantities draw either. This applies to the scalar built-in families
  with such values, alone or `weighted`. For every other family, `z_gen` and
  `z_likelihood` are the same as without the mark.
- The mask has the response's shape: a `Bool` vector for a vector response, one
  `Bool` vector per group for a ragged response. It is indexed the same way as the
  response, so the `z[i] ~ …` cells of a `@plate for` (and the cells of a `plate`
  do-block over `z`) hold out their marked elements. Stan reads the mask as the data
  input `z_held_out`, which the descriptor marks `derived`. Re-binding `z` on a
  traced model keeps the mask when it still fits, and re-binding it as
  `maybecv(:z, z; held_out = mask)` replaces it.
- The mark only removes likelihood terms. Any other read of `z`, such as a lagged
  response used as a covariate, still sees every element.
- A joint family such as `multi_normal` has no per-element densities, so a mask on
  its response is refused; hold out the whole response instead.

::: tip The descriptor sees this
`stan_descriptor(model)` reports it directly. Each input's `held_out` flag is
contagion-aware — marking one input routinely flips several to `held_out` — and the
`operations` set is derived from the same analysis, which is exactly why `:fit`
disappears once every observation is held out. See
[executable model descriptors](authoring.md#executable-model-descriptors).
:::

## Random draws are generated quantities

Stan allows a random number generator only in transformed data, generated
quantities, and functions whose names end in `_rng`. StanBlocks puts every draw in
generated quantities: a call to an `_rng` function is a generated quantity, and so
is everything computed from it. This holds for a builtin such as `normal_rng` and
for your own `@deffun foo_rng`. The parameters a draw reads stay sampled.

```julia
@deffun shift_rng(m::real)::real = normal_rng(m, 1.0)

m3 = @slic begin
    mu0 ~ std_normal()
    b   = shift_rng(mu0)
    y   ~ normal(b, 1.0)
end
```

With a held-out response, `m3(; y = StanBlocks.stan.maybecv(:y, y))`, the draw and
the response's predictive and pointwise-log-likelihood companions all read the same
`b` in generated quantities:

```stan
parameters {
    real mu0;
}
model {
    mu0 ~ std_normal();
}
generated quantities {
    real b = shift_rng(mu0);
    vector[y_n] y_likelihood = normal_lpdfs(y, b, 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, b, 1.0);
}
```

With an ordinary observation, `m3(; y)` fails at trace time. A likelihood term is
evaluated in the model block, and a target that depends on a fresh draw at every
evaluation is not a log density. The error names the draw and the held-out spelling.

The same rule covers the other places a draw can appear:

- A parameter whose prior or declared size reads a draw (`z ~ normal(b, 1.0)`) is
  re-drawn from that prior in generated quantities, like a parameter tainted by a
  cross-validation mark.
- A draw inside a `plate` or `@plate` cell is made once per cell. It is never
  hoisted out of the loop as cell-invariant.
- A draw whose arguments are all data is a generated quantity too: a fresh draw
  per iteration, written to the output. This matches the `~` spelling of the same
  draw, which a likelihood does not reach and so is re-drawn in generated
  quantities. A likelihood cannot condition on a draw (`y_sim ~ normal(mu, 1.0)`
  after `y_sim = vector_std_normal_rng(n)` is an error). To fit simulated data,
  simulate it with the prior-predictive program (omit the response, as in
  section 2), then bind the draw as data.
- An `_rng` call written inline in a held-out observation's right-hand side is
  evaluated separately in each companion. Bind the draw to a name first when both
  companions must read the same draw.

## A note on the two composition tools

Omitting an outcome is not the same as *fixing* a parameter. Binding a sampled name
through an ordinary kwarg (`m(; theta = value)`) turns it into observed data and
keeps its `~` statement as a likelihood; `Base.merge(m, (; theta = value))` instead
**removes** the statement and stores the value as data. See
[Data binding](index.md#data-binding).

## Both size spellings carry the taint

`alpha ~ normal(mu, tau; n = J)` and the typed-LHS `alpha :: vector[J] ~ normal(mu,
tau)` behave identically, including under cross-validation: a `maybecv` mark on the
size input `J` propagates through the declared size either way, so `alpha` relocates
to `generated quantities` — re-drawn from its prior — the same in both spellings. Use
whichever reads better.
