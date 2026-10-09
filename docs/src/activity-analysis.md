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

::: tip The descriptor sees this
`stan_descriptor(model)` reports it directly. Each input's `held_out` flag is
contagion-aware — marking one input routinely flips several to `held_out` — and the
`operations` set is derived from the same analysis, which is exactly why `:fit`
disappears once every observation is held out. See
[executable model descriptors](authoring.md#executable-model-descriptors).
:::

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
