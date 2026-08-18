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
— parameters and the outcome `y` alike — forward-simulates in `generated_quantities`.

```stan
parameters {
}
model {
}
generated quantities {
    real mu = std_normal_rng();
    real tau = std_normal_rng();
    vector[J] alpha = normal_vector_rng(J, mu, tau);
    real sigma = std_normal_rng();
    array[subject_n] real y = normal_rng(alpha[subject], sigma);
}
```

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
(operations are `:transpile`, `:instantiate`, `:predict`, `:pointwise_loglik`), and
the retained parameters simply sample their priors — a population-level predictive.
Mark *one* observation input while another stays bound, and the retained likelihood
keeps the shared population parameters genuinely fitted: that is leave-group-out
cross-validation, expressed by binding alone.

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
