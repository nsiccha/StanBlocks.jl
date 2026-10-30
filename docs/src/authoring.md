# Authoring and feature support

This page describes the current `devibe` authoring surface. It complements the
[API reference](api.md) with the semantic contracts that matter when deciding how
to write a model: where control flow belongs, how distribution combinators work,
what ragged data means, and the exact boundary of `plate`.

## Choose the right authoring surface

| Need | Surface | Contract |
|---|---|---|
| Declare a model or reusable sub-model | `@slic` | Flat, straight-line `~`, `=`, typed declarations, sub-model calls, and an optional trailing `return` |
| Write deterministic control flow | `@deffun` | Inference-first signatures with optional type/shape annotations; Stan-compatible loops, branches, mutation, iteration, and bounded comprehensions |
| Repeat a parameter-generating cell | `plate(...) do ... end` | The compiler-owned model loop; fresh cell bindings are promoted to outer storage |
| Combine a base distribution with observation semantics | `weighted`, `truncated`, `censored`, `interval_censored` | One base-family token; the compiler selects density, pointwise, CDF, and RNG companions |
| Bound a prior with Stan's plain kernel | `restricted(family, args...; lower, upper)` | The bounds go on the declaration; the model block keeps the family's statement |
| Integrate a Gaussian latent out of the model block | `z ~ laplace_marginal(prior) do z ... end` | Stan's embedded Laplace approximation; the do block is the latent's likelihood, and its observations keep their twins |
| Reflect or execute a traced declaration | `stan_descriptor`, `stan_execute` | Inputs, outputs, included definitions, and available operations are derived from the model |

An ordinary Julia `for` or `if` is not legal at model level because Stan must
know the complete parameter layout before execution. Put deterministic control
flow in `@deffun`. Use `plate` only when the repeated body itself declares fresh
parameters or per-cell observations.

## Distribution combinators

All four combinators take a bare base-family token. Do not pass `_lpdf`,
`_lpmf`, `_rng`, or CDF function names yourself:

```julia
@slic (; y_weighted, y_truncated, y_censored, y_interval,
         weight, lo, hi) begin
    mu ~ normal(0, 1)

    y_weighted  ~ weighted(normal, weight, mu, 1)
    y_truncated ~ truncated(normal, mu, 1; lower=lo, upper=hi)
    y_censored  ~ censored(normal, mu, 1; lower=lo, upper=hi)
    y_interval  ~ interval_censored(normal, lo, hi, mu, 1)
end
```

| Combinator | Meaning | Parameter LHS? | Predictive draw |
|---|---|---:|---|
| `weighted(family, w, args...)` | Multiply an observation's log density by data-qualified `w`; `family` may be a bounded combinator | No | Draw from the unweighted family |
| `truncated(family, args...; lower, upper)` | Condition the latent draw to lie inside the supplied bound(s) | Yes | Redrawn until inside the inclusive bounds, then by inverting the base CDF on the range |
| `censored(family, args...; lower, upper)` | Place tail probability on threshold atoms; interior values keep their ordinary density | No | Clamp the base-family draw to the thresholds |
| `interval_censored(family, lower, upper, args...)` | Observe that the latent value lies in `(lower, upper]` | No | Draw from the uncoarsened base family |
| `restricted(family, args...; lower, upper)` | Bound a parameter's declaration and keep the family's plain kernel (Stan's `real<lower=lo> x; x ~ family(args);`) | Only | The truncated draw, where the law allows it (below) |

For `truncated`, `censored` and `restricted`, at least one of `lower=` or `upper=` is
required. An omitted side and an explicit `nothing` are equivalent and are
removed at compile time. Bounds may be scalar or observation-shaped. Invalid
runtime bounds reject the Stan proposal. `lower` and `upper` are Stan reserved
identifiers, so name bound data `lo`/`hi` (or `lloq`/`uloq`) and map them with
`lower=lo, upper=hi` as above.

A `truncated` predictive draw redraws the base family until a draw lands inside
the bounds. After 100 misses it inverts the base CDF on the range instead, so a
range holding little of the base law's mass, such as a count class far above
the drawn mean, still gives an exact draw: an integer draw is the exact
quantile, a real one is bisected to float resolution inside the family's
support. A lower bound inverts the log CCDF, an upper bound the log CDF, and a
two-sided range the log CCDF when it lies above the base median. Stan Math's
`neg_binomial`, `neg_binomial_2` and `binomial` CCDFs are `-Inf` below about
1e-16 of mass, so these draws read the upper tail through the incomplete-beta
symmetry instead. A range whose mass is not representable in floating point
rejects with a message saying so. Before v0.2.6 the draw only redrew and failed
after 100000 attempts, so a range holding less than about 1e-5 of the mass
could not be drawn.

### Bounded priors: `truncated` or `restricted`

A bound on a parameter prior has two laws, and each has its spelling:

```julia
@slic (; y) begin
    a ~ normal(0, 1)
    x ~ truncated(normal(0, 1); lower=a)   # the normalised law
    w ~ restricted(normal(0, 1); lower=a)  # Stan's plain kernel
    y ~ normal(x + w, 1)
end
```

```stan
parameters { real a; real<lower=a> x; real<lower=a> w; }
model {
    a ~ normal(0, 1);
    x ~ lower_conditioning_normal(a, 0, 1);
    w ~ normal(0, 1);
    y ~ normal((x + w), 1);
}
```

- `truncated` divides the density by the mass inside the bounds, so `x` given
  `a` is the normal truncated at `a`, and prior draws match the density. A
  `lower=`/`upper=` keyword on the family (`normal(0, 1; lower=a)`) is this law
  too.
- `restricted` puts the bounds on the declaration and keeps the family's plain
  kernel, as a hand-written Stan model does without `T[,]`: the joint is the
  unbounded joint restricted to the bounds. It needs no CDF companion, so a
  custom family takes it as it is.
- When the bounds and the family's arguments are all data, or a symmetric
  family is cut once at its location (`normal(0, s; lower=0)`), the two laws
  differ by a constant and emit the same model block. They differ where the
  mass inside the bounds moves with a parameter.

`restricted` takes a token or a family call, like the other combinators:
`restricted(normal, mu, s; lower=lo)` is `restricted(normal(mu, s); lower=lo)`.
It works wherever a prior does: a bare or typed LHS, a `plate` cell, a
`@plate for` body, a `vcat` block and a slice statement. Given its arguments
its law is the truncated law, so where the compiler re-draws it in generated
quantities (held out given a fit, or reading a random draw) the draw is the
`truncated` draw. In a prior-only program it is drawn only when the mass
inside its bounds is a constant. Otherwise the program's joint drops that
mass, no independent draw is exact, and the prior stays a sampled parameter
together with what its bounds and arguments read, as a non-exchangeable
`ordered` prior does. `restricted` is a prior: an observation's plain kernel is
the bare family (`y ~ normal(mu, s)`), so `y ~ restricted(…)` on data is an
error, as are a call without a bound and a combinator in its family slot.

`interval_censored` takes its interval positionally, as above, or by keyword on
a family call. The keyword form is the spelling BayesianRegressionModels uses:

```julia
y ~ interval_censored(normal(mu, sigma); upper=hi)            # (y, hi]: y is the lower endpoint
y ~ interval_censored(normal(mu, sigma); lower=lo, upper=hi)  # (lo, hi]
y ~ weighted(interval_censored(normal(mu, sigma); upper=hi), w)
```

Each keyword call emits the same Stan as its positional form
(`interval_censored(normal(mu, sigma), y, hi)`), in a `plate` cell too.
`upper=` is required. The keywords need the family call: in the token form
(`interval_censored(normal, …; upper=hi)`) the endpoints cannot be told apart
from the family's arguments, so that is an error. A declared observation
without data has no value to serve as the lower endpoint and needs `lower=`.

Every combinator refuses a keyword it does not take: `censored(…; uper=hi)`
fails at trace time instead of silently dropping the bound.

`weighted` accepts scalar or vector weights, but weights must be data-qualified;
weighted priors and parameter-dependent weights reject.

`weighted`'s family slot also takes a bounded combinator, so a power-weighted
censored, truncated, or interval-censored observation composes directly. The
call and token spellings emit identical Stan; in the token form the bounds are
written on `weighted` and belong to the nested combinator:

```julia
y ~ weighted(censored(normal(mu, sigma); lower=lo), w)
y ~ weighted(censored, w, normal, mu, sigma; lower=lo)
y ~ weighted(interval_censored(normal(mu, sigma), lo, hi), w)
```

The density and pointwise likelihood are the combinator's, multiplied by `w`;
the predictive draw is the combinator's own unweighted draw. A bound on
`weighted` whose family is a base family is an error, since it names no
combinator. The other combinators
require the relevant CDF companions. A custom `@lpxf foo_lpdf` / `foo_lpmf`
family needs its usual pointwise and RNG companions plus:

| Form | Additional custom companion(s) |
|---|---|
| lower-only `truncated` | `foo_lccdf` |
| upper-only or two-sided `truncated` | `foo_lcdf` |
| lower-only `censored` | `foo_lcdf` |
| upper-only `censored` | `foo_lccdf` |
| two-sided `censored` | `foo_lcdf` and `foo_lccdf` |
| `interval_censored` | `foo_lcdf` |

The older implementation names `conditioned`, `clamped`, and
`interval_evidence` are not public aliases.

### BLOQ observations

Below/above-limit-of-quantification data are normally `censored`, not
`truncated`: the latent observation occurred, but the assay reported a
threshold. Put the threshold value in the observed vector for an unquantifiable
row; a value at `lower` contributes the lower-tail mass, a value at `upper`
contributes the upper-tail mass, and an interior value contributes its ordinary
density. Use `interval_censored` for genuinely binned observations and
`truncated` only when the data-generating process cannot produce values outside
the range.

## Different priors for coordinates of one vector

One vector can take a different prior family on different coordinates. Lay the
blocks end to end with `vcat`:

```julia
@slic (; rate = [1.5, 3.0], y = 0.4) begin
    tau ~ vcat(exponential(rate), normal(0., 1.; lower=0.))
    y ~ normal(sum(tau), 1.)
end
```

A block's length comes from its own shape: a scalar family covers one
coordinate, a vector argument or `n=` a longer block (`exponential(rate)` covers
`length(rate)`). `tau` is declared with the summed length; a typed LHS
(`tau::vector[3] ~ vcat(…)`) must agree with it. `vcat` lowers to one native
Stan statement per block and one declaration that carries every coordinate's
support:

```stan
parameters { vector<lower=0.0>[(rate_n + 1)] tau; }
model {
    tau[1:rate_n] ~ exponential(rate);
    tau[(rate_n + 1)] ~ normal(0.0, 1.0);
}
```

`tau ~ vcat(…)` is a single statement naming the whole vector, so
`Base.merge(base, :(tau ~ vcat(…)))` replaces a prior in one step, including
over a bare `tau::vector[n]` declaration.

The same lowering is available as Stan-style slice statements, after either a
bare declaration or a `flat` declaration that states the support:

```julia
tau::vector[3]                  # each slice brings its own support
tau[1:2] ~ exponential(rate)
tau[3] ~ normal(0., 1.; lower=0.)

tau::vector[3] ~ flat(; lower=0.)   # the declaration states the support
tau[1:2] ~ exponential(rate)
tau[3] ~ normal(0., 1.)             # a half-normal: it stays inside lower=0
```

A slice brings its family's implied support (`exponential` ⇒ `lower=0`) plus
any explicit `lower=`/`upper=`, intersected with a `flat` declaration's. When
every coordinate ends up with the same bound the declaration carries it as a
scalar; otherwise it carries a per-coordinate bound vector, with
`negative_infinity()`/`positive_infinity()` for unbounded coordinates
(`vector<lower=[0.0, 0.0, negative_infinity()]'>[3] tau`). `vcat` blocks of
literal length fix the coordinates even when the declared length is data, so
`tau::vector[n] ~ vcat(normal(0., 1.), exponential(1.))` still gives
`vector<lower=[negative_infinity(), 0.0]'>[n] tau`. When a slice index, a block
length or a bound is data, or slice statements cover a data-length vector, that
vector is built in `transformed data`. Prior-only and held-out programs re-draw each block from its
own family inside the coordinate's support.

Rules:

- Indices are data: integer literals, ranges, or data integer arrays.
- Each coordinate takes exactly one prior; a coordinate sampled twice is an
  error. A vector with a whole-vector prior (`tau::vector[3] ~ normal(…)`) takes
  no slice statements.
- A coordinate no statement covers stays improper-flat on the declared support.
- Per-coordinate bounds apply to a `vector`; a per-coordinate bound must be data,
  and `offset=`/`multiplier=` belong to the declaration, not to a slice.
- An observed `y ~ vcat(…)` (or slices covering every coordinate of a data
  vector) gets `y_gen` and a per-coordinate `y_likelihood`; a joint family over a
  slice (`multi_normal`) has no per-coordinate likelihood, so `y` then gets `y_gen`
  only.

The same statements give a per-cell vector its per-coordinate priors in a
`plate` cell, a `@plate for` loop (as a local or a `tau[:, s]` array) and a
`@scan` step. Each cell is a column of the storage the loop declares, and that
storage carries the support: a scalar bound when every coordinate shares one,
otherwise a `transformed data` bound of the storage's own shape, filled per cell.

```julia
tau_s ~ plate(; outer=n_strata) do s
    tau::vector[n_terms] ~ vcat(exponential(0.5), lognormal(0.0, 0.5))
    tau
end
```

```stan
parameters { matrix<lower=0.0>[n_terms, n_strata] tau_s_tau; }
model {
    for (plate_i__pl_1 in 1:n_strata) {
        tau_s_tau[1, plate_i__pl_1] ~ exponential(0.5);
        tau_s_tau[2, plate_i__pl_1] ~ lognormal(0.0, 0.5);
    }
}
```

In a `plate` cell, slice statements on a bare cell declaration
(`tau::vector[2]`, then `tau[1] ~ …`) emit the same program; in a `@plate for`
body an indexed left-hand side names an array the loop fills, so write `vcat`
there. A bare cell declaration alone is an improper-flat cell parameter. Not
supported yet, each refused at trace time:
coordinates with different support in a plate inside a called sub-model (only
shared support works there), a `vcat` local in a nested annotated loop, and
observing a plate positional's cell with `vcat` (`yi ~ vcat(…)`; write slice
statements `yi[1] ~ …`). Before StanBlocks `ac830ba` a cell refused
`tau ~ vcat(…)`, and its slice-statement and bare-declaration spellings failed
with `tracetype not defined for ::vector[Colon(), ::int]`.

Write a custom whole-vector family (an `_lpdf`/`_lpdfs`/sized `_rng` triad with
`autokwargs` support) only for a genuinely joint prior.

## Bounded `ordered` and `positive_ordered` vectors

Stan declares no bound on `ordered` or `positive_ordered`. A typed LHS can
still carry one, written or implied by its family:

```julia
@slic (; y = 0.4) begin
    c::ordered[3] ~ uniform(0., 1.)     # implied lower=0, upper=1
    y ~ normal(c[1], 1.)
end
```

A sampled one is declared as `c_free`, an `ordered` vector of the same shape,
and `c` is the transformed parameter that one of Stan's constraint transforms
makes of it. Each transform is increasing, so `c` stays ordered, and adds its
own log-Jacobian:

```stan
parameters { ordered[3] c_free; }
transformed parameters { ordered[3] c = lower_upper_bound_jacobian(c_free, 0.0, 1.0); }
```

| Constraint | `c` |
|---|---|
| `lower=L` | `lower_bound_jacobian(c_free, L)` |
| `upper=U` | `upper_bound_jacobian(-c_free, U)` |
| `lower=L, upper=U` | `lower_upper_bound_jacobian(c_free, L, U)` |
| `offset=` / `multiplier=` | `offset_multiplier_jacobian(c_free, offset, multiplier)` |

`positive_ordered` is bounded below at 0 already: a lower bound at or below 0
adds nothing, and an `upper=U` lowers as `lower=0, upper=U`. A plate cell's
`array[n] ordered[k]` carrier lowers the same way. A re-drawn one (prior-only
or held out) is declared without the bound and drawn inside it.

Rules:

- Every constraint is a scalar: per-element bounds would not keep the elements
  ordered.
- `positive_ordered` takes no `offset=`/`multiplier=`, and its `lower=` must be
  a number.
- The sampler's coordinates are `c_free`: BridgeStan lists `c_free` as the
  parameter and `c` as a transformed parameter.

## Matrix parameters and observations of elementwise families

A typed matrix LHS takes an iid prior from any continuous elementwise family,
and a matrix of data is observed the same way:

```julia
@slic (; R, C, Y, M) begin
    A::matrix[R, C] ~ normal(0.0, 2.0)
    B::matrix[R, C] ~ lognormal(0.0, 0.5)
    s ~ exponential(1.0)
    Y ~ normal(M, s)              # Y, M: R×C data matrices
end
```

Stan vectorises these families over no matrix, so StanBlocks supplies matrix
overloads of each family's density, pointwise companion and sized `_rng`. The
model block keeps the statement as written (`A ~ normal(0.0, 2.0);`); its
density is the family's over `to_vector(A)`, with the normalising constants
dropped exactly as `to_vector(A) ~ normal(0.0, 2.0)` drops them.

- Each argument is a scalar or a matrix of the variate's own shape; Stan checks
  that shape at run time. A `vector`, `row_vector` or array argument is refused
  at trace time: write `rep_matrix(mu, n)` to repeat a `vector[m]` location
  across the `n` columns, or `rep_matrix(mu, m)` to repeat a `row_vector[n]`
  one down the `m` rows (`rep_matrix(b', m)` for a `vector[n]` `b`).
- A prior-only or held-out matrix is re-drawn with its own shape, and a matrix
  observation's `Y_gen` and `Y_likelihood` are `matrix`-shaped, one entry per
  element.
- Families: `std_normal`, `normal`, `student_t`, `cauchy`, `lognormal`,
  `exponential`, `gamma`, `inv_gamma`, `beta`, `uniform`, `weibull`, `frechet`,
  `double_exponential`, `logistic`, `gumbel`, `pareto`, `pareto_type_2`,
  `scaled_inv_chi_square`, `chi_square`, `inv_chi_square`, `rayleigh`,
  `von_mises`, `loglogistic`, `skew_normal`, `exp_mod_normal`,
  `skew_double_exponential`.
- Not yet: the distribution combinators (`truncated`, `censored`,
  `interval_censored`, `weighted`, and a `lower=`/`upper=` cut whose normaliser
  depends on a parameter) over a matrix variate, and integer-matrix
  observations of the discrete families. Both fail at trace time. Before the
  matrix overloads (snag `matrix-parameter-94e1f96c`), every matrix form above
  transpiled to a statement stanc rejects (`Ill-typed arguments supplied to
  function "normal": (matrix, real, real)`), and a prior-only one failed at
  trace time.

## Generated observation twins

A normal top-level observation such as `y ~ normal(mu, sigma)` produces two
generated quantities automatically:

- `y_gen`: a posterior-predictive draw;
- `y_likelihood`: the pointwise log-likelihood.

The descriptor tags their meaning and source, so consumers should read
`ModelOutput.generative` and `ModelOutput.source` rather than parse suffixes.
The exact shape depends on how the observation is written:

| Observation form | Predictive draw | Pointwise log-likelihood |
|---|---|---|
| Dense, top-level | `<obs>_gen`, observation-shaped | `<obs>_likelihood`, elementwise |
| Dense, inside `plate` | `<obs>_gen`, filled per cell | Not synthesized |
| Ragged, top-level | Flat `<obs>_gen` over all elements | `<obs>_likelihood`, one aggregate per group |
| Ragged slice, inside `plate` | Flat `<obs>_gen` over all elements | `<obs>_likelihood`, one aggregate per group |

All four forms contribute the same density to the model; the rows differ only
in their generated-quantity capabilities, not in likelihood semantics. An
observation always samples the data itself (`y ~ …`, `y[i] ~ …`, a plate
positional `yi ~ …`): `a = y; a ~ …` is refused, because a name defined by `=`
has nothing to draw when `y` is omitted and the model could no longer simulate
from its prior. Inside a plate cell every `=`-defined name is refused; at model
scope only a pure alias is, and observing transformed data
(`log_y = log(y); log_y ~ …`, the Stan idiom of several posteriordb ports) still
compiles — such a model cannot simulate `y` itself from its prior.

## Missing outcomes

A partly missing continuous outcome vector is imputed automatically:

```julia
m = @slic (; y = [1.0, missing, 3.0, missing]) begin
    mu    ~ normal(0, 10)
    sigma ~ exponential(1)
    y     ~ normal(mu, sigma)
end
```

The observed entries contribute to the likelihood; missing entries are drawn in
generated quantities and merged back into the full vector. This is intentionally
narrow:

- the missing values must be a continuous vector on the LHS of exactly one `~`;
- scalar/factorising observation families work;
- inherently joint families such as `multi_normal` do not support elementwise
  auto-imputation;
- missing predictors and discrete missing outcomes must be handled explicitly.

## Ragged data and container views

A `Vector{<:AbstractVector{<:Real}}` data kwarg becomes a first-class
`RaggedVector` with flat `mem` and inclusive group `ends`. In a model body,
`length(y)` is the number of groups and `y[g]` is the `g`th group:

```julia
groups = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]

@slic (; groups, g = 2, obs = 0.0) begin
    first_group = groups[1]
    selected    = groups[g]
    n_groups    = length(groups)
    obs ~ normal(sum(first_group) + sum(selected) + n_groups, 1)
end
```

A `Vector{<:AbstractMatrix{<:Real}}` data kwarg, such as one design matrix per
group, is a `RaggedMatrix` in the same way: `X[g]` is the `g`th matrix and
`length(X)` the number of matrices, in the model body, in a plate cell, as a
plate positional, and passed to a submodel or a `@deffun`:

```julia
X = [[0.5 -0.2; 0.1 0.3], [0.2 0.4; -0.6 0.1; 0.3 0.7]]
y = [[0.1, -0.3], [0.4, 0.2, -0.1]]

@slic (; X, y) begin
    b1 ~ normal(0, 1)
    b2 ~ normal(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ normal(X[i] * [b1, b2], 1)
    end
end
```

Its values are real, since a Stan `matrix` has no integer form.

Elementwise arithmetic keeps the groups. `2.0 * x`, `a + b * x`, `x / s`,
`x .^ 2`, `-x`, and `x + z` / `x .* z` for two RaggedVectors with the same
group ends all give a `RaggedVector`. The result works anywhere a RaggedVector
does: as a ragged law argument, indexed by group, or passed to an `@inline`
RaggedVector UDF:

```julia
const RaggedVector = StanBlocks.RaggedVector   # not exported; alias it to name it
@deffun @inline exp_rv(x::RaggedVector) = RaggedVector(exp(x.mem), x.ends)

@slic (; y = [[1.0, 2.0], [3.0]], x = [[0.1, 0.2], [0.3]], z = [[1.1, 1.2], [1.3]]) begin
    mu ~ std_normal(); a ~ std_normal(); b ~ std_normal()
    y ~ normal(mu, exp_rv(a + (b - a) * x) + z .^ 2)
end
```

Group `g` of the result is the same arithmetic over each operand's group `g`.
A UDF result is not split this way, because a UDF may combine groups:
`exp_rv(...)` computes `exp` over the whole memory. Inside a ragged law that
memory is computed once, before the group loop, and each group slices it; when
it depends on parameters it is a local variable, not a model output. Two RaggedVector operands must have the same group ends: transpile rejects
known differing values, and Stan checks rebound data at runtime. Forms with no
elementwise meaning are rejected with the supported forms listed: `x * z`,
`s / x`, `x ^ 2`, and a dense vector operand (per group or per element?). Write
`.*`, `./`, and `.^` for the elementwise versions.

An integer array, a range or a comparison mask selects several groups, as in
Julia: `x[idx]` is the groups `idx`, in that order, as a RaggedVector (a
`RaggedMatrix` selects matrices the same way). `sum(x)` adds the groups
elementwise, `x[1] + x[2] + …`, so every group must have the same length (Stan
rejects the program otherwise), and `mean(x)` is `sum(x) / length(x)`. The
total of every entry is `sum(x.mem)`:

```julia
@slic (; ys = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], idx = [1, 3], obs = [6.1, 7.9]) begin
    s ~ normal(1, 1)
    obs ~ normal(s * sum(ys[idx]), 1)   # s * ([1, 2] + [5, 6])
end
```

The selection's offsets depend only on the data, so they are computed once in
`transformed data`. Any other function of a whole RaggedVector fails when the
program is built; apply it to one group, `sd(x[g])`, or to every entry,
`sd(x.mem)`.

`ragged_start`, `ragged_end`, and `ragged_length` expose flat offsets for UDF
size math. `EachCol(X)` and `EachRow(X)` are zero-cost dense views:
`EachCol(X)[j]` lowers to `col(X, j)`, and `EachRow(X)[i]` lowers to
`row(X, i)`. Both work as `plate` inputs.

Top-level varying-size constrained parameters are also supported:

| Declared family with data vector `K` | Representation |
|---|---|
| `simplex[K]`, `ordered[K]`, `positive_ordered[K]` | Flat free vector plus per-group constrain/Jacobian transform and a `RaggedVector` view |
| `cholesky_factor_corr[K]`, square `cholesky_factor_cov[K]` | Flattened free storage plus per-group matrix transform and a ragged matrix view |

Other varying-size constrained matrix families are rejected. These are
model-scope declarations; a varying-size constrained cell inside `plate` is a
different, currently unsupported shape.

### Ragged observations

A top-level ragged observation broadcasts the family across groups without
flattening the density:

```julia
@slic (; ys, mu) begin
    sigma ~ exponential(1)
    ys ~ normal(mu, sigma)
end
```

For a factorising family, each `ys_likelihood[g]` is the sum of the elementwise
densities in group `g`. For a joint family such as `multi_normal`, it is the
group's joint density. Per-element ragged log-likelihoods are not synthesized.
The flat `ys_gen` can be split back into groups using the corresponding
descriptor output's `segments` field.

A custom family used on a ragged observation must provide a sized predictive
companion such as `foo_rng(vector[n], args...)::vector[n]`. Ragged observations
currently require continuous families because their backing storage is real;
integer-valued ragged predictive draws have no carrier. Distribution
combinators compose over ragged observations, and ragged positional arguments
or keyword bounds are sliced per group.

## `plate`: the compiler-owned parameter loop

`plate` maps one straight-line cell body over an outer grid. Positional inputs
are sliced per cell, lexical captures are shared, fresh cell-local `~` and `=`
bindings are promoted to outer storage, and a trailing value expression is
collected as the plate result:

```julia
@slic (; y = randn(6), mu0 = 0.5) begin
    sigma ~ normal(0, 1; lower = 0)
    theta ~ plate(y; outer = 6) do yi
        t  ~ normal(mu0, 1)
        yi ~ normal(t, sigma)
        t
    end
end
```

Here `yi` means `y[i]`, while `mu0` and `sigma` are the same captured values in
every cell. Every positional's length must equal its outer axis (and every
sibling positional): a provable mismatch errors at trace time naming the
plate, the positional, and both lengths. A ragged input counts groups and
`EachCol`/`EachRow` count columns/rows; a `length(X)`-style `outer` over data
carries no trace-time value and stays unchecked. The compiler emits the
necessary loops separately in the Stan blocks that own the declarations, fills,
densities, and generated quantities.

With positional inputs, `outer` may be omitted: the plate counts its first
positional. A parameter or transformed parameter counts by its declared size,
so `plate(lev, ys) do l, y` with `lev::vector[n]` emits the same program as
`plate(lev, ys; outer=(n,))`. Stan admits only data in the sizes of the
plate's declarations and `transformed data` index tables, which
`num_elements(lev)` is not.

### Plate support matrix

| Capability | Status | Exact boundary |
|---|---:|---|
| One-dimensional outer shape | ✅ | `outer=N` and `outer=(N,)` are equivalent |
| N-dimensional outer shape | ✅ | Use a non-empty tuple; with no positional inputs, take one do-block index per axis; tested through four axes |
| Positional input slicing | ✅ | One do-block argument per positional input, indexed over all outer axes |
| Lexically captured values | ✅ | Shared across cells; loop-invariant expressions are hoisted when safe |
| Fresh scalar `~` or `=` binding | ✅ | Collected as a `vector[N]` for one outer axis; additional axes use matrix/array storage. An integer binding (`k = 2 * i + 1`) stays an integer, `array[N] int`, so it can index |
| Fixed `vector[K]` binding/result | ✅ | Collected as `matrix[K,N]` for one outer axis; extra axes add Stan array prefixes |
| Per-coordinate priors on a vector cell | ✅ | `tau ~ vcat(…)`, or slice statements `tau[1] ~ …` on a bare `tau::vector[K]`; shared support is a scalar bound on the storage, mixed support a `transformed data` bound of its shape; inside a called sub-model only shared support |
| Bare cell declaration `tau::vector[K]` | ✅ | An improper-flat cell parameter with no density statement |
| Observing a cell binding defined by `=` (`a = yi; a ~ …`) | ❌ | Refused: observe the data itself (`yi ~ …`, `ys[i] ~ …`), so that omitting it still gives the prior simulation |
| Sampled fixed `int[K]` cell binding | ✅ | Sampled (`~`) side bindings collect as `array[N,K] int` for one outer axis; deterministic int cells (findall/boolean-mask index arrays) still inline at each use |
| Dense `array[T] vector[K]` binding/result | ✅ | Collected as `array[outer...,T] vector[K]`; includes a raw ODE trajectory bound once and consumed inside the cell |
| Varying-length plain `vector[K[i]]` cell | ✅, narrow | One-dimensional outer shape and data-computable lengths only; represented by flat memory whose descriptor output carries that result/member's own inclusive group ends in `segments` |
| Varying-length `int[K[i]]` cell | ✅, narrow | One-dimensional outer shape and data-computable lengths only; flat-memory descriptor output carries inclusive group ends in `segments`, exactly like a varying-length vector cell |
| Fixed native-constrained vector cell | ✅ | `simplex[K]`, `ordered[K]`, and `positive_ordered[K]`; emitted as `array[outer...] <type>[K]` |
| Shared scalar constraints | ✅ | `lower`, `upper`, `offset`, and `multiplier` are preserved when they do not depend on the cell position |
| Cell-position-dependent constraint | ❌ | Rejected because the promoted declaration lives outside the loop |
| Fixed-size `matrix[R, C]` binding/result | ✅ | Collected as `array[outer...] matrix[R, C]` (an `array[A] matrix[R, C]` cell as `array[outer..., A] matrix[R, C]`); a cell reads and writes it as the whole element `f[i]` |
| Matrix whose size varies by cell | ✅, narrow | One-dimensional outer shape and data-computable sizes; each cell is stored column-major in flat memory with transformed-data `f__pl_rows`/`f__pl_cols`, read back with `to_matrix`, a `RaggedMatrix` after the loop; the descriptor output carries the flat group ends in `segments`. A *sampled* matrix of varying size is refused: sample a `vector` of length rows × cols and reshape it with `to_matrix` |
| Other higher-rank cell value | ❌ | `array[T] vector[K]` and matrix cells are the dense rank-2 forms; declare other shared values outside the plate |
| Varying-size constrained cell | ❌ | Declare the ragged constrained parameter at model scope, then index it in the plate |
| Int-array plate result | ❌ | MVP: sampled int side cells promote, but the loop's own return cannot collect as an int array; collect a scalar or vector cell instead |
| Positional `RaggedVector`, `RaggedMatrix`, `EachCol`, `EachRow` | ✅ | Each cell receives one ragged group, one matrix of a vector of matrices, a matrix column, or a matrix row |
| Named/anonymous `@slic` call inside the cell | ✅ | Internal fresh bindings are discovered, namespaced, and promoted per cell |
| A plate inside a called sub-model | ✅, dense | Scalar and fixed-vector cells work and caller sizes are substituted correctly |
| Ragged-vector plate inside a called sub-model | ❌ | Keep the ragged plate at model scope; a cell may still call a sub-model that returns a ragged-sized vector |
| A `@scan` inside a called sub-model, at model scope or per cell | ✅ | Carried arrays live under the sub-model's namespace; an observed scan observes the caller's data passed by keyword (see "A scan inside a called sub-model" below) |
| Several independent plates reusing local names | ✅ | Cell-local bindings are hygienically namespaced by plate result; a namespaced name that is already bound (a data input `theta_t`, a second array-less `@plate for i`) takes a numbered suffix (`theta_t_2`) |
| Two plates under one name | ❌ | One name, one plate: the name labels its carriers, so a second `theta ~ plate(…)` is refused, collecting or not |
| Dependency on a previous cell | ❌ | Cells are independent; a recurrence that samples per step is the annotated `@scan begin … end` loop (below), a deterministic one a `@deffun` recurrence |
| Vararg do-block parameters | ❌ | Use a fixed positional argument list |
| Cell without a result (observation-only) | ✅ | The body ends in a `~`/`=` statement, an annotated loop (`@scan begin … end`), or `nothing`: fresh bindings keep the `<rv>_<name>` namespace and no `rv` is declared, filled or bound |
| Fully dead/prior-only plate | ✅ | The whole plate — fresh samples, collected result and loop — lowers to `generated quantities` with everything it reads; `parameters {}` stays empty |
| Cross-validation taint | ✅ | Taint from `outer` sizes or a cell prior RHS moves the affected cell path to generated quantities |

### Vector cells and shared multivariate priors

```julia
@slic (; n_groups = 8, k = 3) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)
    tau::vector[k] ~ normal(0, 1; lower = 0)

    b::matrix[k, n_groups] ~ plate(; outer = n_groups) do g
        z::vector[k] ~ std_normal()
        diag_pre_multiply(tau, L) * z
    end
end
```

The result `b` and the fresh `z` are logically `matrix[k,n_groups]`, with cells
in columns. For a one-axis plate whose cell is exactly one shared-argument
multivariate sample returned unchanged, StanBlocks can emit a single vectorized
array sample for `multi_normal`, `multi_normal_prec`,
`multi_normal_cholesky`, `multi_student_t`, or
`multi_student_t_cholesky`. Cell-varying arguments keep the ordinary loop.

A deterministic `array[T] vector[K]` cell value uses Stan's natural nested
layout with the plate axes prepended. This lets an ODE solve be named and reused
without wrapping the extraction in a separate `@deffun`:

```julia
pred ~ plate(; outer = N) do i
    trajectory = ode_rk45(rhs, y0[i], 0.0, to_array_1d(ts), theta)
    selected = to_vector(trajectory[:, 1])
    y[i] ~ normal(selected[1], sigma)
    selected
end
```

Here `trajectory` is collected as `array[N,T] vector[K]`, and references inside
the cell lower to `trajectory[i]`, recovering the original
`array[T] vector[K]`. Arbitrary nested-array values remain unsupported.

### Matrix cells

A cell may name a matrix before using it, such as a covariance that depends
on the cell:

```julia
@slic (; y = [[0.1, 0.2, 0.15], [0.2, 0.1], [0.3, 0.05, 0.1, 0.2]]) begin
    s0 ~ normal(0, 1; lower = 0)
    g ~ normal(0, 1)
    @plate for s in eachindex(y)
        v = exp(g) .* (1.0 .+ y[s] .^ 2)
        c = sqrt(s0) .* v
        ccov = diag_matrix(v) .+ c * c' ./ (1.0 + sum(v))
        y[s] ~ multi_normal(g .* v, ccov)
    end
end
```

In a `@plate for` loop `ccov` is a per-iteration local: a Stan local
`matrix[…] loop_s_ccov = …;` inside the model loop and inside the
generated-quantities loop of the `y_gen`/`y_likelihood` twins, computed once in
each and never written to the draws (see "Annotated loops" below).
The do-form stores every cell binding instead: written as
`cells ~ plate(y) do ys … end`, `ccov` is computed once per evaluation and its
size is the length of the cell's group, so it is stored column-major in one
flat `vector` (`cells_ccov`) and read back as
`to_matrix(cells_ccov[...], cells_ccov__pl_rows[i], cells_ccov__pl_cols[i])`.
A stored matrix of fixed size is collected as `array[outer...] matrix[R, C]`
instead.
A matrix built from data alone, such as a per-cell 0/1 matrix summing the
cell's zones into its groups, `A = reshape(agg[c], n_groups[c], n_zones[c])`,
is filled once in `transformed data` and read back the same way.

### Where to put an observation

Prefer a top-level observation over the collected plate result when the family
already supports the collected shape. That spelling gets both generated twins.
For varying categorical logits, for example, collect the per-row logits into a
`matrix[K,N]` and keep the categorical observation top-level:

```julia
@slic (; x, y, N, K) begin
    beta::vector[K-1] ~ std_normal()
    logits::matrix[K,N] ~ plate(x; outer = N) do xi
        append_row(0.0, beta * xi)
    end
    y ~ categorical_logit(logits)
end
```

The plate result LHS names the **collected** type, so `logits::matrix[K,N]`
agrees with the value it binds (the `~` LHS and RHS types match); the do-block
returns one `vector[K]` cell per row and the compiler derives that cell shape by
stripping the `outer` axis. `y_gen` is `int[N]` and `y_likelihood` is
`vector[N]`. Put an observation inside the cell when slicing or cell-specific
structure requires it, accepting the generated-quantity limits in the table
above.

A cell whose last line is a statement (`~` or `=`) has no value to collect,
so it collects no result. A trailing `nothing` does the same:

```julia
@slic (; ts, ys) begin
    sigma ~ exponential(1)
    pk ~ plate(ts, ys) do t, yi
        b ~ normal(0, 1)
        m = b .* t
        yi ~ normal(m, sigma)
    end
end
```

The cell's fresh bindings are still namespaced by the plate's LHS (`pk_b`,
`pk_m`), but `pk` itself is not declared, filled or bound, in the same way
that `node ~ submodel(...)` binds nothing for a sub-model without `return`.
Reading `pk` later is an error that says so, and so is a second plate named
`pk`. Collecting a value that nothing
reads instead (`m` as the trailing expression) adds a generated-quantities
copy of every cell value.

## Annotated loops: `@plate for … end` and `@scan begin … end`

A top-level model loop that introduces parameters is written as an
*annotated* loop. Both forms are sugar over the compiler-owned loop behind
`plate`: the tracer inlines them (neither is a function call), discovers the
body's fresh bindings once, promotes the ones it stores to outer storage and
emits a plain Stan `for` loop into the blocks that own each statement. Inside
the body:

- an LHS indexed by the loop variable (`x[i] ~ …`, `y[i] = …`) is a
  **model-scope array**, registered under its own name and visible after the
  loop; a whole vector cell is written `v[:, i]`;
- a bare fresh name (`t ~ …`, `tmp = …`) is a **per-iteration local** under a
  hygienic name prefixed by the loop variables (`loop_i_t`;
  [Stan names of plate values](#Stan-names-of-plate-values)). A sampled
  `t ~ …` is a per-cell parameter, hoisted to per-cell storage. A computed
  `tmp = …` (also `tmp::T = …`, and the `=` bindings of a `laplace_marginal`
  do block in the body) is per-iteration scratch: a Stan local declared inside
  the loop of each block that reads it (`real loop_i_tmp = …;` in the model
  loop, again in the generated-quantities loop that draws a twin from it, and
  in a transformed-parameters loop that fills an array from it), recomputed
  there and never written to the draws. Only a `tmp` computed from data alone
  is stored, once, in `transformed data`. To keep a value in the draws, index
  it (`tmp[i] = …`). A hygienic name that is already bound, such as a second
  loop's `loop_i_t`, takes a numbered suffix (`loop_i_t_2`). For a namespace of
  your choosing, use the `plate` do-form, whose LHS names it even when the
  cell collects no result; the do-form stores every cell binding, `=` ones
  included;
- `y[i] ~ …` on data is an observation, with the usual `y_gen` twin. Observe
  the data itself: `obs = y[i]; obs ~ …` is refused, because a name defined
  by `=` has nothing to draw when `y` is omitted;
- a write to a name bound outside the loop is rejected — SB models never
  mutate — and every element of a loop-filled array is assigned exactly once.

`@plate for` is the independent-cell loop: reading one of the loop's own
arrays at any index other than the loop variable's is an error.

```julia
@slic (; y = randn(6), mu0 = 0.5) begin
    sigma ~ normal(0, 1; lower = 0)
    @plate for i in 1:6
        x[i] ~ normal(mu0, 1)         # vector[6] x
        t ~ exponential(1)            # per-cell local → vector<lower=0>[6] loop_i_t
        y[i] ~ normal(x[i] + t, sigma)
    end
end
```

`for i in 1:N, j in 1:M` gives an N-dimensional outer shape, and
`for i in eachindex(y)` is the range `1:length(y)`; `for d in doses`
iterates the values of a container (each cell receives one `RaggedVector`
group, and `d ~ …` observes it). On ragged data, `yr[i] ~ …` is the same
observation of group `i`: a flat `yr_gen` and one aggregate `yr_likelihood`
per group, exactly as the `plate` do-form emits them.

`@scan` is the sequential loop: one block whose trailing `for` is the
recurrence and whose preceding statements are the initial state. The loop's
arrays may be read at the current index or at a fixed lag `x[i - k]`
(`k ≥ 1`, a literal or a data int), never ahead; the setup fills `x[1:m]` with
literal indices and the loop starts at `m + 1`, so every element is assigned
exactly once and no lag reads before the setup.

```julia
@slic (; y = randn(5), T = 5) begin
    phi ~ normal(0, 0.5); s ~ exponential(1)
    @scan begin
        h[1] ~ normal(0, 1)
        for t in 2:T
            h[t] ~ normal(phi * h[t-1], s)   # vector[T] h; a plain Stan loop
        end
    end
    y ~ normal(h, 1)
end
```

Several carried arrays in one loop (`u[t] ~ f(u[t-1], v[t-2])`,
`v[t] ~ g(…)`) are just several indexed LHS. Per-step `~` locals are sized by
the iteration count (`vector[T-1] loop_t_eps` for a loop over `2:T`), so the
non-centered spelling `eps ~ std_normal(); x[t] = phi * x[t-1] + s * eps`
has no never-sampled coordinate.

The annotation is a checked contract, not decoration, and it is
mode-agnostic: activity analysis places the whole block as one unit. Bind the
observation and the block is fitted (its per-step `~` are parameters, sampled
in `model` in step order); leave the observation unbound and `parameters {}`
is empty while the block re-draws in `generated quantities` in step order;
mark a size or input with `maybecv` and the whole chain, observation
included, moves to `generated quantities`.

### Observed scans

A scan may observe data and read it back at a lag. This is how an
autoregressive likelihood is written: each observation is conditioned on the
observed past.

```julia
@slic (; y, mu, T) begin
    phi ~ uniform(-1, 1); sigma ~ exponential(1)
    @scan begin
        y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))       # stationary start
        for t in 2:T
            y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
        end
    end
end
```

- The `model` block gets the conditional densities as written, reading the
  observed `y[t-1]`.
- `y_gen` is the whole simulated trajectory: the scan the same body runs when
  `y` is omitted, laid out like `y`. Its lags read `y_gen`, so it is a draw of
  `y` given the parameters, not a one-step-ahead prediction from the observed
  past.
- `y_likelihood` holds each element's conditional log density, so it sums to
  the log likelihood.

Holding `y` out with `maybecv` drops the densities from `model` and keeps both
twins. Each step of `y_gen` is drawn the way that observation's own `_gen` twin
would be: a bounded family is truncated, `censored(…)` clamps the draw (and the
next step reads the clamped value, as the observed model does), and
`weighted(…)` draws from the base family. A deterministic statement that reads the observations and feeds one of
them, such as a carried residual (`err[t] = y[t] - nu`, read as `err[t-1]` by a
moving-average term), is replayed along the trajectory as `err_gen`. Sampled
statements are not re-drawn: like every predictive twin, the trajectory is
drawn given the posterior latents. An array observed per step and never read
back (`y[t] ~ normal(h[t], s)`) keeps its ordinary per-cell `y_gen`.

Each element is observed exactly once and given earlier elements only: the
setup observes `y[1:m]` with literal steps, the loop observes `y[t]` from
`m + 1`, lags reach back at most `m`, a 1-D `y` has exactly as many elements as
the scan runs, and no observation reads its own current value. Each violation is
refused with a message naming `y`. Conditioning on unmodelled initial values
(observing only `y[2:T]`) is not supported yet.

The same holds in a plate cell. Dense data is observed per cell as `y[t, j]`
(the scan step first, then the cell), and its `y_gen` and `y_likelihood` are
matrices like `y`. A group of nested-vector data is observed through the
container clause or as `ys[p][t]`, and its `ys_gen` and `ys_likelihood` are flat
vectors with the data's group `segments`:

```julia
@slic (; ys) begin                       # ys = [randn(3), randn(5), randn(2)]
    phi ~ uniform(-1, 1); sigma ~ exponential(1)
    @plate for y in ys                   # or: @plate for p in eachindex(ys) … ys[p][t] …
        @scan begin
            y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
            for t in 2:length(y)
                y[t] ~ normal(phi * y[t-1], sigma)
            end
        end
    end
end
```

A carried array takes its support coordinate by coordinate, as slice statements
do: with `x[1] ~ exponential(1.)` and an unbounded recurrence
`x[t] ~ normal(x[t-1], 1.)`, only `x[1]` is bounded
(`vector<lower=[0.0, negative_infinity(), negative_infinity()]'>[3] x`). A
recurrence bound that reads the loop index (`lower=L[t]`) is per coordinate
too, built in `transformed data`.

Annotated loops nest: a `@scan` inside a `@plate for j in 1:S` is the
per-subject state-space shape. The enclosing loop promotes the inner arrays
to per-subject storage and the indices compose into one Stan variable
(`matrix[T, S] h`, `h[t, j] ~ normal(phi * h[t-1, j], s)`). Support that every
coordinate of every cell shares stays a scalar bound (`matrix<lower=0.0>[T, S] h`).
Otherwise the bound is a `transformed data` matrix of the same shape, filled
per cell and coordinate:

```julia
@slic (; y = randn(6), S = 2) begin
    @plate for j in 1:S
        @scan begin
            h[1] ~ exponential(1.)               # h[1, j]: lower=0
            for t in 2:3
                h[t] ~ normal(h[t-1], 1.)        # h[2:3, j]: unbounded
            end
        end
    end
    y ~ normal(to_vector(h), 1.)
end
# matrix<lower=h__lower>[3, S] h;  h__lower[1, j] = 0.0;
#                                  h__lower[t, j] = negative_infinity();
```

Such a bound may read data and the loop indices (`lower=L[j]`, `upper=U[t]`),
but not a value the loop body binds per cell (`c = L[j]; … lower=c`): write the
expression in the bound instead.

A nested scan may run to a per-cell length, `for t in 2:T[j]` or
`for t in 2:length(y)` over a container group. Its carried arrays and stored
per-step locals are then stored as the plate's flat memory (one segment per cell, like a
varying-length plate cell), and element `t` of cell `j` is
`h[ragged_start(h__pl_end, j) + t - 1]`. This works in every mode, and support
reaches the flat memory as it does a fixed-length cell: a bound every element
shares is a scalar bound, and a setup and recurrence with different supports
take a per-element bound vector built in `transformed data`.

An integer a nested step computes, such as a time gap `k = d[t] - d[t - 1]` or
a carried int array (`c[t] = c[t - 1] + k`), is collected the same way: as
`array[S, T - 1] int` (one row per cell), or as flat int memory for a per-cell
length. It is data, so it is computed once in `transformed data`.

### A scan inside a called sub-model

A `@scan` may sit in a called `@slic` sub-model, at model scope or per plate
cell. Its carried arrays are declared under the sub-model's namespace (`hh_h`
for `hh ~ latent(; …)`), and the sub-model's own name for them works in the
setup, in the lag reads and after the loop. Per-step locals (a non-centered
innovation, an integer time gap `k = time[t] - time[t-1]`) and per-cell lengths
(`n = T[p]`) work as in the caller, and all three modes behave as at model
scope.

```julia
latent = @slic begin
    @scan begin
        h[1] ~ normal(0, sigma)
        for t in 2:n
            h[t] ~ normal(phi * h[t-1], sigma)
        end
    end
    return h
end
@slic (; y = randn(5)) begin
    phi ~ uniform(-1, 1); sigma ~ exponential(1)
    hh ~ latent(; n = 5, phi, sigma)     # vector[5] hh_h;  vector[5] hh = hh_h;
    y ~ normal(hh, 1.0)
end
```

An observed scan in a sub-model observes the caller's data passed by keyword:
a data variable (`obs = y`), a column (`obs = Y[:, p]`) or a nested-vector
group (`obs = ys[p]`). The program is the one the same scan emits when written
in the caller (`y[t]`, `Y[t, p]`, `ys[p][t]`): the trajectory is `y_gen`,
`Y_gen` or `ys_gen`, and the likelihood `y_likelihood`, `Y_likelihood` or
`ys_likelihood`, twins of the caller's data. A sub-model is then a reusable
observation family over a series, called once per subject:

```julia
ar1 = @slic begin
    @scan begin
        obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
        for t in 2:n
            obs[t] ~ normal(mu[t] + phi * (obs[t-1] - mu[t-1]), sigma)
        end
    end
end
@slic (; ys, mus) begin                  # nested vectors, one series per subject
    phi ~ uniform(-1, 1); sigma ~ exponential(1)
    @plate for p in eachindex(ys)
        node ~ ar1(; obs = ys[p], mu = mus[p], n = length(ys[p]), phi, sigma)
    end
end
```

The group may also reach the cell as a parameter: `cells ~ plate(ys, mus) do
y, m; node ~ ar1(; obs = y, mu = m, n = length(y), phi, sigma) end`, or
`@plate for y in ys … obs = y`, emits the same program (and the parameter may
share the data's name, `plate(ys) do ys`). Likewise an `EachCol(Y)` cell's
column, `plate(EachCol(Y)) do y … obs = y`, is observed as `obs = Y[:, p]`.

Without the keyword, `obs` is the sub-model's own latent array, the
prior-predictive trajectory (`node_obs`). An observed array bound to anything
else (a row `Y[j, :]`, a computed vector) is refused, naming what to bind.

Current limits: a setup fill uses a literal index (`x[1] ~ …`; slice and
nested `@plate` setups are not supported yet); a `@scan` has one axis; an
untyped fresh cell whose distribution arguments are vector-shaped must be
annotated (`v[:, j]::vector[T] ~ …`) until the plate typing snag
`plate-untyped-ve-819ecfff` lands; a typed `~` local, and a typed `=` local
computed from data alone, inside a nested loop are not supported yet; a nested scan that observes a nested-vector group per step
without reading it back (`ys[p][t] ~ normal(h[t], s)`) is refused
(`unsupported statement LHS`), not supported yet; an observed scan written
inline in a `plate(EachCol(Y)) do y` cell over `y` is refused, not supported
yet (pass the column to a sub-model, or index `Y[t, p]`). A bare `for` in a
model body remains an error.

## Stan names of plate values

The Stan name of every value a plate or an annotated loop stores follows from
that value's own name. It does not change when another statement of the loop is
added, removed or reordered, or when another plate comes before it, so a program
run outside Julia (through BridgeStan from Python, say) can rely on it:

| Value | Stan name |
|---|---|
| An array a `@plate for` / `@scan` fills at the loop index (`mu[i] = …`, `x[i] ~ …`) | its own name: `mu`, `x` |
| A stored per-iteration local of a `@plate for` / `@scan` (`t ~ …`, or a `tmp = …` computed from data alone, in transformed data) | `loop_<variables>_<name>`: `loop_i_t` for `for i in …`, `loop_i_j_t` for `for i in …, j in …`, `loop_d_t` for `for d in doses` |
| The collected result of `rv ~ plate(…) do … end` | `rv` |
| A cell binding of that do-form | `<rv>_<name>`: `rv_t` |
| An observation's predictive draw and pointwise log likelihood | `<obs>_gen`, `<obs>_likelihood` |

A computed per-iteration local (`tmp = …` reading a parameter) is not an
output: it is a Stan local of the loops that read it, under the same
`loop_<variables>_<name>` name. A sub-model prefixes the names it declares as usual (`hh_x`),
data it binds itself included (`hh_k` for a `k` fixed with
`Base.merge(sub, (; k = …))`). A local or cell
binding whose name is already bound takes a numbered suffix (`loop_i_t_2` for a
second loop over `i` with a local `t`). An array a loop fills, or a do-form
result, named like a value that is already bound is refused, since a model never
rebinds a name. Names the compiler makes for loop-invariant expressions it lifts
out of a loop (`<prefix>__pl_inv<k>`, numbered in body order) are not part of
this contract.

**Ragged values.** A value whose length differs between cells (a `vector[K[i]]`
or `int[K[i]]` cell) is one flat Stan `vector` (or `array[] int`) under its
name: the cells concatenated in plate order. Its per-cell lengths and inclusive
1-based ends are `transformed data` (`<name>__pl_len`, `<name>__pl_end`), so they
are not in a fit's output: cell `g` is `<name>[ends[g-1]+1:ends[g]]`, with the
ends rebuilt from the data or read from `ModelOutput.segments` in
[`stan_descriptor`](@ref StanBlocks.stan_descriptor). After the loop, `mu` reads
as a `RaggedVector` in the model body, so `mu[g]` is cell `g`.

A ragged data input `y` (a vector of vectors) is passed to Stan the same way: a
tuple of its flat values and inclusive ends, with the two sizes,

```stan
int y_mem_n;    // number of values
int y_ends_n;   // number of groups
tuple(vector[y_mem_n], array[y_ends_n] int) y;
```

so `y = [[0.1, 0.2, 0.3], [0.4]]` is `y_mem_n = 4`, `y_ends_n = 2` and, in
CmdStan JSON, `"y": {"1": [0.1, 0.2, 0.3, 0.4], "2": [3, 4]}`. An integer group
uses `array[y_mem_n] int` for the values. A vector of matrices `X` adds each
matrix's row and column counts,

```stan
tuple(vector[X_mem_n], array[X_ends_n] int, array[X_rows_n] int, array[X_cols_n] int) X;
```

with each matrix column-major in the values: `X = [[1.0 3.0; 2.0 4.0], [5.0 6.0]]`
is `"X": {"1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "2": [4, 6], "3": [2, 1], "4": [2, 2]}`,
and its descriptor input lists the four components in `fields`.

## `@deffun` iteration boundary

Inside `@deffun`, index and value iteration, `enumerate`, `zip`, one-line nested
loops, and rectangular comprehensions with at most two axes are supported.
Filtered, stepped, flattened-ragged, and three-or-more-dimensional
comprehensions reject explicitly. `if`/`elseif`/`else` chains work. As in
Julia, an `if` has no scope of its own: a name first bound inside a branch and
read after the `if` is declared once before it, while a name used only inside
its branch stays local to that branch. UDFs cannot contain `~` or `target +=`.

Bodyful deterministic `@deffun` definitions are Stan-only by default. Add
`@juliacompat` to install a bounded Julia method for a helper that should also
be unit-tested directly. Probability/RNG/ODE families remain Stan-only, and the
opt-in target is not a promise of full Julia runtime parity for every Stan
builtin. `@stanonly` can document the default or opt one definition out of a
surrounding `@juliacompat` group. The math builtins Stan vectorises (`exp`,
`inv`, `log1p`, `fmax`, `fmin`, …) apply elementwise to containers in the Julia
method as in Stan: `inv` of a matrix is its elementwise reciprocal, not its
inverse, and `fmax` of two vectors pairs them up.

The same builtins trace over containers in a `@slic` or `@deffun` body, with
the shapes stanc accepts:

| Builtins | Container forms |
|---|---|
| unary (`exp`, `log1p`, `inv_logit`, `lgamma`, …) | a vector, row vector, matrix or real array, returned as the same container |
| `fmin`, `fmax`, `fdim`, `fmod`, `atan2`, `lbeta`, `lchoose`, `log_diff_exp`, `owens_t`, `gamma_p`, `gamma_q`, `log_modified_bessel_first_kind` | a scalar with one of those containers, or two containers of one kind |
| `binary_log_loss`, `bessel_first_kind`, `bessel_second_kind`, `modified_bessel_first_kind`, `modified_bessel_second_kind` | an int with any of those containers; an int array with a scalar, vector, row vector or real array |
| `fma` | scalars mixed with one vector, row vector or matrix kind |

An int scalar or int array takes a real one's place, as Stan promotes it, so
`fmax(nfit, 1.0)` with `nfit::vector[k]` is a `vector[k]`. Stan has no form
for two different container kinds (`fmax(v, rv)` with a row vector `rv`), a real
Bessel order, or an array `fma`; those fail at trace time. Nested arrays
(`array[] vector`) are not registered yet.

For a raw `matrix`, the final two annotation dimensions are the native matrix
shape and every leading dimension is a Stan array prefix. Thus
`matrix[K,N,J]` emits `array[] matrix`, while `matrix[M,K,N,J]` emits
`array[,] matrix`. These annotations work on arguments, returns, and fresh
locals; the default Julia method represents them as equal-rank dense arrays, so
use cross-target indexing such as `x[m,k,:,:]` when selecting a nested matrix.

## Embedded Laplace: integrating a Gaussian latent out

`z ~ laplace_marginal(prior) do z … end` integrates a Gaussian latent out of the
model block with Stan's embedded Laplace approximation (Stan 2.37 or later).
HMC samples only the remaining parameters; generated quantities draw `z`
afterwards. The do block is the latent's likelihood, written as ordinary
statements:

```julia
@slic (; y, X, k) begin
    s ~ exponential(1.0)
    tau ~ exponential(1.0)
    z::vector[k] ~ laplace_marginal(normal(0.0, tau)) do z
        y ~ normal(X * z, s)
    end
end
```

```stan
model {
    s ~ exponential(1.0);
    tau ~ exponential(1.0);
    target += laplace_marginal(closure_1, (y, X, s), k, closure_2, (tau, k));
}
generated quantities {
    vector[k] z = laplace_latent_rng(closure_1, (y, X, s), k, closure_2, (tau, k));
    vector[y_n] y_likelihood = normal_lpdfs(y, (X * z), s);
    vector[y_n] y_gen = normal_vector_rng(y_n, (X * z), s);
}
```

`closure_1` is the likelihood function `real (vector θ, y, X, s)`: the do block,
with each observation replaced by the log density its own `~` selects.
`closure_2` builds the prior covariance. The model writes no `target +=`;
the compiler does.

- **The prior** is the latent's Gaussian prior, spelled as it would be to sample
  the latent: `std_normal()`, `normal(mu, sigma)` (independent coordinates,
  scalars or vectors), `multi_normal(mu, Sigma)`, `multi_normal_cholesky(mu, L)`
  or `multi_normal_prec(mu, Omega)`. Stan's prior has mean zero, so a non-zero
  `mu` is added to the latent in the likelihood function and to the draw.
- **The latent's length** is its declared `vector[n]`, the prior's `n=`, or the
  length of the prior's first vector or matrix argument.
- **The do block** may bind values with `=` and observe data with any family,
  including the distribution combinators and custom `@lpxf` families. It may not
  sample a parameter or use control flow. Statements the observations do not need
  are left out of the likelihood function. Not supported yet, each refused with a
  message: an observation sub-model embedded in the do block, and a whole ragged
  observation (observe one group per latent, in a plate).
- **Generated quantities:** the do block replays after the draw, reading it. Each
  observation keeps its `<obs>_gen` and `<obs>_likelihood` twins, and the
  descriptor offers `:fit`, `:predict` and `:pointwise_loglik` as for the sampled
  latent. The outputs are those of the sampled spelling: in a `@plate for` a
  binding inside the do block is a loop local (`loop_g_eta`), as `eta = …` is
  there, and in a do-form cell it is stored under the cell's namespace
  (`rv_eta`). A likelihood term after the statement that reads `z` is refused: `z`
  is a draw there, so observe that data inside the do block.
- **Options** (keywords of `laplace_marginal`): `hessian_block_size` defaults to
  the latent's length, a dense Hessian, which is always correct. A smaller value
  declares a block-diagonal Hessian of the log likelihood, for example `1` when
  each observation reads one coordinate; a wrong one gives a wrong or `NaN`
  density. Any of `theta_init` (the Newton start for the latent, default the
  prior mean), `tol`, `max_num_steps`, `solver`, `max_steps_line_search` and
  `allow_fallthrough` selects `laplace_marginal_tol`; the controls left out take
  Stan Math 5.3.0's defaults (`1.49012e-8`, `500`, `1`, `1000`, `1`).
- **In a plate cell** (`plate(…) do`, `@plate for`) the latent is per cell: the
  model loop holds one `target +=` per cell, and a ragged observation keeps its
  flat `<obs>_gen` and per-group `<obs>_likelihood`. A cell latent's length (a
  ragged group's) is the data `int` Stan requires.
- **Nothing to condition on:** with every observation of the do block unbound
  (the prior simulation) or `maybecv`-held-out, the latent is drawn from its
  prior and the block replays as ordinary statements.

The approximation is exact for a Gaussian likelihood: such a model's log density
and gradient equal the analytic marginal's (`test/laplace.jl`). With BridgeStan
2.9.0 (Stan Math 5.3.0) the call returns `NaN` when the likelihood is not
log-concave and the LU solver meets a negative pivot; the fix is the open pull
request [stan-dev/math#3438](https://github.com/stan-dev/math/pull/3438).
Log-concave likelihoods (Gaussian, Poisson-log, Bernoulli-logit) are not
affected.

## Executable model descriptors

`stan_descriptor(model)` turns the traced model into read-only structured data:

```julia
d = stan_descriptor(model; name = :demo)

d.inputs       # type, size, constraints, value, observed/held-out/derived flags, tuple fields
d.outputs      # parameters, generated quantities, generative role, source, segments, dims, constraint values, tuple fields
d.definitions  # exact emitted Stan functions and dependency links
d.operations   # derived :transpile/:instantiate/:fit/:predict/:pointwise_loglik
```

The operation list fails closed and is derived from what the model can actually
do. `:fit` requires both a parameter and a live likelihood; `:predict` requires
a predictive draw; `:pointwise_loglik` requires a likelihood twin.

`ModelOutput.segments` is `nothing` for every dense output. For a ragged
observation twin or an emitted flat-memory carrier belonging to a ragged plate
result/member, it contains that exact output's inclusive 1-based group ends.
Different named members of one plate may use different ragged axes, so use the
`segments` on each output entry itself rather than borrowing a sibling's or an
input's layout. Re-binding a `StanModel`'s data updates these reflected
boundaries.

A ragged vector, a table, and `NamedTuple` or `Tuple` data are Stan tuple
inputs: a ragged `y` is emitted as `tuple(vector[y_mem_n], array[y_ends_n] int)
y`. Such an input's `fields` describes its components, one entry per component
in Stan tuple order, keyed by the Julia field name (`Symbol("1")`, … for a
positional `Tuple`), each with its `type`, `size`, `value` and own `fields`:
`y.fields.mem.size == (:y_mem_n,)`, and an integer ragged vector reports
`mem.type == :int`. Position `i` is the key `"i"` of the input's object in the
CmdStan JSON data. Component sizes such as `y_mem_n`, `y_ends_n` and a table's
`df_nrow` are `derived`, so `required_inputs` lists `y` alone. A tuple output (a
record a `@deffun` returns) carries the same `fields`, with each component's
resolved `dims` in place of `value`. Every other entry reports `fields == (;)`.

`ModelOutput.size` holds the emitted size expressions, which may be composite
(`n_levels - 1`) or name a transformed-data value (`J = maximum(subject)`) that
is not an input. `ModelOutput.dims` is the shape of the output's value, resolved
against the current data: one nonnegative `Int` per axis, `()` for a scalar. It
is `size` evaluated entry by entry, except that a constrained square matrix
declared by one size fills both of its matrix axes: `cholesky_factor_corr[K]`
(and `cholesky_factor_cov[K]`, `cov_matrix[K]`, `corr_matrix[K]`) reports
`(K, K)`, and an array `cholesky_factor_corr[G, K]` reports `(G, K, K)`. A
`vector[n_levels - 1]` at `n_levels == 1` reports `(0,)`, and for any output
that is not a tuple `prod(dims)` is the number of `param_names` entries
BridgeStan lists for it, so an empty output is distinguishable from a missing
one. Those entries run first index fastest, so `reshape(values, dims)` rebuilds
the value. Every top-level Stan size is data-only, so `dims` always exists; a size
computed through a Stan-only `@deffun` makes `stan_descriptor` throw, and
marking that function `@juliacompat` gives it the Julia method it needs.

`constraints` holds each declared `lower`/`upper`/`offset`/`multiplier` as
spelled: a literal, or an emitted expression such as `fmax(0.0, (2 * floor_))`.
`constraint_values` holds the same keys evaluated against the current data, as
`dims` is, so `real<lower=fmax(0.0, (2 * floor_))>` with `floor_ = 0.3` reports
`(lower = 0.6,)`. Evaluation follows Stan, so `lower = k / 2` with the integer
`k = 7` is `3`, and a builtin Stan vectorises applies elementwise:
`vector<lower=fmax(0.0, lo)>[2]` with `lo = [0.1, -0.2]` reports
`(lower = [0.1, 0.0],)`. A per-coordinate bound is an array with one entry per
coordinate, `-Inf` or `Inf` where a coordinate is unbounded. That includes the
bound arrays the compiler fills in `transformed data` for slice statements,
`vcat` priors, plate cells and `@scan` recurrences. A bound that names a data
input is that input's value. A bound that reads a parameter
(`b ~ normal(0, 1; lower = a)`) has no value without a draw and reports
`nothing`. A data-only bound computed through a Stan-only `@deffun` makes
`stan_descriptor` throw, as a size does; `@juliacompat` gives it a value.

```julia
problem = stan_execute(d, :fit)
pred = stan_execute(d, :predict; problem, draws = theta, seed = 123)
ll   = stan_execute(d, :pointwise_loglik; problem, draws = theta, seed = 123)
```

Use `stan_definition` and `stan_definition_closure` to select an included
function and its exact transitive dependency closure. Lookups and execution
raise on absent or ambiguous names/operations instead of guessing.

### Running a program outside Julia

BridgeStan runs `stan_code(model)` without Julia: its Python package compiles
the same program, and nutpie samples a BridgeStan-compiled model from Python.
BridgeStan is the runtime StanBlocks supports. StanBlocks itself evaluates,
samples and replays its programs through it (`instantiate`, `stan_execute`), so
a Python service using BridgeStan gets the same results. Two functions give it
what the program expects:

```julia
write("model.stan", stan_code(model))
write("model.data.json", stan_data_json(model))              # the bound data, as BridgeStan receives it
write("model.interface.json", StanBlocks.JSON.json(stan_interface(model), 2))
```

::: info CmdStan
StanBlocks does not support CmdStan as a runtime and does not shape its programs
for CmdStan's interfaces. The programs are ordinary Stan, but replaying a fit's
draws on new data, as `stan_execute(d, :predict)` does, relies on BridgeStan's
`param_constrain(…; include_tp = true, include_gq = true)`, which recomputes
transformed parameters and generated quantities on whatever data the model was
built with. CmdStan's standalone `generate_quantities` writes generated
quantities only: a transformed parameter, such as a linear predictor for the new
rows, is missing from its output, and CmdStanPy's `stan_variable` then answers
that name with the fit's draws for the fitted rows.
:::

`stan_interface(model)` is plain JSON data. Its `"data"` lists every data
variable in data-block order. A variable the caller supplies has an
`"encoding"`:

- `"value"`: written as is (a matrix as a list of rows);
- `"ragged"`: a list of groups, written as the tuple `{"1": <values
  concatenated>, "2": <inclusive ends>}`;
- `"tuple"`: a record, written as an object keyed by component position.

A size such as `y_mem_n` or `M_n` is `"supplied": false`, and its `"computed"`
rule names the value and axis it measures:
`{"rule": "size", "of": "y", "component": ["mem"], "axis": 1}`.

`"parameters"` lists the parameter variables in BridgeStan's order.
`"outputs"` gives each output's block, `dims` and generative role. For a
ragged output it also gives a `"segments_rule"` naming where its group ends come
from: `{"rule": "groups_of", "of": "idx"}` (the groups of a ragged input),
`{"rule": "lengths", "of": "K"}` (group `g` has `K[g]` values), or the per-cell
length as a Stan expression. `"program_id"` is the descriptor `id`, so a stale
interface file is detectable. A consumer that encodes its own values from the
interface writes the same JSON as `stan_data_json`.

## Current high-level limits

- `@slic` remains flat: no ordinary model-level control flow, mutation, or
  comprehension; the only model-level loops are the annotated `@plate for`
  and `@scan begin … end` forms (and the `plate` primitive they sugar).
- Only functions registered as builtins or defined through the SLIC macro
  surface can be traced.
- `plate` / `@plate for` are independent-cell constructs; `@scan` is the
  sequential form, with backward-only fixed-lag reads.
- General three-dimensional-and-higher Julia containers and matrix-valued plate
  cells are not part of the current surface.
- Ragged integer observations/predictive draws and per-element ragged
  pointwise likelihoods are not synthesized.
- `transpiles(model)` checks tracing only. Use `stanc_check`/`compiles` and,
  for semantic changes, a BridgeStan log-density/gradient check.

See the [Home guide](index.md) for the end-to-end workflow and the
[Sandbox Gallery](gallery.md) for short executable examples. The
[Worked examples](worked-examples.md) contain the longer migrated model
families and case studies, including their build-generated Stan programs.
