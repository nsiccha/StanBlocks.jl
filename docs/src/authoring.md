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
| `truncated(family, args...; lower, upper)` | Condition the latent draw to lie inside the supplied bound(s) | Yes | Rejection-sampled inside the inclusive bounds |
| `censored(family, args...; lower, upper)` | Place tail probability on threshold atoms; interior values keep their ordinary density | No | Clamp the base-family draw to the thresholds |
| `interval_censored(family, lower, upper, args...)` | Observe that the latent value lies in `(lower, upper]` | No | Draw from the uncoarsened base family |

For `truncated` and `censored`, at least one of `lower=` or `upper=` is
required. An omitted side and an explicit `nothing` are equivalent and are
removed at compile time. Bounds may be scalar or observation-shaped. Invalid
runtime bounds reject the Stan proposal. `lower` and `upper` are Stan reserved
identifiers, so name bound data `lo`/`hi` (or `lloq`/`uloq`) and map them with
`lower=lo, upper=hi` as above.

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
`EachCol`/`EachRow` count columns/rows; a `length(X)`-style `outer` carries no
trace-time value and stays unchecked. The compiler emits the necessary loops
separately in the Stan blocks that own the declarations, fills, densities,
and generated quantities.

### Plate support matrix

| Capability | Status | Exact boundary |
|---|---:|---|
| One-dimensional outer shape | ✅ | `outer=N` and `outer=(N,)` are equivalent |
| N-dimensional outer shape | ✅ | Use a non-empty tuple; with no positional inputs, take one do-block index per axis; tested through four axes |
| Positional input slicing | ✅ | One do-block argument per positional input, indexed over all outer axes |
| Lexically captured values | ✅ | Shared across cells; loop-invariant expressions are hoisted when safe |
| Fresh scalar `~` or `=` binding | ✅ | Collected as a `vector[N]` for one outer axis; additional axes use matrix/array storage. An integer binding (`k = 2 * i + 1`) stays an integer, `array[N] int`, so it can index |
| Fixed `vector[K]` binding/result | ✅ | Collected as `matrix[K,N]` for one outer axis; extra axes add Stan array prefixes |
| Observing a cell binding defined by `=` (`a = yi; a ~ …`) | ❌ | Refused: observe the data itself (`yi ~ …`, `ys[i] ~ …`), so that omitting it still gives the prior simulation |
| Sampled fixed `int[K]` cell binding | ✅ | Sampled (`~`) side bindings collect as `array[N,K] int` for one outer axis; deterministic int cells (findall/boolean-mask index arrays) still inline at each use |
| Dense `array[T] vector[K]` binding/result | ✅ | Collected as `array[outer...,T] vector[K]`; includes a raw ODE trajectory bound once and consumed inside the cell |
| Varying-length plain `vector[K[i]]` cell | ✅, narrow | One-dimensional outer shape and data-computable lengths only; represented by flat memory whose descriptor output carries that result/member's own inclusive group ends in `segments` |
| Varying-length `int[K[i]]` cell | ✅, narrow | One-dimensional outer shape and data-computable lengths only; flat-memory descriptor output carries inclusive group ends in `segments`, exactly like a varying-length vector cell |
| Fixed native-constrained vector cell | ✅ | `simplex[K]`, `ordered[K]`, and `positive_ordered[K]`; emitted as `array[outer...] <type>[K]` |
| Shared scalar constraints | ✅ | `lower`, `upper`, `offset`, and `multiplier` are preserved when they do not depend on the cell position |
| Cell-position-dependent constraint | ❌ | Rejected because the promoted declaration lives outside the loop |
| Matrix or other higher-rank cell value | ❌ | Declare shared matrices outside the plate; the sole dense rank-2 cell form is `array[T] vector[K]` |
| Varying-size constrained cell | ❌ | Declare the ragged constrained parameter at model scope, then index it in the plate |
| Int-array plate result | ❌ | MVP: sampled int side cells promote, but the loop's own return cannot collect as an int array; collect a scalar or vector cell instead |
| Positional `RaggedVector`, `EachCol`, `EachRow` | ✅ | Each cell receives one ragged group, matrix column, or matrix row |
| Named/anonymous `@slic` call inside the cell | ✅ | Internal fresh bindings are discovered, namespaced, and promoted per cell |
| A plate inside a called sub-model | ✅, dense | Scalar and fixed-vector cells work and caller sizes are substituted correctly |
| Ragged-vector plate inside a called sub-model | ❌ | Keep the ragged plate at model scope; a cell may still call a sub-model that returns a ragged-sized vector |
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
`array[T] vector[K]`. Matrix-per-cell and arbitrary nested-array values remain
unsupported.

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
body's fresh bindings once, promotes them to outer storage and emits a plain
Stan `for` loop into the blocks that own each statement. Inside the body:

- an LHS indexed by the loop variable (`x[i] ~ …`, `y[i] = …`) is a
  **model-scope array**, registered under its own name and visible after the
  loop; a whole vector cell is written `v[:, i]`;
- a bare fresh name (`t ~ …`, `tmp = …`) is a **per-iteration local**, hoisted
  to per-cell storage under a hygienic name (`x_t`, prefixed by the
  alphabetically first array the loop fills, or `loop_i_t` — the loop
  variables — when it fills none). A hygienic name that is already bound,
  such as a second array-less loop's `loop_i_t`, takes a numbered suffix
  (`loop_i_t_2`). For a namespace of your choosing, use the `plate` do-form,
  whose LHS names it even when the cell collects no result;
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
        t ~ exponential(1)            # per-cell local → vector<lower=0>[6] x_t
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
`v[t] ~ g(…)`) are just several indexed LHS. Per-step locals are sized by the
iteration count (`vector[T-1] x_eps` for a loop over `2:T`), so the
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
`for t in 2:length(y)` over a container group. Its carried arrays and per-step
locals are then stored as the plate's flat memory (one segment per cell, like a
varying-length plate cell), and element `t` of cell `j` is
`h__pl_mem[ragged_start(ends, j) + t - 1]`. This works in every mode, and support
reaches the flat memory as it does a fixed-length cell: a bound every element
shares is a scalar bound, and a setup and recurrence with different supports
take a per-element bound vector built in `transformed data`.

An integer a nested step computes, such as a time gap `k = d[t] - d[t - 1]` or
a carried int array (`c[t] = c[t - 1] + k`), is collected the same way: as
`array[S, T - 1] int` (one row per cell), or as flat int memory for a per-cell
length. It is data, so it is computed once in `transformed data`.

Current limits: a setup fill uses a literal index (`x[1] ~ …`; slice and
nested `@plate` setups are not supported yet); a `@scan` has one axis; an
untyped fresh cell whose distribution arguments are vector-shaped must be
annotated (`v[:, j]::vector[T] ~ …`) until the plate typing snag
`plate-untyped-ve-819ecfff` lands; typed locals inside a nested loop are not
supported yet. A bare `for` in a model body remains an error.

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
surrounding `@juliacompat` group.

For a raw `matrix`, the final two annotation dimensions are the native matrix
shape and every leading dimension is a Stan array prefix. Thus
`matrix[K,N,J]` emits `array[] matrix`, while `matrix[M,K,N,J]` emits
`array[,] matrix`. These annotations work on arguments, returns, and fresh
locals; the default Julia method represents them as equal-rank dense arrays, so
use cross-target indexing such as `x[m,k,:,:]` when selecting a nested matrix.

## Executable model descriptors

`stan_descriptor(model)` turns the traced model into read-only structured data:

```julia
d = stan_descriptor(model; name = :demo)

d.inputs       # type, size, value, observed/held-out/derived flags
d.outputs      # parameters, generated quantities, generative role, source, segments, dims
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
`segments` on each output entry itself; do not parse `__pl_*` names or borrow a
sibling/input layout. Re-binding a `StanModel`'s data updates these reflected
boundaries.

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

```julia
problem = stan_execute(d, :fit)
pred = stan_execute(d, :predict; problem, draws = theta, seed = 123)
ll   = stan_execute(d, :pointwise_loglik; problem, draws = theta, seed = 123)
```

Use `stan_definition` and `stan_definition_closure` to select an included
function and its exact transitive dependency closure. Lookups and execution
raise on absent or ambiguous names/operations instead of guessing.

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
