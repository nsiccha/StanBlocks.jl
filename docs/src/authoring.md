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
| `weighted(family, w, args...)` | Multiply an observation's log density by data-qualified `w` | No | Draw from the unweighted base family |
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
weighted priors and parameter-dependent weights reject. The other combinators
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
| Ragged slice, inside `plate` | Not synthesized | Not synthesized |

All four forms still contribute their density to the model. The last two rows
are different generated-quantity capabilities, not different likelihood
semantics.

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
bindings are promoted to outer storage, and the trailing expression is collected
as the plate result:

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
every cell. The compiler emits the necessary loops separately in the Stan
blocks that own the declarations, fills, densities, and generated quantities.

### Plate support matrix

| Capability | Status | Exact boundary |
|---|---:|---|
| One-dimensional outer shape | ✅ | `outer=N` and `outer=(N,)` are equivalent |
| N-dimensional outer shape | ✅ | Use a non-empty tuple; with no positional inputs, take one do-block index per axis; tested through four axes |
| Positional input slicing | ✅ | One do-block argument per positional input, indexed over all outer axes |
| Lexically captured values | ✅ | Shared across cells; loop-invariant expressions are hoisted when safe |
| Fresh scalar `~` or `=` binding | ✅ | Collected as a `vector[N]` for one outer axis; additional axes use matrix/array storage |
| Fixed `vector[K]` binding/result | ✅ | Collected as `matrix[K,N]` for one outer axis; extra axes add Stan array prefixes |
| Varying-length plain `vector[K[i]]` cell | ✅, narrow | One-dimensional outer shape and data-computable lengths only; represented by flat memory whose descriptor output carries that result/member's own inclusive group ends in `segments` |
| Fixed native-constrained vector cell | ✅ | `simplex[K]`, `ordered[K]`, and `positive_ordered[K]`; emitted as `array[outer...] <type>[K]` |
| Shared scalar constraints | ✅ | `lower`, `upper`, `offset`, and `multiplier` are preserved when they do not depend on the cell position |
| Cell-position-dependent constraint | ❌ | Rejected because the promoted declaration lives outside the loop |
| Matrix or higher-rank cell value | ❌ | Declare shared matrices outside the plate and return a scalar/vector cell |
| Varying-size constrained cell | ❌ | Declare the ragged constrained parameter at model scope, then index it in the plate |
| Positional `RaggedVector`, `EachCol`, `EachRow` | ✅ | Each cell receives one ragged group, matrix column, or matrix row |
| Named/anonymous `@slic` call inside the cell | ✅ | Internal fresh bindings are discovered, namespaced, and promoted per cell |
| A plate inside a called sub-model | ✅, dense | Scalar and fixed-vector cells work and caller sizes are substituted correctly |
| Ragged-vector plate inside a called sub-model | ❌ | Keep the ragged plate at model scope; a cell may still call a sub-model that returns a ragged-sized vector |
| Several independent plates reusing local names | ✅ | Cell-local bindings are hygienically namespaced by plate result |
| Dependency on a previous cell | ❌ | Cells are independent; use a deterministic `@deffun` recurrence for scans/folds |
| Vararg do-block parameters | ❌ | Use a fixed positional argument list |
| Body without trailing value | ❌ | The last expression must be the cell result, not `~` or `=` |
| Fully dead/prior-only plate | ✅ | Fresh prior samples remain parameters so the collected transformed value stays valid |
| Cross-validation taint | ✅ | Taint from `outer` sizes or a cell prior RHS moves the affected cell path to generated quantities |

### Vector cells and shared multivariate priors

```julia
@slic (; n_groups = 8, k = 3) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2)
    tau::vector[k] ~ normal(0, 1; lower = 0)

    b::vector[k] ~ plate(; outer = n_groups) do g
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

## `@deffun` iteration boundary

Inside `@deffun`, index and value iteration, `enumerate`, `zip`, one-line nested
loops, and rectangular comprehensions with at most two axes are supported.
Filtered, stepped, flattened-ragged, and three-or-more-dimensional
comprehensions reject explicitly. `if`/`else` works; write nested `if` blocks
instead of an `elseif` chain. UDFs cannot contain `~` or `target +=`.

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
d.outputs      # parameters, generated quantities, generative role, source, segments
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
  comprehension.
- Only functions registered as builtins or defined through the SLIC macro
  surface can be traced.
- `plate` is an independent-cell construct, not a general loop or scan.
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
