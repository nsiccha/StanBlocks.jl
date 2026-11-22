# Arrays and loops

Represent vectors, matrices, unequal-length groups and sequential latent states
without writing an unsupported model-level loop.

Continuous elementwise families also accept matrix-shaped parameters and
observations. From v0.2.37, `z ~ normal(M, 1.0)` infers `M`'s matrix shape;
the typed form is `z::matrix[R, C] ~ normal(M, 1.0)`. Every argument must be
a scalar or a matrix of that shape.

## Vectors and matrices

A numeric Julia vector binds as a Stan vector when the family expects real
values; an integer count response retains its integer carrier. Matrix products
use `*`, while `.*`, `./` and `.+` request elementwise arithmetic.

```@example arrays
using StanBlocks
matrix_model = @slic (; X=[1.0 0.2; 1.0 0.5; 1.0 0.9], y=[0.1, 0.4, 0.8]) begin
    beta::vector[2] ~ normal(0, 1)
    mu = X * beta
    y ~ normal(mu, 1)
end
show(stdout, MIME"text/plain"(), [(name=o.name, type=o.type, shape=o.dims) for o in stan_descriptor(matrix_model).outputs]); println()
```

An annotation states the carrier and its shape. It is not a declaration that
can subsequently be filled by arbitrary indexed assignments in the model body.
Use a whole-object expression, a function, or one of the annotated loops below.

## Independent cells

`@plate` declares independent cells conditional on their shared inputs. An
indexed sampling statement introduces one latent value per cell. The collected
array can be used by an observation outside the loop.

```@example arrays
independent = @slic (; y=[0.2, -0.1, 0.4]) begin
    @plate for i in eachindex(y)
        theta[i] ~ normal(0, 1)
    end
    y ~ normal(theta, 1)
end
println(stan_code(independent))
```

Within a cell, an indexed left-hand side is an output available after the
loop. A fresh unindexed probability statement introduces a cell-local
parameter. A fresh unindexed assignment is scratch work, not an output array.
One computed from data alone, an integer index array such as
`sort_indices_asc(x)` included, is filled once in `transformed data` and read
back per cell.
Do not read another cell's output from an independent plate.

Cells can also contain vectors or matrices. Fixed matrix dimensions retain
their matrix carrier inside each cell; varying dimensions use flat storage
with layout metadata. Read the descriptor's `dims` and `segments` when
reconstructing values instead of inferring shapes from generated variable
names. The do-block spelling `plate(...) do ... end` broadcasts a reusable
submodel; [partial pooling](../examples/pooling.md) shows that form.

A submodel that returns a tuple, such as `return (; curve = gain * x, scale)`,
can be bound in a cell and read by field (`cell.curve`). A tuple whose
components have the same size in every cell is collected as an array of
tuples. From v0.2.57 one whose component length differs by cell is a local of
the loop instead, recomputed where it is read and never an output: bind a
field as its own cell value (`curve = cell.curve`) to keep it after the loop.

For several independent axes, use one annotated loop such as
`@plate for j in 1:J, s in 1:S` and index outputs as `x[j, s]`.
From v0.2.62 a `plate(...) do` may also be collected in the cell of a one-axis
plate or annotated loop, directly or inside a called submodel, one level deep:
its cells are stored over one flattened inner index, and a cell that returns
a ragged nested result collects a two-level value, read as `x[e][s]`. From
v0.2.63 the descriptor reports both levels of such a value's flat storage:
`segments` ends each inner vector and `groups` each outer group, counted in
inner vectors, as a nested-vector input's `groups` does. Likewise, observe shared
data outside a loop or index it by cell, rather than repeating the entire
observation in every cell.

## Unequal-length groups

Nested vectors preserve group boundaries. From v0.2.61 vectors nested three
deep (readings per subject per endpoint) are one two-level ragged value: `x[e]`
is group `e` as a ragged value and `x[e][s]` one vector. For a ragged observation, the
pointwise log likelihood is one aggregate value per group, while predictive
values use flat storage plus group-end metadata.

```@example arrays
ragged = @slic (; y=[[0.2, -0.1], [0.4]]) begin
    mu ~ normal(0, 1)
    y ~ normal(mu, 1)
end
show(stdout, MIME"text/plain"(), [(name=o.name, role=o.generative, groups=o.segments) for o in stan_descriptor(ragged).outputs]); println()
```

Choose integer nested vectors for count families and real nested vectors for
continuous families. A group-level likelihood cannot in general be split into
independent element contributions: a multivariate family may couple the
entries of the group.

A family's own functions take a whole ragged value too, for example inside a
`@deffun` (from v0.2.55). `foo_lpdf(y, args...)` adds the density of every
group, `foo_lpdfs(y, args...)` returns one aggregate density per group, and
`foo_rng(mus, args...)` draws each group of the ragged argument `mus` from the
family's sized `foo_rng(vector[n], ...)`, returning a ragged value with the same
groups. Ragged arguments are taken group by group; scalars and dense values are
shared by every group. A family that defines its own method for a whole ragged
value keeps it.

```@example arrays
const RaggedVector = StanBlocks.RaggedVector
@deffun group_density(y::RaggedVector, mus::RaggedVector, s::real)::real =
    normal_lpdf(y, mus, s)
grouped = @slic (; y=[[0.2, -0.1], [0.4]], mus=[[0.0, 0.1], [0.3]]) begin
    s ~ exponential(1)
    lp = group_density(y, mus, s)
end
println(stan_code(grouped))
```

## Sequential states

`@scan` gives a recurrence an explicit initial state. Its trailing loop may
read earlier elements at fixed lags. The observation below reads the collected
state outside the recurrence.

```@example arrays
series = @slic (; y=[0.2, -0.1, 0.4], T=3) begin
    phi ~ normal(0, 0.5)
    @scan begin
        state[1] ~ normal(0, 1)
        for t in 2:T
            state[t] ~ normal(phi * state[t-1], 1)
        end
    end
    y ~ normal(state, 1)
end
check = stanc_check(stan_code(series))
@assert check.ok check.output
show(stdout, MIME"text/plain"(), [(name=o.name, shape=o.dims) for o in stan_descriptor(series).outputs if o.kind == :parameter]); println()
```

The initial statements and loop must fill every carried element exactly once;
forward reads are not allowed. When the carried variable is itself observed,
its predictive twin simulates a complete new trajectory, using earlier
simulated values at the lagged positions. A one-step conditional prediction
from observed history is a different operation.

The [partial-pooling example](../examples/pooling.md) uses independent cells;
[epidemic dynamics](../examples/epidemic.md) uses a deterministic ODE function
when a numerical solver, rather than a latent recurrence, defines the states.
