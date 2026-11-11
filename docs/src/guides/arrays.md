# Arrays and loops

Represent vectors, matrices, unequal-length groups and sequential latent states
without writing an unsupported model-level loop.

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
Do not read another cell's output from an independent plate.

Cells can also contain vectors or matrices. Fixed matrix dimensions retain
their matrix carrier inside each cell; varying dimensions use flat storage
with layout metadata. Read the descriptor's `dims` and `segments` when
reconstructing values instead of inferring shapes from generated variable
names. The do-block spelling `plate(...) do ... end` broadcasts a reusable
submodel; [partial pooling](../examples/pooling.md) shows that form.

## Unequal-length groups

Nested vectors preserve group boundaries. For a ragged observation, the
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
