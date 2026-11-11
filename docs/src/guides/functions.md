# Functions and densities

Define deterministic calculations once and include their generated Stan
implementations in a probabilistic model.

## Deterministic functions

`@deffun` defines a Stan function. Argument and result annotations specify
carrier types or dispatch when needed. `@juliacompat` additionally permits a
supported deterministic definition to be called from ordinary Julia.

```@example functions
using StanBlocks
@deffun @juliacompat saturating_response(x::real, half::real)::real = x / (half + x)
show(stdout, MIME"text/plain"(), saturating_response(2.0, 3.0)); println()
```

The same definition is included when a model calls it. Unlike a submodel, a
function introduces no probability statements or latent parameters of its own.

An `@deffun @inline` helper expands into its caller while resolving sibling
helpers and omitted keyword defaults in its defining module. Explicit
arguments resolve in the caller's module, as do statements after the call.

Loading ReactiveKernels alongside StanBlocks enables an optional source
provider for complete named positional `@deffun @juliacompat` methods.
ReactiveKernels' `jax_source` can translate their loops and branches, including
host `prepare_data`; dimension guards remain active. Source follows Julia
dispatch and survives precompilation. Replacing a method stops use of its old
source. Keyword, default, vararg and anonymous arguments retain the existing
lowered-code route. Ordinary dense `matrix_exp` calls use JAX/SciPy's matrix
exponential; padded ragged matrices are unsupported. See the
[ReactiveKernels JAX guide](https://nsiccha.github.io/ReactiveKernels.jl/dev/jax)
for the supported source subset. Native execution and Stan emission retain
their existing implementations.

```@example functions
response = @slic (; x=2.0, y=0.45) begin
    half ~ lognormal(0, 1)
    expected = saturating_response(x, half)
    y ~ normal(expected, 0.1)
end
d = stan_descriptor(response)
show(stdout, MIME"text/plain"(), [(name=f.name, signature=f.signature) for f in d.definitions]); println()
```

The descriptor lists the actual emitted function signatures. `stan_definition`
retrieves one definition, and `stan_definition_closure` retrieves its dependency
closure. Use these APIs when building an editor or exporter; do not maintain a
parallel list of helper names.

## Control flow

A function body may use conditional branches, loops, local buffers and indexed
assignment. A model body instead describes probability statements and uses the
[annotated loops](arrays.md). This distinction keeps a numerical algorithm's
local work separate from inferential parameters.

```@example functions
@deffun absolute_sum(x::vector[n])::real = begin
    total = 0.0
    for i in 1:n
        if x[i] < 0
            total += -x[i]
        else
            total += x[i]
        end
    end
    total
end
with_function = @slic (; x=[-1.0, 0.5], y=1.4) begin
    scale ~ lognormal(0, 1)
    y ~ normal(scale * absolute_sum(x), 1)
end
println(stan_code(with_function))
```

A `@deffun` is Stan-only unless Julia compatibility is requested. Do not call a
Stan-only helper from Julia preprocessing. Keep preprocessing as ordinary Julia
code, or use `@juliacompat` when the required operators have Julia methods.
Module-level numeric constants are not implicit data arguments. Pass values
explicitly; built-in constants such as `pi` use their Julia spelling.

## Probability families

A custom family can provide an aggregate density, pointwise contributions and
a random-number generator. These serve different operations. `@lpxf` registers
the density, while `@lhs` supplies left-hand-side inference when the family
needs to determine a latent carrier. `family_support` reports the observation type and intrinsic support;
`ValueFamily` and `ValueUDF` are alternatives for producers that construct
families as values.

A custom density must have the intended support, normalization and
transformation Jacobian. A function that returns zero can be an explicitly
chosen improper kernel, but it does not establish a uniform probability law
on a transformed space. A density implementation alone also does not supply
a predictive random-number generator. Check the operations the descriptor
actually offers before attempting prediction.

Use the [model API reference](../reference/models.md) for these registration
interfaces. The [epidemic example](../examples/epidemic.md) shows a typed ODE
right-hand side, its emitted Stan definition and a numerical trajectory.
