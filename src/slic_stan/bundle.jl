_slic_bundle_definition(definition, index) = begin
    Meta.isexpr(definition, :(=)) && Meta.isexpr(definition.args[1], :call) || error(
        "compile_slic_bundle: definition $index must be a named SLIC definition " *
        "like `f(args...) = begin ... end`, got `$(definition)`."
    )
    definition.args[1].args[1] isa Symbol || error(
        "compile_slic_bundle: definition $index must have a bare Symbol as its name, " *
        "got `$(definition.args[1].args[1])`."
    )
    definition
end

_slic_bundle_macrocall(args...) = Expr(
    :macrocall,
    GlobalRef(@__MODULE__, Symbol("@slic")),
    LineNumberNode(0, :slic_bundle),
    args...,
)

"""
    compile_slic_bundle(data, definitions, body; name=nothing)

Compile an ordered collection of named SLIC sub-model definitions together
with one parent SLIC model body. `definitions` contains expressions shaped like
`f(args...) = begin ... end` (without an outer `@slic` macro call), and `body`
is the parent model expression. The definitions are installed in their supplied
order in a fresh module, then the parent is traced exactly once in that same
module.

The result is a named tuple with:

- `model`: the traced [`StanModel`](@ref), ready for cheap data rebinding;
- `descriptor`: its [`ModelDescriptor`](@ref), with `name` forwarded to
  [`stan_descriptor`](@ref);
- `code`: the generated Stan source.

This is an integration boundary for already-parsed, trusted SLIC expressions.
It does not parse or sandbox source, and it deliberately preserves `@slic`'s
macro-expansion semantics: macros inside a supplied expression can execute
Julia code during expansion. Validate untrusted source before calling this
function and isolate its execution when it comes from an anonymous user.

# Example

```julia
definitions = [:(latent(scale) = begin
    z ~ normal(0, scale)
    return z
end)]
body = quote
    mu ~ latent(1.0)
    y ~ normal(mu, 1.0)
end

result = compile_slic_bundle((; y=[0.1, -0.2]), definitions, body;
    name=:hierarchical)
result.code
result.descriptor.inputs
```
"""
function compile_slic_bundle(data, definitions, body::Expr; name=nothing)
    checked = map(enumerate(collect(definitions))) do (index, definition)
        _slic_bundle_definition(definition, index)
    end

    workspace = Module(gensym(:StanBlocksSlicBundle))
    for definition in checked
        Core.eval(workspace, _slic_bundle_macrocall(definition))
    end
    slic = Core.eval(workspace, _slic_bundle_macrocall(QuoteNode(data), body))

    # `Core.eval` installed callable methods in a newer world than this compiled
    # function. Keep tracing and all descriptor reads in that newest world.
    Base.invokelatest() do
        model = stan_model(slic)
        descriptor = stan_descriptor(model; name)
        (; model, descriptor, code=stan_code(model))
    end
end

compile_slic_bundle(_data, _definitions, body; name=nothing) = error(
    "compile_slic_bundle: body must be an Expr, got $(typeof(body))."
)
