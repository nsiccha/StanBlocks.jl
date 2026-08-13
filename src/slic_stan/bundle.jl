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

_slic_bundle_udf_definition(definition, index) = begin
    definition isa Expr || error(
        "compile_slic_bundle: UDF definition $index must be an expression accepted by " *
        "`@deffun`, got $(typeof(definition))."
    )
    definition
end

_slic_bundle_with_source_part(x, _source_part) = x
_slic_bundle_with_source_part(x::LineNumberNode, source_part) =
    LineNumberNode(x.line, source_part)
_slic_bundle_with_source_part(x::Expr, source_part) =
    Expr(x.head, map(arg -> _slic_bundle_with_source_part(arg, source_part), x.args)...)

_slic_bundle_part(value, label, index=nothing) = begin
    labeled = value isa Pair
    source_part, expression = labeled ? (first(value), last(value)) : (:slic_bundle, value)
    location = isnothing(index) ? label : "$label $index"
    (source_part isa Symbol || source_part isa AbstractString) || error(
        "compile_slic_bundle: $location source part must be a Symbol or string, " *
        "got $(typeof(source_part))."
    )
    isempty(string(source_part)) && error(
        "compile_slic_bundle: $location source part must not be empty."
    )
    source_part = Symbol(source_part)
    expression = labeled ? _slic_bundle_with_source_part(expression, source_part) : expression
    (; source_part, expression)
end

_slic_bundle_macrocall(macro_name, source_part, args...) = Expr(
    :macrocall,
    GlobalRef(@__MODULE__, macro_name),
    LineNumberNode(0, source_part),
    args...,
)

"""
    compile_slic_bundle(data, definitions, body; udf_definitions=(), name=nothing)

Compile an ordered collection of `@deffun` UDF definitions, named SLIC
sub-model definitions, and one parent SLIC model body. `udf_definitions`
contains expressions accepted by `@deffun`; `definitions` contains expressions
shaped like `f(args...) = begin ... end` (without an outer `@slic` macro call);
and `body` is the parent model expression. UDFs are installed first in their
supplied order, then the named SLIC definitions, in a fresh module. The parent
is traced exactly once in that same module.

Each UDF or SLIC definition, and the parent body, may be written as either a
bare expression or `source_part => expression`, where `source_part` is a
`Symbol` or string. Labeled parts preserve that identity in [`diagnostic`](@ref)
results. Bare expressions retain the legacy source-location behavior, with
`slic_bundle` used when the expression has no source file.

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
    z ~ normal(shift_zero(0.0), scale)
    return z
end)]
udf_definitions = ["shift-zero" => :(shift_zero(x::real)::real = begin
    x + 0.25
end)]
body = "parent" => quote
    mu ~ latent(1.0)
    y ~ normal(mu, 1.0)
end

result = compile_slic_bundle((; y=[0.1, -0.2]), definitions, body;
    udf_definitions, name=:hierarchical)
result.code
result.descriptor.inputs
```
"""
function compile_slic_bundle(data, definitions, body; udf_definitions=(), name=nothing)
    checked_udfs = map(enumerate(collect(udf_definitions))) do (index, definition)
        part = _slic_bundle_part(definition, "UDF definition", index)
        (; part.source_part,
           expression=_slic_bundle_udf_definition(part.expression, index))
    end
    checked_definitions = map(enumerate(collect(definitions))) do (index, definition)
        part = _slic_bundle_part(definition, "definition", index)
        (; part.source_part,
           expression=_slic_bundle_definition(part.expression, index))
    end
    checked_body = _slic_bundle_part(body, "body")
    checked_body.expression isa Expr || error(
        "compile_slic_bundle: body must be an Expr, got $(typeof(checked_body.expression))."
    )

    workspace = Module(gensym(:StanBlocksSlicBundle))
    for definition in checked_udfs
        Core.eval(workspace, _slic_bundle_macrocall(
            Symbol("@deffun"), definition.source_part, definition.expression))
    end
    for definition in checked_definitions
        Core.eval(workspace, _slic_bundle_macrocall(
            Symbol("@slic"), definition.source_part, definition.expression))
    end
    slic = Core.eval(workspace, _slic_bundle_macrocall(
        Symbol("@slic"), checked_body.source_part, QuoteNode(data), checked_body.expression))

    # `Core.eval` installed callable methods in a newer world than this compiled
    # function. Keep tracing and all descriptor reads in that newest world.
    Base.invokelatest() do
        model = stan_model(slic)
        descriptor = stan_descriptor(model; name)
        (; model, descriptor, code=stan_code(model))
    end
end
