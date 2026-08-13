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

const _SLIC_BUNDLE_UDF_METADATA_FIELDS = (:definition, :markers, :assertions)
const _SLIC_BUNDLE_UDF_MARKERS = (:stanonly, :juliacompat, :lhs, :lpxf, :inline)
const _SLIC_BUNDLE_ASSERTION_FIELDS = (:condition, :message)

_slic_bundle_udf_marker_values(markers, index) = begin
    marker_values = if markers isa Symbol
        (markers,)
    elseif markers isa Tuple || markers isa AbstractVector
        Tuple(markers)
    else
        error(
            "compile_slic_bundle: UDF definition $index markers must be a Symbol, " *
            "tuple, or vector, got $(typeof(markers))."
        )
    end
    for marker in marker_values
        marker isa Symbol || error(
            "compile_slic_bundle: UDF definition $index markers must be Symbols, " *
            "got $(typeof(marker))."
        )
        marker in _SLIC_BUNDLE_UDF_MARKERS || error(
            "compile_slic_bundle: UDF definition $index has unknown marker `$marker`; " *
            "expected one of $(join(("`$value`" for value in _SLIC_BUNDLE_UDF_MARKERS), ", "))."
        )
    end
    length(unique(marker_values)) == length(marker_values) || error(
        "compile_slic_bundle: UDF definition $index markers must not contain duplicates."
    )
    (:inline in marker_values && (:lhs in marker_values || :lpxf in marker_values)) && error(
        "compile_slic_bundle: UDF definition $index marker `inline` cannot be combined " *
        "with `lhs` or `lpxf`."
    )
    marker_values
end

_slic_bundle_assertion_macrocalls(assertions, index, source_part) = begin
    (assertions isa Tuple || assertions isa AbstractVector) || error(
        "compile_slic_bundle: UDF definition $index assertions must be a tuple or vector, " *
        "got $(typeof(assertions))."
    )
    map(enumerate(assertions)) do (assertion_index, assertion)
        assertion isa NamedTuple || error(
            "compile_slic_bundle: UDF definition $index assertion $assertion_index must " *
            "be a named tuple, got $(typeof(assertion))."
        )
        unknown = filter(field -> !(field in _SLIC_BUNDLE_ASSERTION_FIELDS), keys(assertion))
        isempty(unknown) || error(
            "compile_slic_bundle: UDF definition $index assertion $assertion_index has " *
            "unknown field `$(first(unknown))`; expected `condition` and optional `message`."
        )
        haskey(assertion, :condition) || error(
            "compile_slic_bundle: UDF definition $index assertion $assertion_index is " *
            "missing required field `condition`."
        )
        condition = assertion.condition
        (condition isa Expr || condition isa Symbol || condition isa Bool) || error(
            "compile_slic_bundle: UDF definition $index assertion $assertion_index " *
            "condition must be an Expr, Symbol, or Bool, got $(typeof(condition))."
        )
        if haskey(assertion, :message)
            assertion.message isa AbstractString || error(
                "compile_slic_bundle: UDF definition $index assertion $assertion_index " *
                "message must be a string, got $(typeof(assertion.message))."
            )
            _slic_bundle_macrocall(
                Symbol("@stan_assert"), source_part, condition, assertion.message)
        else
            _slic_bundle_macrocall(Symbol("@stan_assert"), source_part, condition)
        end
    end
end

_slic_bundle_insert_assertions(definition, assertions, index) = begin
    isempty(assertions) && return definition
    Meta.isexpr(definition, :(=)) && length(definition.args) == 2 || error(
        "compile_slic_bundle: UDF definition $index assertions require one bodyful " *
        "function definition."
    )
    signature, body = definition.args
    Meta.isexpr(body, :block) || error(
        "compile_slic_bundle: UDF definition $index assertions require a `begin ... end` body."
    )
    Expr(:(=), signature, Expr(:block, assertions..., body.args...))
end

_slic_bundle_udf_marker_call(marker, source_part, definition) = Expr(
    :macrocall,
    Symbol("@", string(marker)),
    LineNumberNode(0, source_part),
    definition,
)

_slic_bundle_udf_definition(definition, index, _source_part) = begin
    definition isa Expr || error(
        "compile_slic_bundle: UDF definition $index must be an expression accepted by " *
        "`@deffun` or a marker-metadata named tuple, got $(typeof(definition))."
    )
    definition
end

_slic_bundle_udf_definition(metadata::NamedTuple, index, source_part) = begin
    unknown = filter(field -> !(field in _SLIC_BUNDLE_UDF_METADATA_FIELDS), keys(metadata))
    isempty(unknown) || error(
        "compile_slic_bundle: UDF definition $index marker metadata has unknown field " *
        "`$(first(unknown))`; expected `definition`, optional `markers`, and optional `assertions`."
    )
    haskey(metadata, :definition) || error(
        "compile_slic_bundle: UDF definition $index marker metadata is missing required " *
        "field `definition`."
    )
    definition = metadata.definition
    definition isa Expr || error(
        "compile_slic_bundle: UDF definition $index metadata field `definition` must be " *
        "an expression accepted by `@deffun`, got $(typeof(definition))."
    )
    markers = _slic_bundle_udf_marker_values(
        haskey(metadata, :markers) ? metadata.markers : (), index)
    assertions = _slic_bundle_assertion_macrocalls(
        haskey(metadata, :assertions) ? metadata.assertions : (), index, source_part)
    lowered = _slic_bundle_insert_assertions(definition, assertions, index)
    for marker in reverse(markers)
        lowered = _slic_bundle_udf_marker_call(marker, source_part, lowered)
    end
    lowered
end

_slic_bundle_with_source_part(x, _source_part) = x
_slic_bundle_with_source_part(x::LineNumberNode, source_part) =
    LineNumberNode(x.line, source_part)
_slic_bundle_with_source_part(x::Expr, source_part) =
    Expr(x.head, map(arg -> _slic_bundle_with_source_part(arg, source_part), x.args)...)
_slic_bundle_with_source_part(x::SlicModel, source_part) = begin
    body = _slic_bundle_with_source_part(model(x), source_part)
    body = Meta.isexpr(body, :block) ?
        Expr(:block, LineNumberNode(0, source_part), body.args...) :
        Expr(:block, LineNumberNode(0, source_part), body)
    SlicModel(body, data(x), x.mod)
end

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
    (; source_part, expression, labeled)
end

_slic_bundle_macrocall(macro_name, source_part, args...) = Expr(
    :macrocall,
    GlobalRef(@__MODULE__, macro_name),
    LineNumberNode(0, source_part),
    args...,
)

_slic_bundle_anonymous_submodel(dependency, index) = begin
    labeled = dependency isa Pair && last(dependency) isa Pair
    named = labeled ? last(dependency) : dependency
    named isa Pair || error(
        "compile_slic_bundle: anonymous submodel $index must be `name => body_or_value` " *
        "or `source_part => (name => body_or_value)`, got $(typeof(dependency))."
    )

    binding, value = first(named), last(named)
    binding isa Symbol || error(
        "compile_slic_bundle: anonymous submodel $index must have a bare Symbol as " *
        "its parent binding, got $(typeof(binding))."
    )
    (value isa Expr || value isa SlicModel) || error(
        "compile_slic_bundle: anonymous submodel $index value must be an anonymous " *
        "SLIC body Expr or SlicModel, got $(typeof(value))."
    )
    if value isa Expr && Meta.isexpr(value, :(=)) && Meta.isexpr(value.args[1], :call)
        error(
            "compile_slic_bundle: anonymous submodel $index is a named SLIC definition; " *
            "put it in `definitions` instead."
        )
    end

    part = _slic_bundle_part(
        labeled ? first(dependency) => value : value,
        "anonymous submodel",
        index,
    )
    (; binding, part.source_part, value=part.expression)
end

_slic_bundle_bind_anonymous!(workspace, dependency, index) = begin
    isdefined(workspace, dependency.binding) && error(
        "compile_slic_bundle: anonymous submodel $index parent binding " *
        "`$(dependency.binding)` conflicts with an earlier workspace definition."
    )
    value = dependency.value isa Expr ?
        Core.eval(workspace, _slic_bundle_macrocall(
            Symbol("@slic"), dependency.source_part, dependency.value)) :
        dependency.value
    value isa SlicModel || error(
        "compile_slic_bundle: anonymous submodel $index did not produce a SlicModel."
    )
    Core.eval(workspace, Expr(
        :const,
        Expr(:(=), dependency.binding, QuoteNode(value)),
    ))
    value
end

"""
    compile_slic_bundle(data, definitions, body;
        udf_definitions=(), anonymous_submodels=(), name=nothing)

Compile an ordered collection of `@deffun` UDF definitions, named SLIC
sub-model definitions, anonymous sub-model dependencies, and one parent SLIC
model body. Each `udf_definitions` value is either an expression accepted by
`@deffun` or a named tuple with a required `definition` expression plus optional
`markers` and `assertions`. `definitions` contains expressions shaped like
`f(args...) = begin ... end` (without an outer `@slic` macro call);
`anonymous_submodels` contains `name => body_or_value` pairs, where `name` is
the Symbol used by the parent and the value is either an anonymous SLIC body
expression (without an outer `@slic`) or an existing `SlicModel`; and `body` is
the parent model expression. UDFs are installed first in their supplied order,
then named SLIC definitions, then anonymous dependencies, in a fresh module. The
parent is traced exactly once in that same module.

Structured UDF metadata is the macro-free spelling of trusted compiler-owned
annotations. `markers` accepts one Symbol or a tuple/vector drawn from
`:stanonly`, `:juliacompat`, `:lhs`, `:lpxf`, and `:inline`; `:inline` cannot be
combined with `:lhs` or `:lpxf`. `assertions` is a tuple/vector of named tuples
with a `condition` AST and optional string `message`. Assertions are prepended
to one bodyful definition with the semantics of [`@stan_assert`](@ref). Unknown
fields, markers, duplicates, and malformed assertions fail before evaluation.

Each UDF or SLIC definition, and the parent body, may be written as either a
bare expression or `source_part => expression`, where `source_part` is a
`Symbol` or string. Labeled parts preserve that identity in [`diagnostic`](@ref)
results. Bare expressions retain the legacy source-location behavior, with
`slic_bundle` used when the expression has no source file.

Each anonymous dependency is either `name => body_or_value` or
`source_part => (name => body_or_value)`. A body expression is expanded by
`@slic` inside the owned workspace, so it sees the bundle's earlier definitions
without caller-side `Core.eval`. An existing `SlicModel` keeps its original
defining module. In both forms the parent refers to the dependency by `name`,
and repeated calls keep ordinary anonymous-submodel kwarg binding and hygienic
LHS namespaces.

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
anonymous_submodels = ["local-prior" => (:local_prior => quote
    z ~ normal(location, scale)
    return z
end)]
udf_definitions = ["shift-zero" => :(shift_zero(x::real)::real = begin
    x + 0.25
end), "safe-log" => (;
    definition=:(safe_log(x::real)::real = begin
        log(x)
    end),
    markers=:stanonly,
    assertions=((;
        condition=:(x > 0),
        message="safe_log: x must be positive",
    ),),
)]
body = "parent" => quote
    mu ~ latent(1.0)
    offset ~ local_prior(; location=0.0, scale=1.0)
    y ~ normal(mu + offset, 1.0)
end

result = compile_slic_bundle((; y=[0.1, -0.2]), definitions, body;
    udf_definitions, anonymous_submodels, name=:hierarchical)
result.code
result.descriptor.inputs
```
"""
function compile_slic_bundle(data, definitions, body;
        udf_definitions=(), anonymous_submodels=(), name=nothing)
    checked_udfs = map(enumerate(collect(udf_definitions))) do (index, definition)
        part = _slic_bundle_part(definition, "UDF definition", index)
        expression = _slic_bundle_udf_definition(
            part.expression, index, part.source_part)
        part.labeled && (expression = _slic_bundle_with_source_part(
            expression, part.source_part))
        (; part.source_part,
           expression)
    end
    checked_definitions = map(enumerate(collect(definitions))) do (index, definition)
        part = _slic_bundle_part(definition, "definition", index)
        (; part.source_part,
           expression=_slic_bundle_definition(part.expression, index))
    end
    checked_anonymous = map(enumerate(collect(anonymous_submodels))) do (index, dependency)
        _slic_bundle_anonymous_submodel(dependency, index)
    end
    anonymous_bindings = map(dependency -> dependency.binding, checked_anonymous)
    length(unique(anonymous_bindings)) == length(anonymous_bindings) || error(
        "compile_slic_bundle: anonymous submodel parent bindings must be unique."
    )
    checked_body = _slic_bundle_part(body, "body")
    checked_body.expression isa Expr || error(
        "compile_slic_bundle: body must be an Expr, got $(typeof(checked_body.expression))."
    )

    workspace = Module(gensym(:StanBlocksSlicBundle))
    # UDFs installed in this fresh module must see public transpile-time
    # helpers without depending on which StanBlocks exports happen to exist in
    # `Main`. Bind the package function itself before evaluating definitions.
    Core.eval(workspace, Expr(:const, Expr(:(=), :return_type_of,
        GlobalRef(@__MODULE__, :return_type_of))))
    for definition in checked_udfs
        Core.eval(workspace, _slic_bundle_macrocall(
            Symbol("@deffun"), definition.source_part, definition.expression))
    end
    for definition in checked_definitions
        Core.eval(workspace, _slic_bundle_macrocall(
            Symbol("@slic"), definition.source_part, definition.expression))
    end
    for (index, dependency) in enumerate(checked_anonymous)
        _slic_bundle_bind_anonymous!(workspace, dependency, index)
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
