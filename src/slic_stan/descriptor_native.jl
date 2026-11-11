# Descriptor metadata needs deterministic data computations, including functions
# whose only implementation is in Stan. Plan before evaluating: a failed Julia
# numerical call is never retried with a different solver. One projection batches
# the unsupported expressions; the normal immutable artifact store caches its
# library, while each descriptor supplies its own current data.
@nospecialize

_descriptor_names!(out, x) = out
_descriptor_names!(out, x::Symbol) = (push!(out, x); out)
_descriptor_names!(out, xs::Union{Tuple,NamedTuple,AbstractVector}) =
    (foreach(x -> _descriptor_names!(out, x), xs); out)
_descriptor_names!(out, x::CanonicalExpr) =
    _descriptor_names!(out, (x.args, x.kwargs))
_descriptor_names!(out, x::StanExpr) = _descriptor_names!(out, (expr(x), type(x)))
_descriptor_names!(out, x::StanExpr2{<:types.closure}) =
    _descriptor_names!(out, _closure_captures(x))
_descriptor_names!(out, t::StanType) = begin
    _descriptor_names!(out, stan_size(t))
    for key in (:lower, :upper, :offset, :multiplier, :arg_types)
        haskey(info(t), key) && _descriptor_names!(out, info(t)[key])
    end
    out
end

# Both TD writers and their data dependencies, in program order. Loop indices
# and function locals have no model-level writer and do not enter the slice.
_descriptor_slice(scope, xs) = begin
    names = _descriptor_names!(Set{Symbol}(), xs)
    queue, visited, indices = collect(names), Set{Symbol}(), Set{Int}()
    while !isempty(queue)
        name = pop!(queue)
        name in visited && continue
        push!(visited, name)
        dependencies = Set{Symbol}()
        if haskey(scope.data, name)
            _descriptor_names!(dependencies, type(scope.data[name]))
        else
            for i in get(scope.writers, name, ())
                i in indices && continue
                push!(indices, i)
                _descriptor_names!(dependencies, scope.statements[i].statement)
            end
        end
        union!(names, dependencies)
        append!(queue, dependencies)
    end
    (; names, indices=sort!(collect(indices)))
end

# Most builtin functions have ordinary Julia methods. Others have a `jcall`
# adapter (e.g. rep_vector). A bodyful UDF records its explicit Julia capability
# per signature, and its transitive calls are inspected too.
_descriptor_julia_type(x) = typeof(x)
_descriptor_julia_type(x::StanExpr) = _descriptor_julia_type(type(x))
_descriptor_julia_type(t::StanType{types.func{F}}) where {F} = F
_descriptor_julia_type(t::StanType{<:types.tup}) = begin
    components = _tuple_component_types(t)
    tuple = Tuple{(_descriptor_julia_type(c) for c in components)...}
    components isa NamedTuple ? NamedTuple{keys(components),tuple} : tuple
end
_descriptor_julia_type(t::StanType) = begin
    ct = center_type(t)
    ct <: Union{types.complex,types.any_vector,types.matrix} || return Any
    scalar = ct <: types.int ? Int : ct <: types.complex && !(ct <: types.real) ? ComplexF64 : Float64
    rank = length(index_shape(StanExpr(:_, t)))
    rank == 0 ? scalar : Array{scalar,rank}
end
_descriptor_has_jcall(f::Function, x) = begin
    args = Tuple{(_descriptor_julia_type(a) for a in x.args)...}
    which(_plate_layout_call, Tuple{typeof(f),Module,args.parameters...}) !==
        which(_plate_layout_call, Tuple{Function,Module}) && return true
    hasmethod(f, args) && return true
    fallback = which(jcall, Tuple{Val{:__descriptor_unregistered_call__},Module})
    any(m -> m !== fallback,
        methods(jcall, Tuple{Val{nameof(f)},Module,args.parameters...}))
end
_descriptor_has_jcall(f, x) = true
_descriptor_calls_native(scope, x, seen) = false
_descriptor_calls_native(scope, xs::Union{Tuple,NamedTuple,AbstractVector}, seen) = begin
    native = false
    for x in xs
        native |= _descriptor_calls_native(scope, x, seen)
    end
    native
end
_descriptor_calls_native(scope, x::StanExpr, seen) =
    _descriptor_calls_native(scope, expr(x), seen)
_descriptor_calls_native(scope, x::CanonicalExpr, seen) = begin
    h = head(x)
    h isa Function && endswith(string(nameof(h)), "_rng") && error(
        "stan_descriptor: metadata must be deterministic; it calls `", nameof(h), "`.")
    native = _descriptor_calls_native(scope, (x.args, x.kwargs), seen)
    signature = sig_expr(x)
    signature in seen && return native
    push!(seen, signature)
    capability = _descriptor_julia_by_head(getfield(x, :head), x)
    available = _descriptor_has_jcall(h, x)
    native |= !available
    definition = get(content(block(scope.m, :functions)), signature, nothing)
    # A hand-written Julia twin or jcall adapter supplies the whole operation;
    # it need not implement each call in the corresponding Stan body. An
    # emitted @juliacompat method, however, runs those calls in Julia too.
    if definition !== nothing && (capability === true || !available)
        native |= _descriptor_calls_native(scope, definition.body, seen)
    end
    native
end
_descriptor_td_native(x) = false
_descriptor_td_native(x::StanExpr) = _descriptor_td_native(expr(x))
_descriptor_td_native(x::Union{IfExpr,ElseIfExpr,WhileExpr}) = true
_descriptor_td_native(x::CanonicalExpr) = any(_descriptor_td_native, x.args)

_descriptor_needs_native(scope, x, slice) = begin
    seen = Set{Any}()
    native = _descriptor_calls_native(scope, x, seen)
    for i in slice.indices
        s = scope.statements[i].statement
        native |= _descriptor_td_native(s) | _descriptor_calls_native(scope, s, seen)
    end
    native
end

_descriptor_requests!(out, x::StanExpr) = begin
    append!(out, index_shape(x))
    ends = get(info(type(x)), :ragged_observation_ends, nothing)
    ends === nothing || push!(out, ends)
    for value in values(_descriptor_constraints(x))
        qual(value) === :data && push!(out, value)
    end
    for t in get(info(type(x)), :arg_types, ())
        _descriptor_requests!(out, StanExpr(:_, t))
    end
    out
end

_descriptor_native!(scope) = _with_slic_read() do
    declarations = OrderedDict{Symbol,Any}()
    for name in (:parameters, :transformed_parameters, :generated_quantities)
        _output_symbols(block(scope.m, name), declarations)
    end
    requests = Any[]
    for x in values(declarations)
        always_inline(x) || _descriptor_requests!(requests, x)
    end
    for x in values(scope.data), value in values(_descriptor_constraints(x))
        qual(value) === :data && push!(requests, value)
    end
    # A ragged cell's size can call native code too. Its already-lowered ends
    # array evaluates the cell recipe for every index in one native loop.
    td = _output_symbols(block(scope.m, :transformed_data), OrderedDict{Symbol,Any}())
    for layout in unique(collect(values(scope.layouts)))
        haskey(td, layout.ends) && push!(requests, td[layout.ends])
    end
    selected = Any[]
    for x in unique(requests)
        slice = _descriptor_slice(scope, x)
        _descriptor_needs_native(scope, x, slice) && push!(selected, x)
    end
    isempty(selected) && return nothing
    try
        merge!(scope.native, _descriptor_native_values(scope, selected))
    catch err
        expressions = sprint(io -> print(StanIO(io), Join(selected, ", ")))
        error("stan_descriptor: native evaluation of data-only metadata failed for `",
            expressions, "`. Cause: ", sprint(showerror, err))
    end
    for layout in unique(collect(values(scope.layouts)))
        x = get(td, layout.ends, nothing)
        haskey(scope.native, x) || continue
        scope.env[layout.ends] = scope.native[x]
    end
    nothing
end

# Export dimensions alongside each value: decoding never needs to evaluate its
# shape again in Julia. Constraints belong to the original declaration, not to
# the temporary holding the metadata value (which may legitimately be infinite).
_descriptor_native_type(x::StanExpr) = StanType(center_type(x), stan_size(x))
_descriptor_native_type(x) = type(stan_expr(x))

_descriptor_projection(scope, requests) = begin
    slice = _descriptor_slice(scope, requests)
    occupied = Set(keys(vars(scope.m)))
    union!(occupied, keys(scope.data))
    union!(occupied, keys(scope.writers))
    functions = content(block(scope.m, :functions))
    names = _emission_names(functions)
    for definition in values(functions)
        definition === nothing && continue
        push!(occupied, Symbol(_named_func_name(definition.parent, definition.args, names)))
    end
    counter = 0
    fresh() = begin
        while true
            counter += 1
            name = Symbol("slic_descriptor_value_", counter)
            name in occupied || (push!(occupied, name); return name)
        end
    end
    statements, exports = Any[], Any[]
    for x in requests
        t = _descriptor_native_type(x)
        value = StanExpr(fresh(), t)
        push!(statements, CanonicalExpr(:(=), CanonicalExpr(:(::), value), x))
        dims = Symbol[]
        for s in index_shape(value)
            d = StanExpr(fresh(), StanType(types.int))
            push!(dims, expr(d))
            push!(statements, CanonicalExpr(:(=), CanonicalExpr(:(::), d), s))
        end
        push!(exports, (; x, name=expr(value), dims, integer=center_type(value) <: types.int))
    end
    empty = StanModel(:descriptor_data)
    data = OrderedDict(k => v for (k, v) in pairs(scope.data) if k in slice.names)
    projection_blocks = (; blocks(empty)...,
        functions=block(scope.m, :functions),
        data=StanBlock(:data, data),
        transformed_data=StanBlock(:transformed_data,
            Any[scope.statements[i].statement for i in slice.indices]),
        generated_quantities=StanBlock(:generated_quantities, statements))
    StanModel(meta(empty), vars(empty), projection_blocks), exports
end

_descriptor_native_values(scope, requests) = begin
    projection, exports = _descriptor_projection(scope, requests)
    problem = instantiate(projection; nan_on_error=false)
    sm = problem.model
    names = BridgeStan.param_names(sm; include_tp=true, include_gq=true)
    # BridgeStan requires an RNG handle for any generated-quantities request,
    # including this deterministic projection. No selected expression uses it.
    rng = BridgeStan.StanRNG(sm, 0)
    values = BridgeStan.param_constrain(sm, Float64[]; include_tp=true, include_gq=true, rng)
    indices = Dict{Symbol,Vector{Int}}()
    for (i, name) in enumerate(names)
        push!(get!(Vector{Int}, indices, _bridgestan_stem(name)), i)
    end
    result = Dict{Any,Any}()
    for output in exports
        dims = Tuple(Int(only(values[indices[d]])) for d in output.dims)
        flat = values[get(indices, output.name, Int[])]
        output.integer && (flat = Int.(flat))
        result[output.x] = isempty(dims) ? only(flat) : Base.reshape(flat, dims)
    end
    result
end

@specialize
