module types
    abstract type anything end
    abstract type matrix <: anything end
    abstract type square_matrix <: matrix end
    abstract type cov_matrix <: square_matrix end
    abstract type corr_matrix <: square_matrix end
    abstract type cholesky_factor_cov <: square_matrix end
    abstract type cholesky_factor_corr <: square_matrix end
    abstract type any_vector <: anything end
    abstract type vector <: any_vector end
    abstract type row_vector <: any_vector end
    abstract type simplex <: vector end
    abstract type ordered <: vector end
    abstract type positive_ordered <: vector end
    abstract type complex <: anything end
    abstract type real <: complex end
    abstract type int <: real end
    # A boolean is an `int` (Stan has no bool array — it emits as `array[] int`),
    # so `bool[n]` behaves like `int[n]` under EVERY dispatch (arithmetic, `sum`,
    # comparison tracetypes) — the ONLY specialisation is `getindex`, where a
    # `bool[n]` index means boolean-MASK selection (lowered to `v[findall(mask)]`)
    # rather than integer indexing. Produced by element-wise comparison broadcasts
    # (`cmt .== 1`). Renders as `int` (see the `Base.show` override below).
    abstract type bool <: int end
    abstract type func{T} <: anything end
    # Anonymous lambdas (`(x) -> body`) flow through tracing as values of
    # type `types.closure`. They are *sibling* to `types.func` (not a subtype)
    # because `canonical(::CanonicalExpr{<:StanExpr2{<:types.func}})` rewrites
    # the call head to `info.value` (the bare Function), and we instead want
    # closure dispatch to keep the StanExpr-at-head shape so `inline_body`
    # can pull the closure record off `type(head(x)).info.value`.
    abstract type closure <: anything end
    # `tokenof{T}` wraps a Stan type as a SLIC value, so `real` / `real[n]` /
    # `typeof(x)` can flow through tracing as StanExprs (mirroring how
    # `func{T}` lets functions flow). Stan doesn't support types as
    # parameters — these tokens participate purely in Julia-side dispatch and
    # in Stan-function-name mangling.
    abstract type tokenof{T} <: anything end
    abstract type tup <: anything end
    abstract type ntup <: tup end
    # Marker for `@usertype`-declared bundles. A user type is a named
    # `ntup` with a Julia abstract type tag (defined in the user's module
    # by `@usertype`), enabling normal Julia method dispatch on
    # `<:StanExpr2{<:RaggedVector}` etc. Field bundling, Stan rendering,
    # and `r.mem` access all reuse the existing `ntup` machinery.
    abstract type usertype <: ntup end
    # Side-effect-only return type for `@deffun foo(...)::void = …`.
    # Calls are statements; binding their result is rejected at trace time.
    abstract type void <: anything end
end
function stan_code end
# Types flow through SLIC like functions: wrap once on entry, then dispatch
# via the resulting StanExpr's `tokenof{T}` center type. Sized forms and
# `typeof(x)` hit their own tracetype rules (see below).
forward!(x::Type{<:types.anything}; info) = stan_expr(x)
stan_type(expr, value::Type{T}; kwargs...) where {T<:types.anything} = StanType(
    types.tokenof{T}; value=T, qual=:data, kwargs...
)
Base.show(io::IO, ::Type{T}) where {T<:types.anything} = print(io, T.name.name)
# A `bool` has no Stan spelling — it IS an `int` on the Stan side.
Base.show(io::IO, ::Type{<:types.bool}) = print(io, "int")
Base.show(io::IO, ::Type{T}) where {T<:types.func} = print(io, "func")#.parameters[1].name.name)
Base.show(io::IO, ::Type{<:types.closure}) = print(io, "closure")
Base.show(io::IO, ::Type{<:types.tup}) = print(io, "tuple(...)")
r_ndim(::Type{types.anything}) = 0
r_ndim(::Type{types.matrix}) = 2
r_ndim(::Type{<:types.square_matrix}) = 1
r_ndim(::Type{<:types.any_vector}) = 1
r_ndim(::Type{<:types.complex}) = 0
r_ndim(::Type{<:types.func}) = 0
r_ndim(::Type{<:types.closure}) = 0
r_ndim(::Type{<:types.tokenof}) = 0
r_ndim(::Type{<:types.tup}) = 0
# `usertype <: ntup`, so `r_ndim` already resolves to 0 via the `tup`
# rule. Field access (`r.mem`) routes through `forward!(::GetPropertyExpr)`
# unchanged because the dispatch is on `<:types.ntup`.
r_ndim(::Type{types.void}) = 0
r_ndim(::StanType{T}) where {T} = r_ndim(T)
l_ndim(x::StanType) = stan_ndim(x) - r_ndim(x)
lr_size(x::StanType) = stan_size(x, 1:l_ndim(x)), stan_size(x, 1+l_ndim(x):stan_ndim(x))
canonical(x::CanonicalExpr{<:StanExpr2{<:types.func}}) = CanonicalExpr(type(x.head).info.value, x.args...; x.kwargs...)
backward!(x::StanExpr2{<:types.func}; info) = x
# Closures: pass through `forward!`/`backward!` like `types.func` does, but
# crucially do NOT rewrite the call head via `canonical` — `expand_inline!`
# pulls the closure record off `type(head(x)).info.value` and substitutes
# captures + args into the stored body Expr.
forward!(x::StanExpr2{<:types.closure}; info) = x
# `backward!` only descends into a closure when the enclosing call is itself
# reachable from the likelihood (the `AssignmentExpr`/`SamplingExpr` methods
# recurse into an RHS only once its LHS is `:affects_likelihood`). A LIFTED
# closure flowing to a builtin (`ode_rk45`/`reduce_sum`/`jbroadcasted`) has its
# body emitted as a SEPARATE Stan function with its captures threaded as trailing
# call args (`show(::CanonicalExpr{<:ODESolver})`, `_closure_captures`), so a
# capture that appears ONLY inside such a body is never seen as a plain symbol
# occurrence — without this, a capture-only parameter (e.g. an ODE-RHS `lambda`)
# is optimised into `generated quantities` yet still emitted in `transformed
# parameters`, producing out-of-scope Stan. Reaching here means the closure's
# output IS a downstream-likelihood use, so mark each captured binding
# `:affects_likelihood` (INLINED closures never reach here — `expand_inline!`
# removes them in `forward!`, so their captures keep flowing through their
# spliced bindings).
backward!(x::StanExpr2{<:types.closure}; info) = begin
    foreach(backward!(; info), _closure_captures(x))
    x
end
# A closure StanExpr's `expr` field carries the closure record (a NamedTuple);
# the generic `fetch_data!` NamedTuple fallback would iterate the record and trip
# on its `body::Expr` field, so a specialisation is required. It must NOT be a
# no-op, though: for a LIFTED closure (flowing to a builtin like `ode_rk45`) the
# body becomes a SEPARATE Stan function and the emitter threads the captures as
# trailing call args (`show(::CanonicalExpr{<:ODESolver})`, `_closure_captures`).
# A DATA value captured ONLY inside such a body is therefore never walked as a
# plain symbol occurrence, so dead-data elimination (`fetch_data!(::StanExpr{Symbol})`
# only declares `hasvalue` symbols it visits) drops its `data` declaration while
# the emitter still references it → out-of-scope Stan. Descend into the CAPTURES
# only (not the whole record, which keeps the `body::Expr` trap avoided): a
# captured data symbol gets declared, captured params/derived values are `hasvalue`
# no-ops. This mirrors `backward!(closure)` above. Inlined closures never reach
# here (`expand_inline!` removes them in `forward!`).
fetch_data!(x::StanExpr2{<:types.closure}; info) = foreach(fetch_data!(; info), _closure_captures(x))
# tokenof StanExprs carry a raw Stan type as `expr`; skip recursing into the
# bare `Type{...}` which would hit the generic backward! fallback.
backward!(x::StanExpr2{<:types.tokenof}; info) = x

short_expr(x::Symbol) = x
short_expr(x::StanExpr2{types.anything}) = StanExpr(short_expr(expr(x)), type(x))
short_expr(x::StanExpr) = StanExpr("", StringStanType(sigtype(x)))
short_expr(x::CanonicalExpr) = CanonicalExpr(head(x), short_expr.(x.args)...)
tracetype(x::CanonicalExpr) = begin
    map(x.args) do arg 
        # tracetype not defined for $(head(expr(arg)))$(typeof.(type.(expr(arg).args))) (nargs = $(length(expr(arg).args))).
        center_type(type(arg)) == types.anything && error("""
            `tracetype` not defined for $(short_expr(arg))!
            This is only allowed if this return value does not get used in another expression,
            but it is used in $(short_expr(x)) (nargs = $(length(x.args))).
            """)
                # but needed in $(head(x))$(typeof.(type.(x.args))) (nargs = $(length(x.args))).
    end
    StanType(types.anything)
end

"""
    return_type_of(f, args...) -> StanType

Infer the Stan return type and shape of a SLIC-callable function for the given
representative Julia arguments. The result is the same `StanType` used
by the transpiler and prints as Stan syntax, for example `real`, `vector[3]`,
or `matrix[2, 4]`.

Inside an `@deffun` body, `return_type_of(f, args...)` is a transpile-time type
token and may be used in a computed type annotation:

```julia
@deffun element_type(x::real)::real = x
@deffun copy_like(x::vector[n]) = begin
    out::return_type_of(element_type, x[1])[n]
    for i in 1:n
        out[i] = x[i]
    end
    out
end
```

The query is intentionally bounded to non-inline `@deffun` functions and
`@defsig`-registered SLIC callables whose result is a scalar or sized Stan
container. Inline functions, closures, sub-models, arbitrary Julia functions,
and tuple/user-defined-type results require a full call-site trace and are not
queryable through this API. Inside a UDF, spell symbolic output dimensions
explicitly (`return_type_of(f, x)[n]`) when they are already bound by the
surrounding signature.
"""
function return_type_of end

_return_type_query_arg(i, x::StanExpr) = x
_return_type_query_arg(i, x) = begin
    name = Symbol(:arg, i)
    sx = stan_expr(name, x)
    concrete_size = map(stan_size(sx)) do dim
        hasvalue(dim) ? stan_expr(getvalue(dim), getvalue(dim)) : dim
    end
    StanExpr(name, remake(type(sx), concrete_size...))
end

_return_type_signature(args) = join((sprint(show, type(arg)) for arg in args), ", ")
_supported_return_type(rt::StanType) = begin
    ct = center_type(rt)
    ct <: types.complex || ct <: types.any_vector || ct <: types.matrix
end
_checked_return_type(f, args) = begin
    rt = type(stan_expr(CanonicalExpr(f, args...)))
    center_type(rt) === types.anything && throw(ArgumentError(
        "return_type_of could not infer a SLIC return type for $(f)(" *
        _return_type_signature(args) * "). Only non-inline @deffun functions and " *
        "@defsig-registered SLIC callables are supported."
    ))
    _supported_return_type(rt) || throw(ArgumentError(
        "return_type_of inferred unsupported result `$(sprint(show, rt))` for $(f). " *
        "The public query currently supports scalar and sized-container results."
    ))
    rt
end

return_type_of(f::Function, args...) = _checked_return_type(
    f,
    ntuple(i -> _return_type_query_arg(i, args[i]), length(args)),
)
return_type_of(f, args...) = throw(ArgumentError(
    "return_type_of expects a non-inline @deffun or @defsig-registered Function " *
    "as its first argument, got $(typeof(f))."
))

tracetype(x::CanonicalExpr{typeof(return_type_of),<:Tuple{<:StanExpr2{<:types.func},Vararg{Any}}}) = begin
    farg = x.args[1]
    rt = _checked_return_type(getvalue(farg), x.args[2:end])
    ct = center_type(rt)
    StanType(types.tokenof{ct}, stan_size(rt); value=ct, qual=:data)
end
tracetype(x::CanonicalExpr{typeof(return_type_of)}) = error(
    "return_type_of expects a non-inline @deffun or @defsig-registered function " *
    "as its first argument; closures, inline functions, and sub-models require a " *
    "full call-site trace and are not supported by this query."
)

# Stan defines the elementwise arithmetic operators `+ - * / ^` (and their
# broadcast forms `.+ .- .* ./ .^`) only on `vector` / `row_vector` / `matrix`
# operands — NEVER on plain scalar arrays (`array[] int` / `array[] real` /
# `array[] complex`). A `tracetype` that returned `int[n]` / `real[n]` for such a
# call transpiles PASS but stanc REJECTs it downstream ("Semantic error"): a
# silent miscompile. `_reject_scalar_array_elementwise` fails LOUDLY at trace time
# with the actionable fix instead of emitting invalid Stan. (Correct lowering — an
# @deffun element loop or a `to_vector` round-trip — is a possible future
# enhancement; today the loud rejection is the contract.)
_is_scalar_array(t::StanType) = center_type(t) <: types.complex && l_ndim(t) >= 1
_reject_scalar_array_elementwise(x::CanonicalExpr) = begin
    any(a -> _is_scalar_array(type(a)), x.args) || return nothing
    error(
        "Elementwise arithmetic (`+ - * / ^`, `.+ .- .* ./ .^`) is not supported on scalar ",
        "arrays (`array[] int` / `array[] real`): Stan defines it only on `vector` / ",
        "`row_vector` / `matrix`. Got `", short_expr(x), "`. Compute it element-by-element in ",
        "an @deffun function-body loop, or convert to a `vector` via `to_vector(...)` first."
    )
end
tracetype(x::CanonicalExpr{<:Union{typeof.((+, -, ^, *, /))...}}) = if length(x.args) > 2
    f = head(x)
    tracetype(CanonicalExpr(f, x.args[1], stan_expr(CanonicalExpr(f, x.args[2:end]...))))
else
    _reject_scalar_array_elementwise(x)
    error("tracetype not defined for $(short_expr(x))!")
    StanType(types.anything)
end
# Broadcast `.* ./ .^` route to `Base.BroadcastFunction` (unlike `.+`/`.-`, which
# `broadcast_callee` rewrites to plain `+`/`-` above). Reject scalar-array operands
# the same way, then preserve the generic fallthrough so valid-but-untabulated
# broadcasts (e.g. `matrix .* matrix`) keep their current inferred type.
tracetype(x::CanonicalExpr{<:Base.BroadcastFunction}) = begin
    _reject_scalar_array_elementwise(x)
    invoke(tracetype, Tuple{CanonicalExpr}, x)
end
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 1),))))
)
# Fully selecting the scalar array-prefix of an `array[...] vector/matrix`
# leaves its native vector/matrix core. This general rule covers arbitrary
# array depth (e.g. `array[N] matrix[K,M] x; x[n, :, m]`) beyond the finite
# signature table below while preserving that table's core slice inference.
tracetype(x::CanonicalExpr{<:typeof(getindex),<:Tuple{<:StanExpr,Vararg{Any}}}) = begin
    value = x.args[1]
    nl = l_ndim(type(value))
    indices = x.args[2:end]
    if nl > 0 && length(indices) >= nl && all(i -> i isa StanExpr2{<:types.int,0}, indices[1:nl])
        core_type = remake(type(value), stan_size(type(value))[nl+1:end]...)
        rest = indices[nl+1:end]
        isempty(rest) && return core_type
        core = StanExpr(missing, core_type)
        return tracetype(CanonicalExpr(getindex, core, rest...))
    end
    invoke(tracetype, Tuple{CanonicalExpr}, x)
end
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 1),))), x.args[3])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], x.args[2], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 2),))))
)
_colon_range_expr(x, j) = StanExpr(missing, StanType(types.int, (stan_size(x, min(j, stan_ndim(x))),)))
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], _colon_range_expr(x.args[1], 1), x.args[3])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Any,<:Colon,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], x.args[2], _colon_range_expr(x.args[1], 2), x.args[4])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], _colon_range_expr(x.args[1], 1), x.args[3], x.args[4])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Colon,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], _colon_range_expr(x.args[1], 1), x.args[3], x.args[4])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Any,<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], x.args[2], x.args[3], _colon_range_expr(x.args[1], 3))
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Any,<:Colon,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], x.args[2], _colon_range_expr(x.args[1], 2), x.args[4])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Any,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], _colon_range_expr(x.args[1], 1), x.args[3], x.args[4])
)
# Field access via `Base.getfield(obj, position)` — `forward!(::GetPropertyExpr)`
# lowers `obj.name` to `getfield(obj, find_field_position(name))`. Kept on
# its own dispatch lane (rather than reusing `getindex`) so user-defined
# `Base.getindex(::usertype, ::int)` methods don't accidentally catch
# field accesses on usertypes.
tracetype(x::CanonicalExpr{<:typeof(Base.getfield),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = x.args[1].type.info.arg_types[x.args[2].type.info.value]
# `T[d1, …, dS]`: getindex on a 0-dim type token upgrades it to a sized token.
# Retained `value = T` carries the center Stan type across resizings.
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.tokenof{T},0},Vararg{Any}}}) where {T} = StanType(
    types.tokenof{T}, x.args[2:end]; value=T
)
# `typeof(x)` wraps x's inferred Stan type into a token of the same shape.
tracetype(x::CanonicalExpr{typeof(Base.typeof),<:Tuple{<:StanExpr}}) = begin
    xt = type(x.args[1])
    StanType(types.tokenof{center_type(xt)}, stan_size(xt); value=center_type(xt))
end

tracetype(x::CanonicalExpr{Colon}) = StanType(types.int, (stan_call(+,stan_expr(1,1),stan_call(-,x.args[2],x.args[1])), ))
tracetype(x::BracesExpr) = StanType(types.real, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::VectExpr) = StanType(types.vector, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::TupleExpr) = StanType(types.tup; arg_types=map(type, x.args))
tracetype(x::KwExpr) = type(x.args[2])
tracetype(x::NamedTupleExpr) = StanType(types.ntup; arg_types=(;[
    kw.args[1]=>type(kw.args[2]) for kw in map(expr, x.args)
]...))
tracetype(x::DeclExpr) = x.args[1].type
tracetype(x::ForExpr) = StanType(types.anything)
tracetype(x::WhileExpr) = StanType(types.anything)
tracetype(x::IfExpr) = StanType(types.anything)
tracetype(x::ElseIfExpr) = StanType(types.anything)
tracetype(x::BlockExpr) = error("tracetype(::BlockExpr) not implemented — block expressions don't carry a result type; refactor the caller to trace the final expression instead.")#tracetype(expr(x.args[end]))

autokwargs(::CanonicalExpr) = (;)
# `type=T` kwarg: `forward!(::Type{<:types.anything})` wraps the symbol as a
# `tokenof{T}` StanExpr whose `.value` carries the raw Type. Unwrap so
# `StanType(ct, size; …)` receives a bare Type as it expects.
_unwrap_type_kwarg(ct::StanExpr2{<:types.tokenof,0}) = type(ct).info.value
_unwrap_type_kwarg(ct) = ct
autotype(x::StanExpr) = autotype(type(x); merge(autokwargs(expr(x)), expr(x).kwargs)...)
autotype(x::StanType; kwargs...) = begin
    ct = _unwrap_type_kwarg(get(kwargs, :type, center_type(x)))
    nsize = [
        kwargs[key]
        for key in (:m, :n, :o) if key in keys(kwargs)
    ]
    size = length(nsize) > 0 ? (nsize..., ) : get(kwargs, :size, stan_size(x))
    (ct in (types.anything, types.real)) && (ct = [types.real, types.vector, types.matrix][1+length(size)])
    cons = (;[
        key=>getindex(kwargs, key)
        for key in (:lower, :upper, :offset, :multiplier) if key in keys(kwargs)
    ]...)
    StanType(ct, size; cons...)
end

struct StanFunction3
    docstring::AbstractString
    rv_type::StanType
    # `parent` identifies the function for `func_name` mangling. For
    # @deffun UDFs it's the bare `Function` value; for lifted closures
    # (phase 2-deeper) it's the closure StanExpr, whose `func_name`
    # specialisation produces `closure_<id>` so the synthesised Stan
    # function name matches what call sites already emit.
    parent
    args::NamedTuple
    body::Vector
end

Base.show(io::IO, f::StanFunction3) = autoprint(
    io,
    f.docstring,
    sigtype(f.rv_type), " ", func_name(f.parent, f.args), "(", func_args(f.args), ")",
    StanBlock(Symbol(), f.body)
)

begin
    xiscall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
    xassign(args...) = Expr(:(=), args...)
    xtuple(args...) = Expr(:tuple, args...)
    xref(args...) = Expr(:ref, args...)
    xtyped(args...) = Expr(:(::), args...)
    xpair(args...) = Expr(:call, :(=>), args...)
    xvect(args...) = Expr(:vect, args...)
    xstring(args...) = Expr(:string, args...)
    ensure_xassign(x, default=missing) = Meta.isexpr(x, :(=)) ? x : xassign(x, default)
    ensure_xtuple(x) = Meta.isexpr(x, :tuple) ? x : xtuple(x)
    ensure_xref(x) = Meta.isexpr(x, :ref) ? x : xref(x)
    ensure_xtyped(x, default_symbol=missing; default=:anything) = if Meta.isexpr(x, :(::))
        if length(x.args) == 2
            x.args[1] == Symbol("_") ? xtyped(default_symbol, x.args[2]) : x
        else
            xtyped(default_symbol, x.args[1])
        end
    else
        xtyped(x, default)
    end
    _is_symbol(::Symbol) = true
    _is_symbol(_) = false
    _is_expr(::Expr) = true
    _is_expr(_) = false
    _is_quotenode(::QuoteNode) = true
    _is_quotenode(_) = false
    _is_anything_type(::Type{<:types.anything}) = true
    _is_anything_type(_) = false
    ensure_xpair(x, default) = xiscall(x, :(=>)) ? x : xpair(x, default)
    ensure_xvect(x) = Meta.isexpr(x, :vect) ? x : xvect(x)
    ensure_xreturn(x::Expr) = if x.head in (:block, :macrocall)
        Expr(x.head, x.args[1:end-1]..., ensure_xreturn(x.args[end]))
    elseif x.head == :if
        Expr(x.head, x.args[1], ensure_xreturn.(x.args[2:end])...)
    elseif x.head == :return
        x
    else
        Expr(:return, x)
    end
    ensure_xreturn(x) = Expr(:return, x)

    # Strip type/decl/kw wrapping to recover the bare argument *name* —
    # used wherever we need a call-position arg (typed forms are syntax
    # errors at call position).
    _name_of(a::Symbol) = a
    _name_of(a::Expr) = if a.head === :(::)
        _name_of(a.args[1])
    elseif a.head === :kw
        _name_of(a.args[1])
    else
        a
    end

    # Resolve a type name from a `@deffun`-style signature. Builtin Stan
    # types live in `types` and are spliced as Type *values* (so the
    # generated code doesn't need any user-side import). `@usertype`-
    # declared names aren't in `types`; leave them as the bare Symbol so
    # the surrounding `esc()` resolves them in the user's module via
    # standard Julia scope rules — no registry, no name-leak into `types`.
    gettype(ct::Symbol) = isdefined(types, ct) ? getproperty(types, ct) : ct
    gettype(ct::Expr) = begin
        return :($types.func{$ct})
    end

    xsig_type(x::Expr) = begin
        @assert x.head == :ref "xsig_type expects a `T` or `T[dims...]` `:ref` expression, got `$x` (head `$(x.head)`)."
        ct, size... = x.args
        ct = gettype(ct)
        ndims = length(size)
        if ct == types.anything && ndims == 0
            :(<:$StanExpr2{<:$ct})
        else
            :(<:$StanExpr2{<:$ct, $ndims})
        end
    end
    # Type-token positional args: bare `T` or `T[dims...]` where `T` is a Stan
    # type. Dispatched via `<:StanExpr2{<:types.tokenof{<:T_t}, S}`.
    _is_type_token_sym(x) = _is_symbol(x) && isdefined(types, x) && _is_anything_type(getproperty(types, x))
    _is_type_token(x) = _is_type_token_sym(x) ||
        (Meta.isexpr(x, :ref) && length(x.args) >= 1 && _is_type_token_sym(x.args[1]))
    _type_token_ref(x::Symbol) = xref(x)
    _type_token_ref(x::Expr) = x
    xsig_type_token(x::Expr) = begin
        @assert x.head == :ref "xsig_type_token expects a `T[dims...]` `:ref` expression, got `$x` (head `$(x.head)`)."
        ct, size... = x.args
        ct = gettype(ct)
        :(<:$StanExpr2{<:$types.tokenof{<:$ct},$(length(size))})
    end
    xsig_expr(x::Expr) = begin
        @assert x.head == :ref "xsig_expr expects a `T[dims...]` `:ref` expression, got `$x` (head `$(x.head)`)."
        ct, size... = x.args
        ct = gettype(ct)
        size = xtuple([:($forward!($arg; info)) for arg in canonical.(size)]...)
        :($StanType($ct, $size))
    end

    defsig(x::LineNumberNode; kwargs...) = x
    defsig(x::Expr; source=LineNumberNode(0, :none)) = if x.head == :block
        Expr(:block, defsig.(x.args; source)...)
    else
        @assert xiscall(x, :(=>)) "@defsig expects `ftype => begin ... end`, got `$x`."
        _, ftype, rhs = x.args
        @assert Meta.isexpr(rhs, :block) "@defsig RHS must be a `begin ... end` block of `args => rv` signatures, got `$rhs`."
        Expr(:block, map(sig->defsig(ftype, sig; source), rhs.args)...)
    end
    defsig(ftype, x::LineNumberNode; kwargs...) = x
    defsig(ftype, sig::Expr; source=LineNumberNode(0, :none)) = begin
        @assert xiscall(sig, :(=>)) "@defsig each line of the block must be `(args...) => rv`, got `$sig` for ftype `$ftype`."
        _, lhs, rv = sig.args
        lhs = ensure_xref.(ensure_xtuple(lhs).args)
        rv = ensure_xref(rv)
        arg_types = lhs
        lhs_type = xsig_type.(lhs)
        dim_names = OrderedSet()
        for arg_type in arg_types
            for dim_name in arg_type.args[2:end]
                _is_symbol(dim_name) || continue
                push!(dim_names, dim_name)
            end
        end

        xexpr = :(x::$CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, source, [
            xassign(xtuple(ensure_xlhs.(lhsi.args[2:end])...), :(stan_size(x.args[$i])))
            for (i, lhsi) in enumerate(lhs)
        ]..., :(info = (;$(dim_names...),)), xsig_expr(rv))
        :($stan.tracetype($xexpr) = $xbody)
    end
    funbody(x::Expr) = begin
        @assert x.head == :block "funbody expects a `begin ... end` block, got `$x` (head `$(x.head)`)."
        funbody(x.args)
    end
    funbody(x::AbstractVector) = join(map(funbody, x), "\n")
    funbody(x::LineNumberNode) = ""
    funbody(x::String) = strip(x)
    make_stan_type(x::Symbol) = make_stan_type(xref(x))
    make_stan_type(x::Expr) = begin
        @assert x.head == :ref "make_stan_type expects a `T[dims...]` `:ref` expression, got `$x` (head `$(x.head)`)."
        ct, size... = x.args
        ct = getproperty(types, ct)
        StanType(ct, StanExpr.((size..., ), Ref(StanType(types.int))))
    end
    # A return annotation whose base type is COMPUTED (`typeof(...)` /
    # `return_type_of(...)`, optionally `[dims]`-sized) cannot be built at
    # macro-expansion time (the args aren't bound yet). Route it to the same
    # infer-from-body path as `::anything` — the body's trailing (declared)
    # expression already carries the resolved container type.
    _is_computed_ret_type(rv) = false
    _is_computed_ret_type(rv::Expr) =
        rv.head == :call || (rv.head == :ref && !(rv.args[1] isa Symbol))
    sigtype(x::Symbol) = sigtype(xref(x))
    sigtype(x::Expr) = begin
        @assert x.head == :ref "sigtype expects a `T[dims...]` `:ref` expression, got `$x` (head `$(x.head)`)."
        ct, size... = x.args
        ct = getproperty(types, ct)
        l = length(size) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array[", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigtype(x::Type) = x
    sigtype(x::Type{types.cholesky_factor_corr}) = types.matrix
    sigtype(x::Type{<:types.vector}) = types.vector
    # `bool` is Stan-identical to `int` (it emits as `array[] int`), so its
    # signature type — which drives Stan-function NAMING and DEDUP — must be
    # `int`. Otherwise a generated helper reached with a `bool[]` arg and the same
    # helper reached with an `int[]` arg (e.g. `broadcasted_getindex` inside
    # `jbroadcasted`) would key as two distinct functions yet render to one
    # identical Stan signature → "already declared". `center_type` stays `bool`
    # for the `getindex` mask dispatch; only the Stan-facing signature collapses.
    sigtype(x::Type{<:types.bool}) = types.int
    sigtype(x::StanExpr) = sigtype(x.type)
    sigtype(x::StanType) = begin 
        ct = center_type(x)
        ct == types.anything && @error("Stan compilation will fail: `sigtype($x)` == anything")
        # @assert ct != types.anything
        l = stan_ndim(x) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array[", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigtype(x::StanType{<:types.tup}) = begin 
        io = IOBuffer()
        stan_ndim(x) > 0 && print(io, "array[", join(fill("", stan_ndim(x)), ", "), "] ")
        print(io, "tuple(", join(map(sigtype, x.info.arg_types), ", "), ")")
        String(take!(io))
    end
    always_inline(x) = false
    always_inline(::StanExpr2{<:types.func}) = true
    always_inline(::StanExpr2{<:types.closure}) = true
    # 0-dim type tokens carry no runtime value — they only contribute a name
    # mangle component to the Stan function name. Sized tokens (`real[n]`)
    # _do_ render, as a Stan `tuple(int, ...)` literal at the call site.
    always_inline(::StanExpr2{<:types.tokenof,0}) = true
    expr_replace(x; kwargs...) = get(kwargs, x, x)
    expr_replace(x::Expr; kwargs...) = Expr(x.head, expr_replace.(x.args; kwargs...)...)

    ensure_xlhs(arg::Symbol; hidden=()) = arg in hidden ? Symbol("_") : arg
    ensure_xlhs(::Expr; kwargs...) = Symbol("_")

    hasvararg(args) = length(args) > 0 && Meta.isexpr(args[end], :(...))
    maybedoc(x::AbstractString) = length(strip(x)) == 0 ? "" : strip(replace("\n" * strip(x), "\n"=>"\n// ")) * "\n"
    forward_return!(x; info) = task_local_storage(:_slic_inline_pending, Any[]) do
        # Isolate any inline-call pending statements from this throwaway
        # type-inference trace — they'd otherwise leak into the caller's
        # block (for non-inline UDFs whose tracetype evaluates an inlined
        # call as part of return-type inference).
        info = OrderedDict{Symbol,Any}(pairs(info))
        forward!(x; info)
        info[RV_NAME]
    end
    # Walk a UDF body looking for forms StanBlocks deliberately does not
    # support inside `@deffun` definitions: sampling (`~`) and `target +=`
    # (and its variants). The user-facing rule: UDFs must not introduce
    # parameters or directly manipulate the log density. Errors here surface
    # at macro-expansion time with a clear message rather than later as a
    # downstream symbol-resolution or tracing failure.
    _is_target_compound_assign(x::Expr) = begin
        s = string(x.head)
        # any compound assignment to `target` — `+=`, `.-=`, etc.
        length(s) >= 2 && s[end] == '=' && s != "==" && s != "!=" &&
            s != "<=" && s != ">=" && s != "===" && s != "=>" &&
            length(x.args) >= 1 && x.args[1] === :target
    end
    _reject_udf_forms!(::Any, fname) = nothing
    _reject_udf_forms!(x::Expr, fname) = begin
        if Meta.isexpr(x, :call) && length(x.args) >= 1 && x.args[1] === :~
            error(
                "@deffun ($fname): `~` sampling statements are not allowed in UDF bodies. ",
                "UDFs must not introduce parameters or define likelihoods — keep `~` to `@slic` model bodies."
            )
        end
        if _is_target_compound_assign(x)
            error(
                "@deffun ($fname): `target +=` (and variants) is not allowed — StanBlocks does not support direct log-density manipulation in UDFs."
            )
        end
        foreach(a -> _reject_udf_forms!(a, fname), x.args)
    end

    deffun(x::LineNumberNode; kwargs...) = x
    # Macrocall args may start with a LineNumberNode carrying the call site;
    # otherwise the surrounding source is propagated.
    _macrocall_source(arg::LineNumberNode, source) = arg
    _macrocall_source(_, source) = source
    _is_inline_macrocall(x, sym::Symbol) = Meta.isexpr(x, :macrocall) && (
        x.args[1] === sym ||
        (Meta.isexpr(x.args[1], :.) && length(x.args[1].args) == 2 &&
         _is_quotenode(x.args[1].args[2]) && x.args[1].args[2].value === sym)
    )
    _is_lpxf_macrocall(x) = _is_inline_macrocall(x, Symbol("@lpxf"))
    _is_lhs_macrocall(x) = _is_inline_macrocall(x, Symbol("@lhs"))
    _is_at_inline_macrocall(x) = _is_inline_macrocall(x, Symbol("@inline"))
    # Doc-macrocall recognition. Julia's parser/lowering can emit several
    # shapes for a `@doc`-style docstring attached to a definition:
    # `GlobalRef(Core, @doc)` (post-lowering form for `"""..."""` followed by
    # an expr — what `slic_macroexpand` produces), bare `Symbol("@doc")`
    # (hand-written `@doc "..." expr`), or `Core.@doc` as a `:.` expression.
    # `slic.jl`'s `_is_doc_macro_head` covers the same set; mirror it here.
    _is_doc_macrocall(x) = Meta.isexpr(x, :macrocall) && _is_doc_head(x.args[1])
    _is_doc_head(h::GlobalRef) = h == GlobalRef(Core, Symbol("@doc"))
    _is_doc_head(h::Symbol) = h === Symbol("@doc")
    _is_doc_head(h::Expr) = h.head === :. && length(h.args) == 2 &&
        h.args[2] isa QuoteNode && h.args[2].value === Symbol("@doc")
    _is_doc_head(_) = false
    _peel_macros(x) = begin
        is_lhs = false
        is_lpxf = false
        is_inline = false
        cur = x
        while Meta.isexpr(cur, :macrocall) && (
                _is_lhs_macrocall(cur) || _is_lpxf_macrocall(cur) || _is_at_inline_macrocall(cur)
            )
            is_lhs    |= _is_lhs_macrocall(cur)
            is_lpxf   |= _is_lpxf_macrocall(cur)
            is_inline |= _is_at_inline_macrocall(cur)
            cur = cur.args[3]
        end
        (cur, is_lhs, is_lpxf, is_inline)
    end
    _inline_fname(inner, kind::AbstractString) = begin
        fsig = Meta.isexpr(inner, :(=)) ? inner.args[1] : inner
        fcall = Meta.isexpr(fsig, :(::)) ? fsig.args[1] : fsig
        Meta.isexpr(fcall, :call) || error(
            "@deffun: inline @$kind annotation must precede a function call or definition, got $inner"
        )
        f = fcall.args[1]
        _is_symbol(f) || error(
            "@deffun: inline @$kind annotation requires a bare-Symbol function name, got `$f`"
        )
        f
    end
    _lpxf_inline_fname(inner) = _inline_fname(inner, "lpxf")
    # Does AST `x` mention symbol `s` anywhere? Used by both regular-UDF
    # signature-dimension analysis and inline-body size binding so neither path
    # materialises size locals that its own definition never references.
    _ast_mentions(x, s::Symbol) = x isa Symbol ? x === s :
        (x isa Expr ? any(a -> _ast_mentions(a, s), x.args) : false)
    deffun(x::Expr; docstring="", source=LineNumberNode(0, :none), is_lhs=false, is_lpxf=false, is_inline=false, _shim_kwarg_specs=nothing, def_mod=nothing) = if x.head == :block
        seen_lpxf_bases = Set{Symbol}()
        for arg in x.args
            _is_expr(arg) || continue
            inner, _, arg_is_lpxf, _ = _peel_macros(arg)
            arg_is_lpxf || continue
            f = _lpxf_inline_fname(inner)
            base = _lpxf_base(f)
            base in seen_lpxf_bases && error(
                "@deffun: duplicate inline @lpxf annotation for base name `$base`. ",
                "At most one @lpxf-annotated method may register hooks per base function."
            )
            push!(seen_lpxf_bases, base)
        end
        Expr(:block, deffun.(x.args; docstring, source, is_lhs, is_lpxf, is_inline, def_mod)...)
    elseif x.head == :macrocall && (_is_lpxf_macrocall(x) || _is_lhs_macrocall(x) || _is_at_inline_macrocall(x))
        inner_source = _macrocall_source(x.args[2], source)
        new_is_lhs    = is_lhs    || _is_lhs_macrocall(x)
        new_is_lpxf   = is_lpxf   || _is_lpxf_macrocall(x)
        new_is_inline = is_inline || _is_at_inline_macrocall(x)
        deffun(x.args[3]; docstring, source=inner_source, is_lhs=new_is_lhs, is_lpxf=new_is_lpxf, is_inline=new_is_inline, _shim_kwarg_specs, def_mod)
    elseif x.head == :macrocall
        _is_doc_macrocall(x) || error(
            "@deffun: unexpected macrocall head `$(x.args[1])` (expected `@doc` / `@inline` / `@lpxf` / `@lhs`). ",
            "If a new doc-providing or annotation macro should be allowed, extend the predicate at functions.jl:_is_doc_macrocall / _is_inline_macrocall."
        )
        # @assert x.args[3] isa String
        deffun(x.args[4]; docstring=:($maybedoc($(x.args[3]))), source, is_lhs, is_lpxf, is_inline, _shim_kwarg_specs, def_mod)
    else
        # @assert x.head == :(=)
        fsig, body = ensure_xassign(x).args
        fcall, rv = ensure_xtyped(fsig).args
        @assert Meta.isexpr(fcall, :call) "@deffun: function signature must be a `:call` expression like `f(args...)::T`, got `$fcall`."
        f, all_args... = fcall.args

        # Kwargs (`f(x; sigma=1.0, alpha=2.0) = body`) mirror Julia's own
        # lowering: emit a canonical body method
        # `Core.kwcall(kw::ntup, ::typeof(f), x)::T = begin sigma=kw.sigma; …; body end`
        # plus an `@inline` shim `f(x) = Core.kwcall((;sigma=sigma, alpha=alpha), f, x)`
        # whose inline_body carries the kwarg names + defaults so call-site
        # expansion fills them from call-site kwargs or the registered
        # defaults. A kwarg with no default (`f(x; sigma)`) is *required*:
        # its default slot carries the sentinel `missing` and call-site
        # expansion errors if it is omitted. The shim's positional signature (no `:parameters` block)
        # is what gets registered for dispatch; kwargs don't participate in
        # dispatch (Julia's rule). Stan-side the canonical name auto-mangles
        # to `kwcall_f` via `func_name` since the function arg is
        # `always_inline`.
        if !isempty(all_args) && Meta.isexpr(all_args[1], :parameters)
            _is_symbol(f) || error("@deffun: kwargs require a bare-Symbol fname, got `$f`.")
            ismissing(body) && error("@deffun: kwargs require a body.")
            params = all_args[1]
            positional = all_args[2:end]
            kwarg_specs = []
            for p in params.args
                if Meta.isexpr(p, :kw)
                    # Optional kwarg with a default: `sigma=1.0` / `sigma::real=1.0`.
                    nt = p.args[1]
                    kw_name = Meta.isexpr(nt, :(::)) ? nt.args[1] : nt
                    push!(kwarg_specs, (name=kw_name, default=p.args[2]))
                elseif _is_symbol(p) || Meta.isexpr(p, :(::))
                    # Required kwarg with no default: `sigma` / `sigma::real`.
                    # The sentinel `missing` in the `default` slot marks it as
                    # required; call-site expansion (forward.jl) errors — the
                    # SLIC analogue of Julia's `UndefKeywordError` — if omitted.
                    # A real default is always an AST node (Symbol/Expr/literal),
                    # so the *value* `missing` can never collide with one.
                    kw_name = Meta.isexpr(p, :(::)) ? p.args[1] : p
                    push!(kwarg_specs, (name=kw_name, default=missing))
                else
                    error(
                        "@deffun: unrecognised keyword argument `$p`. ",
                        "Write `f(x; sigma)` (required) or `f(x; sigma=1.0)` (with default)."
                    )
                end
            end

            positional_names = [_name_of(p) for p in positional]
            rv_part = rv === :anything ? () : (rv,)

            # Canonical body method: `Core.kwcall(kw::ntup, ::typeof(f), positional...) = begin (unpack); body end`.
            # `Core.kwcall` is spliced as a bare Function value at args[1] of
            # the inner `:call` Expr; SLIC resolves it via `forward!(::Function)`.
            # Stan-side `func_name(::typeof(Core.kwcall))` mangles call sites
            # to `kwcall_<f>` via the type-token of `f`.
            kw_unpacks = [Expr(:(=), s.name, Expr(:., :kw, QuoteNode(s.name))) for s in kwarg_specs]
            canonical_body = Expr(:block, source, kw_unpacks..., body.args...)
            canonical_call = Expr(:call, Core.kwcall,
                Expr(:(::), :kw, :ntup),
                Expr(:(::), Expr(:call, :typeof, f)),
                positional...)
            canonical_sig = isempty(rv_part) ? canonical_call : Expr(:(::), canonical_call, rv_part[1])
            canonical_def = Expr(:(=), canonical_sig, canonical_body)

            # Inline shim: positional-only signature; kwargs live in the
            # inline_body metadata. Body constructs the kwcall NT and
            # delegates.
            shim_call = Expr(:call, f, positional...)
            shim_sig = isempty(rv_part) ? shim_call : Expr(:(::), shim_call, rv_part[1])
            nt_construct = Expr(:tuple, Expr(:parameters,
                [Expr(:kw, s.name, s.name) for s in kwarg_specs]...))
            shim_body = Expr(:block, source,
                Expr(:call, Core.kwcall, nt_construct, f, positional_names...))
            shim_def = Expr(:(=), shim_sig, shim_body)
            inline_shim = Expr(:macrocall, Symbol("@inline"), source, shim_def)

            return Expr(:block,
                # `function f end` first so the canonical method's
                # `::typeof(f)` dispatch can reference it.
                Expr(:function, f),
                deffun(canonical_def; docstring, source, is_lhs=false, is_lpxf=false, is_inline=false, def_mod),
                deffun(inline_shim; docstring, source, is_lhs, is_lpxf, is_inline=true,
                    _shim_kwarg_specs=kwarg_specs, def_mod),
            )
        end

        # Default positional args (`f(x, y=1.0)`): emit one `@inline`
        # trampoline per omitted-suffix arity that fills the default(s),
        # then fall through to register the full method below. This mirrors
        # Julia's "default args = sugar for multiple methods" semantics.
        default_idxs = findall(a -> Meta.isexpr(a, :kw), all_args)
        if !isempty(default_idxs)
            ismissing(body) && error("@deffun: default positional args require a body.")
            first_default = minimum(default_idxs)
            all(i -> Meta.isexpr(all_args[i], :kw), first_default:length(all_args)) ||
                error("@deffun: default positional args must be trailing — got `$fcall`.")
            _is_symbol(f) || error("@deffun: defaults require a bare-Symbol fname, got `$f`.")
            n = length(all_args)
            stripped = [Meta.isexpr(a, :kw) ? a.args[1] : a for a in all_args]
            arg_names_only = [_name_of(s) for s in stripped]
            rv_part = rv === :anything ? () : (rv,)
            defs = []
            # Trampolines: `f(x, ...)` for each k in [first_default-1, n-1].
            for k in (first_default - 1):(n - 1)
                tramp_args = stripped[1:k]
                full_call_args = Any[arg_names_only[1:k]...]
                for i in (k+1):n
                    push!(full_call_args, all_args[i].args[2])
                end
                tramp_call = Expr(:call, f, tramp_args...)
                tramp_sig = isempty(rv_part) ? tramp_call : Expr(:(::), tramp_call, rv_part[1])
                # `@deffun` expects bodies to be `begin ... end` blocks.
                tramp_body = Expr(:block, source, Expr(:call, f, full_call_args...))
                tramp_def = Expr(:(=), tramp_sig, tramp_body)
                # `@inline` so the trampoline doesn't emit a Stan function;
                # the call to `f` with defaults filled goes straight through.
                push!(defs, Expr(:macrocall, Symbol("@inline"), source, tramp_def))
            end
            # The full method, with defaults stripped to plain args.
            full_call = Expr(:call, f, stripped...)
            full_sig = isempty(rv_part) ? full_call : Expr(:(::), full_call, rv_part[1])
            push!(defs, Expr(:(=), full_sig, body))
            return Expr(:block, [
                deffun(d; docstring, source, is_lhs, is_lpxf, is_inline, def_mod) for d in defs
            ]...)
        end

        args = all_args
        # Trailing `!` in the function name is a synonym for `@inline`.
        # The Julia mutation-convention name is preserved verbatim — the only
        # actionable thing for SLIC is "inline this UDF at every call site."
        f_via_bang = _is_symbol(f) && endswith(string(f), "!")
        f_via_bang && (is_inline = true)
        f_is_lpxf_named = _is_symbol(f) && endswith(string(f), r"_lp[md]f")
        (f_is_lpxf_named || (_is_symbol(f) && endswith(string(f), r"_l?c?cdf"))) && (rv = :real)
        if (is_lhs || is_lpxf) && !f_is_lpxf_named
            error(
                "@deffun: @lhs/@lpxf annotation requires a `_lpdf`/`_lpmf`-suffixed function name, got `$f`"
            )
        end
        if is_inline && (is_lhs || is_lpxf)
            error(
                "@deffun: @inline / `!` cannot be combined with @lhs / @lpxf — inlined UDFs do not register the lpxf/likelihood/rng triad."
            )
        end
        if is_inline && ismissing(body)
            error("@deffun @inline / `!`: inline UDFs require a body, not a signature stub.")
        end
        ftype = :(typeof($f))
        args, vararg = if hasvararg(args)
            args[1:end-1], args[end]
        else
            args, nothing
        end
        is_token = Bool[_is_type_token(arg) for arg in args]
        args = map(zip(args, is_token, eachindex(args))) do (arg, tok, i)
            if tok
                xtyped(Symbol("anontok__", i), _type_token_ref(arg))
            else
                ensure_xtyped(arg, Symbol("arg__", i))
            end
        end
        arg_names = map(arg->arg.args[1], args)
        sig_names = copy(arg_names)
        arg_types = map(arg->ensure_xref(arg.args[2]), args)
        lhs_type = map(zip(arg_types, is_token)) do (at, tok)
            tok ? xsig_type_token(at) : xsig_type(at)
        end
        if !isnothing(vararg)
            push!(sig_names, vararg.args[1])
            # push!(arg_types, vararg.args[1])
            push!(lhs_type, :(Vararg{Any}))
        end

        # Collect every signature-dimension binding as a candidate first.  A
        # named dimension only belongs in the emitted function preamble when
        # this UDF's body / return annotation references it, or when a later
        # occurrence needs it as the reference side of a runtime shape check.
        # Keeping the use analysis local to this definition avoids emitting an
        # `int n = dims(x)[1];` merely because `n` appeared in the signature.
        fun_size_candidates = OrderedDict()
        required_fun_sizes = Set{Symbol}()
        # Keep the semantic source of each signature-size name alongside the
        # emitted Stan binding. During anonymous UDF tracing, `n` and
        # `dims(x)[1]` are distinct syntax even though the function preamble
        # explicitly equates them; typed-assignment validation needs that
        # relationship to compare shapes accurately.
        fun_size_aliases = OrderedDict{Symbol,String}()
        # When a dim name appears in multiple args (e.g.
        # `f(x::vector[n], y::vector[n])`), the first occurrence binds it
        # via `int n = dims(x)[1];` and each subsequent occurrence becomes
        # a runtime shape check that aborts with a `reject` message
        # naming the offending arg / dim.
        fun_checks = String[]
        for (arg_name, arg_type, tok) in zip(arg_names, arg_types, is_token)
            for (i, dim_name) in enumerate(arg_type.args[2:end])
                _is_symbol(dim_name) || continue
                dim_name == :(_) && continue
                dim_name in arg_names && continue
                if haskey(fun_size_candidates, dim_name)
                    # Token-arg dims are encoded differently; skip the
                    # check for those — usually internal dispatch glue.
                    tok && continue
                    push!(required_fun_sizes, dim_name)
                    access = string("dims(", arg_name, ")[", i, "]")
                    msg = string("\"", f, ": dim mismatch — `", arg_name,
                                 "` dim ", i, " (= \", ", access,
                                 ", \") does not match `", dim_name, "` (= \", ", dim_name, ", \")\"")
                    push!(fun_checks, string("if (", access, " != ", dim_name, ") reject(", msg, ");"))
                else
                    access = if tok
                        # Stan has no 1-element tuple type — single-dim tokens are passed
                        # as a plain `int`, so unpack without `.1` indexing.
                        ndims = length(arg_type.args) - 1
                        ndims == 1 ? string(arg_name) : string(arg_name, ".", i)
                    else
                        string("dims(", arg_name, ")[", i, "]")
                    end
                    fun_size_aliases[dim_name] = access
                    fun_size_candidates[dim_name] = "int $dim_name = $access;"
                end
            end
        end
        for dim_name in keys(fun_size_candidates)
            (_ast_mentions(body, dim_name) || _ast_mentions(rv, dim_name)) || continue
            push!(required_fun_sizes, dim_name)
        end
        fun_sizes = OrderedDict(
            dim_name => binding
            for (dim_name, binding) in pairs(fun_size_candidates)
            if dim_name in required_fun_sizes
        )
        hidden_size_names = union(
            Set(arg_names),
            setdiff(Set(keys(fun_size_candidates)), required_fun_sizes),
        )
        deconstruct = Expr(:block, 
            xassign(xtuple(arg_names..., (isnothing(vararg) ? () : (vararg,))...), :(x.args)), 
            [
                xassign(xtuple(ensure_xlhs.(args_type.args[2:end]; hidden=hidden_size_names)...), :($stan_size($args_name)))
                for (args_name, args_type) in zip(arg_names, arg_types)
            ]..., 
            :(info = (;$(sig_names...), $(keys(fun_sizes)...),))
        )
        size_aliases = (; fun_size_aliases...)
        anon_deconstruct = Expr(
            :block, 
            deconstruct.args..., 
            :(info = merge($anon_info(info), (; __size_aliases__ = $size_aliases))),
            :((;$(sig_names...), $(keys(fun_sizes)...),) = info),
            :(info = $OrderedDict{Symbol,Any}(pairs(info)))
        )

        stmts = []
        rv_expr = xsig_expr(ensure_xref(rv))
        stan_fundef = nothing
        # For inline UDFs we keep the original (un-`ensure_xreturn`'d) body
        # for substitution; the canonicalized + return-wrapped form is only
        # needed for the regular fundef path.
        original_body = body
        # Inline UDFs re-trace their body in the CALLER's scope, where the size
        # names (`n` in `x::vector[n]`) — which the non-inline path binds via the
        # emitted `int n = dims(x)[i];` preamble — are undefined. Prepend explicit
        # `n = dims(x)[i]` bindings so an inline body's `out::vector[n]` (fresh
        # result) resolves. Only for dims the body actually uses, once per dim
        # (first arg wins); non-token args only (token dims are dispatch glue).
        # The inline rename machinery turns each into a fresh local, so these
        # never leak into the caller's scope.
        inline_body_ast = original_body
        if !ismissing(original_body) && Meta.isexpr(original_body, :block)
            _size_binds = Any[]
            _seen_dims = Set{Symbol}()
            for (an, at, tok) in zip(arg_names, arg_types, is_token)
                tok && continue
                for (i, dn) in enumerate(at.args[2:end])
                    (_is_symbol(dn) && dn !== :(_) && !(dn in arg_names) && !(dn in _seen_dims)) || continue
                    _ast_mentions(original_body, dn) || continue
                    push!(_seen_dims, dn)
                    push!(_size_binds, :($dn = dims($an)[$i]))
                end
            end
            isempty(_size_binds) || (inline_body_ast = Expr(:block, _size_binds..., original_body.args...))
        end
        if !ismissing(body)
            @assert Meta.isexpr(body, :block) "@deffun: function body must be a `begin ... end` block, got `$body`."
            _reject_udf_forms!(body, f)
            # `void` UDFs run for side effects only — don't wrap the trailing
            # expression in a `return`, and force the inferred return type.
            if rv != :void
                body = ensure_xreturn(body)
            end
            sig_rv = if rv == :anything || _is_computed_ret_type(rv)
                rv_expr = :($forward_return!($(canonical(body)); info).type)
                :($rv_expr)
            else
                make_stan_type(rv)
            end
            stan_fundef = :($StanFunction3(
                $docstring,
                $sig_rv,
                $f,
                (;$(sig_names...), ),vcat(
                    $(collect(values(fun_sizes))),
                    $(fun_checks),
                    $forward!($(canonical(body)); info)
                )
            ))
        end

        xexpr = :(x::$CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        if _is_symbol(f)
            push!(stmts, :(function $f end))
            # Fail LOUDLY at load time if this StanBlocks-context builtin name was
            # not also added to the `@builtin_module` manifest (see
            # `_assert_builtin_registered`). No-op for user-side `@deffun`, whose
            # stub lands in the user's own module and resolves without a manifest.
            def_mod === nothing || push!(stmts, :($stan._assert_builtin_registered($(QuoteNode(f)), $def_mod)))
        end
        # Capture + inject the user function's defining module into
        # `info[:__mod__]` so nested `forward!`s inside the body (e.g.
        # `rv_expr = forward_return!(body; info)` when `rv == :anything`)
        # resolve names against the user's module rather than the
        # default `Main`. Without this, a user-side @deffun whose body
        # recurses into another user-side function (e.g.
        # `_lpmfs(...) = ... _lpmf(...)`) fails the inner symbol lookup
        # because the lookup falls back to `Main` and only checks
        # `StanBlocks.builtin` first. `fundef` already does this below;
        # tracetype needs it for the same reason.
        #
        # Prefer the defining module the `@deffun` macro threads in via
        # `def_mod` (= the macro's `__module__`). Deriving it instead from
        # `parentmodule(typeof(head(x)))` is wrong for a Base-extended
        # `@deffun Base.foo(...)=...`, where it yields `Base` (not the user
        # module), so the body's sibling lookup runs with `mod=Base` and the
        # user-module sibling isn't found. We fall back to that `parentmodule`
        # derivation only when `def_mod` isn't threaded (e.g. a direct
        # internal `deffun` call) — for a normal `@deffun f(...)=...` the two
        # agree, so the fallback never changes established behavior.
        capture_mod = isnothing(def_mod) ? :(__fundef_mod__ = $parentmodule(typeof($head(x)))) : :(__fundef_mod__ = $def_mod)
        inject_mod = :(info[$(QuoteNode(:__mod__))] = __fundef_mod__)
        # `tracetype`'s `info` is typically a NamedTuple (immutable), so
        # convert to OrderedDict before injecting `:__mod__`. (`fundef`
        # uses `anon_deconstruct` which already does the conversion.)
        promote_info = :(info = $OrderedDict{Symbol,Any}(pairs(info)))
        if is_inline
            # Inline UDFs do not produce a Stan function (no `functions {}`
            # entry) and do not register `tracetype` — the call site fully
            # substitutes the body and re-traces it in the caller's scope, so
            # neither tracetype-based dispatch nor the fundef payload is ever
            # consulted. We do register `inline_body` keyed on the same arg
            # signature so the call-site lookup picks the right method.
            arg_names_tuple = Expr(:tuple, [QuoteNode(n) for n in arg_names]...)
            vararg_qn = isnothing(vararg) ? :(nothing) : QuoteNode(vararg.args[1])
            kwarg_meta = if _shim_kwarg_specs === nothing
                :(())
            else
                # Pair each kwarg name (Symbol) with its default value Expr
                # — defaults are quoted as AST so they can be canonicalised
                # and forwarded at the call site. A *required* kwarg carries
                # the sentinel `missing` (not an AST) here; `expand_inline!`
                # errors if such a kwarg is omitted at the call site.
                Expr(:tuple, [
                    Expr(:tuple, QuoteNode(s.name), QuoteNode(s.default))
                    for s in _shim_kwarg_specs
                ]...)
            end
            push!(stmts, :($stan.inline_body($xexpr) = (
                arg_names = $arg_names_tuple,
                vararg_name = $vararg_qn,
                body = $(QuoteNode(inline_body_ast)),
                kwargs = $kwarg_meta,
                source = $source,
            )))
        else
            push!(stmts, quote
                $stan.tracetype($xexpr) = $(Expr(:block, source, capture_mod, deconstruct, promote_info, inject_mod, rv_expr))
            end)
            if !ismissing(body)
                push!(stmts, :($stan.fundef($xexpr) = $(Expr(:block, source, capture_mod, anon_deconstruct, inject_mod, stan_fundef))))
            end
        end
        _is_symbol(f) || return Expr(:block, stmts...)
        if is_lhs
            isempty(arg_types) && error(
                "@deffun: @lhs requires an explicit observation argument so the base ",
                "tracetype can infer the sampled value's type. ",
                "`@lhs $f(args...)` has no signature info to drive the inference — ",
                "either drop the `@lhs` annotation or pin the obs type, e.g. ",
                "`@lhs $f(y::vector[n], ...)`."
            )
            base_f = _lpxf_base(f)
            base_ftype = :(typeof($base_f))
            base_xexpr = :(_x::$CanonicalExpr{<:$base_ftype,<:Tuple{$(lhs_type[2:end]...)}})
            y_type = arg_types[1] == :(anything[]) ? :(real[]) : arg_types[1]
            y_expr = StanExpr(
                missing,
                StanType(getproperty(types, y_type.args[1]), ntuple(i->StanExpr(missing, StanType(types.int)), length(y_type.args)-1))
            )
            reconstruct = :(x = $CanonicalExpr($f, $y_expr, _x.args...))
            push!(stmts, :(function $base_f end))
            push!(stmts, quote
                $stan.tracetype($base_xexpr) = $(Expr(:block, source, reconstruct, deconstruct, xsig_expr(y_type)))
                $stan.fundef($base_xexpr) = nothing
            end)
        end
        if is_lpxf
            push!(stmts, lpxf_register(f; source))
        end
        Expr(:block, stmts...)
    end
    
end

fundef(x) = nothing
sig_expr(x) = x
sig_expr(x::Union{Tuple,NamedTuple,Vector}) = map(sig_expr, x)
sig_expr(x::CanonicalExpr) = remake(x, sig_expr(x.args)...)
sig_expr(x::StanExpr) = StanExpr(:_, sig_expr(type(x)))
sig_expr_size(x::StanExpr) = StanExpr(:_, StanType(types.int, ()))
sig_expr(x::StanType) = StanType(sigtype(center_type(x)), map(sig_expr_size, stan_size(x)))
sig_expr(x::StanType{<:types.tup}) = StanType(center_type(x), map(sig_expr_size, stan_size(x)); arg_types=sig_expr(info(x).arg_types))
sig_expr(x::StanType{<:types.func}) = StanType(center_type(x), map(sig_expr_size, stan_size(x)); value=sig_expr(info(x).value))
# A closure arg's function-dedup key MUST preserve its per-site `id`.
# `func_name(::StanExpr2{<:types.closure})` mangles the emitted Stan function
# name on that id (`f_closure_<id>`), so two DISTINCT closure literals with
# identical bodies — distinct ids → distinct names `f_closure_1`/`f_closure_2`
# — must collect as TWO definitions. The generic `sig_expr(::StanExpr)` wipes
# `expr` to `:_` and the generic `sig_expr(::StanType)` drops `info.value`,
# losing the id: both keys collapsed to one, so only `f_closure_1` was defined
# while the second call site still emitted `f_closure_2(...)` → stanc rejects
# with "undeclared identifier". Keying on the id realigns dedup with naming.
sig_expr(x::StanExpr2{<:types.closure}) = StanExpr(type(x).info.value.id, sig_expr(type(x)))
fetch_functions!(x::CanonicalExpr; info) = begin
    sx = sig_expr(x)
    sx in keys(info) && return
    info[sx] = fundef(x)
    isnothing(info[sx]) && return
    fetch_subfunctions!(info[sx].body; info)
end
fetch_functions!(x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    fetch_functions!(expr(lpxf_expr(lhs, rhs)); info)
    if qual(lhs) == :data || lqual(lhs) == :undefined
        fetch_functions!(expr(likelihood_expr(lhs, rhs)); info)
        # Mirror the gq push: wrap `lhs` into a tokenof token so the per-shape
        # `*_rng` @deffun overloads dispatch. Without this, `rng_expr` misses
        # entirely — it no longer accepts plain StanExprs as the first arg.
        lhs_ct = center_type(lhs)
        token = StanExpr(lhs_ct, StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
        fetch_functions!(expr(rng_expr(token, rhs)); info)
    end
end
fetch_subfunctions!(;info) = x->fetch_subfunctions!(x; info)
fetch_subfunctions!(x; info) = nothing
fetch_subfunctions!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_subfunctions!(;info), x)
fetch_subfunctions!(x::StanExpr; info) = fetch_subfunctions!((expr(x), type(x)); info)
fetch_subfunctions!(x::StanType; info) = fetch_subfunctions!((stan_size(x), x.info); info)
fetch_subfunctions!(x::CanonicalExpr; info) = begin 
    fetch_functions!(x; info)
    fetch_subfunctions!((x.args, x.kwargs); info)
end
# `tok` namespaces a call's anonymized-arg placeholders (`_arg<tok>_<i>`) per
# `stan_expr`/`anon_canonical` invocation. Without it, every nesting level
# reused `_arg1`, `_arg2`, …; a UDF whose inferred return type carries a
# symbolic size expression referencing its own params (anonymized to `_argN`)
# would have those `_argN` re-substituted by an INNER call's `deanon_size`
# (whose args are different), aliasing a param to the wrong value. That
# corrupted symbolic sizes threaded through HOFs (e.g. `getindex_slice`'s
# `vector[max(0, ends-start+1)]`), surfacing later as
# `tracetype not defined for (anything - anything)`. The per-call `tok` keeps
# each level's placeholders distinct so `deanon_size` only ever substitutes
# the placeholders it actually introduced. The counter lives in per-trace
# task-local storage (`_next_anon_id` + the seed-scope are defined centrally in
# tracing.jl); the names never reach Stan output — they are always deanonymized
# away, so the only out-of-trace caller (ad-hoc `stan_expr(::CanonicalExpr)`)
# is determinism-irrelevant and falls to the lazy `get!` default.
anon_arg(x::StanExpr, i::Int, tok) = StanExpr(Symbol(:_arg, tok, :_, i), type(x))
anon_arg(x, i::Int, tok) = x
anon_canonical(x::CanonicalExpr, tok=_next_anon_id()) = remake(x, ntuple(i -> anon_arg(x.args[i], i, tok), length(x.args))...)
anon_canonical(x::CanonicalExpr{Colon}, tok=_next_anon_id()) = x   # needs real args for range size
anon_canonical(x::BlockExpr, tok=_next_anon_id()) = x               # args is Vector, not Tuple
anon_canonical(x::CanonicalExprV{:nt}, tok=_next_anon_id()) = x     # preserve named tuple structure
anon_info(x::NamedTuple) = (;[
    key=>anon_expr(key, value)
    for (key, value) in pairs(x)
]...)
anon_expr(key, x) = error("anon_expr not defined for key `$key` with value of type `$(typeof(x))` — add a specialization if this type needs an anon representation.")
anon_expr(key, x::Tuple) = begin
    idxs = cumsum(map(!always_inline, x))
    ([
        anon_expr(Symbol(key, idx), xi)
        for (idx, xi) in zip(idxs, x)
    ]...,)
end
anon_expr(key, x::StanExpr) = StanExpr(key, StanType(center_type(x), ([
    StanExpr("dims($key)[$i]", StanType(types.int))
    for (i, s) in enumerate(stan_size(x))
]...,)))
anon_expr(key, x::StanExpr2{<:types.func}) = StanExpr(type(x).info.value, type(x))
# Closures have no Stan-side existence (always_inline), but their full
# `StanType{<:types.closure}` — including `info.value = closure_record` —
# must reach the receiver UDF body so `inline_body` dispatch can pull the
# record off and substitute. The anon `expr` is the param name as a
# Symbol; the type is preserved verbatim.
anon_expr(key, x::StanExpr2{<:types.closure}) = StanExpr(key, type(x))
# Sized type tokens expose their dims as tuple fields of the Stan arg (e.g.
# `name.1`, `name.2`) so the function body can reference them by the original
# Julia name via `int n = name.1;` preamble (see @deffun `fun_sizes`).
anon_expr(key, x::StanExpr2{<:types.tokenof,S}) where {S} = StanExpr(
    key,
    StanType(center_type(x), ([
        StanExpr(string(key, ".", i), StanType(types.int))
        for i in 1:S
    ]...,); value=type(x).info.value),
)
anon_expr(key, x::StanExpr2{<:types.tup}) = begin
    StanExpr(key, StanType(center_type(x); arg_types=([
        anon_expr(Symbol(key, ".", i), StanExpr(:_, arg_type)).type
        for (i, arg_type) in enumerate(x.type.info.arg_types)
    ]...,)))
end
anon_expr(key, x::StanExpr2{<:types.ntup}) = begin
    # arg_types = x.type.info.arg_types
    StanExpr(key, StanType(center_type(x); arg_types=(;[
        name=>anon_expr(Symbol(key, ".", i), StanExpr(:_, arg_type)).type
        for (i, (name, arg_type)) in enumerate(pairs(x.type.info.arg_types))
    ]...,)))
end
func_name(x::Symbol) = x
func_name(x::QuoteNode) = func_name(x.value)
func_name(x::Expr) = if x.head == :.
    func_name(x.args[end])
else
    error("func_name(::Expr) only handles `:.` heads (module-qualified refs), got head `$(x.head)` in `$x`.")
end
func_name(f, args) = begin
    rv = join(vcat(func_name(f), func_name(args)), "_")
    suffix_idxs = findfirst(r"_(rng|u?lp(m|d)fs?)_", rv)
    if isnothing(suffix_idxs) 
        rv
    else
        suffix = rv[suffix_idxs[1]:suffix_idxs[end]-1]
        base = rv[1:suffix_idxs[1]-1] * rv[suffix_idxs[end]:end]
        base * suffix
    end
end
func_name(args::NamedTuple) = func_name(values(args))
func_name(args::Tuple) = mapreduce(func_name, vcat, args; init=[])
func_name(x) = []
func_name(x::StanExpr) = always_inline(x) ? [func_name(type(x).info.value)] : []
# Closures: each `(x) -> body` site gets a fresh monotonic id at construction
# time (see `_make_closure` in slic.jl). The id alone is enough — two
# textually-distinct lambdas at the same site already have distinct ids,
# so HOF receivers (`simple_reduce_sum_helper`, etc.) get distinct mangled
# Stan names per closure they are specialised against.
func_name(x::StanExpr2{<:types.closure}) = ["closure", string(type(x).info.value.id)]
# Usertype StanExprs contribute their nominal tag to the mangled receiver
# name (e.g. `getindex_RaggedVector`), so a `Base.getindex(rv::RaggedVector,
# i::int)` UDF gets a distinct Stan helper name from the generic
# tup-getindex tracetype rule. The argument *itself* is still emitted as
# a regular Stan-side function arg (it's not `always_inline`).
func_name(x::StanExpr2{<:types.usertype}) = [string(center_type(x).name.name)]
# Type tokens always contribute their center type to the mangled Stan name,
# including sized tokens that are rendered at the call site.
func_name(x::StanExpr2{<:types.tokenof}) = [func_name(type(x).info.value)]
func_name(::Type{T}) where {T<:types.anything} = string(T.name.name)
func_name(x::Function) = string(x)
# Operator + renamed-function Stan-name table, generated as an @eval loop
# (mirrors the operator @eval loop in show.jl). Each entry maps a Julia
# function to its Stan-side name fragment used for call-site mangling.
for (f, nm) in (
    (&, "and"), (|, "or"), (>=, "gte"), (>, "gt"), (==, "eq"), (!=, "ne"),
    (<=, "lte"), (<, "lt"), (+, "add"), (-, "sub"), (*, "mul"),
    (/, "div"), (÷, "idiv"), (^, "pow"),
    # Julia functions with different Stan names
    (length, "num_elements"), (minimum, "min"), (maximum, "max"), (abs2, "square"),
)
    @eval func_name(::typeof($f)) = $nm
end
func_args(args::NamedTuple) = Join(mapreduce(func_args, vcat, pairs(args); init=[]), ", ")
func_args(arg::Pair) = func_args(arg...)
func_args(name, ::StanExpr2{<:types.func}) = []
# Closure phase 2: a closure passed to a Stan-emitted UDF lifts its
# captures into positional args appended to the receiver's signature.
# Each captured StanExpr's `expr` is a Symbol (e.g. `:shift`) that already
# matches the body's reference (the capture was substituted verbatim into
# the body), so naming the new Stan arg by the same Symbol makes the body
# resolve naturally inside the function scope.
func_args(name, x::StanExpr2{<:types.closure}) = [
    sigtype(v) * " " * string(k)
    for (k, v) in pairs(type(x).info.value.captures)
]
# Call-site arg expansion: when a closure flows into a `:call` CanonicalExpr's
# args, render its captures in place. Stan-side, the receiver's signature
# already absorbs the captures via `func_args` above; call-site arg lists
# need to thread the actual capture values at matching positions. Closures
# with no captures expand to zero args (transparent); closures with captures
# splice each capture value as a positional arg.
expand_call_args(args) = begin
    rv = Any[]
    for a in args
        _splat_or_keep!(rv, a)
    end
    rv
end
_splat_or_keep!(rv, a::StanExpr2{<:types.closure}) =
    (append!(rv, values(type(a).info.value.captures)); nothing)
_splat_or_keep!(rv, a) = (push!(rv, a); nothing)
# What `Base.show(::CanonicalExpr)` (and its `<:ODESolver` / `<:ReduceSumFunction`
# specialisations) want as the rendered Stan-side arg list: closures expanded
# to their capture values, then `always_inline`-typed StanExprs (functions,
# 0-dim tokens, capture-free closures) filtered out. Single helper so each
# show specialisation stays a one-liner.
stan_call_args(args) = filter(!always_inline, expand_call_args(args))
# Captures of a function-position closure (used by ODE-solver fetch/show:
# closures at args[1] must thread their captures into the builtin's
# trailing args, since `expand_call_args` only operates on the rendered
# args list and the function-position arg gets folded into the receiver
# name instead).
_closure_captures(f::StanExpr2{<:types.closure}) = Tuple(values(type(f).info.value.captures))
_closure_captures(_) = ()
# 0-dim tokens: no Stan-side arg. 1-dim tokens: a plain `int` (Stan has no
# 1-element tuple type). N>1-dim tokens: pack dims into a single
# `tuple(int, …)` parameter; the function body then unpacks fields via `.i`.
func_args(name, ::StanExpr2{<:types.tokenof,0}) = []
func_args(name, ::StanExpr2{<:types.tokenof,1}) = "int $name"
func_args(name, ::StanExpr2{<:types.tokenof,S}) where {S} = "tuple(" * join(fill("int", S), ", ") * ") $name"
func_args(name, value::StanExpr2) = sigtype(value) * " $name"
func_args(name, value::Tuple) = reduce(vcat, [
    func_args(Symbol(name, i), vali)
    for (i, vali) in enumerate(filter(!always_inline, value))
]; init=[])
