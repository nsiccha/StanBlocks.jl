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
backward!(x::StanExpr2{<:types.closure}; info) = x
# A closure StanExpr's `expr` field carries the closure record (a
# NamedTuple). Without a specialisation, `fetch_data!`'s NamedTuple
# fallback would iterate the record and trip on its `body::Expr` field.
# Closures contribute nothing data-side — captured StanExprs are already
# traversed via their original bindings.
fetch_data!(x::StanExpr2{<:types.closure}; info) = nothing
# tokenof StanExprs carry a raw Stan type as `expr`; skip recursing into the
# bare `Type{...}` which would hit the generic backward! fallback.
backward!(x::StanExpr2{<:types.tokenof}; info) = x
# fetch_data!(::StanExpr2{<:types.func}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{<:types.func}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{types.func}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{types.func, 0}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{<:types.func, 0}}; info) = nothing

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
tracetype(x::CanonicalExpr{<:Union{typeof.((+, -, ^, *, /))...}}) = if length(x.args) > 2
    f = head(x)
    tracetype(CanonicalExpr(f, x.args[1], stan_expr(CanonicalExpr(f, x.args[2:end]...))))
else 
    error("tracetype not defined for $(short_expr(x))!")
    StanType(types.anything)
end
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 1),))))
)
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
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = x.args[1].type.info.arg_types[x.args[2].type.info.value]
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
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{<:types.matrix}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.vector, (stan_size(x.args[1], 1),))
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{types.real,2}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.real, (stan_size(x.args[1], 1),))
tracetype(x::BracesExpr) = StanType(types.real, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::VectExpr) = StanType(types.vector, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::TupleExpr) = StanType(types.tup; arg_types=map(type, x.args))
tracetype(x::KwExpr) = type(x.args[2])
# tracetype(x::NamedTupleExpr) = StanType(types.ntup; arg_types=map(type, x.args))
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
    parent::Function
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

    gettype(ct::Symbol) = getproperty(types, ct)
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
    _is_type_token_sym(x) = isa(x, Symbol) && isdefined(types, x) && getproperty(types, x) isa Type{<:types.anything}
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
                isa(dim_name, Symbol) || continue
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
    sigarg(x, name::Symbol) = error("sigarg deprecated fallback hit for `$name` of type `$(typeof(x))`; use func_args instead.")#sigtype(x) * " $name"
    sigarg(::StanExpr2{<:types.func}, ::Symbol) = error("sigarg(::types.func) deprecated; use func_args.")#nothing
    sigarg(x::Tuple, name::Symbol) = error("sigarg(::Tuple) deprecated; use func_args.")#join(ntuple(i->sigarg(x[i], Symbol(name, i)), length(x)), ", ")
    always_inline(x) = false
    always_inline(::StanExpr2{<:types.func}) = true
    always_inline(::StanExpr2{<:types.closure}) = true
    # 0-dim type tokens carry no runtime value — they only contribute a name
    # mangle component to the Stan function name. Sized tokens (`real[n]`)
    # _do_ render, as a Stan `tuple(int, ...)` literal at the call site.
    always_inline(::StanExpr2{<:types.tokenof,0}) = true
    # sigargs(x::Tuple) = filter(!isnothing, map(sigargs, x))
    sigarg(x::Expr) = begin
        @assert x.head == :(::) "sigarg expects a `name::T` expression, got `$x` (head `$(x.head)`)."
        "$(sigtype(x.args[2])) $(x.args[1])"
    end
    # fname(x) = string(x)
    # fname(::typeof(>=)) = "gte" 
    # stan_call(;kwargs...) = x->stan_call(x, ;kwargs...)
    # stan_call(x::Expr; kwargs...) = stan_expr()
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
    _is_inline_macrocall(x, sym::Symbol) = Meta.isexpr(x, :macrocall) && (
        x.args[1] === sym ||
        (Meta.isexpr(x.args[1], :.) && length(x.args[1].args) == 2 &&
         x.args[1].args[2] isa QuoteNode && x.args[1].args[2].value === sym)
    )
    _is_lpxf_macrocall(x) = _is_inline_macrocall(x, Symbol("@lpxf"))
    _is_lhs_macrocall(x) = _is_inline_macrocall(x, Symbol("@lhs"))
    _is_at_inline_macrocall(x) = _is_inline_macrocall(x, Symbol("@inline"))
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
        f isa Symbol || error(
            "@deffun: inline @$kind annotation requires a bare-Symbol function name, got `$f`"
        )
        f
    end
    _lpxf_inline_fname(inner) = _inline_fname(inner, "lpxf")
    deffun(x::Expr; docstring="", source=LineNumberNode(0, :none), is_lhs=false, is_lpxf=false, is_inline=false, _shim_kwarg_specs=nothing) = if x.head == :block
        seen_lpxf_bases = Set{Symbol}()
        for arg in x.args
            isa(arg, Expr) || continue
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
        Expr(:block, deffun.(x.args; docstring, source, is_lhs, is_lpxf, is_inline)...)
    elseif x.head == :macrocall && (_is_lpxf_macrocall(x) || _is_lhs_macrocall(x) || _is_at_inline_macrocall(x))
        inner_source = x.args[2] isa LineNumberNode ? x.args[2] : source
        new_is_lhs    = is_lhs    || _is_lhs_macrocall(x)
        new_is_lpxf   = is_lpxf   || _is_lpxf_macrocall(x)
        new_is_inline = is_inline || _is_at_inline_macrocall(x)
        deffun(x.args[3]; docstring, source=inner_source, is_lhs=new_is_lhs, is_lpxf=new_is_lpxf, is_inline=new_is_inline, _shim_kwarg_specs)
    elseif x.head == :macrocall
        @assert x.args[1] == GlobalRef(Core, Symbol("@doc")) "@deffun: unexpected macrocall head `$(x.args[1])` (expected a `@doc` docstring)."
        # @assert x.args[3] isa String
        deffun(x.args[4]; docstring=:($maybedoc($(x.args[3]))), source, is_lhs, is_lpxf, is_inline, _shim_kwarg_specs)
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
        # defaults. The shim's positional signature (no `:parameters` block)
        # is what gets registered for dispatch; kwargs don't participate in
        # dispatch (Julia's rule). Stan-side the canonical name auto-mangles
        # to `kwcall_f` via `func_name` since the function arg is
        # `always_inline`.
        if !isempty(all_args) && Meta.isexpr(all_args[1], :parameters)
            isa(f, Symbol) || error("@deffun: kwargs require a bare-Symbol fname, got `$f`.")
            ismissing(body) && error("@deffun: kwargs require a body.")
            params = all_args[1]
            positional = all_args[2:end]
            kwarg_specs = []
            for p in params.args
                Meta.isexpr(p, :kw) || error(
                    "@deffun: kwargs must have defaults — got `$p`. ",
                    "Write `f(x; sigma=1.0)` not `f(x; sigma)`."
                )
                nt = p.args[1]
                kw_name = Meta.isexpr(nt, :(::)) ? nt.args[1] : nt
                push!(kwarg_specs, (name=kw_name, default=p.args[2]))
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
                deffun(canonical_def; docstring, source, is_lhs=false, is_lpxf=false, is_inline=false),
                deffun(inline_shim; docstring, source, is_lhs, is_lpxf, is_inline=true,
                    _shim_kwarg_specs=kwarg_specs),
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
            isa(f, Symbol) || error("@deffun: defaults require a bare-Symbol fname, got `$f`.")
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
                deffun(d; docstring, source, is_lhs, is_lpxf, is_inline) for d in defs
            ]...)
        end

        args = all_args
        # Trailing `!` in the function name is a synonym for `@inline`.
        # The Julia mutation-convention name is preserved verbatim — the only
        # actionable thing for SLIC is "inline this UDF at every call site."
        f_via_bang = isa(f, Symbol) && endswith(string(f), "!")
        f_via_bang && (is_inline = true)
        f_is_lpxf_named = isa(f, Symbol) && endswith(string(f), r"_lp[md]f")
        (f_is_lpxf_named || (isa(f, Symbol) && endswith(string(f), r"_l?c?cdf"))) && (rv = :real)
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
        if is_inline && !isa(f, Symbol)
            error("@deffun @inline / `!`: function name must be a bare Symbol, got `$f`.")
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

        fun_sizes = OrderedDict()
        # When a dim name appears in multiple args (e.g.
        # `f(x::vector[n], y::vector[n])`), the first occurrence binds it
        # via `int n = dims(x)[1];` and each subsequent occurrence becomes
        # a runtime shape check that aborts with a `reject` message
        # naming the offending arg / dim.
        fun_checks = String[]
        for (arg_name, arg_type, tok) in zip(arg_names, arg_types, is_token)
            for (i, dim_name) in enumerate(arg_type.args[2:end])
                isa(dim_name, Symbol) || continue
                dim_name == :(_) && continue
                dim_name in arg_names && continue
                if haskey(fun_sizes, dim_name)
                    # Token-arg dims are encoded differently; skip the
                    # check for those — usually internal dispatch glue.
                    tok && continue
                    access = string("dims(", arg_name, ")[", i, "]")
                    msg = string("\"", f, ": dim mismatch — `", arg_name,
                                 "` dim ", i, " (= \", ", access,
                                 ", \") does not match `", dim_name, "` (= \", ", dim_name, ", \")\"")
                    push!(fun_checks, string("if (", access, " != ", dim_name, ") reject(", msg, ");"))
                else
                    fun_sizes[dim_name] = if tok
                        # Stan has no 1-element tuple type — single-dim tokens are passed
                        # as a plain `int`, so unpack without `.1` indexing.
                        ndims = length(arg_type.args) - 1
                        ndims == 1 ? "int $dim_name = $arg_name;" : "int $dim_name = $arg_name.$i;"
                    else
                        "int $dim_name = dims($arg_name)[$i];"
                    end
                end
            end
        end
        deconstruct = Expr(:block, 
            xassign(xtuple(arg_names..., (isnothing(vararg) ? () : (vararg,))...), :(x.args)), 
            [
                xassign(xtuple(ensure_xlhs.(args_type.args[2:end]; hidden=arg_names)...), :($stan_size($args_name)))
                for (args_name, args_type) in zip(arg_names, arg_types)
            ]..., 
            :(info = (;$(sig_names...), $(keys(fun_sizes)...),))
        )
        anon_deconstruct = Expr(
            :block, 
            deconstruct.args..., 
            :(info = $anon_info(info)),
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
        if !ismissing(body)
            @assert Meta.isexpr(body, :block) "@deffun: function body must be a `begin ... end` block, got `$body`."
            _reject_udf_forms!(body, f)
            # `void` UDFs run for side effects only — don't wrap the trailing
            # expression in a `return`, and force the inferred return type.
            if rv != :void
                body = ensure_xreturn(body)
            end
            sig_rv = if rv == :anything
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
        isa(f, Symbol) && push!(stmts, :(function $f end))
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
        capture_mod = :(__fundef_mod__ = $parentmodule(typeof($head(x))))
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
                # and forwarded at the call site.
                Expr(:tuple, [
                    Expr(:tuple, QuoteNode(s.name), QuoteNode(s.default))
                    for s in _shim_kwarg_specs
                ]...)
            end
            push!(stmts, :($stan.inline_body($xexpr) = (
                arg_names = $arg_names_tuple,
                vararg_name = $vararg_qn,
                body = $(QuoteNode(original_body)),
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
        isa(f, Symbol) || return Expr(:block, stmts...)
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
anon_arg(x::StanExpr, i::Int) = StanExpr(Symbol(:_arg, i), type(x))
anon_arg(x, i::Int) = x
anon_canonical(x::CanonicalExpr) = remake(x, ntuple(i -> anon_arg(x.args[i], i), length(x.args))...)
anon_canonical(x::CanonicalExpr{Colon}) = x   # needs real args for range size
anon_canonical(x::BlockExpr) = x               # args is Vector, not Tuple
anon_canonical(x::CanonicalExprV{:nt}) = x     # preserve named tuple structure
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
    rv = join(vcat(func_name(f), func_name(args)...), "_")
    suffix_idxs = findfirst(r"_(rng|u?lp(m|d)fs?)_", rv)
    if isnothing(suffix_idxs) 
        rv
    else
        suffix = rv[suffix_idxs[1]:suffix_idxs[end]-1]
        base = rv[1:suffix_idxs[1]-1] * rv[suffix_idxs[end]:end]
        base * suffix
        # if isnothing(match(r"u?lp(m|d)f$", suffix))
        #     base * suffix
        # else
        #     replace(base, r"(_u?lp(m|d)f)+$"=>"") * suffix
        # end
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
# Type tokens always contribute their center type to the mangled Stan name,
# including sized tokens that are rendered at the call site.
func_name(x::StanExpr2{<:types.tokenof}) = [func_name(type(x).info.value)]
func_name(::Type{T}) where {T<:types.anything} = string(T.name.name)
func_name(x::Function) = string(x)
func_name(::typeof(&)) = "and"
func_name(::typeof(|)) = "or"
func_name(::typeof(>=)) = "gte"
func_name(::typeof(>)) = "gt"
func_name(::typeof(==)) = "eq"
func_name(::typeof(<=)) = "lte"
func_name(::typeof(<)) = "lt"
func_name(::typeof(+)) = "add"
func_name(::typeof(-)) = "sub"
func_name(::typeof(*)) = "mul"
func_name(::typeof(/)) = "div"
# Julia functions with different Stan names
func_name(::typeof(length)) = "num_elements"
func_name(::typeof(minimum)) = "min"
func_name(::typeof(maximum)) = "max"
func_name(::typeof(abs2)) = "square"
func_args(args::NamedTuple) = Join(mapreduce(func_args, vcat, pairs(args); init=[]), ", ")
func_args(arg::Pair) = func_args(arg...)
func_args(name, ::StanExpr2{<:types.func}) = []
# Closures are always inlined at the call site — they produce no Stan-side
# argument when passed as a UDF parameter (their captures are substituted
# into the receiver's specialised body).
func_args(name, ::StanExpr2{<:types.closure}) = []
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