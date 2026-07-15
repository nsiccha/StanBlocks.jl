
deanon_size(s, x, tok) = s
deanon_size(s::StanExpr, x::CanonicalExpr, tok) = _deanon_size_expr(expr(s), s, x, tok)
# Match this call's own placeholders `_arg<tok>_<i>` only — never an inner/outer
# level's (different `tok`), which would alias a param to the wrong arg.
_deanon_size_expr(e::Symbol, s, x, tok) = begin
    pre = string("_arg", tok, "_")
    es = string(e)
    startswith(es, pre) || return s
    suf = SubString(es, lastindex(pre) + 1)
    (!isempty(suf) && all(isdigit, suf)) || return s
    i = parse(Int, suf)
    i <= length(x.args) ? x.args[i] : s
end
_deanon_size_expr(e::CanonicalExpr, s, x, tok) = begin
    new_args = map(a -> deanon_size(a, x, tok), e.args)
    new_args == e.args && return s
    StanExpr(remake(e, new_args...), type(s))
end
_deanon_size_expr(_, s, _, tok) = s
deanon_type(tt::StanType, x::CanonicalExpr, tok) = begin
    sz = stan_size(tt)
    nsz = map(s -> deanon_size(s, x, tok), sz)
    sz == nsz ? tt : StanType(center_type(tt), nsz; [k => v for (k, v) in pairs(info(tt)) if k != :size]...)
end
stan_expr(x::CanonicalExpr) = begin
    tok = _next_anon_id()
    tt = deanon_type(tracetype(anon_canonical(x, tok)), x, tok)
    StanExpr(x, remake(tt; qual=maximum(qual, x.args; init=:data), cv=any(cv, x.args) || cv(tt)))
end
# A @slic sub-model or named sub-model function in call position. For an anonymous
# `SlicModel`, data flows via KEYWORDS — a positional call now errors (its call
# operator points at `Base.merge` for splice overrides / `@slic f(...)=...` for
# positional inputs). For a `SubmodelFn`, positional args ARE the inputs (bound by
# its generated call method; Julia's own dispatch/arity handle them). Either way the
# call yields a `SlicModel`, embedded via the existing `~`-rhs-is-`SlicModel` path.
stan_expr(x::CanonicalExpr{<:Union{SlicModel,SubmodelFn}}) = head(x)(x.args...; x.kwargs...)

backward!(x; info) = error("backward! not defined for value `$x` of type `$(typeof(x))` — no method matches a more specific signature.")
backward!(;info) = x->backward!(x; info)
backward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(backward!(;info), x)
backward!(x::Union{String,Number,LineNumberNode,Symbol,Nothing,Colon}; info) = x
backward!(x::CanonicalExpr; info) = remake(x, backward!(x.args; info)...)
backward!(x::BlockExpr; info) = remake(x, reverse(backward!.(reverse(x.args); info))...)
backward!(x::AssignmentExpr; info) = if lqual(info[expr(x.args[1])]) == :affects_likelihood
    lhs, rhs = x.args
    remake(x, info[expr(x.args[1])], backward!(rhs; info))
elseif qual(x.args[1]) == :parameter
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
else
    x
end
backward!(x::SamplingExpr{<:StanExpr{Symbol}}; info) = if qual(x.args[1]) == :data || lqual(info[expr(x.args[1])]) == :affects_likelihood
    lhs, rhs = x.args
    remake(x, info[expr(x.args[1])], backward!(rhs; info))
else
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
end
backward!(x::SamplingExpr; info) = begin 
    @assert qual(x.args[1]) == :data
    lhs, rhs = x.args
    remake(x, backward!(lhs; info), backward!(rhs; info))
end
backward!(x::ReturnExpr; info) = x
backward!(x::DocumentExpr; info) = remake(x, backward!.(x.args; info)...)
backward!(x::StanExpr; info) = StanExpr(backward!(expr(x); info), backward!(type(x); info))
backward!(x::StanExpr{Symbol}; info) = info[expr(x)] = remake(x; lqual=:affects_likelihood)
backward!(x::StanType; info) = remake(x; lqual=:affects_likelihood)

distribute!(x::BlockExpr; info) = distribute!.(x.args; info)
distribute!(x::Union{LineNumberNode,Nothing}; info) = nothing
distribute!(x::DocumentExpr{<:Any,<:BlockExpr}; info) = distribute!(x.args[2]; info)
distribute!(x; info) = begin
    _push_expr!(info, x)
    for b in distribution_blocks(x; info)
        push!(block(info, b), x; info)
    end
    _pop_expr!(info)
end
qual(x::AssignmentExpr) = qual(x.args[1])
qual(x::SamplingExpr) = qual(x.args[1])
distribution_blocks(x::AssignmentExpr; info) = if qual(x) == :data
    (:transformed_data, )
elseif qual(x) == :parameter
    (:transformed_parameters, )
else
    (:generated_quantities, )
end
distribution_blocks(x::SamplingExpr; info) = if qual(x) == :data
    if cv(x.args[1])
        (:generated_quantities,)
    else
        (:model, :generated_quantities)
    end
elseif qual(x) == :parameter
    (:parameters, :model)
else
    (:generated_quantities, )
end
distribution_blocks(x::ReturnExpr; info) = (:generated_quantities,)
distribution_blocks(x::DocumentExpr; info) = distribution_blocks(x.args[2]; info)
distribution_blocks(::Union{Nothing}; info) = tuple()
distribution_blocks(x::StanExpr{Symbol}; info) = hasvalue(x) ? (:data,) : tuple()
# Void-typed StanExprs (calls to `::void` UDFs at statement position) follow
# their args' qualifier — same as an assignment statement would. Implemented
# via a runtime `center_type` check rather than dispatch on `<:types.void`
# because the `types` submodule is defined later in `functions.jl` and isn't
# in scope when this method's signature is parsed.
distribution_blocks(x::StanExpr; info) = if center_type(x) === types.void
    if qual(x) == :data
        (:transformed_data,)
    elseif qual(x) == :parameter
        (:transformed_parameters,)
    else
        (:generated_quantities,)
    end
else
    error("distribution_blocks not defined for non-void StanExpr at statement position: $x")
end

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
fetch_data!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_data!(;info), x)
fetch_data!(x::Union{Function,String}; info) = nothing 
fetch_data!(x::StanExpr{<:Union{Number,String,Missing}}; info) = nothing 
fetch_data!(x::StanType; info) = fetch_data!(stan_size(x); info)
fetch_data!(x::StanExpr{Symbol}; info) = begin
    hasvalue(x) && push!(block(info, :data), x; info)
end
fetch_data!(x::StanExpr{<:Function}; info) = nothing
fetch_data!(x::StanExpr{<:DataType}; info) = nothing
fetch_data!(x::StanExpr{<:CanonicalExpr}; info) = fetch_data!((type(x), expr(x)); info)
fetch_data!(x::CanonicalExpr; info) = begin
    fetch_functions!(x; info=block(info, :functions).content)
    fetch_data!(x.args; info)
end
fetch_data!(x::CanonicalExprV{:kw}; info) = fetch_data!(x.args[2]; info)
fetch_data!(x; info) = error("fetch_data! not defined for value `$x` of type `$(typeof(x))`.")

Base.get!(b::DocumentExpr{<:Any,<:DeclarativeBlock}, k, x) = get!(content(b.args[2]), k, remake(b, b.args[1], x))
Base.push!(b::DocumentExpr{<:Any,<:ImperativeBlock}, x) = push!(content(b.args[2]), remake(b, b.args[1], x))

Base.push!(b::StanBlock, x; info) = error("Block $(typeof(b)) does not know how to handle $(x)!")
Base.push!(b::StanBlock, x::DocumentExpr; info) = begin
    push!(remake(b, remake(x, x.args[1], b)), x.args[2]; info)
end
Base.push!(b::DeclarativeBlock, x::SamplingExpr; info) = push!(b, x.args[1]; info)
Base.push!(b::DeclarativeBlock, x::StanExpr{Symbol}; info) = begin
    fetch_data!(type(x); info)
    get!(content(b), expr(x), x)
end
Base.push!(b::ImperativeBlock, x; info) = begin 
    fetch_data!(x; info)
    push!(content(b), x)
end
Base.push!(b::ImperativeBlock, x::DocumentExpr; info) = begin
    push!(remake(b, remake(x, x.args[1], b)), x.args[2]; info)
end
Base.push!(b::GeneratedQuantitiesBlock, x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    # if hasvalue(lhs)
    if qual(lhs) == :data
        likelihood_rhs = likelihood_expr(lhs, rhs)
        push!(b, CanonicalExpr(
            :(=),
            StanExpr(Symbol(expr(lhs), "_likelihood"), remake(type(likelihood_rhs); value=missing)),
            likelihood_rhs
        ); info)
        lhs = StanExpr(Symbol(expr(lhs), "_gen"), remake(type(lhs); value=missing))
    end
    # Build a type token carrying the wanted output shape (from lhs, which is
    # either explicitly declared via typed-LHS or inferred by autotype). This
    # token becomes the leading arg to the rng call, letting each `*_rng`
    # @deffun dispatch on the shape.
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct, StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    rng_rhs = rng_expr(token, rhs)
    lhs = StanExpr(expr(lhs), remake(type(rng_rhs); value=missing))
    push!(b, CanonicalExpr(:(=), lhs, rng_rhs); info)
end

function lpxf_expr end
function rng_expr end
function likelihood_expr end

const _LPXF_SUFFIXES = ("_lpdf", "_lpmf", "_lcdf", "_lccdf")
_lpxf_base(name::Symbol) = begin
    s = string(name)
    suffix_idx = findfirst(suf -> endswith(s, suf), _LPXF_SUFFIXES)
    isnothing(suffix_idx) && error(
        "@lpxf/@lhs: `$name` does not end in one of $(_LPXF_SUFFIXES). ",
        "Pass the `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf` function name itself."
    )
    Symbol(s[1:end-length(_LPXF_SUFFIXES[suffix_idx])])
end

lpxf_register(x::LineNumberNode; source=x) = x
lpxf_register(x::Expr; source=LineNumberNode(0, :none)) = if x.head === :block
    Expr(:block, [lpxf_register(arg; source) for arg in x.args]...)
else
    error("@lpxf expects a bare symbol or a `begin … end` block of bare symbols, got `$x`")
end
lpxf_register(x; source=LineNumberNode(0, :none)) = error(
    "@lpxf expects a bare symbol or a `begin … end` block of bare symbols, got `$x`"
)
lpxf_register(name::Symbol; source=LineNumberNode(0, :none)) = begin
    base = _lpxf_base(name)
    rng = Symbol(base, "_rng")
    lpxfs = Symbol(name, "s")
    M = @__MODULE__
    quote
        $source
        function $base end
        function $rng end
        function $lpxfs end
        $M.lpxf_expr(::typeof($base)) = $name
        $M.rng_expr(::typeof($base)) = $rng
        $M.likelihood_expr(::typeof($base)) = $lpxfs
    end
end
