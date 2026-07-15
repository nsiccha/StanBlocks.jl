
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
# The LHS of a compiler-injected slice/element fill (`out[a:b] = rhs`, hoisted
# from an inlined mutating helper or a plate via `_slic_inline_pending`) is a
# getindex expr, not a bare Symbol — but every `info` key is a Symbol. Resolve
# the BASE variable being (partially) filled, "coarse-grained": discard *which*
# elements are written and treat the whole base var as touched. A plain Symbol
# LHS resolves to itself (unchanged behaviour). Assignment-LHS canonicals are
# always getindex (a user-written non-Symbol LHS is rejected pre-`forward!`), so
# descending `args[1]` reaches the base Symbol.
_base_lhs_symbol(x::StanExpr) = _base_lhs_symbol(expr(x))
_base_lhs_symbol(x::Symbol) = x
_base_lhs_symbol(x::CanonicalExpr) = _base_lhs_symbol(x.args[1])
# The declared Symbol of a compiler-injected fresh-result declaration `out::T`
# (a `DeclExpr` = single-arg `CanonicalExprV{:(::)}` whose one arg is the typed
# symbol, itself a `StanExpr{Symbol}` post-`forward!`).
_decl_lhs_symbol(x::DeclExpr) = _base_lhs_symbol(x.args[1])
# True iff `x` was registered via an explicit `DeclExpr` (`forward!(::DeclExpr)`
# sets `fresh_decl=true`). Distinguishes a fresh-declared local/derived var — into
# which a compiler-injected slice-fill IS legal even at `:parameter` qual (it's a
# transformed parameter under construction) — from a SAMPLED Stan parameter, which
# is read-only and must reject the fill. `info(::StanType)` is the type-info
# accessor (kept out of the kwarg-`info` methods below to avoid the name clash).
_is_fresh_decl(x::StanExpr) = get(info(type(x)), :fresh_decl, false)
backward!(x::AssignmentExpr; info) = begin
    lhs = x.args[1]
    key = _base_lhs_symbol(lhs)
    slice = !(expr(lhs) isa Symbol)   # getindex LHS ⇒ compiler-injected partial fill
    if slice
        key in keys(info) || error(
            "Compiler-generated slice-fill of `", key, "[…]` but `", key, "` is not declared ",
            "in model scope — the inlining/plate emitter must register the base variable first."
        )
        (qual(info[key]) == :parameter && !_is_fresh_decl(info[key])) && error(
            "Cannot assign to a slice/element of parameter `", key, "` in the model block — Stan ",
            "parameters are read-only there (this typically comes from inlining a mutating helper ",
            "onto a parameter; fill a local/derived vector instead)."
        )
    end
    if lqual(info[key]) == :affects_likelihood
        lhs2, rhs = x.args
        # Symbol LHS: swap in the updated info entry. Slice LHS: keep the getindex
        # LHS verbatim so the emitter renders `out[a:b] = rhs`.
        remake(x, slice ? lhs2 : info[key], backward!(rhs; info))
    elseif qual(lhs) == :parameter
        lhs2, rhs = x.args
        remake(x, remake(lhs2, qual=:quantities), rhs)
    else
        x
    end
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
# A compiler-injected fresh-result declaration `out::T` must stay INERT here.
# The generic `StanExpr`/`StanExpr{Symbol}` methods below would descend into the
# decl's inner typed symbol and rebind `info[out]` to the decl's *stale* entry —
# clobbering the qual that `forward!` promoted across the fills back to the
# declaration-time `:undefined`, which then mis-routes the whole variable to
# generated quantities. Leave the decl unchanged and the promoted `info` entry
# intact; downstream uses (e.g. the injected `r = out`) still set its `lqual`.
backward!(x::StanExpr{<:DeclExpr}; info) = x
# A compiler-injected `for` loop: the generic method would descend into the head's
# raw-Symbol index assignment and into body statements that reference the loop
# index, but `forward!(::ForExpr)` POPS the index after tracing, so it's absent from
# `info` here. Re-scope the index, `backward!` only the body (the head is structural
# `i = 1:n`), then pop — mirroring `forward!(::ForExpr)`. The base vars filled in the
# body keep their `forward!`-promoted qual (this doesn't touch them).
backward!(x::StanExpr{<:ForExpr}; info) = begin
    fe = expr(x)
    head, body = fe.args
    idx = head.args[1]
    info[idx] = StanExpr(idx, StanType(types.int; qual=:data))   # :data — see forward!(::ForExpr)
    rv = StanExpr(remake(fe, head, backward!(body; info)), type(x))
    pop!(info, idx)
    rv
end
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
# Compiler-injected fresh-result declaration `out::T` and its slice/element fills
# `out[i] = rhs` reach `distribute!` wrapped as NON-void StanExprs (the decl carries
# its declared center type; a fill carries `anything`), so the generic method above
# would error. Route BOTH by the base/declared variable's FINALIZED qual in `info`
# — promoted across every fill in `forward!` and preserved through `backward!` — so
# the declaration and all fills land together in one block (coarse-grained). The
# wrapper's own qual is the stale declaration-time `:undefined`; ignore it.
_qual_blocks(q) = q == :data ? (:transformed_data,) :
    q == :parameter ? (:transformed_parameters,) : (:generated_quantities,)
distribution_blocks(x::StanExpr{<:DeclExpr}; info) = _qual_blocks(qual(info[_decl_lhs_symbol(expr(x))]))
distribution_blocks(x::StanExpr{<:AssignmentExpr}; info) = _qual_blocks(qual(info[_base_lhs_symbol(expr(x))]))
# A compiler-injected `for` loop whose body is fills (Feature-1 ragged-simplex:
# `for(g in 1:G) p_flat[lo:hi] = simplex_jacobian(...)`, G data-sized so it can't
# unroll). Route the WHOLE loop by the coarse (max) qual over the base vars its body
# fills — the same coarse-graining as a bare fill, one level into the body. The loop
# emits intact (`show(::ForExpr)`) into the chosen block.
_fill_base(x::StanExpr) = _fill_base(expr(x))
_fill_base(x::AssignmentExpr) = _base_lhs_symbol(x.args[1])
_fill_base(x) = nothing
_for_body_qual(fe::ForExpr, info) = begin
    q = :undefined
    for stmt in fe.args[2].args
        b = _fill_base(stmt)
        (b isa Symbol && b in keys(info)) && (q = _promote_qual(q, qual(info[b])))
    end
    q
end
distribution_blocks(x::StanExpr{<:ForExpr}; info) = _qual_blocks(_for_body_qual(expr(x), info))

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
fetch_data!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_data!(;info), x)
fetch_data!(x::Union{Function,String}; info) = nothing
# `distribute!` skips LineNumberNodes/Nothing at the block level, but they also live
# INSIDE a compiler-injected loop body (`@inline` line info), which
# `fetch_data!(::StanExpr{<:ForExpr})` recurses into — skip them here too.
fetch_data!(x::Union{LineNumberNode,Nothing}; info) = nothing
fetch_data!(x::StanExpr{<:Union{Number,String,Missing}}; info) = nothing 
fetch_data!(x::StanType; info) = fetch_data!(stan_size(x); info)
fetch_data!(x::StanExpr{Symbol}; info) = begin
    hasvalue(x) && push!(block(info, :data), x; info)
end
fetch_data!(x::StanExpr{<:Function}; info) = nothing
fetch_data!(x::StanExpr{<:DataType}; info) = nothing
fetch_data!(x::StanExpr{<:CanonicalExpr}; info) = fetch_data!((type(x), expr(x)); info)
# A compiler-injected `for` loop: walk the range bound and body for data deps, but
# SKIP the head's raw-Symbol loop index — it isn't data, and the generic
# `fetch_data!(::Symbol)` fallback errors on it. Body references to the index are
# `StanExpr{Symbol}` (valueless → no-op), so only the head's bare index needs skipping.
fetch_data!(x::StanExpr{<:ForExpr}; info) = begin
    fe = expr(x)
    fetch_data!(fe.args[1].args[2]; info)   # the range bound (e.g. `1:n`), not the index Symbol
    fetch_data!(fe.args[2]; info)           # the loop body
end
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
