
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
# A COMPOSITE type (`tup`/`ntup`, incl. every `usertype` built on one) keeps its
# ELEMENT types in `info.arg_types`, and each of those carries its OWN symbolic
# size. `stan_size` of a tup/ntup is EMPTY, so deanonymizing only the top-level
# size leaves those nested sizes anonymized: a `@deffun` returning a (named)
# tuple whose element sizes reference its own params (`mem::int[sum(g(x, a))]`)
# emits `_arg<tok>_<i>` verbatim into Stan — an undeclared identifier in a size
# position, which `fetch_data!` then compounds by declaring as a phantom `data`
# variable. Recurse so an element's size is deanonymized exactly like a scalar's.
deanon_arg_types(at, x::CanonicalExpr, tok) = at
deanon_arg_types(at::Union{Tuple,NamedTuple}, x::CanonicalExpr, tok) =
    map(t -> deanon_type(t, x, tok), at)
deanon_type(tt::StanType, x::CanonicalExpr, tok) = begin
    sz = stan_size(tt)
    nsz = map(s -> deanon_size(s, x, tok), sz)
    at = get(info(tt), :arg_types, nothing)
    nat = deanon_arg_types(at, x, tok)
    sz == nsz && nat === at && return tt
    StanType(center_type(tt), nsz; [
        k => (k === :arg_types ? nat : v) for (k, v) in pairs(info(tt)) if k != :size
    ]...)
end
_tracetype(x, _context) = tracetype(x)
_stan_expr(x::CanonicalExpr, context) = begin
    context = _context_or_new(context)
    tok = _next_anon_id(context)
    tt = deanon_type(_tracetype(anon_canonical(x, tok), context), x, tok)
    StanExpr(x, remake(tt; qual=maximum(qual, x.args; init=:data), cv=any(cv, x.args) || cv(tt)))
end
stan_expr(x::CanonicalExpr) = _stan_expr(x, TraceContext())
# Every compiler-internal conversion that runs while a trace is active must use
# the trace's context. Starting a standalone context here would restart the anon
# token counter at 1; a result type that already contains the outer trace's
# `_arg1_…` placeholder could then be deanonymized against the wrong expression.
_trace_stan_expr(x::CanonicalExpr, info) = _stan_expr(x, _trace_context(info))
_trace_stan_arg(x::CanonicalExpr, info) = _trace_stan_expr(x, info)
_trace_stan_arg(x, _info) = stan_expr(x)
_trace_stan_call(f, args...; info) = _trace_stan_expr(
    CanonicalExpr(f, map(a -> _trace_stan_arg(a, info), args)...), info
)
# A @slic sub-model or named sub-model function in call position. For an anonymous
# `SlicModel`, data flows via KEYWORDS — a positional call now errors (its call
# operator points at `Base.merge` for splice overrides / `@slic f(...)=...` for
# positional inputs). For a `SubmodelFn`, positional args ARE the inputs (bound by
# its generated call method; Julia's own dispatch/arity handle them). Either way the
# call yields a `SlicModel`, embedded via the existing `~`-rhs-is-`SlicModel` path.
_stan_expr(x::CanonicalExpr{<:Union{SlicModel,SubmodelFn}}, _context) =
    head(x)(x.args...; x.kwargs...)
stan_expr(x::CanonicalExpr{<:Union{SlicModel,SubmodelFn}}) = head(x)(x.args...; x.kwargs...)

backward!(x; info) = error("backward! not defined for value `$x` of type `$(typeof(x))` — no method matches a more specific signature.")
backward!(;info) = x->backward!(x; info)
backward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(backward!(;info), x)
backward!(x::Union{String,Number,LineNumberNode,Symbol,Nothing,Colon}; info) = x
backward!(x::CanonicalExpr; info) = remake(x, backward!(x.args; info)...)
backward!(x::BlockExpr; info) = remake(x, reverse(backward!.(reverse(x.args); info))...)
# The LHS of a compiler-injected slice/element fill (`out[a:b] = rhs`, hoisted
# from an inlined mutating helper or a plate via the trace's pending buffer) is a
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
backward!(x::AssignmentExpr; info) = begin
    lhs = x.args[1]
    key = _base_lhs_symbol(lhs)
    slice = !(expr(lhs) isa Symbol)   # getindex LHS ⇒ compiler-injected partial fill
    if slice
        key in keys(info) || error(
            "Compiler-generated slice-fill of `", key, "[…]` but `", key, "` is not declared ",
            "in model scope — the inlining/plate emitter must register the base variable first."
        )
        # CERTIFICATE: a compiler-injected slice-fill is legitimate only into a var
        # declared FRESH via `::` (`forward!(::DeclExpr)` stamps `fresh_decl`) — a
        # transformed-quantity result under construction, whose qual is determined
        # solely by its fills. Any other base is read-only for element assignment in a
        # model block: a SAMPLED parameter (Stan-read-only), a DATA var (Stan-read-only),
        # or an ASSIGNMENT-bound/derived var (its qual is already committed — SSA can't
        # un-commit it in one pass; the settled "pre-committed var" bar). Reject all.
        (_is_fresh_decl(info[key]) && _decl_role(info[key]) == :fill) || let q = qual(info[key])
            q == :parameter ? error(
                "Cannot assign to a slice/element of parameter `", key, "` in the model block — Stan ",
                "parameters are read-only there (this typically comes from inlining a mutating helper ",
                "onto a parameter; fill a freshly-declared local/derived vector instead)."
            ) : error(
                "Compiler-injected slice-fill onto `", key, "`, which is not a fresh declaration (it is ",
                q == :data ? "data" : "an assignment-bound/derived value",
                ", read-only for element assignment in a model block). Slice-fills are supported only into ",
                "a vector declared fresh via `::` inside the inlined body (`out::T`); the emitter must ",
                "target such a result."
            )
        end
    end
    if lqual(info[key]) == :affects_likelihood
        lhs2, rhs = x.args
        # Symbol LHS: swap in the updated info entry. Slice LHS: keep the getindex
        # LHS verbatim so the emitter renders `out[a:b] = rhs`.
        remake(x, slice ? lhs2 : info[key], backward!(rhs; info))
    elseif slice && qual(info[key]) != :data
        lhs2, rhs = x.args
        # A PRIOR-ONLY compiler-injected slice fill (a plate's return `z[i] = w[i]`
        # or cell-local `=`, an inlined helper's `out[i] = …`): nothing downstream
        # of `key` reaches a likelihood, or its `lqual` would read
        # `:affects_likelihood` above. The whole coarse-grained variable — its fresh
        # declaration, every fill, and the enclosing compiler-owned loop — is routed
        # by `info[key]`'s qualifier, NOT by this statement's remade LHS
        # (`distribution_blocks(::StanExpr{<:DeclExpr})`, `_loop_distribution_blocks`).
        # Flip THAT to `:quantities`, so the fill follows its consumers to generated
        # quantities. Its sources are deliberately left UNMARKED: the ordinary
        # prior-only lowering then moves them to generated quantities too — a sampled
        # source becomes an `_rng` draw, an assigned one a plain gq assignment — in
        # source order, ahead of this fill, exactly as a whole-variable assignment's
        # sources do. A source that ALSO reaches a likelihood is marked by that path
        # and stays a parameter, which generated quantities can read anyway.
        #
        # History: `aef6a42` recursed into the RHS here instead, pinning every source
        # as a parameter so a TRANSFORMED-PARAMETERS fill kept its sources in scope.
        # That was the conservative direction, but it made a likelihood-free program
        # sample its prior with NUTS: the plate return fill of a prior-regime PK/QT
        # model kept 21 parameters, a 92-line transformed parameters block and a full
        # adaptation run (snag prior-predictive-7e463983). Moving the fill WITH its
        # sources is what keeps `parameters {}` empty when there is no likelihood.
        # A data-only fill (`qual == :data`, transformed data) needs neither.
        info[key] = remake(info[key]; qual=:quantities)
        remake(x, remake(lhs2, qual=:quantities), rhs)
    elseif qual(lhs) == :parameter
        lhs2, rhs = x.args
        # An ordinary whole-variable prior-only assignment moves to generated
        # quantities together with its dependencies. Recursing into the RHS here
        # would wrongly pin prior-only sources as parameters — notably the
        # missing-data rewrite's `y = merge_missing(..., y_mis, ...)`, whose `y_mis`
        # should be a GQ draw unless completed `y` feeds a downstream likelihood.
        # (A slice fill whose base is data-qualified but whose LHS still carries the
        # declaration-time provisional `:parameter` qualifier lands here too; its
        # routing reads the base's `:data` qualifier, so this is a no-op for it.)
        remake(x, remake(lhs2, qual=:quantities), rhs)
    else
        x
    end
end
# `backward!(::CanonicalExpr)` rebuilds a call from its args and drops its kwargs.
# For a SAMPLING rhs those kwargs carry the author's `lower=`/`upper=` bounds, and
# a cv-flipped parameter that still reads `:affects_likelihood` (the reachability
# pass runs over the unconditioned trace) is re-drawn in generated quantities
# from exactly this rhs — with the bounds gone, `alpha ~ normal(mu, 1; lower=-2)`
# re-drew as a plain `normal_rng`. Put the sampling call's kwargs back so
# `redraw_rng_expr` can honour them; the model block never prints kwargs and the
# density/pointwise companions take positional args only, so nothing else moves.
_restore_sampling_kwargs(new::StanExpr, old::StanExpr) = begin
    (expr(new) isa CanonicalExpr && expr(old) isa CanonicalExpr) || return new
    isempty(expr(old).kwargs) && return new
    StanExpr(remake(expr(new), expr(new).args...; expr(old).kwargs...), type(new))
end
_restore_sampling_kwargs(new, old) = new
backward!(x::SamplingExpr{<:StanExpr{Symbol}}; info) = if qual(x.args[1]) == :data || lqual(info[expr(x.args[1])]) == :affects_likelihood
    lhs, rhs = x.args
    remake(x, info[expr(x.args[1])], _restore_sampling_kwargs(backward!(rhs; info), rhs))
else
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
end
backward!(x::SamplingExpr; info) = begin 
    lhs, rhs = x.args
    key = _base_lhs_symbol(lhs)
    if key in keys(info) && _is_fresh_decl(info[key]) && _decl_role(info[key]) == :sampled
        if lqual(info[key]) == :affects_likelihood && !cv(info[key])
            # A plate parameter used by a later likelihood remains a parameter;
            # propagate that reachability into its prior's RHS exactly like the
            # established bare-Symbol sampling path above.
            #
            # UNLESS it is cv-tainted. Likelihood reachability is computed over
            # the UNCONDITIONED trace, so a held-out likelihood still reads as
            # `:affects_likelihood` here — while cv has already dropped the term
            # that made it one. Without the `cv` test a plate cell parameter
            # sized from held-out data stays in `parameters`, sampled from its
            # prior with nothing informing it: a prior draw dressed up as a fit,
            # emitted silently and stanc-clean. `forward!` has already qualified
            # the declaration `:quantities` (see `_forward_indexed_sampling!` in
            # forward.jl); this is the matching half that turns the `~` itself
            # into the generated-quantities re-draw.
            remake(x, lhs, _restore_sampling_kwargs(backward!(rhs; info), rhs))
        else
            # Prior-only plate locals are generated quantities, matching the
            # existing prior-predictive treatment of unused Symbol samples. The
            # separate outer declaration reads this updated qualifier later.
            info[key] = remake(info[key]; qual=:quantities)
            remake(x, remake(lhs; qual=:quantities), rhs)
        end
    elseif key in keys(info) && get(type(info[key]).info, :ragged_density, false)
        # Informative ragged priors sample a compiler-generated slice of the
        # constrained memory.  Mark both the slice and RHS as likelihood-relevant;
        # the earlier transform fill will then propagate reachability from this
        # memory declaration into the unconstrained free coordinates.
        remake(x, backward!(lhs; info), backward!(rhs; info))
    else
        @assert qual(lhs) == :data
        remake(x, backward!(lhs; info), backward!(rhs; info))
    end
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
backward!(x::StanExpr{Symbol}; info) = begin
    key = expr(x)
    # A symbol occurrence captures the type metadata present when it was
    # forwarded. Compiler-owned ragged density certification is stamped only
    # after the injected lowering finishes, so a later likelihood may reach the
    # memory through an older occurrence. Preserve the current certified entry
    # instead of erasing that durable marker with the stale snapshot.
    #
    # The lookup MUST stay guarded. This method is fundamentally a WRITE — before
    # the ragged-density certificate existed it was exactly
    # `info[expr(x)] = remake(x; lqual=:affects_likelihood)`, which happily BOUND a
    # name `info` had never seen. Reading `info[key]` unconditionally silently
    # turned every such first binding into a `KeyError`; a submodel data input
    # reached through an indexed sampling LHS (`obs[i] ~ normal(...)`, where the
    # generic descent hits bare `obs`) is the live case. Absent name ⇒ bind `x`.
    certified = key in keys(info) && get(type(info[key]).info, :ragged_density, false)
    source = certified ? info[key] : x
    info[key] = remake(source; lqual=:affects_likelihood)
end
backward!(x::StanType; info) = remake(x; lqual=:affects_likelihood)

distribute!(x::BlockExpr; info) = distribute!.(x.args; info)
distribute!(x::Union{LineNumberNode,Nothing}; info) = nothing
distribute!(x::DocumentExpr{<:Any,<:BlockExpr}; info) = distribute!(x.args[2]; info)
# A documented statement whose inlined body is a multi-statement `:block` (a
# submodel call — its declarations/samplings/return land in DIFFERENT Stan
# blocks, so the block has no single distribution target). ONE `#`/`@doc`
# comment gives `document(block)` (method above); STACKED comments — or a
# comment plus a preceding statement in the same body — accrue MORE `:document`
# wrappers: `document(document(…(block)))`. Peel every layer and distribute the
# innermost block's statements individually, exactly as the single-layer method
# and a bare inlined block (`distribute!(::BlockExpr)`) do. Without this, the
# generic method below routes the nested document through `distribution_blocks`,
# whose `:document` method (`passes.jl` ~line 298) peels each layer and finally
# calls `distribution_blocks` on the inner `:block` — for which no method exists
# (snag `doc-submodel-cal`; 3 stacked `#` comments before a submodel call, one
# consumed into the model docstring, leaving `document(document(block))`). A
# `:document` wrapping a NORMAL statement (plain `=`/`~`) never reaches here —
# `_document_inner_block` returns `nothing`, so it keeps flowing through the
# generic `distribute!`/`push!` path, which re-homes and renders its comment.
_document_inner_block(x::DocumentExpr) = _document_inner_block(x.args[2])
_document_inner_block(x::BlockExpr) = x
_document_inner_block(x) = nothing
distribute!(x::DocumentExpr{<:Any,<:DocumentExpr}; info) = begin
    inner = _document_inner_block(x)
    inner === nothing ? _distribute_statement!(x; info) : distribute!(inner; info)
end
distribute!(x; info) = _distribute_statement!(x; info)
_distribute_statement!(x; info) = begin
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
    if !(expr(x.args[1]) isa Symbol)
        # Indexed observation inside a compiler-owned plate loop. The ordinary
        # gq expansion assumes a whole named LHS and cannot write one cell of a
        # DATA variable (Stan data is read-only), so this used to route to the
        # model block only — leaving a plate model with no predictive draw at all
        # (snag build-a-declarat-ab2d2471). It now gets the same gq treatment as
        # a whole-LHS observation whenever a generated-output shape is derivable
        # from the base declaration: the per-cell rng writes into a compiler-owned
        # `<base>_gen` twin of the base's declared type (`_indexed_obs_gen_base`).
        # Dense bases use the native declaration retarget below. A RaggedVector
        # base has no declaration of its own, but its exact compiler-owned
        # `mem[start(ends,g):end(ends,g)]` projection has a separate flat-twin
        # plan (`_indexed_ragged_obs_slice`). Other tuple/usertype carriers keep
        # the model-only routing.
        if _indexed_obs_twin_target(x.args[1]; info) === nothing
            # A cv-tainted held-out obs with no gq twin (a top-level ragged
            # observation whose per-group density loop the broadcast lowered — its
            # predictive/pointwise twins are emitted separately) contributes no
            # model likelihood: emitting it in the model block would reference the
            # density argument's GQ-only backing and stanc-reject as out of scope.
            # Drop it from emission; `backward!` has already run, so any population
            # parameter it touches stays sampled (snag ragged-obs-not-c-5b1180c7).
            cv(x.args[1]) ? () : (:model,)
        elseif cv(x.args[1])
            (:generated_quantities,)
        else
            (:model, :generated_quantities)
        end
    elseif cv(x.args[1])
        (:generated_quantities,)
    else
        (:model, :generated_quantities)
    end
elseif qual(x) == :parameter
    # A plate producer emits the outer declaration separately; an indexed
    # sampling statement contributes only the model-side prior/likelihood.
    expr(x.args[1]) isa Symbol ? (:parameters, :model) : (:model,)
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
# wrapper's own qual is only the provisional declaration-time qualifier; ignore it.
_qual_blocks(q) = q == :data ? (:transformed_data,) :
    q == :parameter ? (:transformed_parameters,) : (:generated_quantities,)
distribution_blocks(x::StanExpr{<:DeclExpr}; info) = begin
    base = info[_decl_lhs_symbol(expr(x))]
    role = _decl_role(base)
    if role in (:unfilled, :sampled)
        qual(base) == :parameter ? (:parameters,) : (:generated_quantities,)
    else
        role == :fill || error(
            "Fresh declaration `", expr(base), "` reached distribution with unknown role `", role, "`."
        )
        _qual_blocks(qual(base))
    end
end
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

# Route a compiler-owned loop FINE-GRAINED by body statement while preserving a
# coarse symbolic runtime loop in every destination block. A plate body commonly
# needs one model loop for indexed `~` statements and one transformed-parameters
# loop for its returned-cell fill. This is still one LOGICAL plate loop; Stan's
# block separation requires cloning its structural head around the relevant
# statement subsets.
_loop_distribution_blocks(::Union{LineNumberNode,Nothing}; info) = ()
_loop_distribution_blocks(x::StanExpr{<:ForExpr}; info) = begin
    blocks = Symbol[]
    for stmt in expr(x).args[2].args
        append!(blocks, _loop_distribution_blocks(stmt; info))
    end
    Tuple(unique(blocks))
end
_loop_distribution_blocks(x; info) = distribution_blocks(x; info)
_loop_distribution_stmt(x, ::Val; info) = x
_loop_distribution_stmts(x, v::Val; info) = Any[_loop_distribution_stmt(x, v; info)]
_loop_distribution_stmt(x::StanExpr{<:ForExpr}, ::Val{B}; info) where {B} = begin
    fe = expr(x)
    head, body = fe.args
    selected = Any[]
    for stmt in body.args
        B in _loop_distribution_blocks(stmt; info) || continue
        append!(selected, _loop_distribution_stmts(stmt, Val(B); info))
    end
    StanExpr(remake(fe, head, remake(body, selected...)), type(x))
end
_loop_distribution_stmts(x::SamplingExpr, ::Val{:generated_quantities}; info) =
    _indexed_gq_assignments(x; info)
distribute!(x::StanExpr{<:ForExpr}; info) = begin
    _push_expr!(info, x)
    try
        fe = expr(x)
        head, body = fe.args
        # Assignment synthesis below needs the declared twin entries (notably
        # their flat/vector types) while it rewrites the grouped statements.
        # Declare them before grouping, still immediately before the eventual
        # gq loop because this distribute call owns both operations.
        for target in unique(_obs_twin_targets(x; info))
            _push_indexed_obs_twin_decls!(block(info, :generated_quantities), target; info)
        end
        grouped = OrderedDict{Symbol,Vector{Any}}()
        for stmt in body.args
            for b in _loop_distribution_blocks(stmt; info)
                b in (:data, :parameters) && error(
                    "Compiler-owned loop body statement `", stmt, "` tried to emit into declarative block `", b,
                    "`. Plate parameters must have an outer declaration before the loop and use indexed sampling inside it."
                )
                append!(get!(() -> Any[], grouped, b),
                    _loop_distribution_stmts(stmt, Val(b); info))
            end
        end
        for (b, stmts) in grouped
            loop = StanExpr(remake(fe, head, remake(body, stmts...)), type(x))
            push!(block(info, b), loop; info)
        end
    finally
        _pop_expr!(info)
    end
    nothing
end

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
fetch_data!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_data!(;info), x)
# Bare (un-wrapped) literals reach here from a StanType's constraint `info`:
# `autokwargs` supplies a DISTRIBUTION-implied bound as a raw Julia value
# (`exponential`→`lower=0.0`), never as a StanExpr like an author-written one.
# Descending into constraints without this method turned every implied bound
# into `fetch_data! not defined for value 0.0`.
fetch_data!(x::Union{Function,String,Number,Missing}; info) = nothing
# `distribute!` skips LineNumberNodes/Nothing at the block level, but they also live
# INSIDE a compiler-injected loop body (`@inline` line info), which
# `fetch_data!(::StanExpr{<:ForExpr})` recurses into — skip them here too.
fetch_data!(x::Union{LineNumberNode,Nothing}; info) = nothing
fetch_data!(x::Union{Number,Missing}; info) = nothing
fetch_data!(x::StanExpr{<:Union{Number,String,Missing}}; info) = nothing 
# A StanType's `info` carries its constraint EXPRESSIONS
# (`lower`/`upper`/`offset`/`multiplier`) alongside its size, and data named only
# inside one of those is just as real a dependency as data named in a size.
# Descending into the size alone let a constraint-only reference leave its data
# out of the `data` block entirely — `theta::vector[J] ~ normal(mu, tau;
# multiplier=tau .^ (1 .- lam))` emitted no `lam` when `lam` appeared nowhere
# else, and stanc rejected the model with `Identifier "lam" not in scope`.
fetch_data!(x::StanType; info) = begin
    fetch_data!(stan_size(x); info)
    fetch_data!(values(constraints(x)); info)
end
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
Base.push!(b::DeclarativeBlock, x::StanExpr{<:DeclExpr}; info) =
    push!(b, info[_decl_lhs_symbol(expr(x))]; info)
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
    is_observation = qual(lhs) == :data
    if is_observation
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
    # An observation's `<obs>_gen` twin draws from the bare family (its likelihood
    # is the bare family); a PARAMETER re-draw must respect the sampled symbol's
    # own `lower=`/`upper=` bounds — see `redraw_rng_expr`.
    rng_rhs = is_observation ? rng_expr(token, rhs) :
        _check_redraw_resolves(lhs, rhs, redraw_rng_expr(token, rhs))
    lhs = StanExpr(expr(lhs), remake(type(rng_rhs); value=missing))
    push!(b, CanonicalExpr(:(=), lhs, rng_rhs); info)
end
# Prior-only indexed plate parameters are lowered cell-wise in generated
# quantities: their outer declaration is already present there, so each loop
# statement becomes `x[i] = dist_rng(...)` rather than trying to synthesize a
# second whole-variable name from the getindex expression.
Base.push!(b::GeneratedQuantitiesBlock,
        x::SamplingExpr{<:StanExpr{<:CanonicalExpr}}; info) = begin
    # Same split as the loop clone (`_loop_distribution_stmt`): a per-cell
    # PARAMETER redraw fills its own outer declaration, a per-cell DATA
    # observation fills the `<base>_gen` twin, which must be declared first.
    target = _indexed_obs_twin_target(x.args[1]; info)
    target === nothing || _push_indexed_obs_twin_decls!(b, target; info)
    for stmt in _indexed_gq_assignments(x; info)
        push!(b, stmt; info)
    end
end
_indexed_rng_assignment(x::SamplingExpr) = begin
    lhs, rhs = x.args
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct,
        StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    # A per-cell PARAMETER re-draw: honour the cell prior's own bounds.
    CanonicalExpr(:(=), lhs, _check_redraw_resolves(lhs, rhs, redraw_rng_expr(token, rhs)))
end
# A sampled symbol that is RE-DRAWN in generated quantities — no likelihood
# reaches it (prior-only / prior-predictive lowering), or it is cv-held-out — is
# the one place a custom family's `_rng` companion becomes mandatory for that
# shape. Without it the failure used to surface at EMISSION as an internal
# `AssertionError: tracetype not defined for tau::anything = foo_rng(::array[] tokenof, …)`,
# naming neither why the symbol is drawn nor what to add (the BRM `brm_ranef_sd`
# shape, snag prior-predictive-7e463983). Catch it where the family is known.
_check_redraw_resolves(lhs, rhs, draw) = begin
    center_type(draw) === types.anything || return draw
    name = _base_lhs_symbol(lhs)
    family = _ragged_base_family(expr(rhs))
    rng = try string(nameof(rng_expr(head(expr(rhs))))) catch; string(nameof(family), "_rng") end
    ct = center_type(lhs)
    nd = stan_ndim(lhs)
    shape = nd == 0 ? string(ct) : string(ct, "[", join(("n" for _ in 1:nd), ", "), "]")
    signature = nd == 0 ? string(rng, "(<the family's args…>)::", shape) :
        string(rng, "(", shape, ", <the family's args…>)::", shape)
    error(
        "`", name, " ~ ", nameof(family), "(…)` is re-drawn in generated quantities — ",
        "no likelihood reaches `", name, "` (a prior-only / prior-predictive program), or ",
        "it is cv-held-out — but its family has no predictive companion for a `", shape,
        "`: `", rng, "` resolved to an untyped (`anything`) result. Add\n    @deffun ",
        signature, " = …\nalongside the density (the sized-token protocol every ",
        "vector-shaped predictive draw uses; see `normal_rng(vector[n], …)` in `builtin.jl`).",
    )
end

# --- Per-cell DATA observations inside a plate: the `<base>_gen` twin ---------
# snag build-a-declarat-ab2d2471. A per-cell observation `y[i] ~ dist(…)` inside
# a compiler-owned plate loop cannot write its predictive draw back into `y`
# (Stan data is read-only), which is why it used to route to `model` only — so a
# plate model got NO generated observation at all, while the same statement
# written against a whole `y` did. The gq clone of the loop instead retargets the
# draw at a compiler-owned twin `<base>_gen`, declared once in `generated
# quantities` with the base variable's OWN declared type — that is the
# "generated-output shape" the plate could not previously derive. The per-cell rng
# call itself is unchanged: `_indexed_rng_assignment`'s token already carries the
# CELL shape, so each `*_rng` overload dispatches exactly as it does for a
# prior-only plate parameter.
#
# A natively declarable base qualifies for the ordinary whole-container retarget.
# RaggedVector is handled separately below: it has no declaration form, but its
# compiler-owned plate accessor exposes enough structure to declare flat storage.
# RaggedMatrix and every other usertype/tuple carrier remain outside this path.
_gen_twin_name(k::Symbol) = Symbol(k, "_gen")
_is_gen_declarable(T) = T isa Type &&
    (T <: types.real || T <: types.any_vector || T <: types.matrix)
# STRICT indexed-LHS matcher. A SAMPLING lhs may be an arbitrary data-qualified
# expression — `(sum(a) + n) ~ normal(...)` is legal — so `_base_lhs_symbol`,
# which blindly descends `args[1]`, is the wrong tool here: it is only correct
# for a compiler-injected FILL, whose lhs is always a getindex. Descend ONLY
# through a genuine `base[i…]` getindex chain, bottoming out in a bare Symbol.
_getindex_chain_base(x::StanExpr) = _getindex_chain_base(expr(x))
_getindex_chain_base(x::Symbol) = x
_getindex_chain_base(x::CanonicalExpr) =
    head(x) === getindex ? _getindex_chain_base(x.args[1]) : nothing
_getindex_chain_base(::Any) = nothing
_indexed_obs_gen_base(lhs; info) = begin
    expr(lhs) isa Symbol && return nothing
    k = _getindex_chain_base(lhs)
    (k isa Symbol && k in keys(info)) || return nothing
    base = info[k]
    qual(base) == :data || return nothing
    _is_gen_declarable(center_type(base)) ? k : nothing
end

# STRICT RaggedVector plate-slice matcher. `_plate_input_accessor` lowers one
# logical group `dv[g]` to exactly
#
#   dv.mem[ragged_start(dv.ends, g):ragged_end(dv.ends, g)]
#
# Match that entire shape — including the same named base and group index at
# both bounds — rather than widening `_getindex_chain_base`. The dense retarget
# must remain getindex-only, and an arbitrary tuple field / user-written slice
# is not enough evidence that flat generated storage is semantically valid.
_literal_int(x::StanExpr) = _literal_int(expr(x))
_literal_int(x::Integer) = Int(x)
_literal_int(::Any) = nothing
_ragged_field_base(x, field::Int) = begin
    x isa StanExpr || return nothing
    e = expr(x)
    e isa CanonicalExpr && head(e) === Base.getfield && length(e.args) == 2 || return nothing
    _literal_int(e.args[2]) == field || return nothing
    base = e.args[1]
    base isa StanExpr && expr(base) isa Symbol ? expr(base) : nothing
end
_ragged_bound_group(x, bound, k::Symbol) = begin
    x isa StanExpr || return nothing
    e = expr(x)
    e isa CanonicalExpr && head(e) === bound && length(e.args) == 2 || return nothing
    _ragged_field_base(e.args[1], 2) === k || return nothing
    e.args[2]
end
_indexed_ragged_obs_slice(lhs; info) = begin
    lhs isa StanExpr || return nothing
    e = expr(lhs)
    e isa CanonicalExpr && head(e) === getindex && length(e.args) == 2 || return nothing
    k = _ragged_field_base(e.args[1], 1)
    (k isa Symbol && k in keys(info)) || return nothing
    base = info[k]
    T = center_type(base)
    qual(base) == :data && T isa Type && T <: RaggedVector || return nothing
    slice = e.args[2]
    se = slice isa StanExpr ? expr(slice) : slice
    se isa CanonicalExpr && head(se) isa Colon && length(se.args) == 2 || return nothing
    lo_group = _ragged_bound_group(se.args[1], builtin.ragged_start, k)
    hi_group = _ragged_bound_group(se.args[2], builtin.ragged_end, k)
    lo_group === nothing || hi_group === nothing ||
        isequal(expr(lo_group), expr(hi_group)) || return nothing
    (; base=k, mem=e.args[1], group=lo_group, slice)
end
_indexed_obs_twin_target(lhs; info) = begin
    k = _indexed_obs_gen_base(lhs; info)
    k === nothing || return (; kind=:dense, base=k)
    spec = _indexed_ragged_obs_slice(lhs; info)
    spec === nothing ? nothing : (; kind=:ragged, base=spec.base)
end
# Rebuild an indexed LHS against a different base symbol, preserving every index
# expression and the cell type (`y[i, j]` ⇒ `y_gen[i, j]`).
# The twin is a gq OUTPUT, so drop the base's data value (else `fetch_data!`
# would re-declare `y_gen` in the data block) and its `:data` qualifier.
_rename_lhs_base(x::StanExpr{Symbol}, newname::Symbol) =
    StanExpr(newname, remake(type(x); value=missing, qual=:quantities))
_rename_lhs_base(x::StanExpr{<:CanonicalExpr}, newname::Symbol) =
    StanExpr(_rename_lhs_base(expr(x), newname), type(x))
# Asserts the same shape `_getindex_chain_base` accepts, so a future widening of
# the matcher cannot silently reintroduce the rewrite of a compound sampling lhs.
_rename_lhs_base(x::CanonicalExpr, newname::Symbol) = begin
    head(x) === getindex || error(
        "internal: `_gen` twin retarget reached a non-getindex lhs node (head ",
        head(x), ") — only a genuine `base[i…]` chain may be renamed.")
    remake(x, _rename_lhs_base(x.args[1], newname), x.args[2:end]...)
end
_indexed_obs_gen_assignment(x::SamplingExpr, k::Symbol) = begin
    lhs, rhs = x.args
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct,
        StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    CanonicalExpr(:(=), _rename_lhs_base(lhs, _gen_twin_name(k)), rng_expr(token, rhs))
end
# One RaggedVector group produces TWO generated-quantities assignments. The draw
# uses the existing sized-token RNG protocol; the likelihood uses the aggregate
# family density, never the elementwise `_lpdfs`, so joint families retain their
# meaning. Dense per-cell observations continue to produce only the draw.
_indexed_ragged_obs_assignments(x::SamplingExpr, spec; info) = begin
    lhs, rhs = x.args
    rhs_expr = expr(rhs)
    rhs_expr isa CanonicalExpr || error(
        "internal: ragged observation twin reached a non-call distribution rhs `", rhs_expr, "`.")
    rhs_expr = _assert_ragged_continuous_family(rhs_expr)
    # Preserve the ordinary sampling diagnostic as the first gate. Because the
    # gq declarations/assignments are built before the model-loop clone is
    # pushed, attempting the sized RNG first would mask an unresolved density
    # signature with the less relevant "missing sized RNG" error.
    density = lpxf_expr(lhs, rhs_expr)
    _check_lpxf_resolves(density)
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct,
        StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    draw = _ragged_group_rng(token, rhs_expr, lhs_ct)
    gen = info[_gen_twin_name(spec.base)]
    gen_lhs = StanExpr(
        CanonicalExpr(getindex, gen, spec.slice),
        remake(type(lhs); value=missing, qual=:quantities),
    )
    lik = info[Symbol(spec.base, "_likelihood")]
    lik_lhs = StanExpr(
        CanonicalExpr(getindex, lik, spec.group),
        remake(type(density); value=missing, qual=:quantities),
    )
    Any[
        CanonicalExpr(:(=), gen_lhs, draw),
        CanonicalExpr(:(=), lik_lhs, density),
    ]
end
_indexed_gq_assignments(x::SamplingExpr; info) = begin
    expr(x.args[1]) isa Symbol && error(
        "Compiler-owned loop contains a non-indexed generated-quantities sample. Plate locals must use an outer declaration and indexed sampling.")
    k = _indexed_obs_gen_base(x.args[1]; info)
    k === nothing || return Any[_indexed_obs_gen_assignment(x, k)]
    spec = _indexed_ragged_obs_slice(x.args[1]; info)
    spec === nothing ? Any[_indexed_rng_assignment(x)] :
        _indexed_ragged_obs_assignments(x, spec; info)
end

# Walk a (possibly nested) compiler-owned loop for the bases needing twins.
_obs_twin_targets(::Union{LineNumberNode,Nothing}; info) = NamedTuple[]
_obs_twin_targets(x; info) = NamedTuple[]
_obs_twin_targets(x::StanExpr{<:ForExpr}; info) = begin
    rv = NamedTuple[]
    for stmt in expr(x).args[2].args
        append!(rv, _obs_twin_targets(stmt; info))
    end
    rv
end
_obs_twin_targets(x::SamplingExpr; info) = begin
    qual(x) == :data || return NamedTuple[]
    target = _indexed_obs_twin_target(x.args[1]; info)
    target === nothing ? NamedTuple[] : NamedTuple[target]
end
# Declare the twin once per model (a base observed by two plates must not emit
# two declarations); registering it in `info` doubles as the dedup key.
_push_obs_gen_decl!(b, k::Symbol; info) = begin
    gen = _gen_twin_name(k)
    gen in keys(info) && return
    t = remake(type(info[k]); value=missing, qual=:quantities)
    decl = StanExpr(gen, t)
    info[gen] = decl
    push!(b, StanExpr(CanonicalExpr(:(::), decl), t); info)
end
_push_ragged_obs_decls!(b, k::Symbol; info) = begin
    gen = _gen_twin_name(k)
    lik = Symbol(k, "_likelihood")
    names = (gen, lik)
    existing = filter(name -> name in keys(info), names)
    if !isempty(existing)
        certified = length(existing) == 2 && all(name ->
            get(type(info[name]).info, :ragged_obs_source, nothing) === k, names)
        certified && return
        error(
            "Ragged plate observation `", k, "[…] ~ …` needs compiler-owned names `",
            gen, "` and `", lik, "`, but ", join(string.(existing), ", "),
            " is already bound in this model. Rename that variable.")
    end
    base = info[k]
    arg_types = type(base).info.arg_types
    mem = StanExpr(CanonicalExpr(Base.getfield, base,
        StanExpr(1, StanType(types.int; value=1, qual=:data))), arg_types.mem)
    ends = StanExpr(CanonicalExpr(Base.getfield, base,
        StanExpr(2, StanType(types.int; value=2, qual=:data))), arg_types.ends)
    gen_size = _trace_stan_call(builtin.num_elements, mem; info)
    lik_size = _trace_stan_call(builtin.num_elements, ends; info)
    gen_type = StanType(types.vector, (gen_size,);
        value=missing, qual=:quantities, ragged_obs_source=k)
    lik_type = StanType(types.vector, (lik_size,);
        value=missing, qual=:quantities, ragged_obs_source=k)
    for decl in (StanExpr(gen, gen_type), StanExpr(lik, lik_type))
        info[expr(decl)] = decl
        push!(b, StanExpr(CanonicalExpr(:(::), decl), type(decl)); info)
    end
end
_push_indexed_obs_twin_decls!(b, target; info) =
    target.kind === :dense ? _push_obs_gen_decl!(b, target.base; info) :
    target.kind === :ragged ? _push_ragged_obs_decls!(b, target.base; info) :
    error("internal: unknown indexed observation twin kind `", target.kind, "`.")

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
