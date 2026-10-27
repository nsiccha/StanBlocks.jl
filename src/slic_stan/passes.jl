# A caller's shape is already in caller scope. Keep it opaque while inferring
# a callee's result: a callee argument/local can have the same spelling as a
# symbol in that shape (e.g. `x` in a ragged slice's length).
struct CallSizeRef
    token::Int
    argument::Int
    fields::Tuple
    dimension::Int
end

deanon_size(s, x, tok) = s
deanon_size(s::StanExpr, x::CanonicalExpr, tok) = _deanon_size_expr(expr(s), s, x, tok)
_deanon_size_expr(e::CallSizeRef, s, x, tok) = begin
    e.token == tok || return s
    T = type(x.args[e.argument])
    for field in e.fields
        T = info(T).arg_types[field]
    end
    stan_size(T)[e.dimension]
end
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
# The size expression's own type is deanonymized too: a result size that reads a
# field of a tuple argument (`matrix[rv.rows[i], rv.cols[i]]`) holds the
# subexpression `rv.rows`, typed `int[<the argument's size>]`, which is a
# `CallSizeRef` until it is rewritten against the call's real argument.
_deanon_size_expr(e::CanonicalExpr, s, x, tok) = begin
    new_args = map(a -> deanon_size(a, x, tok), e.args)
    tt = deanon_type(type(s), x, tok)
    new_args == e.args && tt === type(s) && return s
    StanExpr(remake(e, new_args...), tt)
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
#
# Like the `anon_*` walkers (functions.jl), these and `_stan_expr` run once per
# traced call and do not specialize on the call's concrete expression type.
Base.@nospecializeinfer deanon_arg_types(at, @nospecialize(x::CanonicalExpr), tok) = at
Base.@nospecializeinfer deanon_arg_types(at::Union{Tuple,NamedTuple}, @nospecialize(x::CanonicalExpr), tok) =
    _anymap(t -> deanon_type(t, x, tok), at)
Base.@nospecializeinfer deanon_type(tt::StanType, @nospecialize(x::CanonicalExpr), tok) = begin
    sz = stan_size(tt)
    nsz = map(s -> deanon_size(s, x, tok), sz)
    at = get(info(tt), :arg_types, nothing)
    nat = deanon_arg_types(at, x, tok)
    sz == nsz && nat === at && return tt
    StanType(center_type(tt), nsz; [
        k => (k === :arg_types ? nat : v) for (k, v) in pairs(info(tt)) if k != :size
    ]...)
end
Base.@nospecializeinfer _tracetype(@nospecialize(x), _context) = tracetype(x)
# Stan allows a random-number generator only in transformed data, generated
# quantities, and functions whose names end in `_rng` — the suffix is Stan's own
# marker for an RNG-bearing callable, builtin or user-defined. Every draw is a
# generated quantity: qualify it `:quantities` (the top of
# `data < parameter < quantities`), and every expression and compiler-owned fill
# that reads it takes that qualifier through the ordinary `maximum(qual, …)` /
# `_promote_qual` propagation, so the whole chain routes to generated quantities,
# never to transformed parameters (snag `rng-draw-read-by-9ad81a23`). That holds
# for a draw from data alone too: in transformed data it would be one fixed draw
# per run that no output carries, whereas the `~` spelling of the same draw (a
# prior no likelihood reaches) is already re-drawn in generated quantities.
Base.@nospecializeinfer _rng_qual(@nospecialize(x::CanonicalExpr), q) = q in (:data, :parameter) && _is_rng_callee(head(x)) ? :quantities : q
# Whether a value — or a size it is declared with — reads a random draw. The
# fresh-parameter sites (forward.jl) treat this like `cv` taint, which likewise
# checks the declared size: such a parameter is re-drawn in generated quantities,
# because a model-block density or a `parameters` size cannot read a draw.
_reads_draw(_) = false
_reads_draw(x::StanExpr) = qual(x) == :quantities || _reads_draw(type(x))
_reads_draw(x::StanType) = qual(x) == :quantities || any(_reads_draw, stan_size(x))
_is_rng_callee(f) = false
_is_rng_callee(f::Function) = endswith(String(nameof(f)), "_rng")
# Raw-AST callees (the plate pass inspects source before tracing).
_is_rng_callee(f::Symbol) = endswith(String(f), "_rng")
_is_rng_callee(f::GlobalRef) = _is_rng_callee(f.name)
_is_rng_callee(f::Expr) = f.head === :. && _is_rng_callee(f.args[end])
_is_rng_callee(f::QuoteNode) = _is_rng_callee(f.value)
Base.@nospecializeinfer _stan_expr(@nospecialize(x::CanonicalExpr), context) = begin
    _reject_untyped_args(x)
    _reject_call_kwargs(x)
    context = _context_or_new(context)
    tok = _next_anon_id(context)
    tt = deanon_type(_tracetype(anon_canonical(x, tok), context), x, tok)
    args = Any[x.args...]
    q = _rng_qual(x, maximum(qual, args; init=:data))
    StanExpr(x, remake(tt; qual=q, cv=any(cv, args) || cv(tt)))
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
# The same descent for a SAMPLING lhs, made total. A sampling lhs may be an
# arbitrary data-qualified expression (`(2.0 * sum(v)) ~ normal(…)` is legal), so
# the descent can bottom out in a literal rather than a Symbol: `nothing` then
# means "a computed observation with no base variable". Every Symbol-bottomed lhs
# keys exactly as `_base_lhs_symbol` does.
_sampling_lhs_key(x::StanExpr) = _sampling_lhs_key(expr(x))
_sampling_lhs_key(x::Symbol) = x
_sampling_lhs_key(x::CanonicalExpr) = isempty(x.args) ? nothing : _sampling_lhs_key(x.args[1])
_sampling_lhs_key(::Any) = nothing
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
        # sample its prior with NUTS: a prior-only plate model's return fill kept
        # nearly every parameter sampled behind a transformed parameters block and a
        # full adaptation run (snag prior-predictive-7e463983). Moving the fill WITH its
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
# A prior-only sampled symbol is re-drawn in generated quantities from its
# family's `_rng` — which must produce a value of the symbol's declared CENTER
# type. For `simplex ~ dirichlet` / `cholesky_factor_corr ~ lkj_corr_cholesky`
# the family rng does. No family rng yields a sorted vector, but an `ordered` /
# `positive_ordered` prior whose density is exchangeable has an exact re-draw:
# the sorted iid draw (`_ordered_redraw`). Any other ordered prior stays a
# SAMPLED parameter — valid, descriptor-visible (`dimension > 0`), never a
# silently unsorted draw. Its rejection re-draw is exact only conditional on
# the prior's arguments, and the fitted joint drops the normaliser
# P(sorted | θ) that a whole-program prior draw would need. Ragged constrained
# parameters take their own sampled path (`ragged_density`).
_ordered_center(x) = center_type(x) in (types.ordered, types.positive_ordered)
_gq_redrawable(x, rhs) = !_ordered_center(x) || _exchangeable_prior(rhs)
# `c::ordered[k] ~ f(θ)` has the density ∏ f(c_i | θ) on the ordered region.
# When that product is symmetric in the elements — a native elementwise family
# whose every argument is a scalar — each of the k! orderings of k iid draws
# has the same mass, so the normaliser is 1/k! whatever θ is, and `c` is
# distributed exactly as the sorted iid draw (snag ordered-prior-re-c3734f49).
# A vector argument (`normal(mu_vec, s)`), a joint family (`multi_normal`), a
# distribution HOF and a custom family are not certified: the normaliser can
# depend on θ, or nothing here knows how the family treats the elements.
_exchangeable_prior(rhs::StanExpr) = _exchangeable_prior(expr(rhs))
_exchangeable_prior(rhs) = false
_exchangeable_prior(rhs::CanonicalExpr) = begin
    family = head(rhs)
    isnothing(_distribution_hof(family)) || return false
    family === builtin.std_normal || _elementwise_pointwise(likelihood_expr(family)) || return false
    bounds = filter(in(BOUND_KEYS), keys(rhs.kwargs))
    all(a -> stan_ndim(a) == 0, rhs.args) && all(k -> stan_ndim(rhs.kwargs[k]) == 0, bounds)
end
# The bound a declaration implies without a kwarg: `positive_ordered` IS the
# `lower=0` support, so a family that does not imply it is truncated there.
_center_support(decl) =
    center_type(decl) === types.positive_ordered ? (; lower=stan_expr(0.0)) : (;)
# Only a native family's support is known here. A custom family is its author's
# contract (stanblocks-use §8): an `@lhs @lpxf f_lpdf(x::positive_ordered[n], …)`
# declares the `positive_ordered` type itself, so the implied bound restates the
# family's own support, and truncating there would demand an `f_lccdf` that such
# a family never needs (docs `examples/isba-2024.md`). Its re-draw is exempt the
# same way (`_ordered_redraw`).
_builtin_family(family) = family isa Function && parentmodule(family) === builtin
_implied_support(decl, rhs::StanExpr) =
    expr(rhs) isa CanonicalExpr && _builtin_family(head(expr(rhs))) ? _center_support(decl) : (;)
_implied_support(decl, rhs) = (;)
# A bounded `~` samples from the `truncated` call (`_normalised_truncation`): an
# observation always, a parameter prior that stays in the model block whenever
# its bound cuts the family at a parameter-dependent point. Re-draws keep the bare
# call: `_gq_redraw` truncates them itself.
_truncated_sampling_rhs(decl, rhs) =
    qual(decl) == :data ? _normalised_truncation(rhs; observed=true) :
    qual(decl) == :parameter && !cv(decl) ?
        _normalised_truncation(rhs; support=_implied_support(decl, rhs)) : rhs
# A values-free `maybecv(:y)` observation (tracing.jl `_cv_marks`) is held out
# given a fit like a bound one: reachability treats it as an observation, so
# everything its distribution reads stays a parameter, while its own cv-tainted
# declaration is re-drawn in generated quantities. The storage of a ragged or
# plate-cell one names it as `unbound_observation`.
_cv_observation_storage(T::StanExpr; info) = begin
    name = get(type(T).info, :unbound_observation, nothing)
    name isa Symbol && _is_cv_observation(name; info)
end
backward!(x::SamplingExpr{<:StanExpr{Symbol}}; info) = if qual(x.args[1]) == :data ||
        lqual(info[expr(x.args[1])]) == :affects_likelihood || !_gq_redrawable(x.args...) ||
        _is_cv_observation(expr(x.args[1]); info)
    lhs, rhs = x.args
    decl = info[expr(x.args[1])]
    remake(x, decl, _truncated_sampling_rhs(decl, _restore_sampling_kwargs(backward!(rhs; info), rhs)))
else
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
end
backward!(x::SamplingExpr; info) = begin 
    lhs, rhs = x.args
    key = _sampling_lhs_key(lhs)
    known = key !== nothing && key in keys(info)
    # Slice statements after a `flat(…)` declaration (forward.jl,
    # `_record_slice_support!`) route exactly like a fresh declaration's.
    sliced = known && (
        (_is_fresh_decl(info[key]) && _decl_role(info[key]) == :sampled) ||
        _is_flat_decl(info[key]))
    if sliced
        if (lqual(info[key]) == :affects_likelihood || !_gq_redrawable(info[key], rhs)) && !cv(info[key])
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
            remake(x, lhs, _truncated_sampling_rhs(info[key], _restore_sampling_kwargs(backward!(rhs; info), rhs)))
        else
            # Prior-only plate locals are generated quantities, matching the
            # existing prior-predictive treatment of unused Symbol samples. The
            # separate outer declaration reads this updated qualifier later.
            # A values-free `maybecv` observation's storage is re-drawn the same
            # way, but what its draw reads stays likelihood-reachable.
            _cv_observation_storage(info[key]; info) &&
                (rhs = _restore_sampling_kwargs(backward!(rhs; info), rhs))
            info[key] = remake(info[key]; qual=:quantities)
            remake(x, remake(lhs; qual=:quantities), rhs)
        end
    elseif known && get(type(info[key]).info, :ragged_density, false)
        # Informative ragged priors sample a compiler-generated slice of the
        # constrained memory.  Mark both the slice and RHS as likelihood-relevant;
        # the earlier transform fill will then propagate reachability from this
        # memory declaration into the unconstrained free coordinates.
        remake(x, backward!(lhs; info), backward!(rhs; info))
    else
        @assert qual(lhs) == :data
        traced = backward!(rhs; info)
        restored = _restore_sampling_kwargs(traced, rhs)
        truncated = _truncated_sampling_rhs(lhs, restored)
        remake(x, backward!(lhs; info), truncated === restored ? traced : truncated)
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
    already_reached = key in keys(info) && lqual(info[key]) == :affects_likelihood
    # A declaration whose slice statements set its support (forward.jl,
    # `_finalize_slice_supports!`) gets its bounds after the forward pass, so an
    # occurrence forwarded earlier carries none: keep the current entry, as for
    # the ragged-density certificate. Data an observed `@scan` reads back is
    # marked after its loop is traced, likewise (`scan_observed`).
    certified = key in keys(info) && (get(type(info[key]).info, :ragged_density, false) ||
        !isempty(_slice_records(info[key])) || !isempty(_slice_observations(info[key])) ||
        _scan_observed(info[key]) !== nothing)
    source = certified ? info[key] : x
    reached = remake(source; lqual=:affects_likelihood)
    # cv taint only ever grows. An observation that reads its own base (an
    # earlier `mean(y)`, a `sum(y)` in its density) carries an occurrence
    # forwarded before the observation tainted that base; keep the taint rather
    # than clearing it with that snapshot.
    key in keys(info) && cv(info[key]) && !cv(reached) && (reached = remake(reached; cv=true))
    info[key] = reached
    # A parameter's declaration constraints participate in the unconstrained
    # density through its transform/Jacobian. If a likelihood-reachable value
    # has a bound or affine constraint that names another sampled variable,
    # that dependency is therefore part of the same inferred density closure.
    # Mark first, then descend only on the first visit so repeated references
    # and even an invalid cyclic constraint graph cannot recurse forever.
    already_reached || backward!(values(constraints(type(reached))); info)
    reached
end
backward!(x::StanType; info) = remake(x; lqual=:affects_likelihood)

# Stan forbids integers anywhere in a top-level transformed-parameter type,
# including nested tuples, usertypes, and arrays. Keep such derived values as
# model locals and repeat their deterministic construction in generated
# quantities, where their original types and output names remain legal. The
# dependent transforms must follow them, rather than reading a model local
# from the earlier transformed-parameters block. Plan the complete dependency
# closure before distribution: a compiler-owned result may have several fills.
_contains_integer(t::StanType) = center_type(t) <: types.int ||
    any(_contains_integer, values(get(info(t), :arg_types, ())))
Base.@nospecializeinfer _uses_model_local(@nospecialize(x), names) = false
_uses_model_local(x::StanExpr{Symbol}, names) = expr(x) in names ||
    _uses_model_local(type(x), names)
_uses_model_local(x::StanExpr, names) = _uses_model_local(expr(x), names) ||
    _uses_model_local(type(x), names)
_uses_model_local(x::StanType, names) = _uses_model_local(stan_size(x), names) ||
    _uses_model_local(values(constraints(x)), names) ||
    _uses_model_local(get(info(x), :arg_types, ()), names)
Base.@nospecializeinfer _uses_model_local(@nospecialize(x::CanonicalExpr), names) = _uses_model_local(x.args, names) ||
    _uses_model_local(values(x.kwargs), names)
Base.@nospecializeinfer _uses_model_local(@nospecialize(xs::Union{Tuple,NamedTuple,AbstractVector}), names) = begin
    for x in xs
        _uses_model_local(x, names) && return true
    end
    false
end

_transform_bindings!(bindings, x; info) = nothing
_transform_bindings!(bindings, x::StanExpr; info) =
    _transform_bindings!(bindings, expr(x); info)
_transform_bindings!(bindings, x::CanonicalExpr; info) =
    foreach(a -> _transform_bindings!(bindings, a; info), x.args)
# A for-loop header also uses `=` but keeps a raw Symbol LHS. It binds an
# iteration index, rather than declaring a model-scope transformed value.
_transform_bindings!(bindings, x::AssignmentExpr; info) = nothing
_transform_bindings!(bindings, x::AssignmentExpr{<:StanExpr}; info) = begin
    lhs, rhs = x.args
    key = _base_lhs_symbol(lhs)
    base = get(vars(info), key, lhs)
    q = expr(lhs) isa Symbol ? qual(lhs) : qual(base)
    q == :parameter && push!(bindings, (key, type(base), rhs))
end
_transform_bindings!(bindings, x::DeclExpr; info) = begin
    key = _decl_lhs_symbol(x)
    base = info[key]
    qual(base) == :parameter && _decl_role(base) == :fill &&
        push!(bindings, (key, type(base), nothing))
end
_model_local_transforms(x; info) = begin
    bindings = Tuple{Symbol,StanType,Any}[]
    _transform_bindings!(bindings, x; info)
    names = Set{Symbol}(key for (key, t, _) in bindings if _contains_integer(t))
    isempty(names) && return names
    previous = -1
    while previous != length(names)
        previous = length(names)
        for (key, t, rhs) in bindings
            (_uses_model_local(t, names) || _uses_model_local(rhs, names)) && push!(names, key)
        end
    end
    names
end
_transform_blocks(x; info) =
    _uses_model_local(x, get(meta(info), :_model_local_transforms, ())) ?
        (:model, :generated_quantities) : (:transformed_parameters,)

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
# The wrapped document is matched by its head: an expression node's type names
# a nested call by head only, so it carries no `Tuple{L,R}` for `DocumentExpr`
# to match (snag `stanblocks-trans-f759fe4f`).
distribute!(x::DocumentExpr{<:Any,<:CanonicalExprV{:document}}; info) = begin
    inner = _document_inner_block(x)
    inner === nothing ? _distribute_statement!(x; info) : distribute!(inner; info)
end
distribute!(x; info) = _distribute_statement!(x; info)
_distribute_statement!(x; info) = begin
    _push_expr!(info, x)
    for b in distribution_blocks(x; info)
        push!(block(info, b), _ragged_scoped(b, x; info); info)
    end
    _pop_expr!(info)
end
qual(x::AssignmentExpr) = qual(x.args[1])
qual(x::SamplingExpr) = qual(x.args[1])
# An observation that is not held out is a likelihood term, emitted in the model
# block. A `:quantities` value there can only be an RNG draw or something computed
# from one (`_rng_qual`): a cv-tainted value has already marked the observation
# held out. The model block cannot read it — Stan forbids RNGs there and generated
# quantities are out of scope — and a target that depends on a fresh random draw
# is not a log density. Refuse at trace time instead of emitting stanc-invalid
# Stan (snag `rng-draw-read-by-9ad81a23`).
_reject_draw_in_likelihood(x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    qual(rhs) == :quantities || return nothing
    draws = unique!(_draw_reads(rhs))
    obs = _sampling_lhs_key(lhs)
    # A latent `laplace_marginal` integrated out (laplace.jl) is a draw too, but
    # holding the observation out is not what its author wants.
    latents = _laplace_latent_reads(rhs; info)
    isempty(latents) || error(
        "Observation `", _stan_snippet(lhs), " ~ ", _stan_snippet(rhs), "` reads `", first(latents),
        "`, which `laplace_marginal` integrates out of the model block: outside its do block ",
        "it is a draw in generated quantities. Observe `", something(obs, _stan_snippet(lhs)),
        "` inside the do block of `", first(latents), "`, so that its density is part of the approximation.",
    )
    # A computed observation (`(2.0 * sum(v)) ~ …`) names no variable to hold out.
    remedy = obs === nothing ? "Compute the location without an `_rng` call." : string(
        "To predict `", obs, "` from the draw, hold it out — bind it as ",
        "`StanBlocks.stan.maybecv(:", obs, ", ", obs, ")` — so its likelihood leaves the model ",
        "block and its `", obs, "_gen` / `", obs, "_likelihood` twins read the draw in ",
        "generated quantities. Otherwise compute the location without an `_rng` call.",
    )
    error(
        "Observation `", _stan_snippet(lhs), " ~ ", _stan_snippet(rhs), "` is a likelihood term, ",
        "but it reads ", join(map(d -> "`$d`", draws), ", "), ": a random draw (an `_rng` call ",
        "or a value computed from one). Stan evaluates likelihood terms in the model block, ",
        "where random number generators are not allowed and generated quantities are out of ",
        "scope. ", remedy,
    )
end
# The draws a `:quantities` expression reads: each bound name that holds one, or
# the `_rng` call itself when it appears inline.
_draw_reads(x) = String[]
_draw_reads(x::Union{Tuple,AbstractVector}) = reduce(vcat, map(_draw_reads, x); init=String[])
_draw_reads(x::StanExpr) = qual(x) == :quantities ? _draw_reads(expr(x), x) : String[]
_draw_reads(::Symbol, x::StanExpr) = [_stan_snippet(x)]
_draw_reads(e::CanonicalExpr, x::StanExpr) =
    _is_rng_callee(head(e)) ? [_stan_snippet(x)] : _draw_reads(e.args)
_draw_reads(_, x::StanExpr) = String[]
_stan_snippet(x) = sprint(io -> show(nobreak(StanIO(io)), x))
distribution_blocks(x::AssignmentExpr; info) = if qual(x) == :data
    (:transformed_data, )
elseif qual(x) == :parameter
    _transform_blocks(x; info)
else
    (:generated_quantities, )
end
distribution_blocks(x::SamplingExpr; info) = if qual(x) == :data
    cv(x.args[1]) || _reject_draw_in_likelihood(x; info)
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
    validate_sampled_rhs(x.args...; info)
    # A plate producer emits the outer declaration separately; an indexed
    # sampling statement contributes only the model-side prior/likelihood.
    expr(x.args[1]) isa Symbol ? (:parameters, :model) : (:model,)
else
    (:generated_quantities, )
end
validate_sampled_rhs(lhs, rhs; info) = nothing
# A ragged twin aliases the completed flat draw, after its group loop. Keeping
# this marker inert during backward analysis avoids making a latent response
# likelihood-relevant merely because a predictive alias was requested.
backward!(x::CanonicalExpr{typeof(_unbound_ragged_twin)}; info) = x
distribute!(x::CanonicalExpr{typeof(_unbound_ragged_twin)}; info) = begin
    name, memory = x.args
    memory = info[expr(memory)]
    qual(memory) == :quantities || return nothing
    _assert_twin_free(name, "_gen", info)
    gen = Symbol(name, :_gen)
    twin = StanExpr(gen, remake(type(memory); value=missing))
    info[gen] = twin
    push!(block(info, :generated_quantities), CanonicalExpr(:(=), twin, memory); info)
end
# A hoisted ragged law memory (forward.jl `_ragged_law_memory_hoist`). Anything
# but a parameter-dependent value distributes as the plain assignment it wraps
# (data-only memory lands in `transformed data`). A parameter-dependent one is
# declared as a model-block local and recorded: every generated-quantities
# statement that reads it carries its own copy in a local scope (`_ragged_scoped`).
_ragged_scratch(info) = begin
    context = _trace_context(info)
    context === nothing ? nothing : context.ragged_scratch
end
distribute!(x::CanonicalExpr{typeof(_ragged_scratch_assign)}; info) = begin
    stmt = only(x.args)
    key = _base_lhs_symbol(stmt.args[1])
    scratch = _ragged_scratch(info)
    (qual(info[key]) == :parameter && scratch !== nothing) || return distribute!(stmt; info)
    scratch[key] = stmt
    _push_expr!(info, stmt)
    push!(block(info, :model), stmt; info)
    _pop_expr!(info)
end
_ragged_scoped(b, x; info) = begin
    b === :generated_quantities || return x
    scratch = _ragged_scratch(info)
    (scratch === nothing || isempty(scratch)) && return x
    used = Any[stmt for (k, stmt) in scratch if _uses_model_local(x, (k,))]
    isempty(used) && return x
    StanExpr(CanonicalExpr(_ragged_scope, used..., x), type(x))
end
fetch_data!(x::StanExpr{<:CanonicalExpr{typeof(_ragged_scope)}}; info) =
    foreach(s -> fetch_data!(s; info), expr(x).args)
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
        _transform_blocks(x; info)
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
_qual_blocks(q, x; info) = q == :parameter ? _transform_blocks(x; info) : _qual_blocks(q)
distribution_blocks(x::StanExpr{<:DeclExpr}; info) = begin
    base = info[_decl_lhs_symbol(expr(x))]
    role = _decl_role(base)
    if role in (:unfilled, :sampled)
        qual(base) == :parameter ? (:parameters,) : (:generated_quantities,)
    else
        role == :fill || error(
            "Fresh declaration `", expr(base), "` reached distribution with unknown role `", role, "`."
        )
        _qual_blocks(qual(base), base; info)
    end
end
distribution_blocks(x::StanExpr{<:CanonicalExprV{:(=)}}; info) =
    _qual_blocks(qual(info[_base_lhs_symbol(expr(x))]), x; info)
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
distribution_blocks(x::StanExpr{<:ForExpr}; info) = _qual_blocks(_for_body_qual(expr(x), info), x; info)

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
# A loop-local scratch binding (forward.jl `_loop_scratch`) has no block of its
# own: it follows the statements that read it (`_loop_select`).
_loop_distribution_blocks(::CanonicalExpr{typeof(_loop_scratch)}; info) = ()
# Its declaration is a local, never a fill target, so no slice-fill certificate
# applies (a computed `tmp::typeof(…) = …` is no fresh declaration). It only
# passes likelihood reach on to what it reads, when a likelihood reads it.
backward!(x::CanonicalExpr{typeof(_loop_scratch)}; info) =
    lqual(info[_loop_scratch_name(only(x.args))]) == :affects_likelihood ?
        CanonicalExpr(_loop_scratch, _loop_scratch_backward!(only(x.args); info)) : x
_loop_scratch_backward!(x::DocumentExpr; info) = remake(x, x.args[1], _loop_scratch_backward!(x.args[2]; info))
_loop_scratch_backward!(x::StanExpr; info) = StanExpr(_loop_scratch_backward!(expr(x); info), type(x))
_loop_scratch_backward!(x::AssignmentExpr; info) = remake(x, x.args[1], backward!(x.args[2]; info))
_loop_distribution_stmt(x, ::Val; info) = x
_loop_distribution_stmts(x, v::Val; info) = Any[_loop_distribution_stmt(x, v; info)]
_loop_distribution_stmt(x::StanExpr{<:ForExpr}, v::Val; info) = begin
    fe = expr(x)
    head, body = fe.args
    StanExpr(remake(fe, head, remake(body, _loop_select(body.args, v; info)...)), type(x))
end
# The statements of a loop body that block `B`'s copy of the loop runs, in
# source order: each statement routed to `B`, and each scratch local those
# statements read, directly or through a later scratch local. Every block's copy
# recomputes the scratch it reads, as a local of that copy's loop body.
_loop_select(stmts, v::Val{B}; info) where {B} = begin
    groups = Vector{Any}[]
    scratch = Pair{Int,Symbol}[]
    for (k, stmt) in enumerate(stmts)
        if stmt isa CanonicalExpr{typeof(_loop_scratch)}
            push!(scratch, k => _loop_scratch_name(only(stmt.args)))
            push!(groups, Any[stmt])
        else
            push!(groups, B in _loop_distribution_blocks(stmt; info) ?
                _loop_distribution_stmts(stmt, v; info) : Any[])
        end
    end
    isempty(scratch) && return reduce(vcat, groups; init=Any[])
    needed = Set{Symbol}()
    keep = falses(length(stmts))
    for k in reverse(eachindex(stmts))
        is_scratch = stmts[k] isa CanonicalExpr{typeof(_loop_scratch)}
        is_scratch && !(_loop_scratch_name(only(stmts[k].args)) in needed) && continue
        keep[k] = true
        for (j, name) in scratch
            j < k && _uses_model_local(groups[k], (name,)) && push!(needed, name)
        end
    end
    reduce(vcat, (groups[k] for k in eachindex(stmts) if keep[k]); init=Any[])
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
        for stmt in body.args, b in _loop_distribution_blocks(stmt; info)
            b in (:data, :parameters) && error(
                "Compiler-owned loop body statement `", stmt, "` tried to emit into declarative block `", b,
                "`. Plate parameters must have an outer declaration before the loop and use indexed sampling inside it."
            )
        end
        for b in _loop_distribution_blocks(x; info)
            loop = StanExpr(remake(fe, head, remake(body, _loop_select(body.args, Val(b); info)...)), type(x))
            push!(block(info, b), _ragged_scoped(b, loop; info); info)
        end
    finally
        _pop_expr!(info)
    end
    nothing
end

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
# A side-effect walk: one compiled loop, not a `map` per container type.
Base.@nospecializeinfer fetch_data!(@nospecialize(xs::Union{Tuple,NamedTuple,Vector}); info) =
    (for x in xs; fetch_data!(x; info); end; nothing)
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
# A compiler-built `tokenof{T}` token (the leading slot of a sized `_rng` draw:
# `StanExpr(T, StanType(tokenof{T}, size))`) carries the Stan type itself as
# `expr`, so only its SIZE can name data — and `show` prints that size at the
# call site (`foo_rng((n, k), …)`). Skipping the size left data named only
# there out of the `data` block: a re-drawn `z::matrix[n, k] ~ fam(A)` whose
# `fam_rng` result is sized through `A` emitted `fam_rng((n, k), A)` with no
# `int k;`, and stanc rejected it (snag `gq-re-draw-size-91350b2e`). A 0-dim
# token has an empty size and is a no-op.
fetch_data!(x::StanExpr{<:DataType}; info) = fetch_data!(type(x); info)
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
Base.push!(b::DeclarativeBlock, x::StanExpr{Symbol}; info) = _declare!(b, x; info)
_declare!(b, x; info) = begin
    fetch_data!(type(x); info)
    get!(content(b), expr(x), x)
end
# Stan declares no bound or affine transform on `ordered` / `positive_ordered`:
# `ordered<lower=0>[3] c;` is a stanc syntax error. A parameter whose typed LHS
# carries one (forward.jl, `_ordered_constraints`) is declared as a
# compiler-owned `<name>_free` of type `ordered` and the same shape instead, and
# `<name>` becomes the transformed parameter that one of Stan's own constraint
# transforms makes of it (`_ordered_transform`). Each transform is strictly
# increasing elementwise, so the elements stay ordered, and each `_jacobian`
# builtin adds its own log-Jacobian to the target. The builtins take arrays of
# vectors too, so a plate's `array[n] ordered[k]` carrier lowers the same way.
Base.push!(b::ParametersBlock, x::StanExpr{Symbol}; info) =
    _lowered_ordered(type(x)) ? _declare_lowered_ordered!(b, x; info) : _declare!(b, x; info)
_lowered_ordered(t::StanType) = _ordered_center(t) && !isempty(constraints(t))
_unconstrained_info(t::StanType) = Base.structdiff(info(t), NamedTuple{CONSTRAINT_KEYS})
_declare_lowered_ordered!(b, x; info) = begin
    name, t = expr(x), type(x)
    free_name = Symbol(name, :_free)
    # A later `~` statement on the same parameter declares nothing new.
    haskey(content(b), free_name) && return nothing
    free_name in keys(info) && error(
        "Cannot declare the compiler-owned `", free_name, "` behind the constrained `",
        name, "::", nameof(center_type(t)), "`: the model already defines `", free_name,
        "`. Rename the model's `", free_name, "`.",
    )
    cons = constraints(t)
    _uses_model_local(values(cons), get(meta(info), :_model_local_transforms, ())) && error(
        "The constraint on `", name, "::", nameof(center_type(t)), "` reads a value that ",
        "contains an integer and is therefore computed in the model block, but `", name,
        "` is computed from it in transformed parameters. Compute the bound from data ",
        "or from real-valued parameters only.",
    )
    free = StanExpr(free_name, StanType(types.ordered, stan_size(t), _unconstrained_info(t)))
    info[free_name] = free
    _declare!(b, free; info)
    center_type(t) === types.positive_ordered && !haskey(cons, :lower) &&
        (cons = (; lower=0.0, cons...))
    f, args... = _ordered_transform(free, Val(keys(cons)), cons)
    value = StanExpr(CanonicalExpr(f, map(stan_expr, args)...),
        StanType(center_type(t), stan_size(t); qual=:parameter))
    push!(block(info, :transformed_parameters), CanonicalExpr(:(=), x, value); info)
    nothing
end
_ordered_transform(free, ::Val{(:lower,)}, c) = (builtin.lower_bound_jacobian, free, c.lower)
# `upper - exp(x)` decreases in `x`, so it is applied to `-free`.
_ordered_transform(free, ::Val{(:upper,)}, c) =
    (builtin.upper_bound_jacobian, StanExpr(CanonicalExpr(-, free), type(free)), c.upper)
_ordered_transform(free, ::Val{(:lower, :upper)}, c) =
    (builtin.lower_upper_bound_jacobian, free, c.lower, c.upper)
_ordered_transform(free, ::Val{(:offset,)}, c) =
    (builtin.offset_multiplier_jacobian, free, c.offset, 1.0)
_ordered_transform(free, ::Val{(:multiplier,)}, c) =
    (builtin.offset_multiplier_jacobian, free, 0.0, c.multiplier)
_ordered_transform(free, ::Val{(:offset, :multiplier)}, c) =
    (builtin.offset_multiplier_jacobian, free, c.offset, c.multiplier)
Base.push!(b::ImperativeBlock, x; info) = begin 
    fetch_data!(x; info)
    push!(content(b), x)
end
# A constrained `ordered` / `positive_ordered` declared in a statement block (a
# plate's collected result, a re-drawn plate carrier) is declared without its
# constraint: Stan cannot spell one on these types, and what fills it satisfies
# the constraint already — a lowered parameter (`_declare_lowered_ordered!`) or
# a re-draw inside the bound (`_ordered_redraw`).
Base.push!(b::ImperativeBlock, x::StanExpr{<:DeclExpr}; info) = begin
    fetch_data!(x; info)
    push!(content(b), _lowered_ordered(type(expr(x).args[1])) ? _unconstrained_decl(x) : x)
end
_unconstrained_decl(x) = begin
    decl = expr(x).args[1]
    t = type(decl)
    unconstrained = StanExpr(expr(decl), StanType(center_type(t), stan_size(t), _unconstrained_info(t)))
    StanExpr(remake(expr(x), unconstrained), type(x))
end
Base.push!(b::ImperativeBlock, x::DocumentExpr; info) = begin
    push!(remake(b, remake(x, x.args[1], b)), x.args[2]; info)
end
# A compiler-owned `<stem>_gen` / `<stem>_likelihood` twin colliding with a USER
# variable of the same name must fail HERE — loudly naming the twin — rather
# than emitting a duplicate Stan declaration that only stanc rejects (same-block
# twins are `get!`-deduped at reflection, so the descriptor cannot see the
# collision). A DATA-qualified same-named var is exempt: it lands in another
# block, and the descriptor's input/output clash check reports it with the
# twin-aware message. Guards all three whole-LHS twin sites uniformly.
_assert_twin_free(stem, suffix, info) = begin
    stem isa Symbol || return nothing
    twin = Symbol(stem, suffix)
    twin in keys(info) || return nothing
    qual(info[twin]) == :data && return nothing
    error(
        "Cannot emit the compiler-owned twin `", twin, "` for `", stem, " ~ …`: ",
        "the model already defines `", twin, "`. `<stem>_gen` / ",
        "`<stem>_likelihood` names are reserved for the compiler's predictive ",
        "twins — rename the model's `", twin, "`.",
    )
end
Base.push!(b::GeneratedQuantitiesBlock, x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    # A re-drawn `tau::vector[n] ~ flat(…)` whose coordinates slice statements
    # sample (forward.jl, `_record_slice_support!`): `flat` has nothing to draw
    # from, so only DECLARE `tau` here; each slice's re-draw fills its block.
    if qual(lhs) != :data && expr(lhs) isa Symbol && _is_flat_rhs(rhs) &&
            !isempty(_slice_records(info[expr(lhs)]))
        t = remake(type(info[expr(lhs)]); value=missing, qual=:quantities)
        return push!(b, StanExpr(CanonicalExpr(:(::), StanExpr(expr(lhs), t)), t); info)
    end
    # if hasvalue(lhs)
    is_observation = qual(lhs) == :data
    if is_observation
        likelihood_rhs = likelihood_expr(lhs, rhs)
        _assert_twin_free(expr(lhs), "_likelihood", info)
        push!(b, CanonicalExpr(
            :(=),
            StanExpr(Symbol(expr(lhs), "_likelihood"), remake(type(likelihood_rhs); value=missing)),
            likelihood_rhs
        ); info)
        _assert_twin_free(expr(lhs), "_gen", info)
        lhs = StanExpr(Symbol(expr(lhs), "_gen"), remake(type(lhs); value=missing))
    end
    # Build a type token carrying the wanted output shape (from lhs, which is
    # either explicitly declared via typed-LHS or inferred by autotype). This
    # token becomes the leading arg to the rng call, letting each `*_rng`
    # @deffun dispatch on the shape.
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct, StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    # An observation's `<obs>_gen` twin draws from its own family (a bounded one
    # is already the `truncated` call, `_truncated_sampling_rhs`); a PARAMETER
    # re-draw must respect the sampled symbol's own `lower=`/`upper=` bounds —
    # see `redraw_rng_expr`.
    rng_rhs = is_observation ? rng_expr(token, rhs) : _gq_redraw(lhs, rhs, token)
    lhs = StanExpr(expr(lhs), remake(type(rng_rhs); value=missing))
    push!(b, CanonicalExpr(:(=), lhs, rng_rhs); info)
    # A DECLARED-but-unbound observation (`SlicModel(…, observations)`; snag
    # `unbound-observat-d32ac924`) re-draws above under its own name — and gets
    # the same `<obs>_gen` twin a bound observation has, so downstream code
    # reading `y_gen` works unchanged across fitted/prior programs. The twin
    # ALIASES the one draw (`z_gen = z`): a second independent draw would cost
    # another RNG call and dress simulation up as data-vs-simulation, while
    # emitting ONLY the twin would break downstream reads of `z` (a chained
    # unbound `w ~ family(z, …)` reads the draw). NO `_likelihood` twin —
    # pointwise needs observed values. Bound observations never reach here
    # (they twin in the branch above); a declared stem that stays a sampled
    # parameter never reaches generated quantities at all — both correctly
    # twinless.
    if !is_observation && expr(lhs) isa Symbol && expr(lhs) in _model_observations(info)
        _assert_twin_free(expr(lhs), "_gen", info)
        twin = StanExpr(Symbol(expr(lhs), "_gen"), remake(type(lhs); value=missing))
        push!(b, CanonicalExpr(:(=), twin, lhs); info)
    end
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
    if target !== nothing && target.kind === :dense && _slice_obs_pointwise(target.base; info)
        lhs, rhs = x.args
        lik = _push_slice_obs_likelihood_decl!(b, target.base; info)
        pointwise = likelihood_expr(lhs, rhs)
        push!(b, CanonicalExpr(:(=), _rename_lhs_base(lhs, lik), pointwise); info)
    end
end
# A data vector observed slice by slice (`y[1:2] ~ exponential(s); y[3] ~ …`,
# which is what `y ~ vcat(…)` lowers to) gets the per-coordinate
# `y_likelihood` a whole-vector observation has — when its slices cover every
# coordinate exactly once and each slice's pointwise density is per coordinate.
# A joint family over a slice (`y[1:3] ~ multi_normal(…)`) has one density for
# the whole slice, and an uncovered coordinate is not an observation, so either
# leaves `y` with its `y_gen` twin only.
_slice_obs_pointwise(k::Symbol; info) = begin
    records = _slice_observations(info[k])
    isempty(records) && return false
    n = _slice_int(stan_size(info[k])[1]; values=true)
    n === nothing && return false
    coords = Int[]
    for r in records
        r.coords === nothing && return false
        append!(coords, r.coords)
    end
    sort!(coords) == collect(1:n) || return false
    all(r -> stan_ndim(likelihood_expr(r.lhs, r.rhs)) == stan_ndim(r.lhs), records)
end
_push_slice_obs_likelihood_decl!(b, k::Symbol; info) = begin
    lik = Symbol(k, "_likelihood")
    if lik in keys(info)
        _obs_twin_source(info[lik]) === k && return lik
        _assert_twin_free(k, "_likelihood", info)
    end
    t = StanType(types.vector, stan_size(info[k]); qual=:quantities, obs_twin_source=k)
    decl = StanExpr(lik, t)
    info[lik] = decl
    push!(b, StanExpr(CanonicalExpr(:(::), decl), t); info)
    lik
end
_indexed_rng_assignment(x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct,
        StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    # A cell of an `array[…] ordered[k]` declaration indexes to a plain vector;
    # re-draw it as the declaration's center (`_ordered_redraw`).
    base = _base_lhs_symbol(lhs)
    redrawn = base in keys(info) && _ordered_center(info[base]) ?
        StanExpr(expr(lhs), StanType(center_type(info[base]), stan_size(lhs), type(lhs).info)) : lhs
    # A per-cell PARAMETER re-draw: honour the cell prior's own bounds.
    CanonicalExpr(:(=), lhs, _gq_redraw(redrawn, rhs, token))
end
# The one generated-quantities re-draw of a sampled symbol (top-level or
# per-cell): bounds honoured, and the two families that CANNOT be re-drawn fail
# here with the reason rather than deep inside `tracetype`/emission.
_gq_redraw(lhs, rhs, token) = begin
    _check_redraw_improper(lhs, rhs)
    _ordered_center(lhs) && return _ordered_redraw(lhs, rhs, token)
    _check_redraw_resolves(lhs, rhs, redraw_rng_expr(token, rhs))
end
# An exchangeable ordered prior (`_exchangeable_prior`) re-draws as the sorted iid
# draw: k elements from the family as a plain `vector[k]`, then `sort_asc`. A
# `positive_ordered` one draws inside `lower=0`, which `_user_bounded_redraw`
# truncates only where the family does not imply it already (`exponential`,
# `lognormal`, … draw natively). The result keeps the declared center type, so
# Stan still checks it at the end of generated quantities.
#
# Any other native-family ordered prior reaches here only when its symbol must
# be re-drawn all the same (it is cv-held-out, or its prior reads a draw), and
# the family's own draw is not sorted. Conditional on the prior's arguments its
# law is the family's law restricted to the ordered region, which rejection
# samples exactly (`_ordered_rejection`): repeat the family's own vector draw
# until it is strictly increasing. That covers non-identical elementwise
# arguments, joint families (`multi_normal`) and `truncated`, since only the
# family's draw is needed. A custom family's sized `_rng` is its author's
# contract (stanblocks-use §8) and draws as before.
_ordered_redraw(lhs, rhs, token) = begin
    exchangeable = _exchangeable_prior(rhs)
    if !exchangeable
        _builtin_family(head(expr(rhs))) || return _check_redraw_resolves(lhs, rhs, redraw_rng_expr(token, rhs))
    end
    ct = center_type(lhs)
    rhs_c = expr(rhs)
    support = _center_support(lhs)
    isempty(support) ||
        (rhs = StanExpr(remake(rhs_c, rhs_c.args...; support..., rhs_c.kwargs...), type(rhs)))
    vtoken = StanExpr(types.vector,
        StanType(types.tokenof{types.vector}, stan_size(token); value=types.vector, qual=:data))
    draw = _check_redraw_resolves(lhs, rhs, redraw_rng_expr(vtoken, rhs))
    redrawn = exchangeable ? stan_call(builtin.sort_asc, draw) : _ordered_rejection(draw)
    StanExpr(expr(redrawn), StanType(ct, stan_size(redrawn), type(redrawn).info))
end
# `draw` is the family's resolved vector draw `f_rng(token, args...)` — bounds,
# `truncated` variants and `std_normal` already settled by `redraw_rng_expr` —
# and `ordered_rejection_rng` (builtin.jl) repeats exactly that call. Each
# attempt costs one vector draw; the expected count is 1 / P(sorted | args).
_ordered_rejection(draw) = begin
    call = expr(draw)
    stan_call(builtin.ordered_rejection_rng, call.args[1], stan_expr(head(call)), call.args[2:end]...)
end
# `flat()` is the improper-uniform base measure: it has no `_rng` because there
# is nothing to draw from. A symbol whose only density is `flat` and which no
# likelihood reaches is a prior-only program's unsimulable parameter — say so
# (before this it died as `tracetype not defined … flat_rng(::array[] tokenof)`
# at emission, or as a `jbroadcasted` type error once a bound was involved).
_check_redraw_improper(lhs, rhs) = begin
    _ragged_base_family(expr(rhs)) === builtin.flat || return nothing
    name = _base_lhs_symbol(lhs)
    error(
        "`", name, " ~ flat(…)` is an improper prior and `", name, "` is re-drawn in ",
        "generated quantities — no likelihood reaches it (a prior-only / prior-predictive ",
        "program), or it is cv-held-out — but an improper prior has nothing to draw from. ",
        "Give `", name, "` a proper prior (`", name, " ~ normal(…)`, `uniform(lo, hi)`, …), ",
        "or bind the data whose likelihood informs it.",
    )
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
        "no likelihood reaches `", name, "` (a prior-only / prior-predictive program), ",
        "it is cv-held-out, or its prior reads a random draw — but its family has no ",
        "predictive companion for a `", shape,
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
# The observation a per-cell or per-slice twin declaration was minted for. The
# three sites that declare a twin ahead of its fills (`_push_obs_gen_decl!`,
# `_push_slice_obs_likelihood_decl!`, `_push_ragged_obs_decls!`) stamp it on the
# twin's type. The latter two dedupe on it, and the descriptor classifies the
# twin by it (`_generative_role`). `nothing` for every other declaration.
_obs_twin_source(x::StanExpr) = get(type(x).info, :obs_twin_source, nothing)
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
# The traced shape alone: `(; base, mem, group, slice)` with `base` the traced
# `dv`, else `nothing`. Both bounds must be present and name the same group: a
# user-written `dv.mem[1:2]` or `dv.mem[ragged_start(dv.ends, 1):2]` is not a
# group (before snag `submodel-scan-ob-50a09edb` such a slice matched with
# `group = nothing` and emitted `dv_likelihood[nothing]`).
_ragged_group_projection(x) = begin
    x isa StanExpr || return nothing
    e = expr(x)
    e isa CanonicalExpr && head(e) === getindex && length(e.args) == 2 || return nothing
    k = _ragged_field_base(e.args[1], 1)
    k isa Symbol || return nothing
    slice = e.args[2]
    se = slice isa StanExpr ? expr(slice) : slice
    se isa CanonicalExpr && head(se) isa Colon && length(se.args) == 2 || return nothing
    lo_group = _ragged_bound_group(se.args[1], builtin.ragged_start, k)
    hi_group = _ragged_bound_group(se.args[2], builtin.ragged_end, k)
    (lo_group === nothing || hi_group === nothing ||
        !isequal(expr(lo_group), expr(hi_group))) && return nothing
    (; base=expr(e.args[1]).args[1], mem=e.args[1], group=lo_group, slice)
end
_indexed_ragged_obs_slice(lhs; info) = begin
    p = _ragged_group_projection(lhs)
    p === nothing && return nothing
    k = expr(p.base)
    k in keys(info) || return nothing
    base = info[k]
    T = center_type(base)
    qual(base) == :data && T isa Type && T <: RaggedVector || return nothing
    (; base=k, mem=p.mem, group=p.group, slice=p.slice)
end
_scan_observed(x::StanExpr) = get(type(x).info, :scan_observed, nothing)
# A carried data array a `@scan` observes and reads back (forward.jl,
# `_observed_scan_shadow`): its `<base>_gen` is the shadow trajectory the scan
# itself simulates, so each observation contributes only its pointwise
# conditional log density to a per-element `<base>_likelihood`.
_scan_obs_base(lhs; info) = begin
    expr(lhs) isa Symbol && return nothing
    k = _getindex_chain_base(lhs)
    k === nothing && (k = _ragged_mem_element_base(lhs))
    (k isa Symbol && k in keys(info)) || return nothing
    base = info[k]
    qual(base) == :data && _scan_observed(base) !== nothing ? k : nothing
end
# `ys.mem[k]` — an element of an observed group of nested-vector data, in the
# flat form the scan rewrites `ys[c][i]` to: `ys`, else `nothing`.
_ragged_mem_element_base(lhs) = begin
    e = expr(lhs)
    (e isa CanonicalExpr && head(e) === getindex && length(e.args) == 2) || return nothing
    _ragged_field_base(e.args[1], 1)
end
_indexed_obs_twin_target(lhs; info) = begin
    k = _scan_obs_base(lhs; info)
    k === nothing || return (; kind=:scan, base=k)
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
    lhs_ct = center_type(lhs)
    rhs_expr = _assert_ragged_family_carrier(rhs_expr, lhs_ct)
    # Preserve the ordinary sampling diagnostic as the first gate. Because the
    # gq declarations/assignments are built before the model-loop clone is
    # pushed, attempting the sized RNG first would mask an unresolved density
    # signature with the less relevant "missing sized RNG" error.
    density = lpxf_expr(lhs, rhs_expr)
    _check_lpxf_resolves(density)
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
    k = _scan_obs_base(x.args[1]; info)
    if k !== nothing
        lhs, rhs = x.args
        lik = Symbol(k, "_likelihood")
        target = _getindex_chain_base(lhs) === k ? _rename_lhs_base(lhs, lik) : StanExpr(
            CanonicalExpr(getindex, info[lik], expr(lhs).args[2]),
            remake(type(lhs); value=missing, qual=:quantities))
        return Any[CanonicalExpr(:(=), target, likelihood_expr(lhs, rhs))]
    end
    k = _indexed_obs_gen_base(x.args[1]; info)
    k === nothing || return Any[_indexed_obs_gen_assignment(x, k)]
    spec = _indexed_ragged_obs_slice(x.args[1]; info)
    spec === nothing ? Any[_indexed_rng_assignment(x; info)] :
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
#
# Unlike every other twin, this one may have no `_likelihood` companion reading
# `k` (a per-cell observation, a joint-family slice). Held out, such a `k` leaves
# the model block, and the fills here do not read it either, so `k` drops out of
# the data block and only its sizes stay. The stamp is then the only record
# that `<k>_gen` is `k`'s predictive draw.
_push_obs_gen_decl!(b, k::Symbol; info) = begin
    gen = _gen_twin_name(k)
    gen in keys(info) && return
    t = remake(type(info[k]); value=missing, qual=:quantities, obs_twin_source=k)
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
        certified = length(existing) == 2 &&
            all(name -> _obs_twin_source(info[name]) === k, names)
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
    # `<obs>_gen` carries the per-group predictive DRAW, so its center type must
    # match the observation's own carrier: an integer ragged observation
    # (`mem::array[] int`) draws into `array[] int`, a real one into `vector`.
    # (`_likelihood` below stays real — it holds per-group density scalars.)
    gen_type = StanType(center_type(arg_types.mem), (gen_size,);
        value=missing, qual=:quantities, obs_twin_source=k)
    lik_type = StanType(types.vector, (lik_size,);
        value=missing, qual=:quantities, obs_twin_source=k)
    for decl in (StanExpr(gen, gen_type), StanExpr(lik, lik_type))
        info[expr(decl)] = decl
        push!(b, StanExpr(CanonicalExpr(:(::), decl), type(decl)); info)
    end
end
_push_indexed_obs_twin_decls!(b, target; info) =
    target.kind === :dense ? _push_obs_gen_decl!(b, target.base; info) :
    target.kind === :ragged ? _push_ragged_obs_decls!(b, target.base; info) :
    target.kind === :scan ? _push_scan_obs_likelihood_decl!(b, target.base; info) :
    error("internal: unknown indexed observation twin kind `", target.kind, "`.")
# One real log density per observed element, laid out like the observation: a
# `vector` for a 1-D array (or the flat memory of nested-vector data), a
# `matrix` for a 2-D one.
_push_scan_obs_likelihood_decl!(b, k::Symbol; info) = begin
    lik = Symbol(k, "_likelihood")
    if lik in keys(info)
        _obs_twin_source(info[lik]) === k && return lik
        _assert_twin_free(k, "_likelihood", info)
    end
    base = info[k]
    sz = if center_type(base) <: RaggedVector
        mem = StanExpr(CanonicalExpr(Base.getfield, base,
            StanExpr(1, StanType(types.int; value=1, qual=:data))), type(base).info.arg_types.mem)
        (_trace_stan_call(builtin.num_elements, mem; info),)
    else
        stan_size(base)
    end
    ct = length(sz) == 1 ? types.vector : length(sz) == 2 ? types.matrix : error(
        "internal: an observed @scan array `", k, "` of rank ", length(sz), " has no likelihood layout.")
    t = StanType(ct, sz; qual=:quantities, obs_twin_source=k)
    decl = StanExpr(lik, t)
    info[lik] = decl
    push!(b, StanExpr(CanonicalExpr(:(::), decl), t); info)
    lik
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
