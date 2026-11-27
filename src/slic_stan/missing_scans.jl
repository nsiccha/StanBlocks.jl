# ── A partly missing series an observed `@scan` reads back ─────────────────────
#
# Snag `bind-an-observed-62bb34a3`; user picks on decision `0iazaa6`: a gap
# before a series' last observed entry is imputed (a parameter, as §14 does for
# a missing entry a likelihood reads), and the program uses §14's names.
#
# A series `y` (a vector, or one group per series of nested-vector data) holds
# `missing` entries. Observed entries keep their conditional densities; each
# missing entry after the series' last observed one is drawn in generated
# quantities from its own step law, in step order, its lags reading observed
# values or earlier draws (the conditional forecast); a missing entry before it
# is a parameter whose prior is its step law. At trace time the series is bound
# as data in three parts (`_register_missing_series!`):
#
#     y_obs   the observed values, in order (one group per series when nested)
#     y_ii    one integer per position, laid out like `y`: `k > 0` the `k`-th
#             observed value, `-k` the `k`-th imputed one, `0` a drawn entry
#     y_mis   the imputed entries, `y_mis::vector[y_mis_n] ~ flat()`, whose
#             densities the scan's steps add (only when there are gaps)
#
# and the body's name `y` is bound to `y_ii`, so everything that reads the
# series' shape — a loop bound `length(ys[p])`, a plate over `ys`, a submodel
# keyword `obs = ys[p]` — works as it does with complete data. The scan reads
# the series back through `y__h = merge_scan_missing(y_obs, y_mis, y_ii)`,
# observed and imputed values over the positions (a model-block local, or
# transformed data when nothing is imputed). `_observed_scan_shadow` rewrites
# each observation `y[s] ~ law` of the series (`_missing_scan_rewrite`) into
#
#     if (y_ii[s] > 0) y_obs[y_ii[s]] ~ law       (reading y__h)
#     if (y_ii[s] < 0) y__h[s] ~ law                (the imputed entry's prior)
#     generated quantities:
#     if (y_ii[s] != 0) y[s] = y__h[s]; else y[s] = law_rng(…)   (reading y)
#
# The observation is an ordinary data observation of `y_obs`, so it keeps
# every observation's semantics (bounds, censoring, weights) and gets the
# per-element `y_obs_likelihood`; `y_gen` stays the whole trajectory from the
# setup law. The completed series is declared under the data's own name `y`
# and takes the name `y` in the body from the next model-body statement on
# (`_bind_completed_series!`), so a later statement reads the completed series,
# as one after §14's `y ~ dist(…)` reads its assembled `y`.

function _scan_guard end   # CanonicalExpr(_scan_guard, cond, stmt)
function _scan_target end  # CanonicalExpr(_scan_target, y__h[s] ~ law)
function _scan_pick end    # CanonicalExpr(_scan_pick, cond, lhs, value, law)

_missing_series(x::StanExpr) = _missing_series(type(x))
_missing_series(x::StanType) = get(x.info, :missing_series, nothing)
_missing_series(::Any) = nothing

# ── Ingest ───────────────────────────────────────────────────────────────────
# The names §14 rewrites: a top-level `y ~ dist(…)` observing `y` whole
# (`_try_expand_missing_stmt`).
_elementwise_missing_lhs(raw_body) = begin
    stmts = Meta.isexpr(raw_body, :block) ? raw_body.args : Any[raw_body]
    names = Set{Symbol}()
    for s in stmts
        (Meta.isexpr(s, :call) && length(s.args) >= 3 && s.args[1] === :~ &&
            s.args[2] isa Symbol && Meta.isexpr(s.args[3], :call)) && push!(names, s.args[2])
    end
    names
end
_is_missing_ragged(v) = false
_is_missing_ragged(v::AbstractVector{<:AbstractVector}) = any(g -> any(ismissing, g), v)
# A value whose element type admits `missing` but holds none is complete data.
_narrow_complete(v) = v
_narrow_complete(v::AbstractVector) = begin
    T = eltype(v)
    if T <: AbstractVector
        any(g -> nonmissingtype(eltype(g)) !== eltype(g), v) || return v
        any(g -> any(ismissing, g), v) && return v
        return [convert(Vector{nonmissingtype(eltype(g))}, g) for g in v]
    end
    (nonmissingtype(T) === T || any(ismissing, v)) && return v
    convert(Vector{nonmissingtype(T)}, v)
end

_register_missing_series!(data, info; exclude) = begin
    series = OrderedDict{Symbol,Any}()
    for (key, value) in pairs(data)
        key in exclude && continue
        ragged = _is_missing_ragged(value)
        ragged || _is_partly_missing_vec(value) || continue
        groups = ragged ? collect(value) : Any[value]
        for g in groups
            T = nonmissingtype(eltype(g))
            (T === Union{} || T <: AbstractFloat) || error(
                "`", key, "` holds `missing` entries, and its element type is ", eltype(g), ". ",
                "A missing entry is drawn or imputed from its law, which StanBlocks supports for ",
                "continuous (floating-point) values; convert the observed values to `Float64`.")
        end
        names = (; obs=Symbol(key, :_obs), codes=Symbol(key, :_ii), mis=Symbol(key, :_mis),
            mis_n=Symbol(key, :_mis_n), head=Symbol(key, :__h))
        for n in (values(names)..., Symbol(key, :_gen), Symbol(key, :_obs_likelihood))
            n in keys(data) && error(
                "`", key, "` holds `missing` entries, so the program names `", n, "` itself, ",
                "but `", n, "` is already a data input. Rename that input.")
        end
        obs, codes, nmis = _missing_series_split(value, ragged)
        maybedata!(info, names.obs, obs)
        maybedata!(info, names.codes, codes)
        nmis > 0 && (info[names.mis_n] = maybedata(names.mis_n, nmis))
        meta = (; name=key, names..., n_mis=nmis, ragged)
        info[key] = remake(info[names.codes]; missing_series=meta)
        series[key] = meta
    end
    context = _trace_context(info)
    context === nothing || merge!(context.missing_series, series)
    series
end
# The observed values (one group per series when nested), the position codes
# laid out like the value, and the number of imputed entries.
_missing_series_split(value, ragged) = begin
    groups = ragged ? collect(value) : Any[value]
    nobs = 0; nmis = 0
    obs_groups = Vector{Float64}[]; code_groups = Vector{Int}[]
    for g in groups
        last_obs = something(findlast(!ismissing, g), 0)
        o = Float64[]; c = Int[]
        for (t, v) in enumerate(g)
            if !ismissing(v)
                nobs += 1; push!(o, Float64(v)); push!(c, nobs)
            elseif t < last_obs
                nmis += 1; push!(c, -nmis)
            else
                push!(c, 0)
            end
        end
        push!(obs_groups, o); push!(code_groups, c)
    end
    ragged ? (obs_groups, code_groups, nmis) : (only(obs_groups), only(code_groups), nmis)
end
# Re-binding a traced model's partly missing series (`model(; ys = new)`)
# re-splits the new value into the data the program reads. A program traced
# without imputed entries has no `<y>_mis` to hold new ones.
_missing_series_rebinds(x, kwargs) = begin
    out = Dict{Symbol,Any}()
    context = _trace_context(x)
    context isa TraceContext || return out
    for (name, ms) in context.missing_series
        haskey(kwargs, name) || continue
        value = kwargs[name]
        ragged = value isa AbstractVector{<:AbstractVector}
        ragged == ms.ragged || error(
            "`", name, "` was traced as ", ms.ragged ? "nested vectors" : "a vector",
            " with `missing` entries; re-bind it with the same shape, or trace the model again.")
        obs, codes, nmis = _missing_series_split(value, ragged)
        nmis > 0 && ms.n_mis == 0 && error(
            "`", name, "` now has a `missing` entry before its last observed one, which is ",
            "imputed as a parameter, but the model was traced without such entries. Trace the ",
            "model again with this data.")
        out[ms.obs] = obs
        out[ms.codes] = codes
        ms.n_mis > 0 && (out[ms.mis_n] = nmis)
    end
    out
end
# The imputed entries and the values the scan reads back, ahead of the body.
_missing_series_prelude(raw_body, series; info) = begin
    stmts = Any[]
    for ms in values(series)
        obs = ms.ragged ? :($(ms.obs).mem) : ms.obs
        codes = ms.ragged ? :($(ms.codes).mem) : ms.codes
        rhs = if ms.n_mis > 0
            push!(stmts, :($(ms.mis)::vector[$(ms.mis_n)] ~ flat()))
            :(merge_scan_missing($obs, $(ms.mis), $codes))
        else
            :(merge_scan_missing($obs, $codes))
        end
        push!(stmts, Expr(:call, _ragged_scratch_assign, Expr(:(=), ms.head, rhs)))
    end
    body = Meta.isexpr(raw_body, :block) ? raw_body.args : Any[raw_body]
    Expr(:block, stmts..., body...)
end

# ── The scan rewrite (`_observed_scan_shadow`) ───────────────────────────────
# The completed series of `ms`, declared once (the probe and emit traces of an
# enclosing loop both rewrite the scan).
_missing_completed!(ms; info) = begin
    context = _trace_context(info)
    cmp = get(context.missing_completed, ms.name, nothing)
    cmp === nothing || return cmp
    root = _plate_root_info(info)
    sz = ms.ragged ? (forward!(canonical(:(num_elements($(ms.codes).mem))); info=root),) :
        stan_size(root[ms.codes])
    cmp = StanExpr(ms.name, StanType(types.vector, sz; qual=:quantities, missing_completed_of=ms.name))
    context.missing_completed[ms.name] = cmp
end
_missing_completed_of(x::StanExpr) = get(type(x).info, :missing_completed_of, nothing)
# `(; model, completed)`: the statements that replace one statement of the scan
# in its own part, and those that follow its `y_gen` shadow.
_missing_scan_rewrite(label, partly, replayed, obs_ref, obs_lhs, refs!, flatten; info) = begin
    root = _plate_root_info(info)
    head = Dict{Symbol,Any}(); obsv = Dict{Symbol,Any}(); cmp = Dict{Symbol,Any}()
    for (y, ms) in partly
        root[ms.obs] = remake(root[ms.obs]; scan_observed=_gen_twin_name(ms.name))
        head[y] = root[ms.head]
        obsv[y] = root[ms.obs]
        cmp[y] = _missing_completed!(ms; info)
        # An observation's base is resolved by name, so a scan in a called
        # submodel sees the caller's observed values under their own name, as
        # it sees an observed keyword's data (`_scan_data_views`).
        _plate_bind_local!(info, ms.obs, obsv[y])
    end
    flatk(o) = begin
        o.ragged && return _ragged_offset(:(ragged_start($(o.name).ends, $(only(o.cell)))), o.step)
        isempty(o.cell) || error(label, ": `", o.name, "` holds `missing` entries; a series read in ",
            "a plate cell is supported as nested vectors (`ys[p]`), not as a matrix column.")
        o.step
    end
    code(o) = o.ragged ? _ragged_element(o.name, only(o.cell), o.step) : Expr(:ref, o.name, o.step)
    # Every element reference of a missing series → `f(o)`; any other ragged
    # reference flattened as `_observed_scan_shadow` does; replayed locals
    # renamed by `local_name`.
    mapref(e, f, local_name) = begin
        e isa Symbol && return local_name(e)
        e isa Expr || return e
        o = obs_ref(e)
        if o !== nothing
            o = (; o..., step=mapref(o.step, f, local_name), cell=Any[mapref(c, f, local_name) for c in o.cell])
            haskey(partly, o.name) && return f(o)
            return o.ragged ? _ragged_element(o.name, only(o.cell), o.step) : Expr(:ref, o.name, o.step, o.cell...)
        end
        e.head === :call && return Expr(:call, e.args[1], Any[mapref(a, f, local_name) for a in e.args[2:end]]...)
        e.head === :kw && return Expr(:kw, e.args[1], mapref(e.args[2], f, local_name))
        e.head === :. && return Expr(:., mapref(e.args[1], f, local_name), e.args[2:end]...)
        Expr(e.head, Any[mapref(a, f, local_name) for a in e.args]...)
    end
    headify(e) = mapref(e, o -> Expr(:ref, head[o.name], flatk(o)), identity)
    cmpify(e) = mapref(e, o -> Expr(:ref, cmp[o.name], flatk(o)),
        n -> n in replayed ? Symbol(n, :__cmp) : n)
    reads_missing(s) = any(r -> haskey(partly, r.name), refs!(Any[], _annotated_rhs(s)))
    model(s) = begin
        o = obs_lhs(s)
        if o !== nothing && haskey(partly, o.name)
            ms = partly[o.name]
            c = code(o); k = flatk(o)
            law = headify(_annotated_rhs(s))
            lhs = ms.ragged ? Expr(:ref, Expr(:., obsv[o.name], QuoteNode(:mem)), c) : Expr(:ref, obsv[o.name], c)
            out = Any[Expr(:call, _scan_guard, :($c > 0), Expr(:call, :~, lhs, law))]
            ms.n_mis > 0 && push!(out, Expr(:call, _scan_guard, :($c < 0),
                Expr(:call, _scan_target, Expr(:ref, head[o.name], k), law)))
            return out
        end
        reads_missing(s) || return Any[flatten(s)]
        name = first(r.name for r in refs!(Any[], _annotated_rhs(s)) if haskey(partly, r.name))
        (s isa Expr && s.head === :(=) && _annotated_untyped(s.args[1]) isa Symbol) || error(
            label, ": `", s, "` reads `", name, "`, which holds `missing` entries. Only the ",
            "observations of a partly missing series, and per-step `=` locals they read, may read it ",
            "back yet: its missing entries have no value where this statement runs.")
        Any[Expr(:(=), s.args[1], headify(s.args[2]))]
    end
    completed(s) = begin
        o = obs_lhs(s)
        if o !== nothing && haskey(partly, o.name)
            c = code(o); k = flatk(o)
            return Any[Expr(:call, _scan_pick, :($c != 0), Expr(:ref, cmp[o.name], k),
                Expr(:ref, head[o.name], k), cmpify(_annotated_rhs(s)))]
        end
        if s isa Expr && s.head === :(=) && _scan_stmt_lhs_name(s) in replayed
            lhs = _annotated_untyped(s.args[1])
            lhs isa Symbol || error(
                label, ": `", s, "` fills an array that the observations of a partly missing series ",
                "read back; write it as a per-step local (`", _scan_stmt_lhs_name(s), " = …`, read at ",
                "the current step) instead, so the forecast can recompute it from its draws.")
            return Any[Expr(:(=), Symbol(lhs, :__cmp), cmpify(s.args[2]))]
        end
        Any[]
    end
    (; model, completed)
end

# ── After tracing ────────────────────────────────────────────────────────────
# A consumed series is, from the next model-body statement on, its completed
# series (`forward!(::BlockExpr)` at the root body).
_bind_completed_series!(info) = begin
    context = _trace_context(info)
    context === nothing && return nothing
    root = _plate_root_info(info)
    for (name, cmp) in context.missing_completed
        ms = _missing_series(root[name])
        ms === nothing && continue
        root[name] = cmp
    end
    nothing
end
_finish_missing_series!(traced; info) = begin
    context = _trace_context(info)
    (context === nothing || isempty(context.missing_series)) && return nothing
    _bind_completed_series!(info)
    for (name, ms) in context.missing_series
        haskey(context.missing_completed, name) && continue
        error(
            "`", name, "` holds `missing` entries, but no `~` statement observes it whole and no ",
            "observed `@scan` reads it back. A missing entry is drawn or imputed from the law of ",
            "an outcome: an elementwise `", name, " ~ dist(…)` (§14), or the steps of a `@scan` ",
            "that observes `", name, "` and reads it back (§35). Missing predictors or covariates ",
            "on the right-hand side are out of scope; split the model manually.")
    end
    codes = Dict{Symbol,Any}(ms.codes => ms for ms in values(context.missing_series))
    stray = Any[]
    _stray_series_reads!(stray, traced, codes)
    isempty(stray) && return nothing
    ms, snippet = first(stray)
    error(
        "`", ms.name, "` holds `missing` entries, so only the observed `@scan` that reads it back ",
        "may read its values, but `", snippet, "` reads them (`", ms.codes, "` is `", ms.name,
        "` there). Read `", ms.name, "` after the model-body statement holding that scan, where it ",
        "is the completed series, with its missing entries drawn or imputed.")
end
# The reads of a series' position codes outside the conditions and the
# read-back merge this file emits: a value read of the series elsewhere, with
# the expression that holds it.
_stray_series_reads!(acc, x, codes, outer=nothing) = acc
_stray_series_reads!(acc, x::Union{AbstractVector,Tuple}, codes, outer=nothing) =
    (foreach(a -> _stray_series_reads!(acc, a, codes, outer), x); acc)
_stray_series_reads!(acc, x::StanExpr, codes, outer=nothing) = begin
    e = expr(x)
    if e isa Symbol
        haskey(codes, e) && !codes[e].ragged && push!(acc, (codes[e], something(outer, _stan_snippet(x))))
        return acc
    end
    e isa CanonicalExpr || return acc
    if head(e) === Base.getfield && length(e.args) == 2 && e.args[1] isa StanExpr &&
            expr(e.args[1]) isa Symbol && haskey(codes, expr(e.args[1]))
        _literal_int(e.args[2]) == 1 &&
            push!(acc, (codes[expr(e.args[1])], something(outer, _stan_snippet(x))))
        return acc
    end
    head(e) in (builtin.num_elements, size, length) && length(e.args) == 1 && return acc
    _stray_series_reads!(acc, e, codes, _stan_snippet(x))
end
_stray_series_reads!(acc, x::CanonicalExpr, codes, outer=nothing) =
    _stray_series_reads!(acc, x.args, codes, outer)
# A guarded observation is indexed by the position code; its law is what reads values.
_stray_series_reads!(acc, x::CanonicalExpr{typeof(_scan_guard)}, codes, outer=nothing) = begin
    inner = x.args[2]
    _stray_series_reads!(acc, inner isa SamplingExpr ? inner.args[2] : inner, codes, outer)
end
_stray_series_reads!(acc, x::CanonicalExpr{typeof(_scan_pick)}, codes, outer=nothing) =
    _stray_series_reads!(acc, x.args[2:end], codes, outer)
_stray_series_reads!(acc, x::CanonicalExpr{typeof(_ragged_scratch_assign)}, codes, outer=nothing) = begin
    stmt = only(x.args)
    lhs = _base_lhs_symbol(stmt.args[1])
    any(ms -> ms.head === lhs, values(codes)) && return acc
    _stray_series_reads!(acc, stmt, codes, outer)
end

# ── Tracing the rewritten statements ─────────────────────────────────────────
forward!(x::CanonicalExpr{typeof(_scan_guard)}; info) =
    CanonicalExpr(_scan_guard, forward!(x.args[1]; info), forward!(x.args[2]; info))
forward!(x::CanonicalExpr{typeof(_scan_pick)}; info) =
    CanonicalExpr(_scan_pick, map(a -> forward!(a; info), x.args)...)
# The density of an imputed entry: its step law, which must be an unbounded
# continuous family, since the entry is an unconstrained parameter.
forward!(x::CanonicalExpr{typeof(_scan_target)}; info) = begin
    lhs = forward!(x.args[1]; info)
    law = forward!(x.args[2]; info)
    c = expr(law)
    c isa CanonicalExpr || error("internal: the law of an imputed scan entry is not a call: `", c, "`.")
    f = head(c)
    bounded = !isnothing(_distribution_hof(f)) || _probability_kind(f) !== :lpdf ||
        any(k -> k in BOUND_KEYS, keys(c.kwargs)) ||
        any(v -> !(v isa Real && isinf(v)), values(autokwargs(c)))
    bounded && error(
        "A gap before the last observed entry of a series an observed `@scan` reads back is ",
        "imputed as a parameter, which needs a law with support on the whole real line; ",
        "`", _stan_snippet(law), "` is bounded, censored, weighted or discrete. Leave the entry ",
        "out of the data (the time-gap form of the step law handles an unrecorded step), or ",
        "end the series at its last observed entry.")
    lpxf_expr(lhs, c)   # resolves the family's density, as an observation's does
    CanonicalExpr(_scan_target, CanonicalExpr(:~, lhs, law))
end
backward!(x::CanonicalExpr{typeof(_scan_guard)}; info) = remake(x, x.args[1], backward!(x.args[2]; info))
# The imputed entry's `~` reaches what it reads, as a likelihood does.
backward!(x::CanonicalExpr{typeof(_scan_target)}; info) = begin
    stmt = only(x.args)
    remake(x, remake(stmt, backward!(stmt.args[1]; info), backward!(stmt.args[2]; info)))
end
backward!(x::CanonicalExpr{typeof(_scan_pick)}; info) = x

# ── Placement ────────────────────────────────────────────────────────────────
distribution_blocks(x::CanonicalExpr{typeof(_scan_guard)}; info) = distribution_blocks(x.args[2]; info)
_loop_distribution_blocks(x::CanonicalExpr{typeof(_scan_guard)}; info) = _loop_distribution_blocks(x.args[2]; info)
distribution_blocks(::CanonicalExpr{typeof(_scan_target)}; info) = (:model,)
distribution_blocks(::CanonicalExpr{typeof(_scan_pick)}; info) = (:generated_quantities,)
# `y__h[s] ~ law`: like every other `~` in the program, it drops the law's
# constant terms (a `target += law_lpdf(…)` would keep them).
Base.show(io::IO, x::CanonicalExpr{typeof(_scan_target)}) = print(io, only(x.args))
_guard_if(cond, stmts) = CanonicalExpr(:if, cond, CanonicalExpr(:block, stmts...))
# A model-scope statement, typed so a generated-quantities scope can wrap it
# (`_ragged_scoped`).
_stmt_expr(x) = StanExpr(x, StanType(types.anything))
_loop_distribution_stmts(x::CanonicalExpr{typeof(_scan_guard)}, v::Val; info) = begin
    inner = _loop_distribution_stmts(x.args[2], v; info)
    isempty(inner) ? Any[] : Any[_guard_if(x.args[1], inner)]
end
_loop_distribution_stmts(x::CanonicalExpr{typeof(_scan_pick)}, ::Val{:generated_quantities}; info) =
    Any[_pick_if(x)]
# `if (cond) y[k] = value; else y[k] = draw;` — the draw as the observation's
# own twin draws: a bounded law truncated, a censored one clamped.
_pick_if(x) = begin
    cond, lhs, value, law = x.args
    ct = center_type(lhs)
    token = StanExpr(ct, StanType(types.tokenof{ct}, stan_size(lhs); value=ct, qual=:data))
    draw = rng_expr(token, _normalised_truncation(law; observed=true))
    target = StanExpr(expr(lhs), remake(type(lhs); value=missing, qual=:quantities))
    CanonicalExpr(:if, cond, CanonicalExpr(:block, CanonicalExpr(:(=), target, value)),
        CanonicalExpr(:block, CanonicalExpr(:(=), target, draw)))
end
_obs_twin_targets(x::CanonicalExpr{typeof(_scan_guard)}; info) = _obs_twin_targets(x.args[2]; info)
_obs_twin_targets(x::CanonicalExpr{typeof(_scan_pick)}; info) =
    NamedTuple[(; kind=:missing_completed, base=_missing_pick_name(x))]
_missing_pick_name(x) = _base_lhs_symbol(x.args[2])
# The completed series' declaration, once, ahead of its first fill.
_push_missing_completed_decl!(b, name::Symbol; info) = begin
    context = _trace_context(info)
    (context === nothing || name in context.missing_declared) && return nothing
    push!(context.missing_declared, name)
    cmp = context.missing_completed[name]
    t = remake(type(cmp); value=missing, qual=:quantities)
    push!(b, StanExpr(CanonicalExpr(:(::), StanExpr(name, t)), t); info)
    nothing
end
distribute!(x::CanonicalExpr{typeof(_scan_guard)}; info) = begin
    _push_expr!(info, x)
    inner = x.args[2]
    for b in distribution_blocks(inner; info)
        if b === :generated_quantities
            for target in unique(_obs_twin_targets(inner; info))
                _push_indexed_obs_twin_decls!(block(info, b), target; info)
            end
        end
        stmts = _loop_distribution_stmts(inner, Val(b); info)
        isempty(stmts) || push!(block(info, b), _ragged_scoped(b, _stmt_expr(_guard_if(x.args[1], stmts)); info); info)
    end
    _pop_expr!(info)
end
distribute!(x::CanonicalExpr{typeof(_scan_pick)}; info) = begin
    _push_expr!(info, x)
    b = block(info, :generated_quantities)
    _push_missing_completed_decl!(b, _missing_pick_name(x); info)
    push!(b, _ragged_scoped(:generated_quantities, _stmt_expr(_pick_if(x)); info); info)
    _pop_expr!(info)
end

# The partly missing series a traced model binds, by name: the data name of
# each series' position codes (the descriptor reads its groups there).
_missing_series_codes(m) = begin
    context = _trace_context(m)
    context isa TraceContext || return Dict{Symbol,Symbol}()
    Dict{Symbol,Symbol}(name => ms.codes for (name, ms) in context.missing_series)
end
