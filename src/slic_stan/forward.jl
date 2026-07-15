# Resolve a module-level binding to a SLIC value. `@usertype`-declared types
# live in the user's module as abstract types `<: types.anything`; treat
# them like SLIC type tokens (same path as `vector` / `real`) so the
# constructor call dispatches via the usertype tracetype.
_forward_module_value(v::Function, info) = forward!(v; info)
_forward_module_value(v::SlicModel, info) = v
# A named sub-model function resolves to its (singleton) value, like a `SlicModel`;
# the call is embedded by `stan_expr(::CanonicalExpr{<:SubmodelFn})`.
_forward_module_value(v::SubmodelFn, info) = v
# Built-in mathematical constants (π, ℯ, … — all `Irrational`s) resolve to their
# Float64 value via `forward!(::Irrational)`. Per user decision `3bbtrv`: only
# built-in constants resolve this way, NOT arbitrary module-level numbers (there
# is deliberately no `::Number` method here — a bare `const X = 2.5` referenced
# in a model still errors loudly at `forward!(::Symbol)`).
_forward_module_value(v::Irrational, info) = forward!(v; info)
# `_forward_module_value(::Type{<:types.anything}, _)` is defined after
# `include("functions.jl")` (which defines `module types`); placing it
# here would error at module-load time with `UndefVarError: types`.
_forward_module_value(_, _) = nothing
forward!(x::Function; info) = stan_expr(x)
# `macroexpand` (used by `slic_macroexpand`) hygienically resolves all free
# names — including model-scope SLIC variables and SLIC builtins like
# `mean`/`length` — to `GlobalRef(mod, :name)`. Always try SLIC's own
# resolution chain (model → builtin → mod → Main) first so that e.g.
# `mean` resolves to `stan.builtin.mean` (with a registered tracetype),
# not `Statistics.mean` (which lacks one). Fall back to the GlobalRef's
# native module only when nothing else matches.
forward!(x::GlobalRef; info) = begin
    rv = _try_symbol_lookup(x.name; info)
    rv === nothing || return forward!(rv; info)
    isdefined(x.mod, x.name) || error(
        "Could not resolve $x — not in SLIC scope (model/builtin/mod/Main) and not defined in $(x.mod)."
    )
    forward!(getproperty(x.mod, x.name); info)
end
_try_symbol_lookup(x::Symbol; info) = begin
    x in keys(info) && return info[x]
    _is_builtin_name(x) && return getproperty(builtin, x)
    mod = get_module(info)
    if isdefined(mod, x)
        rv = _resolve_module_value(getproperty(mod, x))
        rv === nothing || return rv
    end
    if mod !== Main && isdefined(Main, x)
        rv = _resolve_module_value(getproperty(Main, x))
        rv === nothing || return rv
    end
    nothing
end
_resolve_module_value(v::Function) = v
_resolve_module_value(v::SlicModel) = v
_resolve_module_value(v::SubmodelFn) = v
# Built-in constants resolve here too (the GlobalRef path), mirroring the
# `_forward_module_value(::Irrational)` symbol path — user decision `3bbtrv`.
_resolve_module_value(v::Irrational) = v
# Type{<:types.anything} method defined after `include("functions.jl")`.
_resolve_module_value(_) = nothing
forward!(x::Colon; info) = x
forward!(x::StanExpr{Symbol}; info) = x
forward!(x::StanExpr; info) = x
forward!(x::CanonicalExpr; info) = begin
    _push_expr!(info, x)
    resolved = CanonicalExpr(forward!(head(x); info), forward!(x.args; info)...; forward!(x.kwargs; info)...)
    s = _get_expr_stack(info)
    isnothing(s) || (s[end] = (resolved, s[end][2]))
    rv = expand_inline_or_trace(resolved; info)
    _pop_expr!(info)
    rv
end
# `inline_body` is the dispatch hook populated by `@deffun @inline f(...)`
# and `@deffun f!(...)`. A non-`nothing` return signals the call site should
# substitute its args into the stored body AST and re-trace in the caller's
# scope rather than emitting a Stan-function call.
inline_body(::Any) = nothing
expand_inline_or_trace(x::CanonicalExpr; info) = begin
    meta = inline_body(x)
    isnothing(meta) && return fold_shape_query(stan_expr(x))
    expand_inline!(x, meta; info)
end
# ── Elementwise arithmetic on scalar arrays → lower to the `jbroadcasted` loop ──
# Stan has no elementwise arithmetic on `array[] int`/`array[] real` (nor a
# dotted `.+`/`.-` — `broadcast_callee` collapses those to plain `+`/`-`), so a
# binary `+`/`-` or a broadcast `.* ./ .^` with a scalar-array operand is lowered
# to the generalised `jbroadcasted` element loop (builtin.jl) instead of emitting
# invalid Stan. `jbroadcasted` handles ANY arg position (array-first,
# scalar-first, arbitrary arity) and INFERS the output container from `f`'s
# per-element return type, so the operator + operands pass straight through — no
# commuting/negating dance (the old array-first-only form needed one, and its
# `s .- arr == -(arr .- s)` identity has since become wrong: `jbroadcasted`'s
# int-array result has no valid Stan unary minus). Both plain and dotted `+`/`-`
# lower (Julia-consistent: `+` on arrays is elementwise); plain `*`/`/`/`^` on
# arrays stay rejected (a matmul/dim error in Julia, not elementwise) via the
# `_reject_scalar_array_elementwise` floor. Only binary (2-arg) operators lower;
# a rare >2-arg form falls through to that reject rather than miscompiling.
_broadcast_op(f) = f
_broadcast_op(f::Base.BroadcastFunction) = f.f
_lower_scalar_array_broadcast(x::CanonicalExpr; info) = begin
    length(x.args) == 2 || return nothing
    l, r = x.args
    (_is_scalar_array(type(l)) || _is_scalar_array(type(r))) || return nothing
    forward!(CanonicalExpr(builtin.jbroadcasted, _broadcast_op(head(x)), l, r); info)
end
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(+),typeof(-)}}; info) = begin
    rv = _lower_scalar_array_broadcast(x; info)
    isnothing(rv) ? fold_shape_query(stan_expr(x)) : rv
end
expand_inline_or_trace(x::CanonicalExpr{<:Base.BroadcastFunction}; info) = begin
    rv = _lower_scalar_array_broadcast(x; info)
    isnothing(rv) ? fold_shape_query(stan_expr(x)) : rv
end
# `isdefined(builtin, x)` returns true even for names inherited from Base
# (e.g. `accumulate!`), which would mask user-defined SLIC UDFs that share
# Base's name. Restrict to bindings actually owned by the `builtin` module.
_is_builtin_name(x::Symbol) = isdefined(builtin, x) && Base.which(builtin, x) === builtin
expand_inline!(x::CanonicalExpr, meta; info) = begin
    arg_names = collect(meta.arg_names)
    n_pos = length(arg_names)
    @assert length(x.args) >= n_pos "@inline: call to $(head(x)) has $(length(x.args)) args, expected ≥ $n_pos."
    subst = Dict{Symbol,Any}()
    for (i, name) in enumerate(arg_names)
        subst[name] = x.args[i]
    end
    if meta.vararg_name !== nothing
        subst[meta.vararg_name] = collect(x.args[n_pos+1:end])
    elseif length(x.args) > n_pos
        error("@inline: call to $(head(x)) has $(length(x.args)) args but the UDF takes exactly $n_pos.")
    end
    # Kwargs (kwarg-shim case): bind each declared kwarg name to either the
    # call-site value or the registered default. Unknown call-site kwargs
    # are an error — there's no schema to dispatch them to.
    kwarg_meta = get(meta, :kwargs, ())
    if !isempty(kwarg_meta)
        callsite_kw = Dict(pairs(x.kwargs))
        declared = Set{Symbol}(s[1] for s in kwarg_meta)
        for (kn, _) in pairs(x.kwargs)
            kn in declared || error(
                "@inline kwcall shim for `$(head(x))`: unknown kwarg `$kn`. ",
                "Declared kwargs: $(join(string.(collect(declared)), ", "))."
            )
        end
        for (kn, default_expr) in kwarg_meta
            subst[kn] = if haskey(callsite_kw, kn)
                callsite_kw[kn]
            else
                forward!(canonical(default_expr); info)
            end
        end
    elseif !isempty(x.kwargs)
        error("@inline call to `$(head(x))` got unexpected kwargs: $(collect(keys(x.kwargs))).")
    end
    stmts = inline_unwrap_block(meta.body, head(x))
    isempty(stmts) && error("@inline UDF $(head(x)): empty body.")
    # Collect names introduced inside the body so each call site gets fresh
    # locals that can't collide with the caller's vars or with sibling
    # expansions of the same UDF.
    locals = Set{Symbol}()
    for s in stmts; _collect_locals!(s, locals); end
    arg_set = Set{Symbol}(arg_names)
    meta.vararg_name !== nothing && push!(arg_set, meta.vararg_name)
    rename = Dict{Symbol,Symbol}()
    if !isempty(locals)
        id = _next_inline_id()
        for name in locals
            name in arg_set && continue
            rename[name] = Symbol(name, "__il_", id)
        end
    end
    # Captures (closure case): each pre-resolved definition-site StanExpr
    # gets substituted into the body just like a positional arg. Skip names
    # the body locally shadows — those land in `rename` and the literal
    # `name = ...` LHS keeps its meaning.
    for (k, v) in get(meta, :captures, ())
        k in locals && continue
        subst[k] = v
    end
    rewritten = [inline_substitute(s, subst, rename, meta.vararg_name) for s in stmts]
    _retrace_inline_body(rewritten, get(meta, :mod, nothing); info)
end

# Re-trace the substituted inline body in the caller's `info`. A *closure*
# meta carries `mod` (its definition module — snapshotted as `cl.mod` when the
# lambda was traced); re-point `info[:__mod__]` at it around the re-trace so
# nested user-module UDF lookups inside the closure body resolve there, mirroring
# `fundef(::closure)`'s `info[:__mod__] = cl.mod` and 92f08ae's def_mod threading.
# Without this, a closure inlined inside a builtin helper (e.g.
# `simple_reduce_sum_helper`, whose module is StanBlocks, calling `f(...)`
# directly) would re-trace with `__mod__ = StanBlocks` and fail to find the
# user's UDF. Save/restore keeps the caller's scope (and sibling expansions of
# the same UDF) unaffected, and nests correctly. Regular `@deffun @inline` UDF
# metas carry no `mod`, so their established caller-scope resolution is unchanged.
# Only `AbstractDict` `info` is overridden: `StanModel`/`SubModel` resolve the
# module elsewhere (and there a closure's `mod` already equals the model's
# module, so no override is needed) — and `StanModel` `setindex!` would set a
# *var*, which must not happen.
_retrace_inline_body(rewritten, mod; info) =
    if mod === nothing || !(info isa AbstractDict)
        _do_retrace_inline_body(rewritten; info)
    else
        had = haskey(info, :__mod__)
        old = had ? info[:__mod__] : nothing
        info[:__mod__] = mod
        try
            _do_retrace_inline_body(rewritten; info)
        finally
            had ? (info[:__mod__] = old) : delete!(info, :__mod__)
        end
    end
_do_retrace_inline_body(rewritten; info) = begin
    pending = _get_inline_pending()
    for s in rewritten[1:end-1]
        result = forward!(canonical(s); info)
        pending !== nothing && push!(pending, result)
    end
    forward!(canonical(rewritten[end]); info)
end

# Per-callsite counter for locals (uniqueness within a trace is the only
# requirement). Lives in per-trace task-local storage — `_next_inline_id` +
# the seed-scope are defined centrally in tracing.jl.

_get_inline_pending() = get(task_local_storage(), :_slic_inline_pending, nothing)

# Unwrap an inline-UDF body into a vector of effective statements. A `:return`
# in the final position is unwrapped to its value; an empty/all-LNN body
# signals the caller to error.
inline_unwrap_block(body, fname) = [body]
inline_unwrap_block(body::Expr, fname) = if body.head === :block
    real = filter(a -> !(a isa LineNumberNode), body.args)
    isempty(real) && return Any[]
    if Meta.isexpr(real[end], :return)
        real[end] = real[end].args[1]
    end
    real
elseif body.head === :return
    [body.args[1]]
else
    [body]
end

# Walk the un-canonicalised body collecting names introduced as new bindings
# (LHS of `=`, `for` indices, `::`-decls, tuple destructuring). Indexed LHSes
# (`a[i] = ...`) and field LHSes don't introduce new names — skip those.
_collect_locals!(x, locals) = nothing
_collect_locals!(x::Expr, locals) = begin
    if x.head === :(=)
        _collect_lhs!(x.args[1], locals)
        _collect_locals!(x.args[2], locals)
    elseif x.head === :for && length(x.args) >= 1 && Meta.isexpr(x.args[1], :(=))
        _collect_lhs!(x.args[1].args[1], locals)
        for a in x.args[2:end]; _collect_locals!(a, locals); end
    elseif x.head === :(::)
        _collect_lhs!(x.args[1], locals)
    else
        for a in x.args; _collect_locals!(a, locals); end
    end
end
_collect_lhs!(x, locals) = nothing
_collect_lhs!(x::Symbol, locals) = push!(locals, x)
_collect_lhs!(x::Expr, locals) = if x.head === :tuple
    for a in x.args; _collect_lhs!(a, locals); end
elseif x.head === :(::)
    _collect_lhs!(x.args[1], locals)
end

# Walk the body AST replacing parameter Symbols with call-site StanExprs and
# local names with their renamed forms. Splat positions referencing the
# vararg name (`args...`) expand into separate arguments at the parent call.
inline_substitute(x, subst, rename, vararg_name) = x
inline_substitute(x::Symbol, subst, rename, vararg_name) = if haskey(subst, x)
    subst[x]
elseif haskey(rename, x)
    rename[x]
else
    x
end
inline_substitute(x::Expr, subst, rename, vararg_name) = begin
    # `:kw` exprs (e.g. `(;sigma=sigma)` shorthand) have a LHS *name* that
    # must stay literal — only the RHS *value* gets substituted.
    if x.head === :kw && length(x.args) == 2
        return Expr(:kw, x.args[1], inline_substitute(x.args[2], subst, rename, vararg_name))
    end
    new_args = []
    for arg in x.args
        if vararg_name !== nothing && Meta.isexpr(arg, :...) &&
            length(arg.args) == 1 && arg.args[1] === vararg_name
            for v in subst[vararg_name]
                push!(new_args, v)
            end
        else
            push!(new_args, inline_substitute(arg, subst, rename, vararg_name))
        end
    end
    Expr(x.head, new_args...)
end
fold_shape_query(x) = x
fold_shape_query(x::StanExpr) = x
# `forward!(::BlockExpr)` pushes a fresh pending-statements buffer for its
# scope, drains it between args, then pops on exit. This is what lets inline
# UDFs hoist multi-statement bodies into the enclosing block without leaking
# into sibling sub-blocks (for / while / if branches, nested blocks).
forward!(x::BlockExpr; info) = task_local_storage(:_slic_inline_pending, Any[]) do
    new_args = Any[]
    pending = task_local_storage(:_slic_inline_pending)
    for arg in x.args
        resolved = forward!(arg; info)
        if !isempty(pending)
            append!(new_args, pending)
            empty!(pending)
        end
        # Bare-symbol/number/string StanExprs at statement position have no
        # Stan-side meaning — they arise e.g. when an inline UDF whose final
        # expression is just one of its args (`f!(buf) = (mutate; buf)`) is
        # called at statement position. Skip them so we don't emit `name;`.
        _is_inert_block_stmt(resolved) || push!(new_args, resolved)
    end
    remake(x, new_args...)
end
_is_inert_block_stmt(x) = false
_is_inert_block_stmt(x::StanExpr) = _is_inert_expr(expr(x))
_is_inert_expr(::Symbol) = true
_is_inert_expr(::Number) = true
_is_inert_expr(::AbstractString) = true
_is_inert_expr(_) = false
_is_submodel_info(::SubModel) = true
_is_submodel_info(_) = false
_is_getindex_expr(::CanonicalExprV{:getindex}) = true
_is_getindex_expr(_) = false
_is_canonical_expr(::CanonicalExpr) = true
_is_canonical_expr(_) = false
# `_is_ntup_stan_expr(::StanExpr2{<:types.ntup})` is defined after
# `include("functions.jl")` for the same load-order reason as
# `_forward_module_value(::Type{<:types.anything}, _)` above.
_is_ntup_stan_expr(_) = false
_is_assign_canonical(::CanonicalExprV{:(=)}) = true
_is_assign_canonical(_) = false
_is_block_canonical(::CanonicalExprV{:block}) = true
_is_block_canonical(_) = false
_check_assignment_rhs(name, ::SlicModel) = error(
    "`$name = <submodel>(...)` is not supported — sub-models can only be embedded via `~`. ",
    "Use `$name ~ <submodel>(...)` instead.")
_check_assignment_rhs(_name, ::StanExpr) = nothing
_check_assignment_rhs(name, resolved) = error(
    "`$name = <rhs>`: rhs forwarded to a value of type `$(typeof(resolved))`, expected `StanExpr`.")

forward!(x::AssignmentExpr{Symbol}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    resolved = forward!(rhs; info)
    _check_assignment_rhs(name, resolved)
    center_type(resolved) === types.void && error(
        "`$name = <void-call>(...)`: cannot bind the return value of a void UDF. ",
        "Drop the `$name = ` and call as a statement."
    )
    forward!(remake(x, name, resolved); info)
end
maybe_lazy_size(key::Symbol, i, sizei; info) = sizei
is_simple_size(x::StanExpr) = is_simple_size(expr(x))
is_simple_size(x::CanonicalExpr{<:Union{typeof.((+,-,*,÷))...}}) = all(is_simple_size, x.args)
is_simple_size(x::CanonicalExpr) = false
is_simple_size(x::Symbol) = true
is_simple_size(x::Number) = true
is_simple_size(x) = false#error(typeof(x))
maybe_lazy_size(key::Symbol, i, sizei::StanExpr{<:CanonicalExpr}; info) = if is_simple_size(sizei) || qual(sizei) == :data
    sizei
else
    forward!(canonical(:(dims($key)[$i])); info)
end
forward!(x::AssignmentExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args 
    @assert name ∉ keys(info)
    info[name] = StanExpr(name, remake(type(rhs); value=missing))
    @assert center_type(rhs) != types.anything "tracetype not defined for $name = $(short_expr(rhs))!"
    rv = remake(x, info[name], rhs)
    info[name] = StanExpr(name, remake(type(rhs), [
        maybe_lazy_size(name, i, sizei; info)
        for (i, sizei) in enumerate(stan_size(type(rhs)))
    ]...; value=missing))
    rv
end
# Fallback: a compiler-injected slice/element fill `out[i] = rhs` (getindex LHS —
# Symbol LHS is fully handled by the two methods above; a user-written non-Symbol
# LHS is rejected pre-`forward!`). Forward the args, then PROMOTE the base
# variable's qual by the rhs qual. This is what lets `distribute!` route the whole
# coarse-grained variable — its bare declaration + every fill — to ONE block:
# `info[base].qual` finalizes to `max` over the fills' rhs quals. `:undefined` (the
# bare decl's default, before any fill) is bottom; the reals order
# `data < parameter < quantities` (lexicographically AND semantically).
_promote_qual(cur::Symbol, new::Symbol) =
    cur === :undefined ? new : (new === :undefined ? cur : max(cur, new))
forward!(x::AssignmentExpr; info) = begin
    # Keep the local/raw key before forwarding: inside a `SubModel`, forwarding
    # rewrites the emitted base to its flattened parent name while `keys(info)`
    # intentionally remains the local-name view.
    local_key = _base_lhs_symbol(x.args[1])
    fwd = stan_expr(remake(x, forward!(x.args; info)...))
    lhs, rhs = expr(fwd).args
    k = local_key in keys(info) ? local_key : _base_lhs_symbol(lhs)
    if k in keys(info)
        base = info[k]
        if _is_fresh_decl(base)
            role = _decl_role(base)
            role == :sampled && error(
                "Cannot fill `", k, "[…]` after it has been classified as a plate parameter by indexed sampling."
            )
            if role == :unfilled
                # User decision `1dd0eww`: a bare declaration starts life as a
                # flat-prior parameter, then its FIRST certified indexed fill
                # reclassifies it as a transformed fill target.  Reset (rather
                # than promote) the qualifier to that first RHS.
                info[k] = remake(base; decl_role=:fill, qual=qual(rhs))
            else
                role == :fill || error("Fresh declaration `", k, "` has unknown declaration role `", role, "`.")
                info[k] = remake(base; qual=_promote_qual(qual(base), qual(rhs)))
            end
        else
            info[k] = remake(base; qual=_promote_qual(qual(base), qual(rhs)))
        end
    end
    fwd
end
forward!(x::SamplingExpr{Symbol}; info) = begin
    name, rhs = x.args
    rhs = forward!(rhs; info)::Union{StanExpr,SlicModel}
    forward!(remake(x, name, rhs); info)
end
forward!(x::SamplingExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args
    if name in keys(info)
        q = stan.qual(info[name])
        q == :data || error("Sampling statement `$name ~ ...` has LHS bound to a $q-qualified value — only data-qualified LHS is supported here (submodel kwargs typically refer to caller-provided data).")
        stan.cv(rhs) && (info[name] = remake(info[name]; cv=true))
    else
        autotype = stan.autotype(rhs)
        cv = stan.cv(autotype) || stan.cv(rhs)
        qual = cv ? :quantities : :parameter
        info[name] = StanExpr(name, remake(autotype; qual, cv))
    end
    remake(x, info[name], rhs)
end
forward!(x::SamplingExpr{Symbol,<:SlicModel}; info) = begin
    name, rhs = x.args
    forward!(rhs; info=SubModel(info, name, Dict()))
end
# A native constrained center type carries an implicit Stan constraint
# transform (simplex/ordered/positive_ordered — proper subtypes of `vector` —
# and cov_matrix/corr_matrix/cholesky_factor_* — subtypes of `square_matrix`).
# Plain `vector`/`row_vector`/`matrix`/`int`/… are NOT native-constrained.
_is_native_constrained_ct(T) = T isa Type && (
    (T <: types.vector && T !== types.vector) || T <: types.square_matrix
)
forward!(x::SamplingExpr{<:DeclExpr}; info) = begin
    decl, rhs_raw = x.args
    name = decl.args[1]
    name isa Symbol || error(
        "Typed-LHS sampling currently requires a Symbol LHS, got `$name`. ",
        "For more complex LHS shapes, use a bare assignment + sampling pair."
    )
    type_expr = decl.args[2]
    ct, sizes... = if _is_getindex_expr(type_expr)
        type_expr.args
    else
        (type_expr,)
    end
    ct isa Symbol || error("Typed-LHS sampling: type center must be a Symbol, got `$ct`")
    ct_resolved = gettype(ct)
    sizes_forwarded = isempty(sizes) ? () : Tuple(forward!(collect(sizes); info))
    # Stan requires every native constrained-type size (`simplex[N]`, `ordered[N]`,
    # …) to be a SCALAR int. A NON-scalar size — e.g. `p::simplex[Ks]` with a data
    # int-vector `Ks` — is a RAGGED / varying-per-group constrained parameter. Stan
    # cannot declare `simplex[Ks]` natively, so we desugar: a flat improper-uniform
    # free param + a compiler-injected per-group constrain loop (`simplex_jacobian`)
    # + a `RaggedVector` pairing, reusing the loop-fill routing landed on
    # `slic-model-slice-b3a85769` @ 29c3b59. See brief 2026-07-15T18-26-44-152-1b6sc6m.
    if _is_native_constrained_ct(ct_resolved) && any(sz -> stan_ndim(type(sz)) > 0, sizes_forwarded)
        return _forward_ragged_constrained!(name, ct, collect(sizes), rhs_raw; info)
    end
    _is_canonical_expr(rhs_raw) || error(
        "Typed-LHS sampling `$name::$(pretty_type_expr(type_expr)) ~ rhs` requires rhs to be a distribution call"
    )
    head_resolved = forward!(head(rhs_raw); info)
    args_resolved = collect(forward!(rhs_raw.args; info))
    kwargs_resolved = forward!(rhs_raw.kwargs; info)
    rhs_canonical = CanonicalExpr(head_resolved, args_resolved...; kwargs_resolved...)
    # Fold the RHS distribution's constraints into the DECLARED type: the
    # explicit `lower=`/`upper=`/`offset=`/`multiplier=` kwargs AND the
    # distribution-implied ones (`exponential`→lower=0, `beta`→[0,1], …). The
    # bare-LHS path does exactly this via `autotype(rhs)` (functions.jl §219);
    # the typed-LHS path historically built the decl type from center+sizes
    # ONLY, silently dropping the constraint — so a positive scale
    # `tau::vector[K] ~ normal(0,1; lower=0)` emitted an UNCONSTRAINED
    # `vector[K] tau`. Native-constrained centers (simplex/cholesky/…) carry no
    # bound autokwargs and take no user bounds, so `cons` is empty for them.
    cons_kw = merge(autokwargs(rhs_canonical), (; kwargs_resolved...))
    cons = (;[
        key => getindex(cons_kw, key)
        for key in (:lower, :upper, :offset, :multiplier) if key in keys(cons_kw)
    ]...)
    base_lhs_type = StanType(ct_resolved, sizes_forwarded; cons...)
    cv_args = any(stan.cv, args_resolved)
    qual = cv_args ? :quantities : :parameter
    info[name] = StanExpr(name, remake(base_lhs_type; qual, cv=cv_args))
    rhs_stan = StanExpr(rhs_canonical, info[name].type)
    remake(x, info[name], rhs_stan)
end
# Desugar a RAGGED native-constrained parameter (`p::simplex[Ks] ~ …`, `Ks` a data
# int-vector of per-group dims) into constructs SB can emit. Stan cannot declare
# `simplex[Ks]` natively, so we replicate what Stan does for a native simplex —
# unconstrained free coords + the constrain transform's jacobian — but per group:
#   • flat improper-uniform free param   `p_free ~ flat(;n=sum(Ks .- 1))`
#   • per-group offsets in transformed data via `cumulative_sum` (data-qualified)
#   • a FRESH result vector filled by a compiler-injected per-group constrain loop
#     calling the built-in `<ct>_jacobian` (the jacobian accumulates directly in
#     `transformed parameters`; routing landed on slic-model-slice @ 29c3b59)
#   • a `RaggedVector(mem, ends)` pairing bound to `name`
# All statements are injected via `_slic_inline_pending` (they never enter the raw
# body, so they bypass `_reject_model_control_flow`) and routed to their blocks by
# `distribute!`. NOTE (scope decision 1mfltua): the RHS informative prior is NOT yet
# applied — `p[g] ~ dist` is a `~`-in-loop = the Feature-2 `~`-aware superset, not
# built. For now the ragged param carries only its uniform base measure (jacobian);
# `rhs_raw` is intentionally unused pending 1mfltua.
_forward_ragged_constrained!(name, ct, sizes, rhs_raw; info) = begin
    length(sizes) == 1 || error(
        "Ragged constrained `$name::$ct[…]`: expected a single vector size (the ",
        "per-group dims), got $(length(sizes)). Rectangular `$ct[a,b]` and ragged ",
        "`$ct[Ks]` are distinct shapes; a mixed form is unsupported."
    )
    Ks  = sizes[1]
    jac = Symbol(ct, :_jacobian)
    # Stan-valid unique names (gensym's `#` is an illegal Stan identifier char);
    # `_next_inline_id` is the per-trace counter SB uses for inlined-local renames.
    id = _next_inline_id()
    pfree = Symbol(:p_free, "__rc_", id); pmem = Symbol(:p_mem, "__rc_", id)
    cend  = Symbol(:c_end, "__rc_", id);  fend = Symbol(:f_end, "__rc_", id)
    g     = Symbol(:g, "__rc_", id)
    stmts = Any[
        :($pfree ~ flat(; n = sum($Ks .- 1))),          # improper-uniform free coords → parameters
        :($cend = cumulative_sum($Ks)),                  # constrained group ends (data → TD)
        :($fend = cumulative_sum($Ks .- 1)),             # free group ends (data → TD)
        :($pmem :: vector[sum($Ks)]),                    # fresh result → auto fresh_decl cert
        :(for $g in 1:length($Ks)                        # injected ForExpr — 29c3b59 routes it
              $pmem[$cend[$g] - $Ks[$g] + 1 : $cend[$g]] =
                  $jac($pfree[$fend[$g] - $Ks[$g] + 2 : $fend[$g]])
          end),
        :($name = RaggedVector($pmem, $cend)),           # ragged view (representation bae1167)
    ]
    _do_retrace_inline_body(stmts; info)
end
# Indexed sampling is the plate producer's declaration certificate.  The
# producer first emits an outer-sized fresh declaration, then rewrites a
# cell-local `x ~ dist` to `x[i] ~ dist` inside its compiler-owned loop.  The
# first such use classifies the declaration as a parameter; indexed data LHSs
# remain ordinary observations.  An indexed parameter that did NOT come from a
# fresh declaration stays out of this internal path.
forward!(x::SamplingExpr{<:CanonicalExprV{:getindex}}; info) = begin
    lhs_raw, rhs_raw = x.args
    k = _base_lhs_symbol(lhs_raw)
    k in keys(info) || error(
        "Plate-generated indexed sampling of `", k, "[…]` requires the plate emitter to register its outer declaration first."
    )
    lhs = forward!(lhs_raw; info)
    rhs = forward!(rhs_raw; info)::StanExpr
    base = info[k]
    if _is_fresh_decl(base)
        role = _decl_role(base)
        role == :fill && error(
            "Cannot sample `", k, "[…]` after the fresh declaration was classified as a transformed fill target."
        )
        role in (:unfilled, :sampled) || error(
            "Fresh declaration `", k, "` has unknown declaration role `", role, "`."
        )
        role == :unfilled && (info[k] = remake(base; decl_role=:sampled, qual=:parameter))
    else
        q = qual(base)
        q == :data || error(
            "Indexed sampling of `", k, "[…]` has a ", q,
            "-qualified base that is not a compiler-generated fresh plate declaration."
        )
        cv(rhs) && (info[k] = remake(base; cv=true))
    end
    remake(x, lhs, rhs)
end
# --- Public `plate` do-block emitter (Feature 2, surface decision n35u3c) -----
# Lowers `rv ~ plate(iter1, …; outer=(N,)) do a1, …; body…; cell_output; end`
# into the plate PRODUCER CONTRACT that the `~`-aware routing (landed on
# slic-model-slice-b3a85769) consumes: outer `DeclExpr`s + a compiler-injected
# certified `ForExpr` with indexed fresh samples, indexed observations, and an
# indexed return fill — injected via `_slic_inline_pending` (so they bypass
# `_reject_model_control_flow`, same as `_forward_ragged_constrained!`).
#
# Semantics (n35u3c): positional iterables are PER-CELL slices bound to the
# do-block params (`a_k` ⇒ `iter_k[i]`); lexical captures stay SHARED; each
# fresh `~`/`=` LHS in the body is promoted to an outer array indexed by the
# loop var; the trailing expression is each cell's output, filled into `rv`.
# MVP scope (surface these as limitations): 1-D `outer=(N,)` only; scalar-per-
# cell fresh vars (⇒ `vector[N]` decls); no vararg params; the body must end in
# a value expression; fresh/param names must not collide with function names
# (uniform symbol substitution).
# (name, per_cell_type) for a plate body statement, or nothing. `z::T ~ dist` /
# `z::T = expr` ⇒ (z, T); `t ~ dist` / `w = expr` ⇒ (t, nothing).
_plate_fresh_info(s) = begin
    lhs = if s isa Expr && s.head === :call && length(s.args) >= 3 && s.args[1] === :~
        s.args[2]                         # `lhs ~ dist`  (raw `~` ⇒ Expr(:call, :~, lhs, rhs))
    elseif s isa Expr && s.head === :(=)
        s.args[1]                         # `lhs = rhs`
    else
        return nothing
    end
    lhs isa Symbol && return (lhs, nothing)
    (lhs isa Expr && lhs.head === :(::) && lhs.args[1] isa Symbol) && return (lhs.args[1], lhs.args[2])
    nothing
end
# Strip a typed-LHS annotation on a plate body stmt's LHS: `z::T ~ dist` ⇒
# `z ~ dist` (the per-cell type is captured separately for the outer decl; the
# indexed form `z[idx]`/`z[:,idx]` carries its type from the decl).
_strip_plate_decl(s) =
    if s isa Expr && s.head === :call && length(s.args) >= 3 && s.args[1] === :~ &&
            s.args[2] isa Expr && s.args[2].head === :(::)
        Expr(:call, :~, s.args[2].args[1], s.args[3:end]...)
    elseif s isa Expr && s.head === :(=) && s.args[1] isa Expr && s.args[1].head === :(::)
        Expr(:(=), s.args[1].args[1], s.args[2:end]...)
    else
        s
    end
# Uniform symbol substitution over a raw body AST. Only names in `m` are
# rewritten; dist/function names + captures are absent from `m`, so they pass
# through untouched. LHS symbols are rewritten too (`t ~ …` ⇒ `t[i] ~ …`).
_subst_syms(x, m) = x
_subst_syms(x::Symbol, m) = get(m, x, x)
_subst_syms(x::Expr, m) = Expr(x.head, Any[_subst_syms(a, m) for a in x.args]...)

# `vector[K]` per-cell size K, from a raw `:ref` (body decls) or canonical
# `getindex` (typed plate LHS) type expr; nothing for scalar/unsupported.
_plate_vector_size(ct) =
    if ct isa Expr && ct.head === :ref && length(ct.args) == 2 && ct.args[1] === :vector
        ct.args[2]
    elseif ct isa CanonicalExprV{:getindex} && length(ct.args) == 2 && ct.args[1] === :vector
        ct.args[2]
    else
        nothing
    end
# Outer collection decl + per-cell accessor for a var of per-cell type `ct` over
# N cells. Scalar (ct===nothing) ⇒ `vector[N]` / `f[idx]`; `vector[K]` ⇒
# `matrix[K, N]` / `f[:, idx]` (a column per cell — F's routing keys on the base
# symbol, so a column-slice sample/fill routes like a single-index one).
_plate_outer_decl(f, ct, N) = begin
    ct === nothing && return :($f :: vector[$N])
    K = _plate_vector_size(ct)
    K === nothing && error("plate: unsupported per-cell type `$ct` for `$f` — scalar or `vector[K]` only (MVP).")
    :($f :: matrix[$K, $N])
end
_plate_cell_index(f, ct, idx) = ct === nothing ? :($f[$idx]) : :($f[:, $idx])

# Per-cell StanType of a fresh var, for the result-inference probe: an annotated
# `z::vector[K]` ⇒ `vector[K]`; a bare scalar fresh ⇒ `real` (the MVP's scalar
# assumption). Used only to TYPE the trailing expression, never emitted.
_plate_fresh_cell_type(ct; info) = begin
    ct === nothing && return StanType(types.real)
    K = _plate_vector_size(ct)
    K === nothing && error("plate: unsupported fresh per-cell type `$ct` for result inference.")
    StanType(types.vector, (forward!(K; info),))
end
# FULL AUTO TYPE INFERENCE (goal §5): infer a BARE plate result's per-cell shape
# from the do-block trailing expression, so `b ~ plate(...)` needs no
# `b::vector[K]` annotation. Temporarily binds the per-cell names (loop index,
# do-block params at their slice types, fresh vars at their per-cell types) into
# `info`, then a pure-expression `forward!` of the trailing expression computes
# its `tracetype`; captures are already in `info`. Pops every temp binding (the
# `forward!(::ForExpr)` add-index/pop idiom) so the real retrace sees an
# unperturbed scope. Returns a synthetic per-cell type expr for the existing
# `_plate_outer_decl`/`_plate_cell_index`: `nothing` (scalar ⇒ `vector[N]`) or
# `:(vector[K])` (⇒ `matrix[K, N]`).
_infer_plate_rv_ct(ret_expr, fresh, params, iterables, idx; info) = begin
    bound = Symbol[]
    _bind!(nm, ty) = (info[nm] = StanExpr(nm, ty); push!(bound, nm))
    try
        _bind!(idx, StanType(types.int; qual=:data))
        if isempty(iterables) && length(params) == 1
            _bind!(params[1], StanType(types.int; qual=:data))               # `do i` ⇒ cell index
        else
            for (a, it) in zip(params, iterables)
                _bind!(a, type(forward!(canonical(:($it[$idx])); info)))     # per-cell slice
            end
        end
        for (f, ct) in fresh
            _bind!(f, _plate_fresh_cell_type(ct; info))
        end
        T = type(forward!(canonical(ret_expr); info))
        stan_ndim(T) == 0 && return nothing
        (center_type(T) <: types.vector && stan_ndim(T) == 1) &&
            return :(vector[$(expr(stan_size(T)[1]))])
        error("plate: inferred cell-output type has ndim $(stan_ndim(T)); only scalar or vector[K] supported for now.")
    finally
        for nm in bound
            pop!(info, nm)
        end
    end
end

forward!(x::SamplingExpr{Symbol,<:CanonicalExprV{:plate}}; info) =
    _forward_plate!(x.args[1], nothing, x.args[2]; info)
# Typed-LHS plate result `b::vector[K] ~ plate(…)` ⇒ vector cell output collected
# as `matrix[K, N]`. The DeclExpr LHS carries the per-cell result type.
forward!(x::SamplingExpr{<:DeclExpr,<:CanonicalExprV{:plate}}; info) = begin
    decl = x.args[1]
    (decl.args[1] isa Symbol) || error("plate: typed-LHS result must name a Symbol, got `$(decl.args[1])`.")
    _forward_plate!(decl.args[1], decl.args[2], x.args[2]; info)
end

_forward_plate!(rv::Symbol, rv_ct, plate; info) = begin
    lambda = plate.args[1]
    lambda isa CanonicalExprV{:->} || error(
        "plate: `$rv ~ plate(…)` requires a `do … end` block, got `$(typeof(lambda))`."
    )
    lhs_raw, body_raw = lambda.args
    (body_raw isa Expr && body_raw.head === :block) || error(
        "plate: the do-block body must be a `:block` Expr, got `$(typeof(body_raw))`."
    )
    params, vararg = _parse_lambda_lhs(lhs_raw)
    vararg === nothing || error("plate: vararg do-block params (`args...`) are not supported yet.")
    iterables = collect(plate.args[2:end])

    # Plate size N: explicit `outer=(N,)` (1-D MVP) wins; else the first iterable's length.
    outer = get(plate.kwargs, :outer, nothing)
    N = if outer !== nothing
        (outer isa CanonicalExprV{:tuple} && length(outer.args) == 1) || error(
            "plate: only a 1-D `outer=(N,)` is supported for now."
        )
        outer.args[1]
    elseif !isempty(iterables)
        :(length($(iterables[1])))
    else
        error("plate: cannot size the plate — pass `outer=(N,)` or at least one iterable.")
    end

    (length(params) == length(iterables) || (isempty(iterables) && length(params) == 1)) || error(
        "plate: $(length(params)) do-block params vs $(length(iterables)) positional iterables — ",
        "positional args are per-cell slices and must match 1:1, or use a single `do i` (no iterables)."
    )

    id  = _next_inline_id()
    idx = Symbol(:plate_i, "__pl_", id)
    subst = Dict{Symbol,Any}()
    if isempty(iterables) && length(params) == 1
        subst[params[1]] = idx                          # `do i` ⇒ the cell index
    else
        for (a, it) in zip(params, iterables)
            subst[a] = :($it[$idx])                      # per-cell slice
        end
    end

    stmts = Any[s for s in body_raw.args if !(s isa LineNumberNode)]
    isempty(stmts) && error("plate: empty do-block body.")
    body_stmts, ret_expr = stmts[1:end-1], stmts[end]
    _plate_fresh_info(ret_expr) === nothing || error(
        "plate: the do-block must END with a cell-output VALUE expression, not a `~`/`=` statement."
    )

    # Fresh per-cell vars WITH their per-cell types (scalar or `vector[K]`).
    fresh = Tuple{Symbol,Any}[]
    for s in body_stmts
        fi = _plate_fresh_info(s)
        fi === nothing && continue
        name, ct = fi
        (name in params || any(fc -> fc[1] === name, fresh)) && continue
        push!(fresh, (name, ct))
    end
    for (f, ct) in fresh
        subst[f] = _plate_cell_index(f, ct, idx)         # scalar ⇒ f[idx]; vector[K] ⇒ f[:, idx]
    end

    # Full auto type inference (goal §5): a BARE `b ~ plate(...)` (no `b::vector[K]`)
    # gets its result shape from the do-block trailing expression. A typed-LHS
    # `b::vector[K]` (rv_ct given) stays an explicit override.
    rv_ct === nothing && (rv_ct = _infer_plate_rv_ct(ret_expr, fresh, params, iterables, idx; info))

    # Strip typed-LHS annotations, then substitute names ⇒ per-cell accessors.
    loop_body = Any[_subst_syms(_strip_plate_decl(s), subst) for s in body_stmts]
    push!(loop_body, :($(_plate_cell_index(rv, rv_ct, idx)) = $(_subst_syms(ret_expr, subst))))

    injected = Any[]
    for (f, ct) in fresh
        push!(injected, _plate_outer_decl(f, ct, N))
    end
    push!(injected, _plate_outer_decl(rv, rv_ct, N))
    # NB: build the iteration spec as `Expr(:(=), idx, 1:N)` — a hand-built
    # `Expr(:for, :(idx in 1:N), …)` yields an `:in` CALL spec, not the `:(=)`
    # form `forward!(::ForExpr)` asserts (a quoted `for` auto-normalizes it).
    push!(injected, Expr(:for, Expr(:(=), idx, :(1:$N)), Expr(:block, loop_body...)))
    _do_retrace_inline_body(injected; info)
end
forward!(x::SamplingExpr; info) = begin
    lhs, rhs = forward!(x.args; info)
    forward!(remake(x, lhs, rhs::StanExpr); info)
end
forward!(x::SamplingExpr{<:Any,<:StanExpr}; info) = begin
    lhs, rhs = x.args 
    @assert stan.qual(lhs) == :data
    remake(x, lhs, rhs)
end
forward!(x::ReturnExpr; info) = _forward_return!(x, info)
_forward_return!(x::ReturnExpr, info::SubModel) =
    forward!(CanonicalExpr(:(=), name(info), forward!(x.args[1]; info)); info=parent(info))
_forward_return!(x::ReturnExpr, info::StanModel) =
    forward!(CanonicalExpr(:(=), :MODEL_RV, forward!(x.args[1]; info)); info)
_forward_return!(x::ReturnExpr, info) = let rv = forward!(x.args[1]; info)
    info[RV_NAME] = rv
    remake(x, rv)
end
forward!(x::DocumentExpr; info) = remake(x, forward!(x.args; info)...)
forward!(x::TupleExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::KwExpr; info) = stan_expr(remake(x, x.args[1], forward!(x.args[2]; info)))
forward!(x::NamedTupleExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::GetPropertyExpr; info) = begin
    @assert length(x.args) == 2
    obj, name = forward!(x.args; info)
    @assert _is_ntup_stan_expr(obj) "Trying to access property `$name` of object of type without named properties ($(type(obj)))!"
    names = keys(obj.type.info.arg_types)
    @assert name in names
    # Field access lowers to `Base.getfield(obj, position)` so it stays
    # *distinct* from user-defined `Base.getindex(::usertype, ::int)`.
    # Both end up routing to "obj.N" Stan-side via specialised rules
    # below — but the tracetype and method-dispatch lanes don't conflict.
    return forward!(CanonicalExpr(:getfield, x.args[1], findfirst(==(name), names)); info)
end
forward!(x::BracesExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::VectExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::DeclExpr; info) = begin
    @assert length(x.args) == 2
    lhs, type_ann = x.args
    ct, s... = if _is_getindex_expr(type_ann)
        type_ann.args
    else
        (type_ann, )
    end
    t = if ct isa Symbol
        StanType(gettype(ct), forward!.(s; info))
    else
        # Computed type annotation (`typeof(...)` / `return_type(...)`, optionally
        # `[dims]`-sized). Forward the base to a `tokenof{CT}` token, take its
        # center type CT, and pick the NATURAL container for the given dims via
        # `autotype` (real→vector, int→array[] int — matching jbroadcasted's
        # inference). Explicit `[dims]` override the token's own size; otherwise
        # the token carries the size (e.g. `typeof(some_vector)`).
        tok = forward!(ct; info)
        _decl_computed_type(tok, s; info)
    end
    # Fresh-declaration cert marker (Feature-1/plate): a SYMBOL-typed model
    # declaration is provisionally a flat-prior parameter; its first certified
    # indexed use dynamically selects the final role (`:fill` resets the qual
    # from its RHS; `:sampled` stays a parameter) — decision `1dd0eww`, which
    # lets one compiler-injected decl serve both inline fills and plate params.
    # A COMPUTED-type decl (`::typeof(...)`/`return_type`, merged from devibe)
    # keeps its natural qual from `_decl_computed_type` — no fresh-decl override.
    ct isa Symbol && (t = remake(t; fresh_decl=true, decl_role=:unfilled, qual=:parameter))
    rv = StanExpr(lhs, t)
    lhs isa Symbol || return StanExpr(expr(forward!(lhs; info)), t)
    info[lhs] = rv
    # `SubModel.setindex!` flattens the symbol into the parent. Put that stored
    # value into the declaration AST too, so later backward/distribution lookup
    # uses the same name as the parent model's `info` key.
    stan_expr(remake(x, info[lhs]))
end
# `types` is defined in functions.jl (included AFTER this file), so the token
# check lives in the body (resolved at trace time), not the signature.
_decl_computed_type(tok, s; info) = begin
    tt = type(tok)
    center_type(tt) <: types.tokenof || error(
        "type-annotation expression must evaluate to a type token (e.g. `typeof(...)` ",
        "/ `return_type(...)`), got a value of Stan type `$(sigtype(tt))`."
    )
    cct = tt.info.value
    sz = isempty(s) ? stan_size(tt) : Tuple(forward!.(s; info))
    autotype(StanType(cct, sz))
end
forward!(x::ForExpr; info) = begin
    @assert length(x.args) == 2
    head, body = x.args
    @assert _is_assign_canonical(head)
    @assert _is_block_canonical(body)
    idx = head.args[1]
    @assert idx isa Symbol
    # `:data`-qualify the loop index: it's a deterministic integer. Without an
    # explicit qual it defaults to `:undefined`, which is lexicographically the
    # LARGEST qual symbol and so poisons `stan_expr`'s `maximum(qual, args)` for any
    # body expression referencing the index (`x[i]` → `max(:parameter, :undefined)`
    # → `:undefined`), breaking qual-promotion of a model-body loop's fill target.
    # Irrelevant to forward-only UDF bodies (no qual routing), so no regression there.
    info[idx] = StanExpr(idx, StanType(types.int; qual=:data))
    # A SubModel stores the index under a flattened parent name. The loop head
    # must declare that SAME emitted symbol used by forwarded body references;
    # keeping the raw local `idx` would produce `for(i ...) body[prefix_i]`.
    emitted_idx = expr(info[idx])
    idx_range = forward!(head.args[2]; info)
    body = forward!(body; info)
    pop!(info, idx)
    stan_expr(remake(x, remake(head, emitted_idx, idx_range), body))
end
forward!(x::WhileExpr; info) = begin
    @assert length(x.args) == 2
    head, body = x.args
    @assert _is_block_canonical(body)
    # body = forward!(body; info)
    stan_expr(remake(x, forward!(x.args; info)...))
end
forward!(x::IfExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::ElseIfExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::BreakExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::ContinueExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::QuoteExpr; info) = x.args[1]
forward!(x::StringExpr; info) = join(map(stan_code, forward!(x.args; info)))
