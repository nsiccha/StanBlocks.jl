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
# Guardrail against the silent `@deffun`/`@builtin_module` drift. A bare-symbol
# `@deffun name(...)` registered inside StanBlocks's own builtin context emits
# its dispatch stub (`function name end`) into the *StanBlocks* module — which
# symbol resolution never consults (`forward!(::Symbol)` checks info → builtin →
# user-model-module → Main). Such a name is therefore reachable from a user
# `@slic` body ONLY if it is ALSO bound in the `builtin` submodule via the
# `@builtin_module` manifest. The two registrations are separate by design
# (centralising the module declaration keeps Revise happy — see the GLM note in
# builtin.jl), so a name added to `@deffun` but not the manifest fails silently
# at the user's call site ("Could not find <name> in model, builtin, … or
# Main!"). This turns that drift into a loud LOAD-TIME error naming the missing
# manifest entry. Scoped to `def_mod === @__MODULE__` (StanBlocks): a user-side
# `@deffun` emits its stub into the *user's* module, which IS `get_module(info)`
# at resolution, so user UDFs resolve without a manifest entry and are not flagged.
_assert_builtin_registered(name::Symbol, def_mod::Module) = begin
    def_mod === (@__MODULE__) || return nothing
    _is_builtin_name(name) && return nothing
    error(
        "@deffun registered `", name, "` as a StanBlocks builtin, but it is not in ",
        "the `@builtin_module` manifest (src/slic_stan/builtin.jl): `_is_builtin_name(:",
        name, ")` is false, so a user `@slic` body cannot resolve it. Add `", name,
        "` to the `@builtin_module [ … ]` list."
    )
end
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
    # call-site value or the registered default. A kwarg declared without a
    # default (sentinel `missing`) is required — omitting it at the call site
    # errors. Unknown call-site kwargs are an error too — there's no schema to
    # dispatch them to.
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
            elseif default_expr === missing
                # Required kwarg (declared without a default) omitted at the
                # call site — the SLIC analogue of Julia's `UndefKeywordError`.
                error(
                    "@inline kwcall shim for `$(head(x))`: required keyword argument `$kn` ",
                    "was not provided (it has no default). Pass `$kn=…` at the call site."
                )
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

# Ragged informative priors are emitted as compiler-owned indexed sampling
# statements over the constrained view.  Keep the certificate task-local: an
# ordinary user-authored `derived[i] ~ dist(...)` remains rejected by the
# generic already-bound-LHS rule, while the lowering below may deliberately add
# density to the transformed ragged value.
_ragged_density_targets() =
    get(task_local_storage(), :_slic_ragged_density_targets, ())
_is_ragged_density_target(name::Symbol) = name in _ragged_density_targets()

# Trace-time marker used only while rewriting distribution arguments for one
# ragged group.  `builtin.jl` supplies the type-aware expansion after the
# RaggedVector/RaggedMatrix usertypes exist: ragged args become `arg[g]`, while
# shared scalars/dense values pass through unchanged.
function _ragged_group_arg end
_ragged_group_rhs(rhs::CanonicalExpr, g) = begin
    args = map(a -> CanonicalExpr(_ragged_group_arg, a, g), rhs.args)
    CanonicalExpr(head(rhs), args...; rhs.kwargs...)
end
_ragged_rhs_is_flat(rhs) = _is_canonical_expr(rhs) && head(rhs) === :flat &&
    isempty(rhs.args) && isempty(rhs.kwargs)

# A plate's second trace runs inside the real model scope, but every discovered
# cell-local binding is represented there by an outer array.  Keep the mapping
# task-local so normal tracing stays untouched and submodels can participate:
# their local `z` is mapped to the flattened outer key (`t_z`) before indexing.
_plate_context() = get(task_local_storage(), :_slic_plate_context, nothing)
_plate_root_info(info::StanModel) = info
_plate_root_info(info::SubModel) = _plate_root_info(parent(info))
_plate_global_name(::StanModel, name::Symbol) = name
_plate_global_name(info::SubModel, name::Symbol) =
    _plate_global_name(parent(info), supname(info, name))
_plate_context_entry(name::Symbol; info) = begin
    ctx = _plate_context()
    ctx === nothing && return nothing
    global_name = _plate_global_name(info, name)
    haskey(ctx.cell_types, global_name) || return nothing
    accessor = get(ctx.cell_accessors, global_name, nothing)
    (global_name=global_name, cell_type=ctx.cell_types[global_name], idxs=ctx.idxs, accessor)
end
_plate_promoted_lhs(name::Symbol; info) = begin
    entry = _plate_context_entry(name; info)
    entry === nothing && return nothing
    canonical(entry.accessor === nothing ?
        _plate_cell_index(entry.global_name, entry.cell_type, entry.idxs) : entry.accessor)
end
_plate_promoted_reference(name::Symbol, info::Union{StanModel,SubModel}) = begin
    lhs = _plate_promoted_lhs(name; info)
    lhs === nothing && return nothing
    # Resolve the indexed expression against the top-level outer declaration.
    # Disable the context for this one lookup so resolving its base Symbol does
    # not recursively promote itself.
    task_local_storage(:_slic_plate_context, nothing) do
        forward!(lhs; info=_plate_root_info(info))
    end
end
_plate_bind_local!(::StanModel, _name, _value) = nothing
_plate_bind_local!(info::SubModel, name, value) = (locals(info)[name] = value)

_forward_plate_assignment!(x, name, lhs, rhs_raw; info) = begin
    rhs = forward!(rhs_raw; info)
    _check_assignment_rhs(name, rhs)
    center_type(rhs) === types.void && error(
        "`$name = <void-call>(...)`: cannot bind the return value of a void UDF. ",
        "Drop the `$name = ` and call as a statement."
    )
    # `lhs` is already the promoted outer accessor. Suppress promotion while
    # forwarding it, otherwise its base Symbol becomes indexed a second time.
    fwd = task_local_storage(:_slic_plate_context, nothing) do
        forward!(remake(x, lhs, rhs); info=_plate_root_info(info))
    end
    _plate_bind_local!(info, name, expr(fwd).args[1])
    fwd
end

_forward_plate_sampling!(x, name, lhs, rhs::StanExpr; info) = begin
    fwd = task_local_storage(:_slic_plate_context, nothing) do
        forward!(remake(x, lhs, rhs); info=_plate_root_info(info))
    end
    _plate_bind_local!(info, name, fwd.args[1])
    fwd
end

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
        if _plate_context() !== nothing && resolved isa BlockExpr
            # A submodel embedded inside a plate is an inline expansion. The
            # loop router consumes a flat statement list, so splice the traced
            # submodel block here rather than leaving a nested BlockExpr.
            append!(new_args, resolved.args)
        elseif !_is_inert_block_stmt(resolved)
            push!(new_args, resolved)
        end
    end
    remake(x, new_args...)
end
_is_inert_block_stmt(x) = false
_is_inert_block_stmt(x::StanExpr) = _is_inert_expr(expr(x))
_is_inert_expr(::Symbol) = true
_is_inert_expr(::Number) = true
_is_inert_expr(::AbstractString) = true
_is_inert_expr(_) = false
_is_model_decl_scope(::Union{StanModel,SubModel}) = true
_is_model_decl_scope(_) = false
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

# A bounded one-dimensional comprehension in an `@deffun` body is syntax sugar
# for the local declaration + indexed-fill loop authors otherwise write by hand:
#
#     [f(i) for i in lo:hi]
#
# becomes (using a per-trace fresh result name):
#
#     result::typeof(f(lo))[hi - lo + 1]
#     for i in lo:hi
#         result[i - lo + 1] = f(i)
#     end
#     result
#
# Keep the accepted surface intentionally narrow.  A fixed `lo:hi` range gives
# both Stan's loop bounds and the output size without inventing collection
# iteration semantics; scalar elements keep the output one-dimensional.  The
# dedicated errors below prevent Julia's richer generator/filter ASTs from
# falling into the generic `forward!` error or, worse, being partly lowered.
_contains_comprehension(::Any) = false
_contains_comprehension(::ComprehensionExpr) = true
_contains_comprehension(x::CanonicalExpr) = any(_contains_comprehension, x.args)

# Parse one comprehension's generator into its value expression and the list of
# `(binding, source)` iteration specs. A single symbol-bound `lo:hi` generator is
# the base case; multi-generator (`for i in …, j in …`) forms yield several specs
# (N-D result), and a tuple binding (`(i, xi)`) marks an enumerate/zip source.
function _comprehension_generator_specs(x::ComprehensionExpr)
    length(x.args) == 1 || error(
        "@deffun array comprehension: malformed comprehension `", x, "`."
    )
    # Chained/flattened generators `[expr for i in … for j in …]` produce a
    # ragged/flattened 1-D result (its length depends on the inner ranges). That
    # ragged-result path is a deferred follow-up; reject it specifically rather than
    # via the generic "expected one generator" error.
    x.args[1] isa CanonicalExprV{:flatten} && error(
        "@deffun array comprehension: chained/flattened generators ",
        "`[expr for i in … for j in …]` are not supported — they build a ragged/flattened ",
        "result. Use the product form `[expr for i in …, j in …]` for a rectangular N-D ",
        "result, or write explicit nested loops (or `plate`) to assemble a ragged result."
    )
    x.args[1] isa GeneratorExpr || error(
        "@deffun array comprehension: expected exactly one generator, got `", x.args[1], "`. ",
        "Only `[expr for i in lo:hi]` (and its enumerate/zip/N-D forms) is supported."
    )
    generator = x.args[1]
    length(generator.args) >= 2 || error(
        "@deffun array comprehension: malformed generator `", generator, "`."
    )
    value = generator.args[1]
    _contains_comprehension(value) && error(
        "@deffun array comprehension: nested comprehensions are not supported. ",
        "Write the nested loops explicitly."
    )
    specs = Tuple{Any,Any}[]
    for iteration in generator.args[2:end]
        iteration isa FilterExpr && error(
            "@deffun array comprehension: filtered generators are not supported. ",
            "Write an explicit loop when the output length depends on a condition."
        )
        iteration isa AssignmentExpr || error(
            "@deffun array comprehension: expected a generator binding `i = lo:hi`, got `", iteration, "`."
        )
        idx, range = iteration.args
        (idx isa Symbol || idx isa TupleExpr) || error(
            "@deffun array comprehension: the generator must bind one Symbol or a `(a, b, …)` tuple, got `", idx, "`."
        )
        push!(specs, (idx, range))
    end
    value, specs
end

_is_literal_one(x) = x == 1
_is_literal_one(x::StanExpr) = _is_literal_one(expr(x))

# ── Value-iteration (`for xi in <container>` / `[expr for xi in <container>]`) ──
# A `lo:hi` iteration source forwards to a 2-arg `CanonicalExpr{Colon}` and takes
# the index-iteration path. Any OTHER indexable container (vector/array, or a
# RaggedVector/RaggedMatrix/EachCol/EachRow view usertype) desugars to
# index-iteration binding the user's loop variable to the element:
#     for xi in c          →  for _vi in 1:length(c); xi = c[_vi]; <body>
#     [f(xi) for xi in c]  →  [f(c[_vi]) for _vi in 1:length(c)]
# `length(c)` and `c[_vi]` already dispatch per container type, so no per-type
# loop logic lives here — the same desugar covers every supported container.
_is_bounded_colon(x) =
    x isa StanExpr && expr(x) isa CanonicalExpr{Colon} && length(expr(x).args) == 2
# A Colon-headed range is NEVER value-iterable: a 2-arg `lo:hi` takes the index
# path, and any other arity is a stepped range (`lo:step:hi`) — unsupported, and
# must reject rather than be mistaken for a container (it has ndim ≥ 1).
_is_value_iterable(x::StanExpr) =
    !(expr(x) isa CanonicalExpr{Colon}) &&
    (stan_ndim(type(x)) >= 1 || center_type(type(x)) <: types.usertype)
_is_value_iterable(::Any) = false

function _fresh_value_index(info)
    id = _next_inline_id()
    vi = Symbol(:value_index__vi_, id)
    while vi in keys(info)
        id = _next_inline_id()
        vi = Symbol(:value_index__vi_, id)
    end
    vi
end

# Loop bound: `1:length(container)`. `length` dispatches per container type —
# `num_elements(x)` for vectors/arrays, `cols(X)` / `num_elements(ends)` for the
# view usertypes — so one form covers every supported container.
#
# LIMITATION (parameter-sized results): when the container is a UDF *parameter*
# arg, `length` emits `num_elements(x)`, which Stan rejects as a non-data size in
# the model-side result declaration (`transformed parameters { vector[num_elements(x)] z; }`).
# The equivalent index form `[f(x[i]) for i in 1:n]` avoids this only because the
# user names the signature dim `n`, which the UDF binds as `int n = dims(x)[1]`
# and the return type tracks to the arg's concrete size; a value generator never
# mentions `n`, so functions.jl's size-requirement analysis leaves it unbound and
# unavailable here. Recovering it would couple this desugar to that macro-time
# analysis. Value-iteration over DATA containers (the common case) is unaffected.
_value_iter_count(container) = CanonicalExpr(Colon(), 1, CanonicalExpr(length, container))
_value_iter_elem(container, vi) = CanonicalExpr(getindex, container, vi)

# ── Save/restore a set of names around a scoped loop body ─────────────────────
# Loop indices and element bindings must not leak into the enclosing block. Values
# in `info` are always StanExprs, so `nothing` is a safe "was-unbound" sentinel.
_save_scope(info, names) = Dict{Symbol,Any}(n => (n in keys(info) ? info[n] : nothing) for n in names)
_restore_scope!(info, saved) = for (n, old) in saved
    old === nothing ? (n in keys(info) && pop!(info, n)) : (info[n] = old)
end

# ── Destructuring iteration: `enumerate(c)` / `zip(a, b, …)` ──────────────────
# `for (i, xi) in enumerate(c)` / `for (ai, bi) in zip(a, b)` (and their
# comprehension forms) bind a TUPLE of names per step. `enumerate`/`zip` survive
# canonicalisation as symbol-headed `CanonicalExprV{:enumerate}`/`{:zip}` calls
# (they are never forwarded on their own). Both desugar to ordinary index
# iteration binding the element name(s) to `container[idx]`; `length`/`getindex`
# dispatch per container, so every supported container works with no per-type
# logic. Returns `(idx::Symbol, range, preludes)` where `range` is a raw `lo:hi`
# canonical expr (re-forwarded by each consumer) and `preludes` is a
# `Vector{Pair{Symbol,<raw element expr>}}` of `name => container[idx]` bindings.
_checked_iterable(c, ctx, what) = begin
    _is_value_iterable(c) || error(
        "@deffun `", ctx, "`: ", what, " must be an indexable container, got Stan type `",
        sigtype(type(c)), "`."
    )
    c
end
# `1:min(length(c1), length(c2), …)` — zip stops at the shortest container.
_zip_count(containers) = CanonicalExpr(Colon(), 1, foldl(
    (acc, c) -> CanonicalExpr(min, acc, CanonicalExpr(length, c)),
    containers[2:end]; init=CanonicalExpr(length, containers[1])
))
function _destructure_iteration(lhs::TupleExpr, source_raw; info)
    names = lhs.args
    all(n -> n isa Symbol, names) || error(
        "@deffun destructuring iteration: the tuple binding must contain only names, got `", lhs, "`."
    )
    if source_raw isa CanonicalExprV{:enumerate}
        length(source_raw.args) == 1 || error(
            "@deffun `enumerate(c)`: expected exactly one container argument, got ", length(source_raw.args), "."
        )
        length(names) == 2 || error(
            "@deffun `enumerate(c)`: bind exactly two names `(i, xi)` (index and element), got `", lhs, "`."
        )
        iname, ename = names
        container = _checked_iterable(forward!(source_raw.args[1]; info), "enumerate(c)", "the container `c`")
        # `enumerate`'s first slot IS the 1-based position, so it doubles as the
        # loop index; the element name binds to `container[i]`.
        return iname, _value_iter_count(container), Pair{Symbol,Any}[ename => _value_iter_elem(container, iname)]
    elseif source_raw isa CanonicalExprV{:zip}
        length(source_raw.args) >= 1 || error("@deffun `zip(…)`: expected at least one container argument.")
        length(names) == length(source_raw.args) || error(
            "@deffun `zip(a, b, …)`: the binding has ", length(names), " names but zip has ",
            length(source_raw.args), " containers — they must match."
        )
        containers = [
            _checked_iterable(forward!(a; info), "zip(…)", string("container ", k))
            for (k, a) in enumerate(source_raw.args)
        ]
        vi = _fresh_value_index(info)
        preludes = Pair{Symbol,Any}[names[k] => _value_iter_elem(containers[k], vi) for k in eachindex(names)]
        return vi, _zip_count(containers), preludes
    else
        error(
            "@deffun destructuring `for (…) in <source>`: a tuple binding requires `enumerate(…)` ",
            "or `zip(…)` as the iteration source, got `", source_raw, "`."
        )
    end
end

# Normalize one comprehension generator binding into an index-iteration plan:
# `(idx::Symbol, range, preludes)`. A tuple binding routes through the shared
# enumerate/zip destructuring; a plain `i in lo:hi` keeps the user's index and has
# no preludes; a container value-iterates with a fresh index and one element
# prelude. `range` is raw for the container/tuple cases (re-forwarded by the
# caller) and an already-forwarded StanExpr for the plain-range case.
function _comprehension_iter(lhs, source_raw; info)
    lhs isa TupleExpr && return _destructure_iteration(lhs, source_raw; info)
    lhs isa Symbol || error(
        "@deffun array comprehension: the generator must bind a Symbol or `(a, b, …)` tuple, got `", lhs, "`."
    )
    source = forward!(source_raw; info)
    _is_bounded_colon(source) && return lhs, source, Pair{Symbol,Any}[]
    _is_value_iterable(source) || error(
        "@deffun array comprehension: the generator must use one bounded `lo:hi` range ",
        "or an indexable container. Got an iteration source of Stan type `",
        sigtype(type(source)), "` (stepped-range and filtered generators are unsupported)."
    )
    vi = _fresh_value_index(info)
    vi, _value_iter_count(source), Pair{Symbol,Any}[lhs => _value_iter_elem(source, vi)]
end

# Fresh, non-underscore-leading comprehension result name.
function _fresh_comprehension_result(info)
    id = _next_inline_id()
    name = Symbol(:comprehension_result__lc_, id)
    while name in keys(info)
        id = _next_inline_id()
        name = Symbol(:comprehension_result__lc_, id)
    end
    name
end
# Output index for one dimension: the loop index for a `1:hi` range, else the
# dense offset `(i - lo) + 1`.
_result_index(emitted_idx, lo; info) = _is_literal_one(lo) ? emitted_idx :
    forward!(CanonicalExpr(+, CanonicalExpr(-, emitted_idx, lo), 1); info)

# One comprehension lowers to a typed result local + a (possibly nested) fill loop:
#
#     [f(i, j) for i in 1:n, j in 1:m]
#   →
#     result::<matrix|array[n,m]>[n, m]
#     for i in 1:n; for j in 1:m; result[i, j] = f(i, j); end; end
#     result
#
# `_comprehension_generator_specs` yields one `(binding, source)` per generator;
# `_comprehension_iter` normalises each to `(idx, range, preludes)` (handling
# `lo:hi`, value-iterated containers, and enumerate/zip). All indices + element
# bindings are in scope when the element expression is forwarded, so a
# multi-generator comprehension is exactly N nested single-index loops. SLIC scalar
# containers are at most 2-D, so a 3-D+ comprehension rejects loudly.
function forward!(x::ComprehensionExpr; info)
    value_raw, specs = _comprehension_generator_specs(x)
    pending = _get_inline_pending()
    pending === nothing && error(
        "@deffun array comprehension lowering requires a statement context; ",
        "bind or return the comprehension from an @deffun body."
    )
    ndim = length(specs)
    ndim <= 2 || error(
        "@deffun array comprehension: ", ndim, "-dimensional comprehensions are unsupported ",
        "(SLIC scalar containers are at most 2-D). Write explicit nested loops filling a typed result."
    )
    plans = [_comprehension_iter(lhs, src; info) for (lhs, src) in specs]
    ranges = [forward!(p[2]; info) for p in plans]
    result_sizes = Tuple(stan_size(r, 1) for r in ranges)

    scope_names = Symbol[]
    for p in plans
        push!(scope_names, p[1])
        append!(scope_names, Symbol[pp.first for pp in p[3]])
    end
    saved = _save_scope(info, scope_names)
    emitted_idxs = Vector{Any}(undef, ndim)
    try
        # Inline calls (and the element bindings) may contribute pre-statements.
        # Isolate them so they stay INSIDE the innermost loop instead of leaking
        # into the enclosing block.
        value, loop_stmts = task_local_storage(:_slic_inline_pending, Any[]) do
            for (k, p) in enumerate(plans)
                idx_k = p[1]
                info[idx_k] = StanExpr(idx_k, StanType(types.int; qual=:data))
                emitted_idxs[k] = info[idx_k]
                for (v, elem) in p[3]
                    info[v] = forward!(elem; info)
                end
            end
            value = forward!(value_raw; info)
            value isa StanExpr || error(
                "@deffun array comprehension: element expression forwarded to `",
                typeof(value), "`, expected a scalar Stan expression."
            )
            value, copy(task_local_storage(:_slic_inline_pending))
        end
        value_type = type(value)
        (center_type(value_type) <: types.complex && stan_ndim(value_type) == 0) || error(
            "@deffun array comprehension: the element expression must yield one scalar numeric value; ",
            "got Stan type `", sigtype(value_type), "`."
        )

        result_name = _fresh_comprehension_result(info)
        result_type = autotype(StanType(center_type(value_type), result_sizes))
        _is_model_decl_scope(info) && (result_type = remake(
            result_type; fresh_decl=true, decl_role=:unfilled, qual=:parameter
        ))
        info[result_name] = StanExpr(result_name, result_type)
        result = info[result_name]
        declaration = stan_expr(CanonicalExpr(:(::), result))

        result_idxs = [_result_index(emitted_idxs[k], expr(ranges[k]).args[1]; info) for k in 1:ndim]
        fill = forward!(CanonicalExpr(:(=), CanonicalExpr(getindex, result, result_idxs...), value); info)

        # Nest the fill loops from the innermost generator outward.
        body = CanonicalExpr(:block, loop_stmts..., fill)
        for k in ndim:-1:1
            body = stan_expr(CanonicalExpr(:for,
                CanonicalExpr(:(=), emitted_idxs[k], ranges[k]),
                _is_block_canonical(body) ? body : CanonicalExpr(:block, body)))
        end

        push!(pending, declaration, body)
        info[result_name]
    finally
        _restore_scope!(info, saved)
    end
end

_check_assignment_rhs(name, ::SlicModel) = error(
    "`$name = <submodel>(...)` is not supported — sub-models can only be embedded via `~`. ",
    "Use `$name ~ <submodel>(...)` instead.")
_check_assignment_rhs(_name, ::StanExpr) = nothing
_check_assignment_rhs(name, resolved) = error(
    "`$name = <rhs>`: rhs forwarded to a value of type `$(typeof(resolved))`, expected `StanExpr`.")

forward!(x::AssignmentExpr{Symbol}; info) = begin
    name, rhs = x.args
    promoted = _plate_promoted_lhs(name; info)
    promoted === nothing || return _forward_plate_assignment!(x, name, promoted, rhs; info)
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
    promoted = _plate_promoted_lhs(name; info)
    promoted === nothing || return _forward_plate_assignment!(x, name, promoted, rhs; info)
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
# coarse-grained variable — its bare declaration + every fill — to ONE block.
# The first certified fill resets a fresh declaration's provisional `:parameter`
# qualifier to the RHS qualifier; subsequent fills promote across RHS qualifiers.
# For those later fills, `:undefined` is bottom and the real order is
# `data < parameter < quantities` (lexicographically AND semantically).
_promote_qual(cur::Symbol, new::Symbol) =
    cur === :undefined ? new : (new === :undefined ? cur : max(cur, new))

# Typed assignment is an assertion about the RHS, not a request to coerce it.
# Center types follow the SLIC lattice (`int <: real`, constrained vectors
# `<: vector`, …); dimensions must describe the same Stan shape.  A dimension
# can be equal either symbolically or through a known data value (`3` versus a
# data vector's `x_n`, whose trace-time value is also 3).
_typed_assignment_shape_key(x::StanExpr, aliases) = _typed_assignment_shape_key(expr(x), aliases)
_typed_assignment_shape_key(x::CanonicalExpr, aliases) = (
    head(x),
    map(arg -> _typed_assignment_shape_key(arg, aliases), x.args),
    Tuple((k, _typed_assignment_shape_key(v, aliases)) for (k, v) in pairs(x.kwargs)),
)
_typed_assignment_shape_key(x::Expr, aliases) =
    (x.head, map(arg -> _typed_assignment_shape_key(arg, aliases), x.args))
_typed_assignment_shape_key(x::Tuple, aliases) =
    map(arg -> _typed_assignment_shape_key(arg, aliases), x)
# Signature-dimension aliases (`__size_alias_names__`, functions.jl) map an
# emitted Stan access expression back to the signature dimension it is
# GUARANTEED to equal — `dims(loc)[1]` and `dims(scale)[1]` both canonicalize to
# `n` for `f(loc::vector[n], scale::vector[n])`, an equality the emitted UDF
# either establishes (the defining `int n = dims(loc)[1];`) or enforces (the
# `reject` guard on every later occurrence).
#
# Normalization is deliberately ONE-WAY, towards the dimension NAME: a declared
# `n` is already canonical, while an argument-derived RHS type carries the raw
# `dims(...)[i]` fragment. Mapping names FORWARD to accesses as well would
# ping-pong, and could not express the many-to-one relation anyway.
_typed_assignment_shape_key(x::Symbol, aliases) = x
_typed_assignment_shape_key(x::AbstractString, aliases) = get(aliases, Symbol(x), x)
_typed_assignment_shape_key(x, aliases) = x

_typed_assignment_dim_matches(declared::StanExpr, inferred::StanExpr, aliases) =
    isequal(_typed_assignment_shape_key(declared, aliases), _typed_assignment_shape_key(inferred, aliases)) ||
    (hasvalue(declared) && hasvalue(inferred) && isequal(getvalue(declared), getvalue(inferred)))

_typed_assignment_aliases(info::Union{AbstractDict,NamedTuple}) =
    get(info, :__size_alias_names__, NamedTuple())
_typed_assignment_aliases(info) = NamedTuple()

_check_typed_assignment(name, declared::StanType, inferred::StanType; info) = begin
    declared_ct = center_type(declared)
    inferred_ct = center_type(inferred)
    inferred_ct <: declared_ct || error(
        "Typed assignment `", name, "::", declared, " = ...` is incompatible with inferred RHS type `",
        inferred, "`: RHS center `", inferred_ct, "` is not assignable to declared center `", declared_ct, "`."
    )

    declared_size = stan_size(declared)
    inferred_size = stan_size(inferred)
    length(declared_size) == length(inferred_size) || error(
        "Typed assignment `", name, "::", declared, " = ...` is incompatible with inferred RHS type `",
        inferred, "`: declared rank is ", length(declared_size), " but RHS rank is ", length(inferred_size), "."
    )
    aliases = _typed_assignment_aliases(info)
    for i in eachindex(declared_size, inferred_size)
        _typed_assignment_dim_matches(declared_size[i], inferred_size[i], aliases) || error(
            "Typed assignment `", name, "::", declared, " = ...` is incompatible with inferred RHS type `",
            inferred, "`: dimension ", i, " is declared as `", declared_size[i], "` but inferred as `",
            inferred_size[i], "`."
        )
    end
    nothing
end

forward!(x::AssignmentExpr; info) = begin
    lhs_raw = x.args[1]
    if lhs_raw isa DeclExpr && lhs_raw.args[1] isa Symbol
        name = lhs_raw.args[1]
        promoted = _plate_promoted_lhs(name; info)
        promoted === nothing || return _forward_plate_assignment!(x, name, promoted, x.args[2]; info)
    end
    # Keep the local/raw key before forwarding: inside a `SubModel`, forwarding
    # rewrites the emitted base to its flattened parent name while `keys(info)`
    # intentionally remains the local-name view.
    local_key = _base_lhs_symbol(x.args[1])
    fwd = stan_expr(remake(x, forward!(x.args; info)...))
    lhs, rhs = expr(fwd).args
    lhs_raw isa DeclExpr && _check_typed_assignment(local_key, type(lhs), type(rhs); info)
    k = local_key in keys(info) ? local_key : _base_lhs_symbol(lhs)
    if k in keys(info)
        base = info[k]
        # A fill CARRIES cv, exactly like the ordinary expression path
        # (`passes.jl` `cv=any(cv, x.args) || cv(tt)`): cv propagates through
        # every expression it reaches. Without this the plate's collected result
        # inherits its fill's QUALIFIER but not its TAINT, so a cv-tainted cell
        # parameter lands in `generated quantities` while the collection reads as
        # untainted — the downstream likelihood is then kept, referencing a name
        # that is no longer in scope for the model block (stanc reject). The flag
        # is monotone: once tainted, a later untainted fill cannot clear it.
        tainted = cv(base) || cv(rhs)
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
                info[k] = remake(base; decl_role=:fill, qual=qual(rhs), cv=tainted)
            else
                role == :fill || error("Fresh declaration `", k, "` has unknown declaration role `", role, "`.")
                info[k] = remake(base; qual=_promote_qual(qual(base), qual(rhs)), cv=tainted)
            end
        else
            info[k] = remake(base; qual=_promote_qual(qual(base), qual(rhs)), cv=tainted)
        end
    end
    fwd
end
forward!(x::SamplingExpr{Symbol}; info) = begin
    name, rhs = x.args
    # A whole ragged-vector OBSERVATION fed to a top-level distribution is
    # BROADCAST ACROSS its groups — never flattened. `ys ~ dist(mu, args…)` with a
    # data `RaggedVector` LHS lowers to a compiler-owned per-group loop
    #   for g in 1:length(ys)   ys[g] ~ dist(mu[g], args[g]…)
    # (ragged distribution args sliced per group via `_ragged_group_arg`; shared
    # scalar/dense args pass through). This preserves the ragged structure that
    # VECTOR-valued families need — `ys[g] ~ multi_normal(mu[g], Sigma)` is ONE
    # obs per group — while univariate families (`normal`) reduce to the same
    # per-group density Stan already vectorises. It reuses the ragged-prior density
    # loop (`_ragged_group_rhs` + `_trace_ragged_stmts`); the indexed data obs
    # routes to the model block only (no auto-GQ), matching the obs-in-cell plate
    # form. Uses the RAW rhs so the `_ragged_group_arg` markers resolve during the
    # injected trace. See snag ragged-dist-arg-dcffbc1b.
    if name in keys(info) && stan.qual(info[name]) == :data &&
            center_type(info[name]) <: RaggedVector && _is_canonical_expr(rhs)
        return _forward_ragged_obs_broadcast!(name, rhs; info)
    end
    rhs = forward!(rhs; info)::Union{StanExpr,SlicModel}
    forward!(remake(x, name, rhs); info)
end
_forward_ragged_obs_broadcast!(name, rhs_raw; info) = begin
    g = Symbol(:g, "__ro_", _next_inline_id())
    stmt = :(for $g in 1:length($name)
                 $name[$g] ~ $(_ragged_group_rhs(rhs_raw, g))
             end)
    _trace_ragged_stmts([stmt], name; info, certify_density=false)
end
forward!(x::SamplingExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args
    promoted = _plate_promoted_lhs(name; info)
    promoted === nothing || return _forward_plate_sampling!(x, name, promoted, rhs; info)
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
    promoted = _plate_promoted_lhs(name; info)
    if promoted !== nothing
        rhs = forward!(rhs_raw; info)::StanExpr
        return _forward_plate_sampling!(x, name, promoted, rhs; info)
    end
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
    # free param + a compiler-injected per-group `<family>_jacobian` loop
    # + a compile-time ragged carrier, reusing the loop-fill routing landed on
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
# Desugar a RAGGED constrained parameter (`p::simplex[Ks] ~ …`, `Ks` a data
# int-vector of per-group dims) into constructs SB can emit. Vector families use
# a RaggedVector; square cholesky families flatten K×K results and bind a
# RaggedMatrix that reconstructs each matrix on indexed access.
#   • bare improper-uniform free param sized by the flattened free dimensions
#   • per-group offsets in transformed data via `cumulative_sum` (data-qualified)
#   • a FRESH result vector filled by a compiler-injected per-group constrain loop
#     calling the built-in `<ct>_jacobian` (the jacobian accumulates directly in
#     `transformed parameters`; routing landed on slic-model-slice @ 29c3b59)
#   • a compile-time ragged pairing bound to `name`
# All statements are injected via `_slic_inline_pending` (they never enter the raw
# body, so they bypass `_reject_model_control_flow`) and routed to their blocks by
# `distribute!`. Informative RHS distributions add a second compiler-owned loop
# after the view binding: `name[g] ~ dist(group_arg(args, g)...)`. The same
# fine-grained ForExpr router used by plate sends that loop to `model`; ragged
# distribution arguments are indexed per group while shared arguments remain
# unchanged.
_forward_ragged_constrained!(name, ct, sizes, rhs_raw; info) = begin
    _is_canonical_expr(rhs_raw) || error(
        "Ragged constrained `$name::$ct[…] ~ rhs` requires rhs to be a distribution call."
    )
    isempty(rhs_raw.kwargs) || error(
        "Ragged constrained `$name::$ct[…]` does not accept distribution constraint kwargs. ",
        "Put the constraint in the declared center type and pass prior parameters positionally."
    )
    if ct in (:simplex, :ordered, :positive_ordered)
        _forward_ragged_vector_constrained!(name, ct, sizes, rhs_raw; info)
    elseif ct in (:cholesky_factor_corr, :cholesky_factor_cov)
        _forward_ragged_matrix_constrained!(name, ct, sizes, rhs_raw; info)
    else
        error(
            "Ragged constrained `$name::$ct[…]` is not supported yet. Supported ",
            "families are `simplex`, `ordered`, `positive_ordered`, ",
            "`cholesky_factor_corr`, and square `cholesky_factor_cov`."
        )
    end
end

_ragged_density_stmt(name, Ks, rhs_raw, g) = _ragged_rhs_is_flat(rhs_raw) ?
    nothing : :(for $g in 1:length($Ks)
                    $name[$g] ~ $(_ragged_group_rhs(rhs_raw, g))
                end)
_trace_ragged_stmts(stmts, name; info, certify_density::Bool=false) = begin
    targets = (_ragged_density_targets()..., name)
    traced = task_local_storage(:_slic_ragged_density_targets, targets) do
        _do_retrace_inline_body(stmts; info)
    end
    # The task-local certificate above is sufficient during `forward!`, but the
    # backward likelihood-reachability pass runs after tracing has left that
    # dynamic scope.  Stamp the underlying constrained-memory declaration so an
    # indexed `mem[lo:hi] ~ prior(...)` remains recognisable as a compiler-owned
    # ragged density (and not as forbidden user sampling of a derived value).
    if certify_density
        ragged = info[name]
        mem = expr(ragged).args[1]
        mem_key = _base_lhs_symbol(mem)
        root = _plate_root_info(info)
        root[mem_key] = remake(root[mem_key]; ragged_density=true)
    end
    traced
end

_forward_ragged_vector_constrained!(name, ct, sizes, rhs_raw; info) = begin
    length(sizes) == 1 || error(
        "Ragged constrained `$name::$ct[…]`: expected a single vector size (the ",
        "per-group dims), got $(length(sizes)). Rectangular `$ct[a,b]` and ragged ",
        "`$ct[Ks]` are distinct shapes; a mixed form is unsupported."
    )
    Ks  = sizes[1]
    free_drop = ct === :simplex ? 1 : 0
    jac = Symbol(ct, :_jacobian)
    free_sizes = free_drop == 0 ? Ks : :($Ks .- $free_drop)
    # Stan-valid unique names (gensym's `#` is an illegal Stan identifier char);
    # `_next_inline_id` is the per-trace counter SB uses for inlined-local renames.
    id = _next_inline_id()
    pfree = Symbol(:p_free, "__rc_", id); pmem = Symbol(:p_mem, "__rc_", id)
    cend  = Symbol(:c_end, "__rc_", id);  fend = Symbol(:f_end, "__rc_", id)
    g     = Symbol(:g, "__rc_", id)
    free_size_g = free_drop == 0 ? :($Ks[$g]) : :($Ks[$g] - $free_drop)
    free_start = :($fend[$g] - $free_size_g + 1)
    stmts = Any[
        :($pfree :: vector[sum($free_sizes)]),           # bare improper-uniform free coords → parameters
        :($cend = cumulative_sum($Ks)),                  # constrained group ends (data → TD)
        :($fend = cumulative_sum($free_sizes)),           # free group ends (data → TD)
        :($pmem :: vector[sum($Ks)]),                    # fresh result → auto fresh_decl cert
        :(for $g in 1:length($Ks)                        # injected ForExpr — 29c3b59 routes it
              $pmem[$cend[$g] - $Ks[$g] + 1 : $cend[$g]] =
                  $jac($pfree[$free_start : $fend[$g]])
          end),
        :($name = RaggedVector($pmem, $cend)),           # ragged view (representation bae1167)
    ]
    density = _ragged_density_stmt(name, Ks, rhs_raw, g)
    density === nothing || push!(stmts, density)
    _trace_ragged_stmts(stmts, name; info, certify_density=density !== nothing)
end

_forward_ragged_matrix_constrained!(name, ct, sizes, rhs_raw; info) = begin
    length(sizes) == 1 || error(
        "Ragged constrained `$name::$ct[…]`: this increment supports one vector ",
        "of square group sizes. Rectangular ragged cholesky factors need separate ",
        "row and column size vectors and are not implemented yet."
    )
    Ks = sizes[1]
    corr_family = ct === :cholesky_factor_corr
    jac = Symbol(ct, :_jacobian)
    # cholesky_factor_corr[K] has K(K-1)/2 free coordinates; a square
    # cholesky_factor_cov[K] has K(K+1)/2. Both materialise K² matrix cells.
    free_sizes = corr_family ? :(($Ks .* ($Ks .- 1)) .÷ 2) : :(($Ks .* ($Ks .+ 1)) .÷ 2)
    mem_sizes = :($Ks .* $Ks)

    id = _next_inline_id()
    pfree = Symbol(:p_free, "__rcm_", id); pmem = Symbol(:p_mem, "__rcm_", id)
    cend  = Symbol(:c_end, "__rcm_", id);  fend = Symbol(:f_end, "__rcm_", id)
    g     = Symbol(:g, "__rcm_", id)
    free_size_g = corr_family ?
        :(($Ks[$g] * ($Ks[$g] - 1)) ÷ 2) :
        :(($Ks[$g] * ($Ks[$g] + 1)) ÷ 2)
    mem_size_g = :($Ks[$g] * $Ks[$g])
    free_start = :($fend[$g] - $free_size_g + 1)
    mem_start = :($cend[$g] - $mem_size_g + 1)
    free_slice = :($pfree[$free_start : $fend[$g]])
    constrained = corr_family ?
        :($jac($free_slice, $Ks[$g])) :
        :($jac($free_slice, $Ks[$g], $Ks[$g]))

    stmts = Any[
        :($pfree :: vector[sum($free_sizes)]),
        :($cend = cumulative_sum($mem_sizes)),
        :($fend = cumulative_sum($free_sizes)),
        :($pmem :: vector[sum($mem_sizes)]),
        :(for $g in 1:length($Ks)
              $pmem[$mem_start : $cend[$g]] = to_vector($constrained)
          end),
        :($name = RaggedMatrix($pmem, $cend, $Ks, $Ks)),
    ]
    density = _ragged_density_stmt(name, Ks, rhs_raw, g)
    density === nothing || push!(stmts, density)
    _trace_ragged_stmts(stmts, name; info, certify_density=density !== nothing)
end
# Indexed sampling is the plate producer's declaration certificate.  The
# producer first emits an outer-sized fresh declaration, then rewrites a
# cell-local `x ~ dist` to `x[i] ~ dist` inside its compiler-owned loop.  The
# first such use classifies the declaration as a parameter; indexed data LHSs
# remain ordinary observations.  An indexed parameter that did NOT come from a
# fresh declaration stays out of this internal path.
_forward_indexed_sampling!(x; info) = begin
    lhs_raw, rhs_raw = x.args
    k = _base_lhs_symbol(lhs_raw)
    k in keys(info) || error(
        "Plate-generated indexed sampling of `", k, "[…]` requires the plate emitter to register its outer declaration first."
    )
    lhs = forward!(lhs_raw; info)
    rhs = forward!(rhs_raw; info)::StanExpr
    base = info[k]
    if _is_ragged_density_target(k)
        (center_type(base) <: RaggedVector || center_type(base) <: RaggedMatrix) || error(
            "Compiler-certified ragged density target `", k,
            "` is not backed by a RaggedVector/RaggedMatrix."
        )
        return remake(x, lhs, rhs)
    end
    retainted = false
    if _is_fresh_decl(base)
        role = _decl_role(base)
        role == :fill && error(
            "Cannot sample `", k, "[…]` after the fresh declaration was classified as a transformed fill target."
        )
        role in (:unfilled, :sampled) || error(
            "Fresh declaration `", k, "` has unknown declaration role `", role, "`."
        )
        # cv contagion, exactly as the bare-Symbol path does it (see
        # `forward!(::SamplingExpr{Symbol,<:StanExpr})`): a fresh parameter whose
        # DECLARED type is cv-tainted — a plate whose `outer=` size came from
        # held-out data — or whose prior RHS is cv, is predictive-only and must
        # be a generated quantity, never a parameter. Deciding it HERE, in the
        # forward pass, rather than from likelihood reachability in `backward!`,
        # is what keeps the emitted program stanc-clean: the `:quantities`
        # qualifier then propagates through the plate's collected result and
        # everything downstream via the ordinary `_promote_qual` fill path, so
        # the whole chain lands in `generated quantities` TOGETHER. A `backward!`
        # -side fix moves the cell parameter alone and leaves its collection
        # behind in `transformed parameters`, referencing a name declared later.
        if role == :unfilled
            tainted = cv(base) || cv(rhs)
            info[k] = tainted ?
                remake(base; decl_role=:sampled, qual=:quantities, cv=true) :
                remake(base; decl_role=:sampled, qual=:parameter)
            retainted = tainted
        end
    else
        q = qual(base)
        q == :data || error(
            "Indexed sampling of `", k, "[…]` has a ", q,
            "-qualified base that is not a compiler-generated fresh plate declaration."
        )
        if cv(rhs)
            info[k] = remake(base; cv=true)
            retainted = true
        end
    end
    # Re-derive the LHS against the base we may have just tainted.  `lhs` was
    # snapshotted at the top of this function, BEFORE the cv decision, and
    # `distribution_blocks` routes a sampling statement by reading `cv` off the
    # EMITTED LHS expression rather than off `info` — so a stale LHS leaves a
    # held-out per-cell observation `y[i] ~ normal(t[i], s)` sitting in the model
    # block while `t` has already moved to `generated quantities`, which stanc
    # rejects as out of scope.  The bare-Symbol path never hits this because it
    # ends with `remake(x, info[name], rhs)`, re-reading `info` after the taint;
    # an indexed LHS is a compound expression and has to be re-forwarded to pick
    # the base up again.  Guarded on an actual taint so every untainted emission
    # is byte-identical to before.
    retainted && (lhs = forward!(lhs_raw; info))
    remake(x, lhs, rhs)
end
forward!(x::SamplingExpr{<:CanonicalExprV{:getindex}}; info) =
    _forward_indexed_sampling!(x; info)
# Resolve the intentional intersection with the generic
# `SamplingExpr{<:Any,<:StanExpr}` floor. Plate promotion forwards the RHS first,
# then hands the indexed LHS to this same classifier.
forward!(x::SamplingExpr{<:CanonicalExprV{:getindex},<:StanExpr}; info) =
    _forward_indexed_sampling!(x; info)
# --- Public `plate` do-block emitter (Feature 2, surface decision n35u3c) -----
# Lowers `rv ~ plate(iter1, …; outer=(dims...)) do a1, …; body…; cell_output; end`
# into the plate PRODUCER CONTRACT that the `~`-aware routing (landed on
# slic-model-slice-b3a85769) consumes: outer `DeclExpr`s + a compiler-injected
# certified `ForExpr` with indexed fresh samples, indexed observations, and an
# indexed return fill — injected via `_slic_inline_pending` (so they bypass
# `_reject_model_control_flow`, same as `_forward_ragged_constrained!`).
#
# Semantics (n35u3c): positional iterables are PER-CELL slices bound to the
# do-block params (`a_k` ⇒ `iter_k[i1,…]`); lexical captures stay SHARED; each
# fresh `~`/`=` LHS in the body is promoted to an outer collection indexed by
# every loop axis; the trailing expression is each cell's output, filled into
# `rv`. An integer `outer=N` is the 1-D shorthand; tuples emit nested loops.
# Current cell-shape scope: scalar, fixed `vector[K]`, or heterogeneous
# one-dimensional `vector[K[i]]` values; no vararg params; the body must end in
# a value expression.
# Used only to reject a terminal `~`/`=` statement with a clear error. Fresh
# binding discovery itself is trace-based (`_plate_discover` below).
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
# Substitute only do-block parameters with their per-cell input accessors. Fresh
# names are promoted during tracing via `_slic_plate_context`, which also reaches
# names introduced inside an inlined submodel (not present in this raw AST).
_subst_syms(x, m) = x
_subst_syms(x::Symbol, m) = get(m, x, x)
_subst_syms(x::Expr, m) = Expr(x.head, Any[_subst_syms(a, m) for a in x.args]...)

# Resolve a legacy typed plate-result override to the same per-cell StanType the
# discovery trace produces. Bare result LHSs use discovery's `ret_type` directly.
_plate_annotation_type(type_expr; info) = begin
    ct, sizes... = _is_getindex_expr(type_expr) ? type_expr.args : (type_expr,)
    ct isa Symbol || error("plate: result type center must be a Symbol, got `$ct`.")
    StanType(gettype(ct), forward!.(sizes; info))
end

_plate_cell_shape(T, name) = begin
    stan_ndim(T) == 0 && return :scalar
    if center_type(T) <: types.vector && stan_ndim(T) == 1
        # A native-constrained vector cell (simplex/ordered/positive_ordered) is
        # stored as a Stan `array[N…] <ct>[K]` so Stan applies the constraint
        # transform + jacobian per cell; a plain `vector[K]` cell is packed into a
        # dense `matrix[K, N]` column instead (no per-cell constraint).
        return _is_native_constrained_ct(center_type(T)) ? :constrained_vector : :vector
    end
    error(
        "plate: unsupported per-cell type `", sigtype(T), "` for `", name,
        "` — scalar or vector[K] only (MVP)."
    )
end

# A vector cell is ragged when its inferred length depends on the plate's own
# loop index. Discovery already traces size expressions (including through a
# called submodel), so this is an expression-level dependency test rather than
# an AST guess at the do-block source.
_plate_depends_on(x, idxs) = false
_plate_depends_on(x::Symbol, idxs) = x in idxs
_plate_depends_on(x::StanExpr, idxs) = _plate_depends_on(expr(x), idxs)
_plate_depends_on(x::CanonicalExpr, idxs) = any(a -> _plate_depends_on(a, idxs), x.args)
_plate_depends_on(x::Union{Tuple,NamedTuple,AbstractVector}, idxs) =
    any(a -> _plate_depends_on(a, idxs), x)
_plate_is_ragged_cell(T::StanType, idxs) =
    center_type(T) <: types.vector && stan_ndim(T) == 1 &&
    _plate_depends_on(stan_size(T)[1], idxs)

_plate_ragged_plan(f, T::StanType, outer, idxs, id) = begin
    length(outer) == 1 || error(
        "plate: ragged vector cells currently require a one-dimensional `outer`; ",
        "got $(length(outer)) axes for `$f`."
    )
    center_type(T) === types.vector || error(
        "plate: ragged constrained cell `$f::$(sigtype(T))` needs the constrained-",
        "parameter transform path and is not represented as unconstrained flat memory. ",
        "Use a plain `vector[K[i]]` cell here, or declare the ragged constrained ",
        "parameter at model scope."
    )
    qual(stan_size(T)[1]) == :data || error(
        "plate: ragged cell length for `$f` must be data-computable, got a ",
        "$(qual(stan_size(T)[1]))-qualified size expression. Stan parameter ",
        "dimensions cannot depend on parameters or generated quantities."
    )
    mem = Symbol(f, :__pl_mem_, id)
    lens = Symbol(f, :__pl_len_, id)
    ends = Symbol(f, :__pl_end_, id)
    idx = only(idxs)
    accessor = :($mem[ragged_start($ends, $idx):ragged_end($ends, $idx)])
    (; logical=f, cell_type=T, mem, lens, ends,
       size_expr=expr(stan_size(T)[1]), accessor)
end

# Outer collection decl + per-cell accessor from DISCOVERED StanTypes. The
# shipped 1-D shapes stay dense (`vector[N]`, `matrix[K,N]`). Additional outer
# axes become Stan array dimensions around the natural vector/matrix core while
# preserving the logical cell-first shape.
_plate_type_expr(ct::Symbol, sizes) = Expr(:ref, ct, sizes...)
_plate_outer_decl(f, T::StanType, outer) = begin
    isempty(outer) && error("plate: `outer` must contain at least one dimension.")
    shape = _plate_cell_shape(T, f)
    if shape === :scalar
        length(outer) == 1 && return Expr(:(::), f, _plate_type_expr(:vector, outer))
        sizes = Any[outer[3:end]...; outer[1]; outer[2]]
        return Expr(:(::), f, _plate_type_expr(:matrix, sizes))
    end
    # Keep `K` as the traced StanExpr (not its bare `expr`). A submodel arg-derived
    # size resolves to a CALLER-scope name (`n_terms` ⇒ the caller's `P`); as a bare
    # symbol that re-traces cleanly at the root but NOT under `info` (where the `rv`
    # decl emits), so keep the resolved StanExpr and let `forward!(::StanExpr)` pass
    # it through idempotently in either scope. A top-level plate is unaffected (the
    # size renders identically either way).
    K = stan_size(T)[1]
    if shape === :constrained_vector
        # `array[outer…] <ct>[K]`: outer dims lead (Stan array), the constrained
        # core size K trails. `<ct>` (simplex/ordered/positive_ordered) declared
        # in `parameters` gets Stan's native constraint transform + jacobian; in
        # `transformed parameters` (the plate result copy) it is validate-only.
        ct = center_type(T).name.name
        return Expr(:(::), f, _plate_type_expr(ct, Any[outer...; K]))
    end
    length(outer) == 1 && return Expr(:(::), f, _plate_type_expr(:matrix, Any[K, outer[1]]))
    sizes = Any[outer[2:end]...; K; outer[1]]
    Expr(:(::), f, _plate_type_expr(:matrix, sizes))
end

# `lower`/`upper`/`offset`/`multiplier` live in a cell StanType's `info`, not in
# its center type — so unlike the native-constrained centers (simplex/ordered/…)
# they do NOT survive `_plate_outer_decl`'s shape/size/center-type rebuild. Left
# alone they simply vanish on promotion, and the model compiles clean: `stanc`
# accepts the unconstrained declaration and samples a DIFFERENT posterior with no
# diagnostic anywhere. Carry them onto the promoted declaration instead.
#
# A constraint referencing the plate's own per-cell position cannot be promoted
# verbatim: the promoted declaration is emitted OUTSIDE the loop, so a cell
# `multiplier=exp(l)` would render with `l` unbound. Promoting it needs the whole
# constraint vector materialised over the outer axis first, and Stan additionally
# forbids a constraint referencing a transformed parameter — so only the
# data-qualified case is hoistable at all. Until that hoist lands, refuse loudly:
# a wrong model that compiles is the failure mode being removed here.
_plate_promoted_constraints(f, T::StanType, index_aliases) = begin
    cons = constraints(T)
    for (key, value) in pairs(cons)
        _plate_depends_on(value, index_aliases) || continue
        error(
            "plate: cell `", f, "`'s `", key, "=` constraint depends on the plate's ",
            "per-cell position, but the promoted declaration is emitted outside the ",
            "loop, where that position does not exist. Compute the full `", key,
            "` over the outer axis BEFORE the plate and pass that name, or declare ",
            "`", f, "` at model scope."
        )
    end
    cons
end

# Merge promoted constraints into an already-emitted declaration, mirroring the
# tail of `forward!(::DeclExpr)`: re-key on the RAW name `f` so a SubModel's
# `setindex!` flattens it exactly once, then put the stored value back into the
# declaration AST so later passes look the binding up under the same name.
_plate_constrain_decl(f, x, cons; info) = begin
    isempty(cons) && return x
    d = expr(x)
    info[f] = StanExpr(f, remake(type(d.args[1]); cons...))
    stan_expr(remake(d, info[f]))
end

_plate_cell_index(f, T::StanType, idxs) = begin
    isempty(idxs) && error("plate: internal error — missing outer indices for `$f`.")
    shape = _plate_cell_shape(T, f)
    if shape === :scalar
        length(idxs) == 1 && return Expr(:ref, f, idxs[1])
        indices = Any[idxs[3:end]...; idxs[1]; idxs[2]]
        return Expr(:ref, f, indices...)
    end
    # A native-constrained `array[N…] <ct>[K]` cell is indexed by its plate axes
    # as a whole element (`cell[g]`); the plain-vector matrix packing takes a
    # column (`cell[:, g]`) instead.
    shape === :constrained_vector && return Expr(:ref, f, idxs...)
    indices = Any[idxs[2:end]...; Symbol(":"); idxs[1]]
    Expr(:ref, f, indices...)
end

# An isolated copy of the tracing `info`, MIRRORING its structure so bindings
# land in throwaway storage: a StanModel copies its `vars`; a SubModel copies its
# `locals` AND recursively probes its parent, so a fresh binding created inside a
# submodel-scoped plate flattens into the probe-ROOT the same way it will at emit
# time (`t_z`), without polluting the real parent's vars. `blocks` stay SHARED —
# `forward!` only binds vars + returns canonical; block routing is a later pass.
_plate_probe(info::StanModel) = StanModel(meta(info), copy(vars(info)), blocks(info))
_plate_probe(info::SubModel) = SubModel(_plate_probe(parent(info)), name(info), copy(locals(info)))

# Per-cell input accessor for a positional plate iterable. A DENSE iterable slices
# with ordinary `it[idxs...]` (scalar element or dense sub-slice). A CERTIFIED
# RAGGED carrier — nested-vector data, which `stan_type` now mints as a nominal
# `RaggedVector` (a subtype of `ntup`) with `(mem, ends)` fields (the same
# certification `_ragged_group_arg` slices on) — is instead sliced INLINE here into
# its per-group `vector` view, mirroring the ragged OUTPUT accessor
# `mem[ragged_start(ends,i):ragged_end(ends,i)]`. That lets a ragged observed slice
# feed a typed `vector[k]` called cell. Since 197f5be the data ALSO carries the
# nominal tag, so a bare `it[idx]` would route through `getindex_RaggedVector` and
# yield the same group vector — the inline form is kept to avoid a per-cell UDF call
# and to match emit-time `input_subst`. Ragged input is only recognised for a 1-D
# `outer` (single index), matching the ragged-cell scope.
_plate_iterable_type(it::Symbol, info) = it in keys(info) ? type(info[it]) : nothing
_plate_iterable_type(it, info) = nothing
_plate_is_ragged_iterable(::Nothing) = false
_plate_is_ragged_iterable(T::StanType) =
    center_type(T) <: types.ntup && keys(T.info.arg_types) == (:mem, :ends)
_plate_input_accessor(it, idxs, info) = begin
    (length(idxs) == 1 && _plate_is_ragged_iterable(_plate_iterable_type(it, info))) || return Expr(:ref, it, idxs...)
    idx = only(idxs)
    :($it.mem[ragged_start($it.ends, $idx):ragged_end($it.ends, $idx)])
end

# ── Loop-invariant code motion (LICM) over the plate cell body ───────────────
# The cell body is emitted INSIDE the compiler-owned `for` loop, so every
# subexpression that does not depend on the cell — `diag_pre_multiply(tau, L)`,
# `rep_vector(0.0, K)` — is recomputed and AD-taped `prod(outer)` times per
# gradient evaluation. Measured on a BRM correlated-random-effect block
# (`n_groups=1000`, `K=5`, identical posterior, dlp=0): 743 us/grad with the
# scale written inline in the cell body vs 408 us/grad with the SAME scale bound
# to a local before the plate — 1.82x, purely from where the emitter put it
# (snag benchmarked-brm-20aa0361). The emitter already emits the right thing for
# the hand-hoisted spelling, so this pass performs that binding automatically
# and every plate author stops having to know the trick.
#
#   PASS A lifts a whole cell-body ASSIGNMENT whose RHS is cell-invariant out of
#          the loop entirely, so its LHS becomes one shared value instead of a
#          promoted per-cell collection plus a fill loop.
#   PASS B replaces each MAXIMAL cell-invariant CALL subexpression that survives
#          Pass A with a compiler-owned local bound before the loop. Identical
#          expressions share one local.
#
# Two independent gates decide hoistability and BOTH must pass, so the error is
# always "declined to hoist" (emit exactly as before), never "hoisted something
# that was per-cell":
#   1. SYNTACTIC INVARIANCE — the expression mentions no loop index, no do-block
#      parameter and no name the cell body itself binds. Symbol collection
#      deliberately over-approximates (callee names and keyword labels count), so
#      it can only ever be too strict.
#   2. A PROBE TRACE in a throwaway copy of `info` (`_plate_probe` — the same
#      isolation `_plate_discover` uses) that must resolve WITHOUT creating any
#      binding, WITHOUT queueing any pending statement, and must yield a plain
#      declarable Stan value. That single gate is what keeps a `~`-bearing @slic
#      submodel call — whose per-cell parameters must NOT collapse to shared ones
#      — and an inline UDF that hoists its own statements out of this pass,
#      without it having to recognise either shape syntactically.
# `~` statements are never lifted: only the ARGUMENTS of the distribution call
# are eligible, so the sampling effect itself always stays per-cell.
_plate_stmt_lhs(s) =
    if s isa Expr && s.head === :(=)
        s.args[1]
    elseif s isa Expr && s.head === :call && length(s.args) >= 3 && s.args[1] === :~
        s.args[2]
    else
        nothing
    end
_plate_lhs_root(::Nothing) = nothing
_plate_lhs_root(lhs::Symbol) = lhs
_plate_lhs_root(lhs::Expr) =
    (lhs.head === :(::) || lhs.head === :ref) ? _plate_lhs_root(lhs.args[1]) : nothing
_plate_lhs_root(_) = nothing

_plate_collect_syms!(acc, _) = acc
_plate_collect_syms!(acc, x::Symbol) = (push!(acc, x); acc)
_plate_collect_syms!(acc, x::GlobalRef) = (push!(acc, x.name); acc)
_plate_collect_syms!(acc, x::Expr) =
    (for a in x.args; _plate_collect_syms!(acc, a); end; acc)
_plate_is_invariant(x, varying) =
    isdisjoint(_plate_collect_syms!(Set{Symbol}(), x), varying)

# Only a CALL with at least one binding-mentioning argument is worth lifting: a
# bare symbol or literal costs nothing per cell, an index into a shared value
# would become an AD-tracked copy, and a pure constant fold saves nothing.
# `:`/`~`/`&&`/`||` are syntax or statements, not Stan values.
_plate_mentions_binding(::Symbol) = true
_plate_mentions_binding(::GlobalRef) = true
_plate_mentions_binding(x::Expr) = any(_plate_mentions_binding, x.args)
_plate_mentions_binding(_) = false
_plate_hoist_callee_ok(f::Symbol) = !(f in (:~, :(:), :(=), :&&, :||))
_plate_hoist_callee_ok(f::GlobalRef) = _plate_hoist_callee_ok(f.name)
_plate_hoist_callee_ok(_) = false
_plate_is_hoist_candidate(x) =
    x isa Expr && x.head === :call && length(x.args) >= 2 &&
    _plate_hoist_callee_ok(x.args[1]) &&
    any(_plate_mentions_binding, @view x.args[2:end])

# A hoisted local is DECLARED at model scope, so its value must have a plain
# declarable center. Constrained centers are excluded deliberately: emitting
# `simplex[K] tmp = <expr>;` in `transformed parameters` adds a Stan VALIDATION
# the inline expression never paid, which could reject a draw the model used to
# accept — a hoist must not change what the posterior admits.
_plate_hoistable_center(T) = T === types.int || T === types.bool || T === types.real ||
    T === types.vector || T === types.row_vector || T === types.matrix
_plate_probe_hoistable(e; info) = task_local_storage(:_slic_inline_pending, Any[]) do
    probe = _plate_probe(info)
    root = _plate_root_info(probe)
    before = Set(keys(root))
    # A candidate that does not resolve standalone (it reached a cell-local the
    # syntactic gate could not see, or it is not a value at all) is simply NOT
    # hoisted. That is the expected, survivable outcome of a probe — the
    # expression is then emitted verbatim, exactly as before this pass existed —
    # not a swallowed bug.
    v = try
        forward!(canonical(e); info=probe)
    catch
        return false
    end
    isempty(task_local_storage(:_slic_inline_pending)) || return false
    Set(keys(root)) == before || return false
    v isa StanExpr || return false
    _plate_hoistable_center(center_type(v))
end

# Emits each lifted binding into `info` + `pending` (so it lands before the loop
# in the enclosing block) and returns the rewritten `(body_stmts, ret_expr)`.
# Runs BEFORE `_plate_discover`, so a lifted name is already in `info` when
# discovery snapshots `before` and is therefore never promoted to a cell.
_plate_hoist_invariants(body_stmts, ret_expr, rv, params, idxs, id; info) = begin
    pending = _get_inline_pending()
    varying = Set{Symbol}(idxs)
    union!(varying, params)
    for s in body_stmts
        r = _plate_lhs_root(_plate_stmt_lhs(s))
        r === nothing || push!(varying, r)
    end
    emit_hoist!(s) = begin
        emitted = forward!(canonical(s); info)
        pending !== nothing && push!(pending, emitted)
        emitted
    end

    # PASS A — lift whole cell-invariant assignments. Restricted to an UNTYPED
    # Symbol LHS: a typed `S::matrix[K,K] = …` would have to re-trace through the
    # ordinary model-scope declaration path the probe never exercised, and
    # declining costs only the optimisation.
    kept = Any[]
    for s in body_stmts
        name = (s isa Expr && s.head === :(=) && s.args[1] isa Symbol) ? s.args[1] : nothing
        if name !== nothing && !(name in keys(info)) &&
            _plate_is_invariant(s.args[2], varying) && _plate_probe_hoistable(s.args[2]; info)
            emit_hoist!(s)
            delete!(varying, name)
        else
            push!(kept, s)
        end
    end
    body_stmts = kept

    # PASS B — maximal invariant call subexpressions.
    cache = Dict{Any,Symbol}()
    n = Ref(0)
    hoist!(e) = begin
        haskey(cache, e) && return cache[e]
        (_plate_is_hoist_candidate(e) && _plate_is_invariant(e, varying) &&
            _plate_probe_hoistable(e; info)) || return nothing
        n[] += 1
        name = Symbol(rv, :__pl_inv, n[], :_, id)
        emit_hoist!(Expr(:(=), name, e))
        cache[e] = name
    end
    walk(e) = begin
        h = hoist!(e)
        h === nothing || return h
        e isa Expr || return e
        Expr(e.head, Any[walk(a) for a in e.args]...)
    end
    # A distribution call and a void call at statement position ARE the effect —
    # descend into their arguments only, never lift the call itself.
    walk_call_args(e) = (e isa Expr && e.head === :call && length(e.args) >= 2) ?
        Expr(:call, e.args[1], Any[walk(a) for a in @view e.args[2:end]]...) : e
    rewrite(s) =
        if s isa Expr && s.head === :call && length(s.args) == 3 && s.args[1] === :~
            Expr(:call, :~, s.args[2], walk_call_args(s.args[3]))
        elseif s isa Expr && s.head === :(=)
            Expr(:(=), s.args[1], walk(s.args[2]))
        elseif s isa Expr && s.head === :call
            walk_call_args(s)
        else
            s
        end

    (Any[rewrite(s) for s in body_stmts], walk(ret_expr))
end

# TRACE-THEN-PROMOTE discovery (rework, decision 1vujeta): trace the do-block
# body ONCE in an ISOLATED probe scope to discover the fresh params/vars the body
# creates AND their per-cell types — INCLUDING submodel-internal ones (a submodel
# embeds via `SubModel`, flattening `t_z`/`t` into the probe's vars). Fresh names
# are collected in the probe ROOT (global) namespace so a plate nested inside a
# called `@slic` submodel discovers each cell under the SAME flattened name the
# emit-time promotion context (`_plate_global_name`) will look it up by. For a
# top-level plate the probe root IS the probe, so the global names equal the local
# ones. `pending` is isolated in its own task-local; nothing is emitted and the
# probe is discarded. Returns (fresh::Vector{Pair{Symbol,StanType}} in body order
# keyed by global name, ret_type).
_plate_discover(body_stmts, ret_expr, params, iterables, idxs; info::Union{StanModel,SubModel}) =
    task_local_storage(:_slic_inline_pending, Any[]) do
        probe = _plate_probe(info)
        root = _plate_root_info(probe)
        before = Set(keys(root))
        for idx in idxs
            probe[idx] = StanExpr(idx, StanType(types.int; qual=:data))
        end
        if isempty(iterables)
            for (param, idx) in zip(params, idxs)
                # The do-block parameter is an alias for this plate axis, not a
                # distinct Stan variable. Preserve that alias in discovered size
                # expressions (`K[g]` becomes `K[plate_i]`) so ragged dependency
                # analysis sees the same index that emit-time substitution uses.
                probe[param] = StanExpr(idx, StanType(types.int; qual=:data))
            end
        else
            for (a, it) in zip(params, iterables)
                # Bind the do-block param to its per-cell accessor EXPR (not a bare
                # symbol), mirroring the iterable-free `probe[param] = StanExpr(idx, …)`
                # aliasing above. A fresh cell whose size indexes THROUGH this param —
                # e.g. `z::vector[K[g]]` with `g` an index into positional `groups` —
                # then carries the loop-index dependence into discovered size
                # expressions (`K[g]` ⇒ `K[groups[plate_i]]`), so
                # `_plate_is_ragged_cell` classifies it ragged instead of emitting a
                # dense `matrix[K[g], N]` decl that references the out-of-scope `g`.
                # The expr matches emit-time `input_subst`, keeping the two traces
                # consistent (same invariant the iterable-free branch relies on).
                probe[a] = forward!(canonical(_plate_input_accessor(it, idxs, info)); info=probe)
            end
        end
        # Trace as a BLOCK (not per-statement) so submodel embedding binds its
        # flattened result the same way the real model-body trace does.
        isempty(body_stmts) || forward!(canonical(Expr(:block, body_stmts...)); info=probe)
        ret = forward!(canonical(ret_expr); info=probe)
        # idx/param helper bindings flatten to their global names in the root; skip
        # those (and everything present before the trace) so only body-introduced
        # cell bindings remain.
        helpers = Set(_plate_global_name(info, s) for s in Iterators.flatten((idxs, params)))
        fresh = Pair{Symbol,Any}[]
        for k in keys(root)
            (k in before || k in helpers) && continue
            push!(fresh, k => type(root[k]))
        end
        (fresh=fresh, ret_type=type(ret))
    end

# ── Plate emitter entry: the public `rv ~ plate(iters…; outer=…) do … end`. ──
# Trace-then-promote (decision 1vujeta): `_plate_discover` probes the body once to
# find every fresh cell-local binding + the cell result type, then the loop is
# re-traced under the task-local `_slic_plate_context` that maps each cell name to
# its outer array slot. VERIFIED contract boundary (BRM Complete-PLATE snag,
# 2026-07-16) — consumers must not assume more than this is owned:
#   • Cell VALUES: scalar or 1-D `vector[K]` (`_plate_cell_shape`, l.1052);
#     `ndim≥2`/matrix cells error. A NATIVE-constrained 1-D vector center
#     (simplex/ordered/positive_ordered, fixed `K`) IS carried now: it emits a
#     Stan `array[N…] <ct>[K]` parameter (`:constrained_vector`) so Stan applies
#     the per-cell constraint transform + jacobian; a plain `vector[K]` keeps the
#     dense `matrix[K,N]` packing (snag plate-constraine-90607054). Still dropped:
#     `~`-bound scalar constraints (lower/upper on a plain center) and constrained
#     MATRIX families (cholesky/cov/corr) — declare those at model scope.
#   • RAGGED cells: 1-D plain-vector with a DATA-computable per-cell length only
#     (`_plate_is_ragged_cell` / `_plate_ragged_plan`). N-D/arbitrary raggedness
#     and ragged CONSTRAINED cells (varying-`K` simplex/…) are rejected — Stan
#     cannot declare `array[N] simplex[K[g]]`.
#   • Per-cell LIKELIHOOD: the pointwise DENSITY (lpdf/lpmf) loop is compiler-owned
#     — an indexed data-LHS `obs[i] ~ dist(...)` routes to the model block — and a
#     cv-flipped per-cell PARAMETER redraws in GQ (`_indexed_rng_assignment`).
#     Per-cell OBSERVATION posterior-predictive RNG IS now synthesized too (snag
#     build-a-declarat-ab2d2471, superseding the 2026-07-16 boundary note): the gq
#     clone of the loop writes each draw into a compiler-owned `<obs>_gen` twin
#     declared with the observation's own type (`_indexed_obs_gen_base` /
#     `_push_obs_gen_decl!`, passes.jl). NOT covered: the pointwise
#     `<obs>_likelihood` vector (the whole-LHS expansion's other half — the cell
#     shape does not fix its container), and a RAGGED observation base, which has
#     no declarable Stan twin and keeps the model-only routing.
#   • cv/GQ taint does NOT flow through the outer sized declaration (same limit as
#     typed-LHS ranefs — cv section / parked override feature); vararg do-block
#     params (l.1193) and reduce_sum lowering are unimplemented.
# The StanBlocks primer's plate sections hold the acceptance-ladder roadmap.
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

    # Explicit `outer` accepts an int or a non-empty tuple. Without it, infer the
    # shipped 1-D shape from the first positional iterable.
    outer = get(plate.kwargs, :outer, nothing)
    outer_dims = if outer !== nothing
        dims = outer isa CanonicalExprV{:tuple} ? collect(outer.args) : Any[outer]
        isempty(dims) && error("plate: `outer=()` is invalid; pass an int or a non-empty tuple.")
        dims
    elseif !isempty(iterables)
        Any[:(length($(iterables[1])))]
    else
        error("plate: cannot size the plate — pass `outer=N`, `outer=(dims...)`, or at least one iterable.")
    end

    if isempty(iterables)
        length(params) == length(outer_dims) || error(
            "plate: without positional iterables, bind one do-block index per outer axis ",
            "($(length(params)) params for $(length(outer_dims)) axes)."
        )
    else
        length(params) == length(iterables) || error(
            "plate: $(length(params)) do-block params vs $(length(iterables)) positional iterables — ",
            "positional args are per-cell slices and must match 1:1."
        )
    end

    id = _next_inline_id()
    idxs = if length(outer_dims) == 1
        Symbol[Symbol(:plate_i, "__pl_", id)]
    else
        Symbol[Symbol(:plate_i, axis, "__pl_", id) for axis in eachindex(outer_dims)]
    end
    input_subst = Dict{Symbol,Any}()
    if isempty(iterables)
        for (param, idx) in zip(params, idxs)
            input_subst[param] = idx                    # `do i,j` ⇒ the cell indices
        end
    else
        for (a, it) in zip(params, iterables)
            input_subst[a] = _plate_input_accessor(it, idxs, info)   # per-cell scalar/slice; ragged → group vector view
        end
    end

    stmts = Any[s for s in body_raw.args if !(s isa LineNumberNode)]
    isempty(stmts) && error("plate: empty do-block body.")
    body_stmts, ret_expr = stmts[1:end-1], stmts[end]
    _plate_fresh_info(ret_expr) === nothing || error(
        "plate: the do-block must END with a cell-output VALUE expression, not a `~`/`=` statement."
    )

    # HYGIENE (multiple-plate snag): a fresh cell-local (`z ~ std_normal()`) emits
    # a model-scope Stan parameter named exactly `z` and binds `info[:z]`. Two
    # independent plates that reuse the same cell-local name then collide — the
    # second plate's discovery probe copies `info` (now holding `z`) and rejects
    # its own `z ~ …` as "LHS bound to a parameter-qualified value", and even
    # discovery aside, two `vector[N] z;` declarations are invalid Stan. Give every
    # plate its own NAMESPACE by prefixing each fresh cell-local with the plate's
    # result name `rv` (`z` → `b_subject_z`): `rv` is required and already unique
    # in model scope (asserted below), so distinct plates get distinct carriers,
    # and the emitted Stan reads as "the `z` of `b_subject`". The rename runs
    # BEFORE discovery, so a submodel-internal flattened name (`cell_z`, derived
    # from the direct binding `cell`) inherits the prefix (`rv_cell_z`) for free —
    # matching how submodel flattening already namespaces. Only genuinely fresh
    # names are renamed: a captured/model-scope reference (already in `info`) is
    # left alone, and a do-block PARAMETER used as an observation LHS
    # (`yi ~ normal(t, sigma)`) is a sliced INPUT, not a cell-local, so it is
    # excluded too (it lowers via `input_subst` to `y[plate_i]`).
    fresh_rename = Dict{Symbol,Symbol}()
    for s in body_stmts
        fi = _plate_fresh_info(s)
        fi === nothing && continue
        f = fi[1]
        (f in params || f in keys(info) || haskey(fresh_rename, f)) && continue
        fresh_rename[f] = Symbol(rv, :_, f)
    end
    if !isempty(fresh_rename)
        body_stmts = Any[_subst_syms(s, fresh_rename) for s in body_stmts]
        ret_expr = _subst_syms(ret_expr, fresh_rename)
    end

    # Lift every cell-INVARIANT binding and subexpression out of the loop before
    # anything else looks at the body, so discovery never sees a lifted name and
    # never promotes it to a per-cell collection. Emits into `info` + `pending`.
    body_stmts, ret_expr =
        _plate_hoist_invariants(body_stmts, ret_expr, rv, params, idxs, id; info)

    # Trace once in isolation to discover EVERY fresh binding — including
    # submodel-internal flattened names — and the cell result type. The emit
    # trace below uses these StanTypes rather than re-parsing LHS syntax.
    discovered = _plate_discover(body_stmts, ret_expr, params, iterables, idxs; info)
    fresh = discovered.fresh
    rv_type = rv_ct === nothing ? discovered.ret_type : _plate_annotation_type(rv_ct; info)

    # An `array[] int` cell-local is an EPHEMERAL per-cell INDEX array — a
    # `findall`/boolean-mask result such as an explicit `idx = findall(c .== 1)`, or
    # the index `y[c .== 1]` lowers to. It is NEVER collected across cells (its length
    # is `sum(mask)`, inherently ragged, and an int-array cell OUTPUT is unsupported
    # anyway), so it must NOT become an outer collection. It also cannot survive as a
    # loop-local: `distribute!` duplicates the plate loop into a transformed-data copy
    # (where the data-derived index is computed) and a model copy (where the obs uses
    # it), and a loop-local does not cross between those scopes (stanc "Identifier not
    # in scope"). So drop it from the promoted set and INLINE its defining `name = rhs`
    # into every use, recomputing the index in whichever block routes the use. The
    # boolean-mask SUGAR (`y[c .== 1]`) has no source binding to inline — its `findall`
    # is kept inline inside a plate by `expand_inline_or_trace` (builtin.jl) for the
    # same reason — so here we only drop its discovered `boolmask_idx_*` entry. (Snag
    # plate-cell-int: a per-cell index array feeding cmt-keyed do-block obs.)
    inline_int_names = Set{Symbol}(
        f for (f, T) in fresh if center_type(T) <: types.int && stan_ndim(T) >= 1
    )
    if !isempty(inline_int_names)
        fresh = Pair{Symbol,Any}[p for p in fresh if !(p.first in inline_int_names)]
        inline_map = Dict{Symbol,Any}()
        kept = Any[]
        for s in body_stmts
            fi = _plate_fresh_info(s)
            if fi !== nothing && s isa Expr && s.head === :(=) &&
                _plate_global_name(info, fi[1]) in inline_int_names
                inline_map[fi[1]] = s.args[2]
            else
                push!(kept, s)
            end
        end
        body_stmts = Any[_subst_syms(s, inline_map) for s in kept]
        ret_expr = _subst_syms(ret_expr, inline_map)
    end

    # `cell_types`/`fresh` are keyed by the GLOBAL (root-namespace) name of each
    # discovered cell binding; `rv` stays the do-block's LOCAL result name. For a
    # top-level plate these coincide, so the collision check compares rv's global
    # name against the fresh set.
    cell_types = Dict{Symbol,Any}(fresh)
    global_rv = _plate_global_name(info, rv)
    haskey(cell_types, global_rv) && error("plate: result `$rv` collides with a fresh binding in the do-block.")
    rv in keys(info) && error("plate: result `$rv` is already bound in model scope.")
    all_cell_types = copy(cell_types)
    all_cell_types[rv] = rv_type

    # A cell is ragged when its length depends on the per-cell position. That
    # position is aliased by the loop index AND the do-block param, each of which a
    # SubModel flattens (`g` ⇒ `sub_g`, `plate_i` ⇒ `sub_plate_i`) — so raggedness
    # must be tested against ALL of those names, not just the raw indices, or a
    # submodel-scoped ragged size (`K[sub_g]`) escapes detection and emits invalid
    # Stan (`matrix[K[sub_g], …]`).
    index_aliases = Set{Symbol}()
    for s in Iterators.flatten((idxs, params))
        push!(index_aliases, s)
        push!(index_aliases, _plate_global_name(info, s))
    end
    ragged_plans = Dict{Symbol,Any}()
    for (f, T) in all_cell_types
        _plate_is_ragged_cell(T, index_aliases) || continue
        ragged_plans[f] = _plate_ragged_plan(f, T, outer_dims, idxs, id)
    end
    # A ragged cell's flat-memory carriers (lens/ends/mem) are declared in the
    # plate's own `info` scope and indexed by the raw loop index; hoisting them out
    # of a called submodel (parent-scope carriers + flattened index) is not wired
    # yet. Fixed `vector[K]` and scalar cells ARE supported inside a submodel.
    (info isa SubModel && !isempty(ragged_plans)) && error(
        "plate: ragged vector cells inside a called @slic submodel are not supported yet — ",
        "use a fixed `vector[K]` cell here, or lift the ragged plate to model scope."
    )
    cell_accessors = Dict{Symbol,Any}(
        f => plan.accessor for (f, plan) in ragged_plans if haskey(cell_types, f)
    )

    # Positional do-block params still lower syntactically to per-cell input
    # accessors. Fresh names stay untouched: the task-local promotion context
    # rewrites them while tracing, including names hidden inside submodels.
    loop_body = Any[_subst_syms(s, input_subst) for s in body_stmts]
    ret_cell = _subst_syms(ret_expr, input_subst)
    rv_accessor = haskey(ragged_plans, rv) ?
        ragged_plans[rv].accessor : _plate_cell_index(rv, rv_type, idxs)
    push!(loop_body, :($rv_accessor = $ret_cell))

    # NB: build each iteration spec as `Expr(:(=), idx, 1:N)` — a hand-built
    # `Expr(:for, :(idx in 1:N), …)` yields an `:in` CALL spec, not the `:(=)`
    # form `forward!(::ForExpr)` asserts (a quoted `for` auto-normalizes it).
    loop = nothing
    for (idx, dim) in reverse(collect(zip(idxs, outer_dims)))
        body = loop === nothing ? Expr(:block, loop_body...) : Expr(:block, loop)
        loop = Expr(:for, Expr(:(=), idx, :(1:$dim)), body)
    end

    # Hoist declarations into the enclosing block exactly like an inline UDF,
    # then trace ONLY the loop under promotion. `forward!(::ForExpr)` binds the
    # loop index before its body, so promoted references can resolve `f[idx]`.
    pending = _get_inline_pending()
    # Ragged vector cells get a data-sized flat-memory carrier. First materialise
    # each cell length in transformed data (this accepts arbitrary data-only size
    # expressions, not merely `K[i]`), then cumulative ends and flat storage.
    # The logical RaggedVector binding is installed after the emit trace so it
    # captures the memory declaration's final promoted qualifier.
    for plan in values(ragged_plans)
        emitted = forward!(canonical(:($(plan.lens) :: int[$(outer_dims[1])])); info)
        pending !== nothing && push!(pending, emitted)
    end
    if !isempty(ragged_plans)
        sizing_body = Any[
            :($(plan.lens)[$(idxs[1])] = $(plan.size_expr))
            for plan in values(ragged_plans)
        ]
        sizing_loop = Expr(
            :for,
            Expr(:(=), idxs[1], :(1:$(outer_dims[1]))),
            Expr(:block, sizing_body...),
        )
        emitted = forward!(canonical(sizing_loop); info)
        pending !== nothing && push!(pending, emitted)
        for plan in values(ragged_plans)
            emitted = forward!(canonical(:($(plan.ends) = cumulative_sum($(plan.lens)))); info)
            pending !== nothing && push!(pending, emitted)
            emitted = forward!(canonical(:($(plan.mem) :: vector[sum($(plan.lens))])); info)
            pending !== nothing && push!(pending, emitted)
        end
    end
    # Outer collection declarations. FRESH cell collections carry GLOBAL names, so
    # emit them at the ROOT to bind each once without re-flattening. The `rv` result
    # collection keeps its LOCAL name and emits under `info`, so a SubModel flattens
    # it to the parent AND binds the local alias the submodel's `return rv` needs.
    # For a top-level plate `root === info`, so this reduces to the original loop.
    root = _plate_root_info(info)
    # A submodel-scoped plate emits its FRESH cell collections at the ROOT (under
    # already-global names) but the `rv` result under `info`. The `outer=` dims are
    # submodel-LOCAL names: a data arg like `n_groups` ALIASES the caller's binding
    # (e.g. the caller's `G`) rather than gaining a `<sub>_` prefix, so the raw
    # symbol resolves under `info` but NOT at the root. Resolve each dim under `info`
    # ONCE, up front, to the StanExpr it names — matching how the cell core size `K`
    # (`stan_size(T)[1]`, kept as a StanExpr by `_plate_outer_decl`) already comes
    # resolved from the traced cell type. Because `forward!(::StanExpr)` is
    # idempotent, these resolved dims re-trace cleanly whether the decl is emitted at
    # the root (fresh cells) or under `info` (the `rv`), so both targets name the
    # caller's binding rather than a raw local that need not exist there. A top-level
    # plate keeps the raw dims (identity), leaving its emitted Stan byte-for-byte
    # unchanged.
    plate_outer = info isa SubModel ?
        task_local_storage(:_slic_plate_context, nothing) do
            Any[forward!(canonical(d); info) for d in outer_dims]
        end : outer_dims
    for (f, T) in all_cell_types
        haskey(ragged_plans, f) && continue
        tgt = f === rv ? info : root
        cons = _plate_promoted_constraints(f, T, index_aliases)
        emitted = forward!(canonical(_plate_outer_decl(f, T, plate_outer)); info=tgt)
        emitted = _plate_constrain_decl(f, emitted, cons; info=tgt)
        pending !== nothing && push!(pending, emitted)
    end
    # The emit trace forwards each promoted cell accessor at the root, where the
    # loop index is bound under its flattened name; key the context idxs to match.
    global_idxs = Symbol[_plate_global_name(info, idx) for idx in idxs]
    ctx = (idxs=global_idxs, cell_types=cell_types, cell_accessors=cell_accessors)
    emitted_loop = task_local_storage(:_slic_plate_context, ctx) do
        forward!(canonical(loop); info)
    end
    for plan in values(ragged_plans)
        forward!(canonical(:($(plan.logical) = RaggedVector($(plan.mem), $(plan.ends)))); info)
    end
    emitted_loop
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
        # Computed type annotation (`typeof(...)` / `return_type_of(...)`, optionally
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
    ct isa Symbol && _is_model_decl_scope(info) &&
        (t = remake(t; fresh_decl=true, decl_role=:unfilled, qual=:parameter))
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
_decl_computed_size_alias(size, info) = begin
    aliases = _typed_assignment_aliases(info)
    size_key = _typed_assignment_shape_key(size, NamedTuple())
    # Only a raw `dims(arg)[i]` / `tok.i` access fragment can name a signature
    # dimension; anything else (a symbol, a compound size expression) is left
    # alone. `name in keys(info)` keeps this honest for a dimension whose
    # `int n = …;` binding was pruned as unused (§R5 addendum) — there is no
    # `n` in scope to render.
    size_key isa AbstractString || return size
    name = get(aliases, Symbol(size_key), nothing)
    (name === nothing || !(name in keys(info))) && return size
    info[name]
end

_decl_computed_type(tok, s; info) = begin
    tt = type(tok)
    center_type(tt) <: types.tokenof || error(
        "type-annotation expression must evaluate to a type token (e.g. `typeof(...)` ",
        "/ `return_type_of(...)`), got a value of Stan type `$(sigtype(tt))`."
    )
    cct = tt.info.value
    raw_size = isempty(s) ? stan_size(tt) : Tuple(forward!.(s; info))
    # Anonymous UDF args carry dimensions as raw `dims(arg)[i]` expressions.
    # Reuse the signature's emitted size binding (`int n = dims(arg)[i]`) so a
    # computed declaration emits `vector[n]`, not a quoted string dimension.
    sz = Tuple(_decl_computed_size_alias(size, info) for size in raw_size)
    autotype(StanType(cct, sz))
end
forward!(x::ForExpr; info) = begin
    @assert length(x.args) == 2
    head, body = x.args
    @assert _is_block_canonical(body)
    # One-line nested loops `for i in r1, j in r2, … ; body end` parse with a
    # `:block` head holding the successive `i = r1`, `j = r2` bindings; desugar to
    # genuinely nested single loops.
    _is_block_canonical(head) && return _forward_nested_for!(head, body; info)
    @assert _is_assign_canonical(head)
    idx = head.args[1]
    # Tuple binding ⇒ destructuring iteration (`for (i, xi) in enumerate(c)` /
    # `for (ai, bi) in zip(a, b)`); the enumerate/zip source is intercepted raw,
    # never forwarded on its own.
    idx isa TupleExpr && return _forward_destructured_for!(x, idx, head.args[2], body; info)
    @assert idx isa Symbol
    # Forward the iteration source first. A bounded `lo:hi` range takes the
    # index-iteration path below; any other indexable container value-iterates
    # (`for xi in c` → `for _vi in 1:length(c); xi = c[_vi]; <body>`). The range
    # never references `idx`, so forwarding it before binding `idx` is safe.
    idx_range = forward!(head.args[2]; info)
    if !_is_bounded_colon(idx_range)
        _is_value_iterable(idx_range) || error(
            "@deffun `for $idx in <iterable>`: iteration source is neither a bounded ",
            "`lo:hi` range nor an indexable container (Stan type `",
            sigtype(type(idx_range)), "`)."
        )
        return _forward_value_for!(x, idx, idx_range, body; info)
    end
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
    body = forward!(body; info)
    pop!(info, idx)
    stan_expr(remake(x, remake(head, emitted_idx, idx_range), body))
end

# Re-forward `x` (a ForExpr) as an ordinary index loop `for idx in range_raw` whose
# body is prefixed with `prelude_stmts` (the element bindings), scoping `idx` and
# the prelude names to the loop. Re-forwarding takes the bounded-range path above.
# Shared by value-iteration (`for xi in c`) and enumerate/zip destructuring.
function _reforward_index_loop!(x, idx, range_raw, prelude_stmts, scope_names, body; info)
    head, _ = x.args
    saved = _save_scope(info, scope_names)
    try
        forward!(remake(x, remake(head, idx, range_raw),
            remake(body, prelude_stmts..., body.args...)); info)
    finally
        _restore_scope!(info, saved)
    end
end

# `for xi in <container>`: desugar to `for _vi in 1:length(c); xi = c[_vi]; <body>`.
_forward_value_for!(x, var, container, body; info) = begin
    vi = _fresh_value_index(info)
    _reforward_index_loop!(x, vi, _value_iter_count(container),
        Any[CanonicalExpr(:(=), var, _value_iter_elem(container, vi))], Symbol[var], body; info)
end

# `for (…) in enumerate(c)/zip(a, b)`: destructure the tuple binding into a loop
# index + element bindings (see `_destructure_iteration`), then re-forward.
_forward_destructured_for!(x, lhs, source_raw, body; info) = begin
    idx, range_raw, preludes = _destructure_iteration(lhs, source_raw; info)
    _reforward_index_loop!(x, idx, range_raw,
        Any[CanonicalExpr(:(=), v, e) for (v, e) in preludes],
        Symbol[idx; Symbol[p.first for p in preludes]], body; info)
end

# `for i in r1, j in r2, … ; body end`: the `:block` head lists successive
# `var in range` bindings; desugar to genuinely nested single loops and re-forward.
# Each binding may be any supported iteration source (range, container, enumerate,
# zip), so nested loops compose with the destructuring/value-iteration forms.
_as_block(b::BlockExpr) = b
_as_block(b) = CanonicalExpr(:block, b)
function _forward_nested_for!(head, body; info)
    binds = head.args
    isempty(binds) && error("@deffun `for`: empty iteration head `", head, "`.")
    all(_is_assign_canonical, binds) || error(
        "@deffun one-line nested `for`: each `,`-separated clause must bind `var in range`, got `", head, "`."
    )
    nested = body
    for bind in reverse(binds)
        nested = CanonicalExpr(:for, bind, _as_block(nested))
    end
    forward!(nested; info)
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
