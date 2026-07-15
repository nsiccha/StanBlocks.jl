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
forward!(x::AssignmentExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
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
    base_lhs_type = StanType(ct_resolved, sizes_forwarded)
    _is_canonical_expr(rhs_raw) || error(
        "Typed-LHS sampling `$name::$(pretty_type_expr(type_expr)) ~ rhs` requires rhs to be a distribution call"
    )
    head_resolved = forward!(head(rhs_raw); info)
    args_resolved = collect(forward!(rhs_raw.args; info))
    kwargs_resolved = forward!(rhs_raw.kwargs; info)
    rhs_canonical = CanonicalExpr(head_resolved, args_resolved...; kwargs_resolved...)
    cv_args = any(stan.cv, args_resolved)
    qual = cv_args ? :quantities : :parameter
    info[name] = StanExpr(name, remake(base_lhs_type; qual, cv=cv_args))
    rhs_stan = StanExpr(rhs_canonical, info[name].type)
    remake(x, info[name], rhs_stan)
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
    lhs, type = x.args
    ct, s... = if _is_getindex_expr(type)
        type.args
    else
        (type, )
    end
    @assert ct isa Symbol
    ct = gettype(ct)
    t = StanType(ct, forward!.(s; info))
    rv = StanExpr(lhs, t)
    lhs isa Symbol || return StanExpr(expr(forward!(lhs; info)), t)
    info[lhs] = rv 
    stan_expr(remake(x, rv))
end
forward!(x::ForExpr; info) = begin
    @assert length(x.args) == 2
    head, body = x.args
    @assert _is_assign_canonical(head)
    @assert _is_block_canonical(body)
    idx = head.args[1]
    @assert idx isa Symbol
    info[idx] = StanExpr(idx, StanType(types.int))
    idx_range = forward!(head.args[2]; info)
    body = forward!(body; info)
    pop!(info, idx)
    stan_expr(remake(x, remake(head, idx, idx_range), body))
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
