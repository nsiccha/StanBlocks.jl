stan_call(f, args...) = stan_expr(CanonicalExpr(f, map(stan_expr, args)...))
"""
"""
function stan_expr end
stan_expr(x::StanExpr; kwargs...) = weak_remake(x; kwargs...)
stan_expr(x; kwargs...) = stan_expr(x, x; kwargs...)
stan_expr(expr, value; kwargs...) = StanExpr(expr, stan_type(expr, value; kwargs...))
stan_expr(x, value::StanExpr; kwargs...) = weak_remake(value; kwargs...)
maybedata(expr, value; kwargs...) = stan_expr(expr, value; qual=:data, kwargs...)
maybedata(expr, value::Function; kwargs...) = stan_expr(value, value; qual=:data, kwargs...)
maybecv(expr, value) = stan_expr(expr, value; cv=true)

"""
    stan_model(slic::SlicModel) -> StanModel

Trace `slic` end-to-end (forward / backward / distribute passes), returning the
inferred [`StanModel`](@ref StanBlocks.StanModel) — a fully resolved
representation of the Stan blocks plus the data dictionary.

Tracing is the expensive step. A `StanModel` is **cheap to re-data**: call
`model(; new_kwargs...)` to swap the data dict without re-tracing. Use
this in preference to repeatedly calling [`stan_instantiate`](@ref
StanBlocks.stan_instantiate) on the original `SlicModel`.

Errors during tracing are wrapped in a [`StanBlocksError`](@ref
StanBlocks.StanBlocksError) tagged with `phase = :transpile`.
"""
function stan_model end
# Internal alias for the sibling error type (StanBlocks.jl defines it before this
# include). Reference it DIRECTLY — not via `parentmodule(@__MODULE__)`. That
# indirection was correct only while SLIC lived in the nested `module stan`
# (parent = StanBlocks); after the hoist (1a3a4ae) `@__MODULE__` IS StanBlocks, so
# under `include`-into-Main-from-source `parentmodule` resolves to `Main` and
# `Main.StanBlocksError` is undefined (a normal `using` load is masked because a
# top-level package module is its own parent).
const _StanBlocksError = StanBlocksError
_is_stanblocks_error(e::_StanBlocksError) = true
_is_stanblocks_error(_) = false

# --- Per-trace monotonic id counters (thread-safe + deterministic) -----------
# `_next_inline_id`/`_next_closure_id`/`_next_anon_id` hand out the ids used for
# inlined-UDF local renames (`name__il_<id>`, forward.jl), lifted-closure Stan
# fn names + comments (`// lifted closure (id <id>)`, closures.jl), and per-call
# anon-arg placeholders (`_arg<tok>_<i>`, functions.jl). These ids must be
# unique WITHIN a single transpilation but must NOT carry across transpilations:
# a module-global counter would (a) data-race under concurrent tracing and (b)
# make the emitted Stan depend on how many prior inlines/closures ran this
# session — non-deterministic generation that also defeats the `hash(stan_code)`
# cache. So the counters live in PER-TASK storage, seeded FRESH per trace by
# `_with_trace_counters` (called from the `stan_model` wrapper below). The lazy
# `get!` is the get-with-default for any call OUTSIDE a trace scope (only the
# anon counter is reachable there, via ad-hoc `stan_expr(::CanonicalExpr)`;
# determinism is irrelevant there since anon placeholders are always
# deanonymized away) — it returns a task-local `Ref`, so still thread-safe.
_trace_counter(key) = get!(() -> Ref(0), task_local_storage(), key)
_next_inline_id()  = (_trace_counter(:_slic_inline_counter)[]  += 1)
_next_closure_id() = (_trace_counter(:_slic_closure_counter)[] += 1)
_next_anon_id()    = (_trace_counter(:_slic_anon_counter)[]    += 1)
# Seed all three counters fresh (scoped, restored on exit) around one trace.
_with_trace_counters(body) =
    task_local_storage(:_slic_inline_counter, Ref(0)) do
        task_local_storage(:_slic_closure_counter, Ref(0)) do
            task_local_storage(:_slic_anon_counter, Ref(0)) do
                body()
            end
        end
    end

stan_model(x::SlicModel; info=StanModel()) = begin
    _expr_stack = Any[]
    _current_lnn = Ref{Any}(nothing)
    info = remake(info; _expr_stack, _current_lnn)
    task_local_storage(:_slic_expr_stack, _expr_stack) do
        task_local_storage(:_slic_current_lnn, _current_lnn) do
            task_local_storage(:_slic_inline_pending, Any[]) do
                _with_trace_counters() do
                    try
                        distribute!(backward!(forward!(x; info); info); info)
                        remake(info; docstring=get(x.data, :docstring, ""))
                    catch e
                        _is_stanblocks_error(e) && rethrow()
                        bt = catch_backtrace()
                        # Keep the raw (exception, backtrace, expr_stack) tuple so the
                        # display layer can show the Julia traceback alongside the
                        # SLIC-level expression trace.
                        throw(_StanBlocksError(:transpile, "model", (e, bt, copy(_expr_stack))))
                    end
                end
            end
        end
    end
end
maybedata!(x::StanModel, key, value) = x[key] = maybedata(key, value)
maybedata!(x::StanModel, key, value::AbstractString) = nothing
maybedata!(x::SubModel, key, value) = locals(x)[key] = maybedata(key, value)

# --- Partly-missing-vector imputation helpers --------------------------------

# True when a data value is a vector with at least one `missing` entry.
_is_partly_missing_vec(v) = false
_is_partly_missing_vec(v::AbstractVector{T}) where {T>:Missing} = any(ismissing, v)

# Distributions that couple all elements of their outcome vector (off-diagonal
# covariance means obs and mis entries cannot be independently conditioned).
# A partly-missing LHS with any of these is an error.
_joint_dist_names() = Set{Symbol}([
    :multi_normal, :multi_normal_prec, :multi_normal_cholesky,
    :multi_gp, :multi_gp_cholesky,
    :multi_student_t, :multi_student_t_cholesky,
    :wishart, :inv_wishart, :wishart_cholesky, :inv_wishart_cholesky,
    :lkj_corr, :lkj_corr_cholesky, :gaussian_dlm_obs,
])

# Scan `data` for partly-missing vectors.  For each one found:
#   - validate continuous element type
#   - check for split-name collision
#   - register y_obs / y_ii_obs / y_ii_mis in `info` (split vars as data)
#   - register y_ii_mis_n explicitly so it can be used as a size symbol
#   - return a Dict mapping original name → split metadata
_scan_and_register_missing_vars!(data, info) = begin
    result = OrderedDict{Symbol,NamedTuple}()
    for (key, value) in pairs(data)
        _is_partly_missing_vec(value) || continue
        eltype(value) <: Union{Missing,<:AbstractFloat} || error(
            "Partly-missing vector `$(key)`: auto-imputation only supports continuous " *
            "(floating-point) outcome vectors — Stan has no discrete parameters. " *
            "Got element type $(eltype(value)). For discrete missing data, split the " *
            "model manually or marginalise out the missing entries."
        )
        obs_vals = Float64[v for v in value if !ismissing(v)]
        ii_obs   = Int[i for (i, v) in enumerate(value) if !ismissing(v)]
        ii_mis   = Int[i for (i, v) in enumerate(value) if ismissing(v)]
        n_obs = length(ii_obs); n_mis = length(ii_mis)
        # Guard against accidental collision with user-defined data keys.
        for (suffix, kind) in (
            (:_obs, "y_obs (observed values)"),
            (:_ii_obs, "y_ii_obs (observed indices)"),
            (:_ii_mis, "y_ii_mis (missing indices)"),
        )
            sk = Symbol(key, suffix)
            sk in keys(data) && error(
                "Partly-missing vector `$(key)`: auto-generated split name `$(sk)` " *
                "($(kind)) already exists in the data dict. Rename the conflicting key."
            )
        end
        # Register the three split data arrays; their _n sizes are auto-named
        # by stan_type (e.g. y_obs → y_obs_n, y_ii_mis → y_ii_mis_n).
        maybedata!(info, Symbol(key, :_obs),    obs_vals)
        maybedata!(info, Symbol(key, :_ii_obs), ii_obs)
        maybedata!(info, Symbol(key, :_ii_mis), ii_mis)
        # Make the missing-entry count explicitly resolvable as a size symbol
        # (y_ii_mis_n == n_mis); used in the typed-LHS `~` for y_mis.
        info[Symbol(key, :_ii_mis_n)] = maybedata(Symbol(key, :_ii_mis_n), n_mis)
        result[key] = (;obs_vals, ii_obs, ii_mis, n_obs, n_mis)
    end
    result
end

# Walk the raw body block (before canonical/forward!), replacing each
# `y ~ dist(args...)` where `y` is in `missing_vars` with a 3-statement
# expansion: typed-LHS parameter ~, data ~ for obs likelihood, assembly.
# Hoists compound dist args into temporaries to avoid double-eval.
_expand_missing_stmts(body::Expr, missing_vars) = begin
    @assert body.head == :block
    new_args = Any[]
    seen = Set{Symbol}()
    for arg in body.args
        replacement = _try_expand_missing_stmt(arg, missing_vars, seen)
        if replacement !== nothing
            append!(new_args, replacement)
        else
            push!(new_args, arg)
        end
    end
    unseen = setdiff(keys(missing_vars), seen)
    isempty(unseen) || error(
        "Partly-missing vector(s) " *
        join([string("`", k, "`") for k in sort!(collect(unseen))], ", ") *
        " never appear as the LHS of a `~` statement. Auto-imputation only supports " *
        "missing OUTCOMES (left-hand side of `~`). Missing predictors/covariates on " *
        "the right-hand side are out of scope — split the model manually."
    )
    Expr(:block, new_args...)
end

_try_expand_missing_stmt(arg, missing_vars, seen) = begin
    # Match: Expr(:call, :~, name::Symbol, Expr(:call, dist_fn, dist_args...))
    Meta.isexpr(arg, :call) || return nothing
    arg.args[1] === :~ || return nothing
    length(arg.args) >= 3 || return nothing
    name = arg.args[2]
    name isa Symbol && name in keys(missing_vars) || return nothing
    dist_call = arg.args[3]
    Meta.isexpr(dist_call, :call) || return nothing
    dist_fn = dist_call.args[1]
    dist_fn in _joint_dist_names() && error(
        "Partly-missing vector `$(name) ~ $(dist_fn)(...)`: inherently-joint " *
        "distributions cannot be element-wise imputed — obs and mis entries are " *
        "coupled through the off-diagonal covariance. Use a conditional-normal " *
        "formula or an explicit obs/mis split instead."
    )
    name in seen && error(
        "Partly-missing vector `$(name)` appears as the LHS of more than one `~` " *
        "statement. Auto-imputation requires exactly one sampling statement per " *
        "missing-data vector."
    )
    push!(seen, name)
    raw_dist_args = dist_call.args[2:end]
    obs_sym   = Symbol(name, :_obs)
    mis_sym   = Symbol(name, :_mis)
    ii_obs    = Symbol(name, :_ii_obs)
    ii_mis    = Symbol(name, :_ii_mis)
    mis_n     = Symbol(name, :_ii_mis_n)
    # Hoist compound args (non-Symbol) so each is evaluated once, not twice.
    hoisted_stmts = Any[]
    final_args    = Any[]
    for (i, darg) in enumerate(raw_dist_args)
        if darg isa Symbol
            push!(final_args, darg)
        else
            tmp = Symbol(name, :_arg_, i)
            push!(hoisted_stmts, :($tmp = $darg))
            push!(final_args, tmp)
        end
    end
    mis_dist = Expr(:call, dist_fn, [:(maybe_index($a, $ii_mis)) for a in final_args]...)
    obs_dist = Expr(:call, dist_fn, [:(maybe_index($a, $ii_obs)) for a in final_args]...)
    # Typed-LHS ~ introduces y_mis as a parameter sized by y_ii_mis_n.
    typed_lhs = :($mis_sym::vector[$mis_n])
    stmts = Any[
        hoisted_stmts...,
        Expr(:call, :~, typed_lhs, mis_dist),
        Expr(:call, :~, obs_sym,   obs_dist),
        :($name = merge_missing($obs_sym, $mis_sym, $ii_obs, $ii_mis)),
    ]
    stmts
end
_control_flow_kind(::ForExpr) = "for"
_control_flow_kind(::WhileExpr) = "while"
_control_flow_kind(::IfExpr) = "if"
_control_flow_kind(::ElseIfExpr) = "elseif"
_control_flow_kind(::BreakExpr) = "break"
_control_flow_kind(::ContinueExpr) = "continue"
_control_flow_kind(::ComprehensionExpr) = "comprehension"
_reject_model_control_flow(x) = x
_reject_model_control_flow(x::CanonicalExpr) = (foreach(_reject_model_control_flow, x.args); x)
_reject_model_control_flow(x::Union{ForExpr,WhileExpr,IfExpr,ElseIfExpr,BreakExpr,ContinueExpr,ComprehensionExpr}) = error(
    "`$(_control_flow_kind(x))` control flow is not supported in @slic model bodies — ",
    "move the logic into an @deffun function body, or use a vectorised form."
)
# Slice/element assignment (`v[a:b] = …`, `v[i] = …`) is declare-then-fill: the
# indexed LHS forwards to a non-Symbol expr that is never registered in `info`,
# so the backward pass would otherwise die with a raw `KeyError` from the
# OrderedDict. Reject it loudly here (BEFORE `forward!`, model-bodies only — so
# it fires even when the RHS wouldn't resolve, and never touches @deffun bodies,
# which ARE traced forward-only and DO support indexed assignment). §R3.
_reject_model_control_flow(x::AssignmentExpr{<:CanonicalExprV{:getindex}}) = error(
    "Indexed/slice assignment `", x.args[1].args[1], "[…] = …` is not supported in a @slic model ",
    "body — only whole-variable assignment `name = …` is. Declare-then-fill (`v[a:b] = expr`) is ",
    "available only inside an @deffun function body; move it there and call it (e.g. `v = fill_v(…)`)."
)
"""
Descends forwards through the expression tree. Basically _always_ returns a StanExpr.
"""
function forward! end
forward!(x::SlicModel; info=StanModel()) = begin
    info = remake(info; mod=x.mod)
    missing_vars = _scan_and_register_missing_vars!(data(x), info)
    for (key, value) in pairs(data(x))
        key in keys(missing_vars) && continue   # split vars already registered above
        maybedata!(info, key, value)
    end
    raw_body = model(x)
    raw_body = isempty(missing_vars) ? raw_body : _expand_missing_stmts(raw_body, missing_vars)
    body = canonical(raw_body)
    _reject_model_control_flow(body)
    forward!(body; info)
end

isexpr(h) = Base.Fix2(isexpr, h)
isexpr(x, h) = false
isexpr(x::CanonicalExpr, h) = head(x) == h
canonical(x) = x
canonical(x::Expr) = if x.head === :->
    # Lambdas (`(x) -> body`): keep `lhs` and `body` as raw, un-canonicalised
    # AST. `forward!(::CanonicalExprV{:->, ...})` snapshots the raw body for
    # later substitution — `inline_substitute` walks Expr/Symbol, not
    # CanonicalExpr.
    CanonicalExpr(x.head, x.args...)
elseif x.head === :do
    # `f(args) do x; body end` parses to `Expr(:do, Expr(:call, f, args...),
    # Expr(:->, Expr(:tuple, x), body))`. Desugar to the equivalent
    # `f((x) -> body, args...)` by injecting the lambda as the call's first
    # positional arg, then re-canonicalise.
    @assert length(x.args) == 2 "canonical(:do): expected 2 args (call, lambda), got $(length(x.args))."
    call_expr, lambda = x.args
    @assert Meta.isexpr(call_expr, :call) "canonical(:do): first arg must be a `:call` Expr, got `$call_expr`."
    @assert Meta.isexpr(lambda, :->) "canonical(:do): second arg must be a `:->` lambda Expr, got `$lambda`."
    canonical(Expr(:call, call_expr.args[1], lambda, call_expr.args[2:end]...))
else
    CanonicalExpr(x.head, canonical.(x.args)...)
end
ensure_kw(x::CanonicalExprV{:kw}) = x
ensure_kw(x::Symbol) = CanonicalExpr(:kw, x, x)
ensure_kw(x::CanonicalExprV{:.}) = CanonicalExpr(:kw, kw_name(x), x)
kw_name(x::CanonicalExprV{:.}) = kw_name(x.args[2])
kw_name(x::QuoteNode) = x.value
canonical(x::CanonicalExprV{:parameters}) = if all(isexpr(:kw), x.args)
    x
else
    CanonicalExpr(x.head, ensure_kw.(x.args)...)
end
canonical(x::CanonicalExprV{:call}) = begin
    f = broadcast_callee(x.args[1])
    args = []
    kwargs = []
    for arg in x.args[2:end]
        if isexpr(arg, :parameters)
            @assert all(isexpr(:kw), arg.args) "canonical(:call): `:parameters` block must contain only `:kw` entries, got `$(arg.args)`."
            for argi in arg.args
                push!(kwargs, argi.args[1]=>argi.args[2])
            end
        elseif isexpr(arg, :kw) 
            push!(kwargs, arg.args[1]=>arg.args[2])
        else
            push!(args, arg)
        end
    end
    CanonicalExpr(f, args...; kwargs...)
end
canonical(x::CanonicalExprV{:tuple}) = begin
    @assert length(x.args) > 0 "canonical(:tuple): empty tuple expression `$x`."
    if any(Base.Fix2(isexpr, :parameters), x.args)
        @assert length(x.args) == 1 "canonical(:tuple): a `:parameters`-bearing tuple must have exactly one arg (the parameters block), got $(length(x.args)) in `$x`."
        @assert all(isexpr(:kw), x.args[1].args) "canonical(:tuple): named-tuple parameters block must contain only `:kw` entries, got `$(x.args[1].args)`."
        CanonicalExpr(:nt, x.args[1].args...)
    else
        x
    end
end
canonical(x::CanonicalExprV{:macrocall}) = begin
    head = x.args[1]
    is_doc = head == GlobalRef(Core, Symbol("@doc")) || head == Symbol("@doc")
    @assert is_doc "canonical(:macrocall): only `@doc` macrocalls are supported in SLIC bodies, got `$(head)`."
    CanonicalExpr(:document, x.args[3:4]...)
end
canonical(x::CanonicalExprV{Symbol("'")}) = CanonicalExpr(:adjoint, x.args...)
canonical(x::CanonicalExprV{:ref}) = CanonicalExpr(:getindex, x.args...)
# Broadcasting operators in `@slic` / `@deffun` bodies. Julia's dotted operators
# all denote element-wise ops; `broadcast_callee` rewrites the call's callee
# symbol so each transpiles to valid Stan, and raises a descriptive error for
# any dotted operator that is not (yet) handled:
#   `.+` / `.-`        -> plain `+` / `-`  — Stan's `+`/`-` already broadcast
#                         scalars over vectors/matrices; there is no dotted form.
#   `.*` / `./` / `.^` / `.÷` -> `Base.BroadcastFunction` — Stan's arithmetic is
#                         matrix-oriented, so element-wise variants need this form.
# Resolved once, at the `:call` canonicalization site, so adding a broadcast
# operator is a single new method below.
broadcast_callee(f) = f
broadcast_callee(f::Symbol) = broadcast_callee(f, Val(f))
broadcast_callee(::Symbol, ::Val{Symbol(".+")}) = +
broadcast_callee(::Symbol, ::Val{Symbol(".-")}) = -
broadcast_callee(::Symbol, ::Val{Symbol(".*")}) = .*
broadcast_callee(::Symbol, ::Val{Symbol("./")}) = ./
broadcast_callee(::Symbol, ::Val{Symbol(".^")}) = .^
broadcast_callee(::Symbol, ::Val{Symbol(".÷")}) = Base.BroadcastFunction(÷)
broadcast_callee(f::Symbol, ::Val) = begin
    s = string(f)
    if startswith(s, '.') && length(s) > 1 && s != ".."
        error(
            "SLIC: broadcast operator `$f` is not supported in `@slic`/`@deffun` bodies. " *
            "Supported broadcast operators: `.+` `.-` `.*` `./` `.^` `.÷`. " *
            "Stan's plain `+`, `-`, `*` already broadcast scalars over vectors/matrices, " *
            "so a non-dotted operator is often what you want. To add support for `$f`, " *
            "define a matching `broadcast_callee(::Symbol, ::Val{Symbol(\"$s\")})` method in tracing.jl."
        )
    end
    f
end
pretty_type_expr(T::Symbol) = string(T)
pretty_type_expr(ref::CanonicalExprV{:getindex}) = string(ref.args[1], "[", join(ref.args[2:end], ", "), "]")

# TODO: task-local storage is used here to propagate _expr_stack/_current_lnn through
# @deffun calls, which rebuild `info` from scratch. Cleaner alternatives:
# - Thread info through stan_expr (invasive: changes signature everywhere + codegen)
# - Carry refs as CanonicalExpr kwargs (requires codegen changes in functions.jl)
_get_expr_stack(info) = something(_expr_stack(info), get(task_local_storage(), :_slic_expr_stack, nothing), Some(nothing))
_get_lnn_ref(info) = something(_current_lnn(info), get(task_local_storage(), :_slic_current_lnn, nothing), Some(nothing))
_get_lnn(info) = _deref_lnn(_get_lnn_ref(info))
_deref_lnn(r::Ref) = r[]
_deref_lnn(_) = nothing
_push_expr!(info, x) = (s = _get_expr_stack(info); isnothing(s) || push!(s, (x, _get_lnn(info))); nothing)
_pop_expr!(info) = (s = _get_expr_stack(info); isnothing(s) || pop!(s); nothing)

forwards!(;info) = x->forwards!(x; info)
forwards!(x; info) = [forward!(x; info)]
forwards!(x::SplatExpr; info) = [forward!(x.args[1]; info)...]
forward!(x; info) = error("forward! not defined for value `$x` of type `$(typeof(x))` — no method matches a more specific signature.")
forward!(;info) = x->forward!(x; info)
forward!(x::Tuple; info) = (mapreduce(forwards!(;info), vcat, x; init=[])...,)
forward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(forward!(;info), x)
forward!(x::Union{Number,Function,Nothing}; info) = x
# Wrap string literals as StanExprs so they can be passed as args to
# Stan functions like `reject` / `print` and survive `tracetype`'s
# `center_type(type(arg))` walk. Renders quoted at emission time.
forward!(x::AbstractString; info) = stan_expr(x, x; qual=:data)
forward!(x::LineNumberNode; info) = (_set_lnn!(_get_lnn_ref(info), x); x)
_set_lnn!(r::Ref, x) = (r[] = x; nothing)
_set_lnn!(_, _) = nothing
forward!(x::QuoteNode; info) = x.value
# Built-in mathematical constants (π, ℯ, γ, φ, catalan — all `Irrational`s)
# resolve to their `Float64` value. Per user decision `3bbtrv` (comment: "only
# pi and e or other built in constants"): the built-in constants resolve;
# arbitrary module-level numbers do not (no `::Number` resolution method).
forward!(x::Irrational; info) = forward!(Float64(x); info)
forward!(x::Number; info) = maybedata(x, x)
get_module(info::StanModel) = get(info.meta, :mod, Main)
get_module(info::AbstractDict) = get(info, :__mod__, Main)
get_module(info::NamedTuple) = get(info, :__mod__, Main)
get_module(info::SubModel) = get_module(parent(info))
# Plate emission installs a task-local promotion context while re-tracing its
# compiler-owned loop.  The forward layer adds the StanModel/SubModel method;
# this floor keeps ordinary symbol resolution independent of that feature.
_plate_promoted_reference(x, info) = nothing
forward!(x::Symbol; info) = begin
    if x in keys(info)
        promoted = _plate_promoted_reference(x, info)
        promoted === nothing || return promoted
        return info[x]
    end
    _is_builtin_name(x) && return forward!(getproperty(builtin, x); info)
    mod = get_module(info)
    if isdefined(mod, x)
        Mx = getproperty(mod, x)
        rv = _forward_module_value(Mx, info)
        rv === nothing && error("Found $x in $(mod), but is of type $(typeof(Mx))!")
        return rv
    end
    if mod !== Main && isdefined(Main, x)
        Mx = getproperty(Main, x)
        rv = _forward_module_value(Mx, info)
        rv === nothing && error("Found $x in Main, but is of type $(typeof(Mx))!")
        return rv
    end
    error("Could not find $(x) in model, builtin, $(mod) or Main!")
end
