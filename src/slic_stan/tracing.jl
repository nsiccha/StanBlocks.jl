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
const _StanBlocksError = parentmodule(@__MODULE__).StanBlocksError
_is_stanblocks_error(e::_StanBlocksError) = true
_is_stanblocks_error(_) = false
stan_model(x::SlicModel; info=StanModel()) = begin
    _expr_stack = Any[]
    _current_lnn = Ref{Any}(nothing)
    info = remake(info; _expr_stack, _current_lnn)
    task_local_storage(:_slic_expr_stack, _expr_stack) do
        task_local_storage(:_slic_current_lnn, _current_lnn) do
            task_local_storage(:_slic_inline_pending, Any[]) do
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
maybedata!(x::StanModel, key, value) = x[key] = maybedata(key, value)
maybedata!(x::StanModel, key, value::AbstractString) = nothing
maybedata!(x::SubModel, key, value) = locals(x)[key] = maybedata(key, value)
_control_flow_kind(::ForExpr) = "for"
_control_flow_kind(::WhileExpr) = "while"
_control_flow_kind(::IfExpr) = "if"
_control_flow_kind(::ElseIfExpr) = "elseif"
_control_flow_kind(::BreakExpr) = "break"
_control_flow_kind(::ContinueExpr) = "continue"
_reject_model_control_flow(x) = x
_reject_model_control_flow(x::CanonicalExpr) = (foreach(_reject_model_control_flow, x.args); x)
_reject_model_control_flow(x::Union{ForExpr,WhileExpr,IfExpr,ElseIfExpr,BreakExpr,ContinueExpr}) = error(
    "`$(_control_flow_kind(x))` control flow is not supported in @slic model bodies — ",
    "move the logic into an @deffun function body, or use a vectorised form."
)
"""
Descends forwards through the expression tree. Basically _always_ returns a StanExpr.
"""
function forward! end
forward!(x::SlicModel; info=StanModel()) = begin
    info = remake(info; mod=x.mod)
    for (key, value) in pairs(data(x))
        maybedata!(info, key, value)
    end
    body = canonical(model(x))
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
#   `.*` / `./` / `.^` -> `Base.BroadcastFunction` — Stan's `*`/`/`/`^` are matrix
#                         ops, so the element-wise variant needs the dotted Stan op.
# Resolved once, at the `:call` canonicalization site, so adding a broadcast
# operator is a single new method below.
broadcast_callee(f) = f
broadcast_callee(f::Symbol) = broadcast_callee(f, Val(f))
broadcast_callee(::Symbol, ::Val{Symbol(".+")}) = +
broadcast_callee(::Symbol, ::Val{Symbol(".-")}) = -
broadcast_callee(::Symbol, ::Val{Symbol(".*")}) = .*
broadcast_callee(::Symbol, ::Val{Symbol("./")}) = ./
broadcast_callee(::Symbol, ::Val{Symbol(".^")}) = .^
broadcast_callee(f::Symbol, ::Val) = begin
    s = string(f)
    if startswith(s, '.') && length(s) > 1 && s != ".."
        error(
            "SLIC: broadcast operator `$f` is not supported in `@slic`/`@deffun` bodies. " *
            "Supported broadcast operators: `.+` `.-` `.*` `./` `.^`. " *
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
forward!(x::Irrational; info) = error("forward! not defined for irrational `$x` — only `π` is handled; convert to a concrete numeric value first.")
forward!(x::Irrational{:π}; info) = forward!(Float64(pi); info)
forward!(x::Number; info) = maybedata(x, x)
get_module(info::StanModel) = get(info.meta, :mod, Main)
get_module(info::AbstractDict) = get(info, :__mod__, Main)
get_module(info::NamedTuple) = get(info, :__mod__, Main)
get_module(info::SubModel) = get_module(parent(info))
forward!(x::Symbol; info) = begin
    x in keys(info) && return info[x]
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
