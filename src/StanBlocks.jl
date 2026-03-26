module StanBlocks

export @stan, @model, @parameters, @transformed_parameters, @generated_quantities, @bsum, with_gradient
export @slic, @defsig, @deffun
export stan_code, stan_model, stan_instantiate
export StanBlocksError

using LinearAlgebra, Statistics, Distributions, LogExpFunctions, JSON

# --- Error type for StanBlocks computations (defined early so submodules can use it) ---

"""
    StanBlocksError <: Exception

Wraps errors that occur during transpilation, compilation, or evaluation of Stan models.

# Fields
- `phase::Symbol`: the pipeline stage where the error occurred (`:transpile`, `:compile`, or `:evaluate`)
- `context::String`: a description of what was being processed (e.g. `"model: eight_schools"`)
- `cause::Any`: the underlying error, typically an `(exception, backtrace)` tuple
"""
struct StanBlocksError <: Exception
    phase::Symbol        # :transpile, :compile, :evaluate
    context::String      # e.g. "model: eight_schools"
    cause::Any           # (exception, backtrace) tuple
end

include("wrapper.jl")
include("macros.jl")
include("functions.jl")
include("slic_stan/slic.jl")

julia_implementation(key; kwargs...) = missing
slic_implementation(key; kwargs...) = nothing
stan_implementation(key; kwargs...) = missing
include("check.jl")

# --- Error display for StanBlocks computations ---

_cause_error(e::StanBlocksError) = e.cause isa Tuple ? e.cause[1] : e.cause
_cause_backtrace(e::StanBlocksError) = e.cause isa Tuple ? e.cause[2] : []
_cause_expr_stack(e::StanBlocksError) = e.cause isa Tuple && length(e.cause) >= 3 ? e.cause[3] : []

unwrap_error(e::Base.TaskFailedException) = unwrap_error(e.task.exception)
unwrap_error(e::CompositeException) = unwrap_error(first(e.exceptions))
unwrap_error(e::StanBlocksError) = unwrap_error(_cause_error(e))
unwrap_error(e) = e

_filter_bt(bt) = filter(bt) do frame
    file = string(frame.file)
    !any(p -> occursin(p, file), (
        "StanBlocks.jl/src", "HTMXObjects.jl/src",
        "DynamicObjects.jl/src",
        "/Oxygen/", "/HTTP/", "task.jl", "lock.jl",
        "essentials.jl", "dict.jl",
    ))
end

function _format_frame(frame)
    try
        if frame.linfo isa Core.MethodInstance
            sig = frame.linfo.specTypes
            params = fieldtypes(sig)
            fname = string(frame.func)
            arg_strs = ["::$(p)" for p in params[2:end]]
            return fname * "(" * join(arg_strs, ", ") * ")"
        end
    catch
    end
    return string(frame.func)
end

# Extract expression-trace frames from the backtrace:
# find forward!/distribute!/backward! calls on CanonicalExpr/StanExpr types
# and format the first type arg using short_expr-style rendering.
_is_transpile_func(name) = name in (:forward!, :distribute!, :backward!, :tracetype, :lpxf_expr, :rng_expr, :likelihood_expr)

function _type_short_expr(T::Type)
    if T <: stan.CanonicalExpr
        H, A = T.parameters
        head_str = _type_head_str(H)
        A <: Tuple || return head_str
        arg_strs = _type_short_expr.(fieldtypes(A))
        return "$head_str($(join(arg_strs, ", ")))"
    elseif T <: stan.StanExpr
        E, ST = T.parameters
        # StanType{real, 0} → extract the center type from the type parameter
        ct = ST <: stan.StanType ? ST.parameters[1] : ST
        return "::$ct"
    elseif T == Symbol
        return "<Symbol>"
    elseif T <: Number
        return "::$T"
    else
        return string(T)
    end
end

function _type_head_str(H::Type)
    H <: Val && return string(H.parameters[1])
    if H <: Function
        s = string(H)
        s = replace(s, r"^typeof\((.*)\)$" => s"\1")
        # StanBlocks.stan.builtin.normal → normal
        s = replace(s, r"^.*\." => "")
        return s
    end
    return string(H)
end

function _expr_trace(frames)
    trace = String[]
    seen = Set{String}()
    for frame in frames
        _is_transpile_func(frame.func) || continue
        frame.linfo isa Core.MethodInstance || continue
        params = fieldtypes(frame.linfo.specTypes)
        length(params) >= 2 || continue
        T = params[2]
        (T <: stan.CanonicalExpr || T <: stan.StanExpr) || continue
        s = try _type_short_expr(T) catch; continue end
        s in seen && continue
        push!(seen, s)
        push!(trace, s)
    end
    trace
end

Base.show(io::IO, e::StanBlocksError) = showerror(io, e)
Base.show(io::IO, ::MIME"text/plain", e::StanBlocksError) = showerror(io, e)

function _showerror_header(io::IO, e::StanBlocksError)
    print(io, "StanBlocksError [$(e.phase)]: $(e.context)\n")
    print(io, "  Caused by: ")
    showerror(io, unwrap_error(e))
end

# 2-arg: full vanilla backtrace
function Base.showerror(io::IO, e::StanBlocksError)
    _showerror_header(io, e)
    orig_bt = _cause_backtrace(e)
    isempty(orig_bt) || Base.show_backtrace(io, orig_bt)
end

# 3-arg: compact expression trace + filtered stacktrace
function Base.showerror(io::IO, e::StanBlocksError, bt)
    _showerror_header(io, e)
    # Expression stack (from push/pop during forward!)
    expr_stack = _cause_expr_stack(e)
    if !isempty(expr_stack)
        println(io, "\n\n  While processing:")
        for (i, x) in enumerate(reverse(expr_stack))
            println(io, "   [$i] $x")
        end
    end
    # Type-based expression trace (from stacktrace type parameters)
    orig_bt = _cause_backtrace(e)
    isempty(orig_bt) && return
    frames = Base.stacktrace(orig_bt)
    etrace = _expr_trace(frames)
    if !isempty(etrace)
        println(io, "\n  Type trace:")
        for (i, s) in enumerate(etrace)
            println(io, "   [$i] $s")
        end
    end
    filtered = _filter_bt(frames)
    if !isempty(filtered)
        println(io, "\n  Stacktrace (user code):")
        for (i, frame) in enumerate(filtered)
            println(io, "   [$i] $(_format_frame(frame)) at $(frame.file):$(frame.line)")
        end
    end
end

end # module StanBlocks
