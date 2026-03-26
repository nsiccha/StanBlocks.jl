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

function _showerror_body(io::IO, e::StanBlocksError)
    print(io, "StanBlocksError [$(e.phase)]: $(e.context)\n")
    print(io, "  Caused by: ")
    showerror(io, unwrap_error(e))
    expr_stack = _cause_expr_stack(e)
    if !isempty(expr_stack)
        println(io, "\n\n  While processing:")
        for (i, item) in enumerate(reverse(expr_stack))
            x, lnn = item isa Tuple ? item : (item, nothing)
            loc = lnn isa LineNumberNode ? " at $(lnn.file):$(lnn.line)" : ""
            println(io, "   [$i] $x$loc")
        end
    end
    orig_bt = _cause_backtrace(e)
    isempty(orig_bt) && return
    etrace = _expr_trace(Base.stacktrace(orig_bt))
    if !isempty(etrace)
        println(io, "\n  Type trace:")
        for (i, s) in enumerate(etrace)
            println(io, "   [$i] $s")
        end
    end
end

Base.showerror(io::IO, e::StanBlocksError) = _showerror_body(io, e)

function Base.showerror(io::IO, e::StanBlocksError, bt; kwargs...)
    try
        _showerror_body(io, e)
    catch internal_err
        print(io, "StanBlocksError [$(e.phase)]: $(e.context)")
        print(io, "\n  (internal error in showerror: ")
        showerror(io, internal_err)
        print(io, ")")
    end
end

end # module StanBlocks
