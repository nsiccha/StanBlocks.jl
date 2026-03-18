module StanBlocks

export @stan, @model, @parameters, @transformed_parameters, @generated_quantities, @bsum, with_gradient
export @slic, @defsig, @deffun
export stan_code, stan_model, stan_instantiate
export StanBlocksError

using LinearAlgebra, Statistics, Distributions, LogExpFunctions, JSON

# --- Error type for StanBlocks computations (defined early so submodules can use it) ---

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

_cause_error(e::StanBlocksError) = e.cause isa Tuple ? first(e.cause) : e.cause
_cause_backtrace(e::StanBlocksError) = e.cause isa Tuple ? last(e.cause) : []

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
    if frame.linfo isa Core.MethodInstance
        sig = frame.linfo.specTypes
        params = fieldtypes(sig)
        fname = string(frame.func)
        arg_strs = ["::$(p)" for p in params[2:end]]
        return fname * "(" * join(arg_strs, ", ") * ")"
    end
    return string(frame.func)
end

function Base.showerror(io::IO, e::StanBlocksError)
    root = unwrap_error(e)
    print(io, "StanBlocksError [$(e.phase)]: $(e.context)\n")
    print(io, "  Caused by: ")
    showerror(io, root)
    orig_bt = _cause_backtrace(e)
    if !isempty(orig_bt)
        frames = Base.stacktrace(orig_bt)
        filtered = _filter_bt(frames)
        if !isempty(filtered)
            println(io, "\n\n  Stacktrace (user code):")
            for (i, frame) in enumerate(filtered)
                println(io, "   [$i] $(_format_frame(frame)) at $(frame.file):$(frame.line)")
            end
        end
    end
end

# 3-arg method: suppress framework backtrace since we show our own
Base.showerror(io::IO, e::StanBlocksError, ::Any) = showerror(io, e)

end # module StanBlocks
