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
_cause_expr_stack(e::StanBlocksError) = e.cause isa Tuple && length(e.cause) >= 3 ? e.cause[3] : []

unwrap_error(e::Base.TaskFailedException) = unwrap_error(e.task.exception)
unwrap_error(e::CompositeException) = unwrap_error(first(e.exceptions))
unwrap_error(e::StanBlocksError) = unwrap_error(_cause_error(e))
unwrap_error(e) = e

function _short_repr(x, limit=200)
    s = try sprint(show, x; context=:limit=>true) catch e; "<display error>" end
    length(s) > limit ? s[1:prevind(s, limit)] * "…" : s
end

function _format_cause(phase, context, cause_error, expr_stack)
    sprint() do io
        print(io, "StanBlocksError [$(phase)]: $(context)\n")
        print(io, "  Caused by: ")
        showerror(io, cause_error)
        if !isempty(expr_stack)
            println(io, "\n\n  While processing:")
            for (i, item) in enumerate(reverse(expr_stack))
                x, lnn = item isa Tuple ? item : (item, nothing)
                loc = lnn isa LineNumberNode ? " at $(lnn.file):$(lnn.line)" : ""
                println(io, "   [$i] $(_short_repr(x))$loc")
            end
        end
    end
end

Base.show(io::IO, e::StanBlocksError) = showerror(io, e)
Base.show(io::IO, ::MIME"text/plain", e::StanBlocksError) = showerror(io, e)

function Base.showerror(io::IO, e::StanBlocksError)
    print(io, e.cause isa String ? e.cause : _format_cause(e.phase, e.context, unwrap_error(e), _cause_expr_stack(e)))
end

function Base.showerror(io::IO, e::StanBlocksError, bt; kwargs...)
    try
        showerror(io, e)
    catch internal_err
        print(io, "StanBlocksError [$(e.phase)]: $(e.context)")
        print(io, "\n  (internal error in showerror: ")
        showerror(io, internal_err)
        print(io, ")")
    end
end

end # module StanBlocks
