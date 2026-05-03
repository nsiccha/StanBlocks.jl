module StanBlocks

export @slic, @defsig, @deffun, @lpxf, @lhs
export stan_code, stan_model, stan_instantiate
export StanBlocksError

using JSON, StanLogDensityProblems, LogDensityProblems, Markdown

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

include("slic_stan/slic.jl")

slic_implementation(key; kwargs...) = nothing

# --- Markdown/Quarto display ---
include("quarto.jl")

# --- Error display for StanBlocks computations ---

_cause_error(e::StanBlocksError) = e.cause isa Tuple ? e.cause[1] : e.cause
_cause_bt(e::StanBlocksError) = e.cause isa Tuple && length(e.cause) >= 2 ? e.cause[2] : nothing
_cause_expr_stack(e::StanBlocksError) = e.cause isa Tuple && length(e.cause) >= 3 ? e.cause[3] : []

unwrap_error(e::Base.TaskFailedException) = unwrap_error(e.task.exception)
unwrap_error(e::CompositeException) = unwrap_error(first(e.exceptions))
unwrap_error(e::StanBlocksError) = unwrap_error(_cause_error(e))
unwrap_error(e) = e

_safe_showerror(io, e, bt) = try
    bt === nothing ? showerror(io, e) : showerror(io, e, bt)
catch err
    @warn "showerror threw; falling back to repr" cause_type=typeof(e) inner=typeof(err) inner_msg=sprint(showerror, err)
    try
        print(io, sprint(show, e))
    catch err2
        print(io, "<error rendering ", typeof(e), ": showerror threw ", typeof(err),
              " (", sprint(showerror, err), "); repr threw ", typeof(err2), ">")
    end
end

function _format_cause(io::IO, phase, context, cause_error, expr_stack, bt)
    print(io, "StanBlocksError [", phase, "]: ", context, "\n")
    print(io, "  Caused by: ")
    _safe_showerror(io, cause_error, bt)
    if !isempty(expr_stack)
        println(io, "\n\n  While processing:")
        for (i, item) in enumerate(reverse(expr_stack))
            x, lnn = item isa Tuple ? item : (item, nothing)
            loc = lnn isa LineNumberNode ? (" at ", lnn.file, ":", lnn.line) : ()
            print(io, "   [", i, "] ", x, loc..., "\n")
        end
    end
end

Base.show(io::IO, e::StanBlocksError) = showerror(io, e)
Base.show(io::IO, ::MIME"text/plain", e::StanBlocksError) = showerror(io, e)

function Base.showerror(io::IO, e::StanBlocksError)
    if e.cause isa String
        print(io, e.cause)
    else
        _format_cause(io, e.phase, e.context, unwrap_error(e), _cause_expr_stack(e), _cause_bt(e))
    end
end

function Base.showerror(io::IO, e::StanBlocksError, bt; kwargs...)
    # Prefer the stored backtrace (captured at the original throw site); fall
    # back to the bt handed in by Julia's top-level error printer.
    stored = _cause_bt(e)
    effective_bt = stored === nothing ? bt : stored
    if e.cause isa String
        print(io, e.cause)
    else
        _format_cause(io, e.phase, e.context, unwrap_error(e), _cause_expr_stack(e), effective_bt)
    end
end

end # module StanBlocks
