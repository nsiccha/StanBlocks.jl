module StanBlocks

export @slic, @defsig, @deffun, @stanonly, @lpxf, @lhs, @stan_assert
export stan_code, stan_model, stan_instantiate, stanc_check
export StanBlocksError

using OrderedCollections, JSON, StanLogDensityProblems, LogDensityProblems, Markdown
using BridgeStan

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

# Back-compat alias: SLIC internals used to live in `module stan` (so
# downstream code referenced `StanBlocks.stan.X`). The submodule was
# hoisted; this self-alias keeps `StanBlocks.stan.X === StanBlocks.X`
# so existing call sites keep working. Defined before `include` so
# that `@deffun` blocks inside `slic_stan/builtin.jl` (which expand
# `$stan.tracetype` etc. at quote-eval time) see the binding.
const stan = @__MODULE__

include("slic_stan/slic.jl")

slic_implementation(key; kwargs...) = nothing

# --- Markdown/Quarto display ---
include("quarto.jl")

# --- Error display for StanBlocks computations ---

# `StanBlocksError.cause` is either a 3-tuple `(err, bt, expr_stack)`
# (transpile-time errors capture all three) or a bare exception/string
# (everything else). Three small accessors with a Tuple-typed fast path.
_cause_error(e::StanBlocksError) = _cause_error_inner(e.cause)
_cause_bt(e::StanBlocksError) = _cause_bt_inner(e.cause)
_cause_expr_stack(e::StanBlocksError) = _cause_expr_stack_inner(e.cause)
_cause_error_inner(c::Tuple) = c[1]
_cause_error_inner(c) = c
_cause_bt_inner(c::Tuple) = length(c) >= 2 ? c[2] : nothing
_cause_bt_inner(_) = nothing
_cause_expr_stack_inner(c::Tuple) = length(c) >= 3 ? c[3] : []
_cause_expr_stack_inner(_) = []

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

_split_stack_item(t::Tuple) = t
_split_stack_item(x) = (x, nothing)
_stack_loc(lnn::LineNumberNode) = (" at ", lnn.file, ":", lnn.line)
_stack_loc(_) = ()

function _format_cause(io::IO, phase, context, cause_error, expr_stack, bt)
    print(io, "StanBlocksError [", phase, "]: ", context, "\n")
    print(io, "  Caused by: ")
    _safe_showerror(io, cause_error, bt)
    if !isempty(expr_stack)
        println(io, "\n\n  While processing:")
        for (i, item) in enumerate(reverse(expr_stack))
            x, lnn = _split_stack_item(item)
            print(io, "   [", i, "] ", x, _stack_loc(lnn)..., "\n")
        end
    end
end

Base.show(io::IO, e::StanBlocksError) = showerror(io, e)
Base.show(io::IO, ::MIME"text/plain", e::StanBlocksError) = showerror(io, e)

function Base.showerror(io::IO, e::StanBlocksError)
    _showerror_cause(io, e, e.cause, _cause_bt(e))
end

function Base.showerror(io::IO, e::StanBlocksError, bt; kwargs...)
    # Prefer the stored backtrace (captured at the original throw site); fall
    # back to the bt handed in by Julia's top-level error printer.
    stored = _cause_bt(e)
    effective_bt = stored === nothing ? bt : stored
    _showerror_cause(io, e, e.cause, effective_bt)
end

# Bare-string cause: print the message verbatim. Anything else: format the
# full cause/expression-stack trace.
_showerror_cause(io, _e, cause::AbstractString, _bt) = print(io, cause)
_showerror_cause(io, e, _cause, bt) =
    _format_cause(io, e.phase, e.context, unwrap_error(e), _cause_expr_stack(e), bt)

end # module StanBlocks
