module StanBlocks

export @slic, @defsig, @deffun, @stanonly, @lpxf, @lhs, @stan_assert
export return_type_of, stan_code, stan_model, stan_instantiate, stanc_check
export stan_descriptor, stan_definition, stan_definition_closure, stan_operation, stan_execute
export ModelDescriptor, ModelInput, ModelOutput, ModelDefinition, ModelOperation
export StanBlocksError, StanBlocksDiagnostic, diagnostic

using OrderedCollections, JSON, StanLogDensityProblems, LogDensityProblems, Markdown
using BridgeStan
import Tables   # light interface package: lets a DataFrame / Tables.jl source be a data kwarg (see `_table_stan_type`)

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

"""
    StanBlocksDiagnostic

Machine-readable location and classification for a SLIC failure.

`source_part` identifies the caller-supplied source/editor part. `line` and
`column` are one-based when known and `nothing` when the compiler has no span
at that granularity. `code` is a stable, coarse failure class; `message` is a
concise user-facing description of the immediate cause.
"""
struct StanBlocksDiagnostic
    source_part::Union{Nothing,String}
    line::Union{Nothing,Int}
    column::Union{Nothing,Int}
    code::Symbol
    message::String
end

_diagnostic_source_part(::Nothing) = nothing
_diagnostic_source_part(source) = begin
    value = string(source)
    isempty(value) || value == "none" ? nothing : value
end
_diagnostic_position(value) = value isa Integer && value > 0 ? Int(value) : nothing
_diagnostic_message(err) = begin
    rendered = try
        sprint(showerror, err)
    catch
        string(nameof(typeof(err)))
    end
    strip(first(split(rendered, '\n'; limit=2)))
end
_diagnostic_from_error(code, err, lnn=nothing) = StanBlocksDiagnostic(
    lnn isa LineNumberNode ? _diagnostic_source_part(lnn.file) : nothing,
    lnn isa LineNumberNode ? _diagnostic_position(lnn.line) : nothing,
    nothing,
    code,
    _diagnostic_message(err),
)
function _adjust_diagnostic(d::StanBlocksDiagnostic; source_part=nothing, line_offset=0)
    line = isnothing(d.line) ? nothing : _diagnostic_position(d.line + Int(line_offset))
    StanBlocksDiagnostic(
        isnothing(source_part) ? d.source_part : _diagnostic_source_part(source_part),
        line,
        d.column,
        d.code,
        d.message,
    )
end

"""
    diagnostic(error; source_part=nothing, line_offset=0)

Return a [`StanBlocksDiagnostic`](@ref) for a `StanBlocksError` or Julia
`Meta.ParseError`, and `nothing` for unsupported exceptions. `source_part`
can label a parser error (or supply a fallback label); `line_offset` maps a
synthetic parser wrapper back to the caller's source, e.g. `-1` for a leading
`begin` line.

Stable SLIC codes are `:slic_parse_error`, `:slic_trace_error`, and
`:slic_lowering_error`.
"""
diagnostic(::Any; source_part=nothing, line_offset=0) = nothing
function diagnostic(err::Base.Meta.ParseError; source_part=nothing, line_offset=0)
    details = err.detail
    idx = findfirst(d -> d.level === :error, details.diagnostics)
    idx === nothing && return _adjust_diagnostic(
        StanBlocksDiagnostic(
            _diagnostic_source_part(source_part), nothing, nothing,
            :slic_parse_error, _diagnostic_message(err),
        ); source_part, line_offset,
    )
    item = details.diagnostics[idx]
    line, column = Base.JuliaSyntax.source_location(details.source, item.first_byte)
    part = isnothing(source_part) ? details.source.filename : source_part
    _adjust_diagnostic(StanBlocksDiagnostic(
        _diagnostic_source_part(part),
        _diagnostic_position(line),
        _diagnostic_position(column),
        :slic_parse_error,
        string(item.message),
    ); line_offset)
end

_source_file_missing(file) = file === :none || string(file) == "none"
_inherit_source_file(x, _source) = x
_inherit_source_file(x::LineNumberNode, source::LineNumberNode) =
    _source_file_missing(x.file) && !_source_file_missing(source.file) ?
        LineNumberNode(x.line, source.file) : x
_inherit_source_file(x::Expr, source::LineNumberNode) =
    Expr(x.head, map(arg -> _inherit_source_file(arg, source), x.args)...)

_first_source_lnn(::Any) = nothing
_first_source_lnn(x::LineNumberNode) = x
function _first_source_lnn(x::Expr)
    for arg in x.args
        lnn = _first_source_lnn(arg)
        lnn === nothing || return lnn
    end
    nothing
end
_last_source_lnn(::Any) = nothing
_last_source_lnn(x::LineNumberNode) = x
function _last_source_lnn(x::Expr)
    for arg in Iterators.reverse(x.args)
        lnn = _last_source_lnn(arg)
        lnn === nothing || return lnn
    end
    nothing
end

_stack_lnn(item::Tuple) = length(item) >= 2 && item[2] isa LineNumberNode ? item[2] : nothing
_stack_lnn(_) = nothing
function _diagnostic_lnn(expr_stack, current, fallback)
    # `current` advances while tracing inside a named submodel / @deffun body,
    # whereas an outer expression-stack entry still points at the parent call.
    # Prefer the active child location, then fall back through the stack.
    candidates = Any[current]
    append!(candidates, (_stack_lnn(item) for item in Iterators.reverse(expr_stack)))
    push!(candidates, fallback)
    line_lnn = findfirst(x -> x isa LineNumberNode && x.line > 0, candidates)
    file_lnn = findfirst(x -> x isa LineNumberNode && !_source_file_missing(x.file), candidates)
    line = line_lnn === nothing ? 0 : candidates[line_lnn].line
    file = file_lnn === nothing ? :none : candidates[file_lnn].file
    line == 0 && file === :none ? nothing : LineNumberNode(line, file)
end
function _diagnostic_lnn_for_source(expr_stack, current, fallback)
    fallback isa LineNumberNode || return _diagnostic_lnn(expr_stack, current, fallback)
    _source_file_missing(fallback.file) && return _diagnostic_lnn(expr_stack, current, fallback)
    candidates = Any[current]
    append!(candidates, (_stack_lnn(item) for item in Iterators.reverse(expr_stack)))
    for candidate in candidates
        candidate isa LineNumberNode || continue
        candidate.file == fallback.file || continue
        candidate.line > 0 && return candidate
    end
    fallback
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
_cause_diagnostic(e::StanBlocksError) = _cause_diagnostic_inner(e.cause)
_cause_error_inner(c::Tuple) = c[1]
_cause_error_inner(c) = c
_cause_bt_inner(c::Tuple) = length(c) >= 2 ? c[2] : nothing
_cause_bt_inner(_) = nothing
_cause_expr_stack_inner(c::Tuple) = length(c) >= 3 ? c[3] : []
_cause_expr_stack_inner(_) = []
_cause_diagnostic_inner(c::Tuple) =
    length(c) >= 4 && c[4] isa StanBlocksDiagnostic ? c[4] : nothing
_cause_diagnostic_inner(_) = nothing

unwrap_error(e::Base.TaskFailedException) = unwrap_error(e.task.exception)
unwrap_error(e::CompositeException) = unwrap_error(first(e.exceptions))
unwrap_error(e::StanBlocksError) = unwrap_error(_cause_error(e))
unwrap_error(e) = e

_phase_diagnostic_code(::Val{:transpile}) = :slic_transpile_error
_phase_diagnostic_code(::Val{:compile}) = :stan_compile_error
_phase_diagnostic_code(::Val{:evaluate}) = :stan_evaluate_error
_phase_diagnostic_code(::Val) = :stanblocks_error
function diagnostic(e::StanBlocksError; source_part=nothing, line_offset=0)
    stored = _cause_diagnostic(e)
    d = if stored === nothing
        lnn = _diagnostic_lnn(_cause_expr_stack(e), nothing, nothing)
        _diagnostic_from_error(_phase_diagnostic_code(Val(e.phase)), unwrap_error(e), lnn)
    else
        stored
    end
    _adjust_diagnostic(d; source_part, line_offset)
end

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

# First line of `x`, truncated — for a one-line marker inside a diagnostic.
_brief(x, n=160) = begin
    s = split(string(x), '\n'; limit=2)[1]
    length(s) <= n ? s : string(first(s, n), "…")
end

# An expression-stack entry is a SLIC AST node, and the `Base.show` methods
# that render one ARE the Stan emitter — they legitimately `error(...)` on a
# node they cannot emit. Entries reach here in shapes the emitter never has to
# handle in anger: `forward!` pushes the RAW node and only swaps in the resolved
# form once head *and* args have traced, so a failure early in a call (an
# unresolvable callee, say) leaves un-canonicalised lambda bodies and bare
# `Expr`s on the stack — `func_name(::Expr)` then rejects a `:tuple`/`:block`
# head from inside `showerror`. Rendering an entry must never be able to
# destroy the diagnostic it decorates, so each is rendered into a scratch
# buffer and committed only once it succeeded: a throw costs that one entry's
# text, not the `Caused by:` line, and never leaves a half-written line in `io`.
_show_stack_entry(x) = try
    sprint(print, x)
catch err
    inner = try
        _brief(sprint(showerror, err))
    catch
        string(typeof(err))
    end
    string("<unrenderable ", _brief(typeof(x)), ": ", inner, ">")
end

function _format_cause(io::IO, phase, context, cause_error, expr_stack, bt)
    print(io, "StanBlocksError [", phase, "]: ", context, "\n")
    print(io, "  Caused by: ")
    _safe_showerror(io, cause_error, bt)
    if !isempty(expr_stack)
        println(io, "\n\n  While processing:")
        for (i, item) in enumerate(reverse(expr_stack))
            x, lnn = _split_stack_item(item)
            print(io, "   [", i, "] ", _show_stack_entry(x), _stack_loc(lnn)..., "\n")
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
