struct Join
    iterator
    delim
end
Base.show(io::IO, x::Join) = join(io, x.iterator, x.delim)

abstract type WrappedIO <: IO end
Base.parent(io::WrappedIO) = io.parent
Base.write(io::WrappedIO, arg) = error("WrappedIO `write` not defined for argument of type `$(typeof(arg))`; add a specialized Base.write method.")#write(parent(io), arg)
Base.write(io::WrappedIO, arg::Char) = write(parent(io), arg)
Base.write(io::WrappedIO, arg::Symbol) = write(parent(io), arg)
Base.write(io::WrappedIO, arg::Array) = write(parent(io), arg)
Base.write(io::WrappedIO, arg::Union{SubString{String}, String}) = write(parent(io), arg)
Base.write(io::WrappedIO, arg::UInt8) = write(parent(io), arg)
struct StanIO{P} <: WrappedIO
    parent::P
    info
    StanIO(parent; kwargs...) = new{typeof(parent)}(parent, kwargs)
end
remake(io::StanIO, p=parent(io); kwargs...) = StanIO(p; io.info..., kwargs...)
current_indent(io) = repeat("    ", get(io, :_autoprint_indent, 0))
current_indent(io::StanIO) = repeat("    ", current_indent_level(io))
current_indent_level(io::StanIO) = get(io.info, :current_indent_level, 0)
indent(io) = IOContext(io, :_autoprint_indent => 1 + get(io, :_autoprint_indent, 0))
indent(io::StanIO) = remake(io; current_indent_level=1+current_indent_level(io))
maybe_indent(io, x::StanBlock) = indent(io)
maybe_indent(io, x::FunctionsBlock) = io
nobreak(io) = StanIO(io; maybreak=false)
nobreak(io::StanIO) = remake(io; maybreak=false)
maybreak(io) = true
maybreak(io::StanIO) = get(io.info, :maybreak, true)
line_limit(io) = 100


_autoprint_buf(io::StanIO) = (buf = IOBuffer(); (remake(nobreak(io), buf), buf))
_autoprint_buf(io) = (buf = IOBuffer(); (buf, buf))
_is_join(::Join) = true
_is_join(_) = false
autoprint(io, args...) = if maybreak(io)
    bufio, buf = _autoprint_buf(io)
    print(bufio, args...)
    rv = String(take!(buf))
    if length(rv) <= line_limit(io)
        print(io, rv)
    else
        idx = findfirst(_is_join, args)
        iio = indent(io)
        print(io, args[1:idx-1]...)
        print(io, "\n", current_indent(iio))
        print(iio, Join(args[idx].iterator, rstrip(args[idx].delim) * "\n" * current_indent(iio)))
        print(io, "\n", current_indent(io))
        print(io, args[idx+1:end]...)
    end
else
    print(io, args...)
end
Base.show(io::StanIO, x::StanModel) = begin
    print(io, maybedoc(get(x.meta, :docstring, "")))
    print(io, Join(blocks(x), "\n"))
end
Base.show(io::StanIO, x::StanExpr) = _show_stan_expr(io, type(x), x)
_show_stan_expr(io, ::StringStanType, x) = print(io, expr(x), "::", type(x))
_show_stan_expr(io, _t, x) = print(io, expr(x))
# String-literal StanExprs render as a quoted Stan string.
Base.show(io::StanIO, x::StanExpr{<:AbstractString}) = print(io, '"', expr(x), '"')
# Sized type tokens render as Stan tuple literals at the call site. 0-dim
# tokens are always_inline'd away, so no call-site form is needed for them.
# 1-dim tokens render as a bare int (Stan has no 1-element tuple type).
Base.show(io::StanIO, x::StanExpr2{<:types.tokenof,1}) = autoprint(io, stan_size(x)[1])
Base.show(io::StanIO, x::StanExpr2{<:types.tokenof}) = autoprint(io, "(", Join(stan_size(x), ", "), ")")
Base.show(io::StanIO, ::Colon) = print(io, ":")
Base.show(io::IO, x::StanModel) = show(StanIO(io), x)
Base.show(io::IO, x::SlicModel; mayfail=true) = try
    print(io, stan_model(x))
catch e
    mayfail && return print(io, "SlicModel: Something went wrong:", e)
    rethrow(e)
end
Base.show(io::IO, x::StanBlock) = if true#length(content(x)) > 0
    print(io, name(x), " {\n")
    map(stmt->block_print(maybe_indent(io, x), x, stmt), collect(values(content(x))))
    print(io, current_indent(io), "}")
end
line_terminator(x::StanExpr) = line_terminator(expr(x))
line_terminator(x) = ";\n"
line_terminator(x::String) = endswith(rstrip(x), ";") ? "\n" : ";\n"
line_terminator(x::IfExpr) = "\n"
line_terminator(x::WhileExpr) = "\n"
line_terminator(x::ForExpr) = "\n"
block_print(io, ::StanBlock, ::LineNumberNode) = nothing
block_print(io, ::StanBlock, x) = print(io, current_indent(io), x, line_terminator(x))
block_print(io, ::StanBlock, x::SamplingExpr{<:Any,<:StanExpr{<:CanonicalExpr{typeof(flat)}}}) = nothing
block_print(io, b::StanBlock, x::BlockExpr) = map(stmt->block_print(io, b, stmt), x.args)
block_print(io, ::DeclarativeBlock, x) = !always_inline(x) && print(io, current_indent(io), type(x), " ", expr(x), line_terminator(x))
block_print(io, b::DeclarativeBlock, x::DocumentExpr) = begin
    print(io, current_indent(io), commentstring(x.args[1]))
    block_print(io, b, x.args[2])
end
block_print(io, ::FunctionsBlock, x) = isnothing(x) || print(io, x, "\n")
constraints(x::StanType) = (;[
    key=>getindex(info(x), key)
    for key in (:lower, :upper, :offset, :multiplier) if key in keys(info(x))
]...)
Base.show(io::IO, x::StanExpr) = print(io, expr(x), "::", type(x))
Base.show(io::IO, x::StanType) = begin 
    l, r = lr_size(x)
    length(l) > 0 && autoprint(io, "array[", Join(l, ", "), "] ")
    print(io, center_type(x))
    cons = constraints(x)
    length(cons) > 0 && autoprint(io, "<", Join(map((k,v)->Join((k,v), "="), keys(cons), values(cons)), ", "), ">")
    length(r) > 0 && autoprint(io, "[", Join(r, ", "), "]")
end
Base.show(io::IO, x::StanType{<:types.tup}) = begin 
    stan_ndim(x) > 0 && autoprint(io, "array[", Join(stan_size(x), ", "), "] ")
    autoprint(io, "tuple(", Join(x.info.arg_types, ", ") , ")")
end
Base.show(io::IO, x::AssignmentExpr{<:StanExpr{Symbol}}) = begin
    name, rhs = x.args
    @assert center_type(rhs) != types.anything "tracetype not defined for $name = $(short_expr(rhs))!"
    print(io, type(rhs), " ", name, " = ", rhs)
end
Base.show(io::IO, x::AssignmentExpr) = print(io, x.args[1], " = ", x.args[2])

prettystring(f) = " $f "
prettystring(f::Base.BroadcastFunction) = " .$(f.f) "
Base.show(io::IO, x::CanonicalExpr) = begin
    fname = func_name(head(x), x.args)
    # `stan_call_args` expands closure args to their captures and drops
    # always_inline args (functions, 0-dim tokens, capture-free closures).
    # `func_name` still operates on the *unexpanded* args so the mangled
    # receiver name reflects the closure's identity, not just its captures.
    fargs = stan_call_args(x.args)
    is_lxxf = endswith(string(fname), r"_lp[md]f|_l?c?cdf")
    if is_lxxf && length(fargs) > 1
        autoprint(io, fname, "(", fargs[1], " | ", Join(fargs[2:end], ", "), ")")
    else
        autoprint(io, fname, "(", Join(fargs, ", "), ")")
    end
end
Base.show(io::IO, x::CanonicalExpr{<:ODESolver}) = autoprint(io, head(x), "(", Join(
    # ODE solvers fold the function arg into the receiver name (mangled
    # via `func_name(args[1], rest)`), then emit the remaining args. If
    # `args[1]` is a closure, its captures must *also* be threaded as
    # trailing args so Stan's ODE solver forwards them to the lifted
    # function — `expand_call_args` only catches closures in the args
    # being rendered, not the function-position arg.
    (func_name(x.args[1], x.args[2:end]), stan_call_args(x.args[2:end])...,
     _closure_captures(x.args[1])...), ", "
), ")")
commentstring(x::String) = "// " * replace(x, "\n"=>"\n    // ") * "\n"
# A docstring arg can arrive traced (a `StanExpr` wrapping the String) rather
# than a bare String — e.g. a doc comment on a sub-model whose body was forwarded.
# Unwrap to the underlying value and re-dispatch (loud MethodError if it isn't a
# String, matching the bare-String-only contract).
commentstring(x::StanExpr) = commentstring(expr(x))
Base.show(io::IO, x::DocumentExpr) = print(io, commentstring(x.args[1]), current_indent(io), x.args[2])
Base.show(io::IO, x::ReturnExpr) = print(io, "return ", x.args[1])
Base.show(io::IO, x::TupleExpr) = autoprint(io, "(", Join(x.args, ", "), ")")
Base.show(io::IO, x::NamedTupleExpr) = autoprint(io, "(", Join([arg.args[2] for arg in expr.(x.args)], ", "), ")")
Base.show(io::IO, x::VectExpr) = autoprint(io, "[", Join(x.args, ", "), "]'")
Base.show(io::IO, x::DeclExpr) = print(io, type(x.args[1]), " ", expr(x.args[1]))
Base.show(io::IO, x::Colon2Expr) = print(io, Join(x.args, ":"))
Base.show(io::IO, x::ColonExpr) = print(io, Join(x.args, ":"))
Base.show(io::IO, ::BreakExpr) = print(io, "break")
Base.show(io::IO, x::ContinueExpr) = print(io, "continue ", x.args[1])
Base.show(io::IO, x::ForExpr) = begin
    head, body = x.args
    idx, rhs = head.args 
    print(io, "for(", idx, " in ", rhs, ")", StanBlock(Symbol(), body.args))
end
Base.show(io::IO, x::WhileExpr) = begin
    head, body = x.args
    print(io, "while(", head, ")", StanBlock(Symbol(), body.args))
end
_else_branch(_x, e::BlockExpr) = StanBlock(Symbol(), e.args)
_else_branch(x, e) = error("if/elseif rendering: else branch is not a BlockExpr (got `$(typeof(e))` from `$x`).")

Base.show(io::IO, x::IfExpr) = begin
    print(io, "if(", x.args[1], ")", StanBlock(Symbol(), x.args[2].args))
    if length(x.args) == 3
        print(io, " else", _else_branch(x, x.args[3]))
    end
end
Base.show(io::IO, x::CanonicalExpr{typeof(adjoint)}) = print(io, "(", x.args[1], "')")
Base.show(io::IO, x::CanonicalExpr{typeof(range)}) = autoprint(io, "linspaced_vector(", Join((x.args[end], x.args[1], x.args[2]), ", "), ")")
Base.show(io::IO, x::CanonicalExpr{typeof(getindex)}) = autoprint(io, x.args[1], "[", Join(x.args[2:end], ", "), "]")
# Field access (`obj.name`, lowered via `forward!(::GetPropertyExpr)` to
# `Base.getfield(obj, position)`) renders as Stan's positional tuple
# access `obj.N`.
Base.show(io::IO, x::CanonicalExpr{typeof(Base.getfield),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = print(io, x.args[1], ".", x.args[2])
# User-defined `Base.getindex(::usertype, ::int)` methods (e.g.
# `RaggedVector` group access) route through a real Stan helper function
# rather than Stan's native `r[i]` (which on a tuple-rendered usertype
# would mean positional field access — the wrong semantics). The mangled
# helper name is produced by `func_name` with the usertype tag
# contributing.
Base.show(io::IO, x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.usertype}, <:StanExpr2{<:types.int}}}) = autoprint(io,
    func_name(getindex, x.args), "(", Join(stan_call_args(x.args), ", "), ")"
)
for f in (-,+,*,\,/,^,.*,./,.^,<,<=,==,!=,>=,>,&,|)
    @eval Base.show(io::IO, x::CanonicalExpr{typeof($f)}) = autoprint(io, "(", Join(x.args, prettystring($f)), ")")
    @eval Base.show(io::IO, x::CanonicalExpr{typeof($f),Tuple{A}}) where {A} = print(io, "(", string($f), x.args[1], ")")
end
Base.show(io::IO, x::CanonicalExpr{typeof(÷)}) = autoprint(io, "(", Join(x.args, " %/% "), ")")
for f in (Meta.quot(:(~)), Meta.quot(:(=)))
    @eval Base.show(io::IO, x::CanonicalExprV{$f}) = print(io, Join(x.args, prettystring($f)))
end
Base.show(io::IO, x::SamplingExpr) = print(io, Join(x.args, " ~ "))

for f in (:+=,:-=,:*=)
    qf = Meta.quot(f)
    @eval forward!(x::CanonicalExprV{$qf}; info) = stan_expr(remake(x, forward!(x.args; info)...))
    @eval Base.show(io::IO, x::CanonicalExprV{$qf}) = print(io, Join(x.args, prettystring($qf)))
end
@eval forward!(x::CanonicalExprV{:(.=)}; info) = stan_expr(remake(x, forward!(x.args; info)...))
@eval Base.show(io::IO, x::CanonicalExprV{:(.=)}) = print(io, Join(x.args, " = "))
