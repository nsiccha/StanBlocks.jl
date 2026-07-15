# Methods that reference `types.anything` (defined inside `functions.jl`'s
# `module types` block) live here, AFTER the include — module-load-time
# resolution of the type parameter requires `types` to already exist.
_forward_module_value(v::Type{<:types.anything}, info) = forward!(v; info)
_resolve_module_value(v::Type{<:types.anything}) = v
_is_ntup_stan_expr(::StanExpr2{<:types.ntup}) = true

# --- Closures (`(x) -> body`) ---
# Defined here, after `functions.jl`, because the dispatch and constructor
# both reference `types.closure` in method signatures / type-parameter
# positions, which the load-order in StanBlocks.jl resolves only once
# `functions.jl` has registered the `types` module.

# Per-`:->` counter so each `(x) -> body` site gets an id used both for
# `func_name` mangling (so HOF receivers specialise per closure) and for
# debugging closure flow through the tracer. Lives in per-trace task-local
# storage — `_next_closure_id` + the seed-scope are defined centrally in
# tracing.jl (so the id is fresh per transpilation, not session-global).

# Walk the un-canonicalised body collecting *all* Symbols that appear (as
# uses or definitions). The closure builder subtracts bound names (params
# + locals) and intersects with the current `info` scope to produce the
# captures dict.
_collect_all_syms!(x, syms) = nothing
_collect_all_syms!(x::Symbol, syms) = (push!(syms, x); nothing)
_collect_all_syms!(x::Expr, syms) = (foreach(a -> _collect_all_syms!(a, syms), x.args); nothing)

# Parse the LHS of `(args) -> body` into (arg_names, vararg_name).
# Phase 1 handles bare `Symbol`, `Expr(:tuple, syms...)`, and a trailing
# `args...`. Typed params and kwargs are out of scope.
_parse_lambda_lhs(lhs::Symbol) = ([lhs], nothing)
_parse_lambda_lhs(lhs::Expr) = if lhs.head === :tuple
    arg_names = Symbol[]
    vararg = nothing
    for (i, a) in enumerate(lhs.args)
        if Meta.isexpr(a, :...) && length(a.args) == 1 && a.args[1] isa Symbol
            i == length(lhs.args) || error(
                "closure: vararg `$(a.args[1])...` must be the last parameter, got $lhs."
            )
            vararg = a.args[1]
        elseif a isa Symbol
            push!(arg_names, a)
        else
            error("closure: only bare-Symbol params (and a trailing `args...`) are supported in phase 1, got `$a` in `$lhs`.")
        end
    end
    (arg_names, vararg)
elseif lhs.head === :(...)  && length(lhs.args) == 1 && lhs.args[1] isa Symbol
    (Symbol[], lhs.args[1])
else
    error("closure: unsupported lambda LHS `$lhs` (head `$(lhs.head)`). Phase 1 supports `x -> ...`, `(x, y) -> ...`, and `(args...) -> ...`.")
end

# Trace-time entry point for `(x) -> body`. Snapshots free vars in `body`
# that resolve in the current `info` scope and packages them with the raw
# body Expr as a `types.closure` StanExpr. The receiver UDF (or
# `inline_body` dispatch on the closure StanExpr at head) substitutes
# captures + call-site args and re-traces.
forward!(x::CanonicalExprV{:->,A}; info) where {A} = begin
    length(x.args) == 2 || error(
        "closure: malformed lambda — expected 2 args (lhs, body), got $(length(x.args)): $x."
    )
    lhs, body = x.args
    body isa Expr || error(
        "closure: lambda body must be an Expr (a `:block`), got `$(typeof(body))`. Make sure `canonical(::Expr)` is preserving `:->` args raw."
    )
    arg_names, vararg_name = _parse_lambda_lhs(lhs)

    bound = Set{Symbol}(arg_names)
    vararg_name !== nothing && push!(bound, vararg_name)
    locals = Set{Symbol}()
    _collect_locals!(body, locals)
    union!(bound, locals)

    all_syms = Set{Symbol}()
    _collect_all_syms!(body, all_syms)
    free = setdiff(all_syms, bound)

    # Snapshot every free var that's already in scope. Names that aren't in
    # `info` (e.g. builtin function refs like `sin`) stay as Symbols so they
    # re-resolve at the closure's call site through the regular SLIC path.
    # Iterate `info` (an OrderedDict) so capture ordering is deterministic —
    # matters for phase 2 where captures become positional Stan args.
    captures = OrderedDict{Symbol,Any}()
    for k in keys(info)
        k in free || continue
        captures[k] = info[k]
    end

    id = _next_closure_id()
    source = something(_get_lnn(info), LineNumberNode(0, :none))
    record = (
        arg_names  = Tuple(arg_names),
        vararg_name = vararg_name,
        body       = body,
        captures   = captures,
        kwargs     = (),
        source     = source,
        id         = id,
        # Snapshot the model's module so the lifted-closure `fundef` path
        # (phase 2-deeper, used when a closure flows directly to a Stan
        # builtin like `ode_rk45`) can resolve user-module names while
        # tracing the closure body in a fresh `info` scope.
        mod        = get_module(info),
    )
    # The closure's qual must reflect its captures' quals — at any HOF
    # call site, `stan_expr(::CanonicalExpr)` computes its qual via
    # `maximum(qual, x.args)`, which decides which Stan block the call
    # lands in. A closure that captures a parameter (e.g. `shift`) must
    # therefore *itself* be parameter-qual so the surrounding call routes
    # to `transformed parameters` / `model` / `generated quantities`,
    # not `transformed data`.
    closure_qual = isempty(captures) ? :data : maximum(qual, values(captures); init=:data)
    StanExpr(record, StanType(types.closure; value=record, qual=closure_qual))
end

# A call whose head is a closure StanExpr: pull the record off the head's
# StanType and feed `expand_inline!` directly. Same machinery as
# `@deffun @inline` UDFs — the closure record just lives in the StanExpr
# rather than in a `inline_body(::CanonicalExpr{<:typeof(f)})` method.
inline_body(x::CanonicalExpr{<:StanExpr2{<:types.closure}}) = type(head(x)).info.value

# --- Custom types via `@usertype` (tagged ntups) ---

# `Foo(field_values...)`: dispatch on `tokenof{<:usertype}` head. Build a
# tagged ntup-shaped StanType whose `arg_types` named tuple matches the
# Julia struct's `fieldnames` paired with the call args' types.
tracetype(x::CanonicalExpr{<:StanExpr2{<:types.tokenof{<:types.usertype}}}) = begin
    T = type(head(x)).info.value
    fields = fieldnames(T)
    length(fields) == length(x.args) || error(
        "$T constructor expects $(length(fields)) field(s) " *
        "($(join(fields, ", "))), got $(length(x.args))."
    )
    StanType(T; arg_types=(;[fields[i] => type(x.args[i]) for i in eachindex(fields)]...))
end

# Stan-side render of a usertype constructor call: emit a positional Stan
# tuple literal, mirroring how `(;a, b)` named-tuple literals render. The
# usertype's nominal tag is *Julia-side only* — Stan sees a plain
# `tuple(T1, T2, ...)` value.
Base.show(io::IO, x::CanonicalExpr{<:StanExpr2{<:types.tokenof{<:types.usertype}}}) =
    autoprint(io, "(", Join(x.args, ", "), ")")

"""
    @usertype struct RaggedVector
        mem  :: vector
        ends :: int[]
    end

Declare a custom Stan-renderable record type. Lowers to a real Julia
`struct` whose abstract supertype is
`StanBlocks.stan.types.usertype` (added automatically); field type
annotations are SLIC types and are kept only for documentation —
fields are stored as `Any` so plain Julia construction works for
data plumbing. Method dispatch on the type tag (`Base.length(r::RaggedVector)`)
works via standard Julia. Stan-side, values render as positional
tuples; field access (`r.mem`) reuses the existing `ntup` machinery.
"""
macro usertype(struct_def)
    Meta.isexpr(struct_def, :struct) || error(
        "@usertype: expected `struct ... end`, got `\$struct_def`."
    )
    is_mutable, sig, body = struct_def.args
    is_mutable && error("@usertype: mutable structs not supported.")

    typename, supertype = _usertype_sig(sig)

    # SLIC field type annotations are not real Julia types — strip them so
    # the lowered struct accepts any Julia value at those slots (we only
    # construct via SLIC tracing, which never hits the real constructor).
    new_body_args = Any[_usertype_field(stmt, typename) for stmt in body.args]
    new_body = Expr(:block, new_body_args...)
    new_sig  = :($typename <: $supertype)
    esc(Expr(:block, Expr(:struct, false, new_sig, new_body), typename))
end

# `@usertype` type signature → (typename, supertype). Default supertype is the
# generic SLIC `usertype` tag.
_usertype_sig(sig::Symbol) = (sig, :($StanBlocks.stan.types.usertype))
function _usertype_sig(sig::Expr)
    Meta.isexpr(sig, :<:) && sig.args[1] isa Symbol ||
        error("@usertype: type signature must be `Foo` or `Foo <: SomeType`, got `$sig`.")
    (sig.args[1], sig.args[2])
end
_usertype_sig(sig) =
    error("@usertype: type signature must be `Foo` or `Foo <: SomeType`, got `$sig`.")

# `@usertype` field statement → bare field name (or LineNumberNode passed through).
_usertype_field(stmt::LineNumberNode, _) = stmt
_usertype_field(stmt::Symbol, _) = stmt
function _usertype_field(stmt::Expr, typename)
    Meta.isexpr(stmt, :(::)) && stmt.args[1] isa Symbol ||
        error("@usertype $typename: each field must be `name :: type` or `name`, got `$stmt`.")
    stmt.args[1]
end
_usertype_field(stmt, typename) =
    error("@usertype $typename: each field must be `name :: type` or `name`, got `$stmt`.")

# Phase 2-deeper: closure passed directly to a Stan builtin (no SLIC UDF
# to inline into) gets lifted to a top-level Stan function. The builtin's
# `fetch_functions!` specialisation builds a `CanonicalExpr(closure, params...,
# captures...)` shape that this `fundef` consumes — matching what a
# `@deffun` UDF would consume but with the closure StanExpr at head and
# the captures appended as trailing positional args.
fundef(x::CanonicalExpr{<:StanExpr2{<:types.closure}}) = begin
    cl = type(head(x)).info.value
    arg_names = collect(cl.arg_names)
    n_params = length(arg_names)
    capture_names = collect(keys(cl.captures))
    sig_names = vcat(arg_names, capture_names)
    n_total = length(sig_names)
    @assert length(x.args) >= n_total "fundef(closure): expected ≥ $n_total args (params + captures), got $(length(x.args))."

    # Anonymise the call args (mirrors the `anon_deconstruct` path used
    # by `@deffun`-registered fundefs) so the body emits Stan code with
    # the function's parameter names rather than caller-side expressions.
    info_nt = (;[name => x.args[i] for (i, name) in enumerate(sig_names)]...)
    info = OrderedDict{Symbol,Any}(pairs(anon_info(info_nt)))
    info[:__mod__] = cl.mod

    body_with_return = ensure_xreturn(cl.body)
    body_block = forward!(canonical(body_with_return); info)
    rv_type = type(info[RV_NAME])

    args_nt = (;[name => info[name] for name in sig_names]...)

    StanFunction3(
        "// lifted closure (id $(cl.id))\n",
        rv_type,
        head(x),
        args_nt,
        [body_block],
    )
end

# `f = (x) -> body` binds the *closure StanExpr verbatim* into `info` —
# the regular `AssignmentExpr{Symbol,<:StanExpr}` path replaces `expr` with
# the bound name and wipes `info.value=missing`, which would destroy the
# closure record. Closures live entirely in trace state; emit no Stan-side
# statement (returning `nothing` makes `forward!(::BlockExpr)` skip it via
# `_is_inert_block_stmt`).
forward!(x::AssignmentExpr{Symbol,<:StanExpr2{<:types.closure}}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    name in keys(info) && error(
        "closure: rebinding `$name` is not supported — closures are SLIC-side compile-time aliases for an anonymous lambda."
    )
    info[name] = rhs
    nothing
end
_is_inert_block_stmt(::Nothing) = true

# Defensive: if a closure StanExpr ever ends up with `expr::Symbol` (e.g.
# from a code path that reuses the regular assignment shape), the generic
# `forward!(::StanExpr{Symbol})` clashes with `forward!(::StanExpr2{<:types.closure})`.
# This more-specific method removes the ambiguity. Same passthrough behaviour.
forward!(x::StanExpr{Symbol,<:StanType{<:types.closure}}; info) = x

# `slic_macroexpand` and the user-facing macros must exist before
# `include("builtin.jl")`, which uses `@deffun` extensively at load time.
const _SLIC_RESERVED_MACROS = (Symbol("@doc"), Symbol("@lpxf"), Symbol("@lhs"), Symbol("@inline"), Symbol("@stanonly"))

_is_reserved_slic_macro(::Any) = false
_is_reserved_slic_macro(head::Symbol) = head in _SLIC_RESERVED_MACROS
_is_reserved_slic_macro(head::GlobalRef) = head.name in _SLIC_RESERVED_MACROS
_is_reserved_slic_macro(head::Expr) = head.head === :. &&
    length(head.args) == 2 && head.args[2] isa QuoteNode &&
    head.args[2].value in _SLIC_RESERVED_MACROS

slic_macroexpand(mod::Module, x) = x
slic_macroexpand(mod::Module, x::Expr) = if x.head === :macrocall
    head = x.args[1]
    if _is_reserved_slic_macro(head)
        Expr(:macrocall, head, x.args[2], (slic_macroexpand(mod, a) for a in x.args[3:end])...)
    else
        slic_macroexpand(mod, macroexpand(mod, x; recursive=false))
    end
else
    Expr(x.head, (slic_macroexpand(mod, a) for a in x.args)...)
end

# `reject` and `print` are the natural home for Julia-style string
# interpolation (`"x = $x"`). Stan's variadic argument list is the
# right Stan-side surface, so we lower a single interpolated-string arg
# into multiple positional args at AST level: `reject("x = $x")` →
# `reject("x = ", x)`.
_is_reject_or_print(x) = false
_is_reject_or_print(x::Symbol) = x === :reject || x === :print
_is_reject_or_print(x::Expr) = x.head === :. && length(x.args) == 2 &&
    x.args[2] isa QuoteNode && x.args[2].value in (:reject, :print)

lower_string_interp(x) = x
lower_string_interp(x::Expr) = if x.head === :call && length(x.args) >= 2 && _is_reject_or_print(x.args[1])
    new_args = Any[x.args[1]]
    for a in x.args[2:end]
        if Meta.isexpr(a, :string)
            append!(new_args, [lower_string_interp(p) for p in a.args])
        else
            push!(new_args, lower_string_interp(a))
        end
    end
    Expr(:call, new_args...)
else
    Expr(x.head, [lower_string_interp(a) for a in x.args]...)
end

# A leading string literal inside the `@slic begin ... end` block is the
# model docstring. We split it out here and stash it in the data dict
# under `:docstring` — the existing `stan_model` / `Base.show(::StanIO,
# ::StanModel)` path already renders it as a leading `// ...` comment in
# the generated Stan source. (For Julia-side `?m` lookup, users still
# write the standard `\"\"\"docstring\"\"\" m = @slic …` form, which
# `Core.@doc` attaches to the binding independently.)
_is_doc_macro_head(head) = head == GlobalRef(Core, Symbol("@doc")) ||
    head == Symbol("@doc") ||
    (head isa Expr && head.head === :. && length(head.args) == 2 &&
        head.args[2] isa QuoteNode && head.args[2].value === Symbol("@doc"))

extract_leading_docstring(model) = ("", model)
extract_leading_docstring(model::Expr) = if model.head === :block
    real_idx = findfirst(a -> !(a isa LineNumberNode), model.args)
    real_idx === nothing && return ("", model)
    real = model.args[real_idx]
    if real isa AbstractString
        # Bare leading string literal in the block.
        new_args = copy(model.args)
        deleteat!(new_args, real_idx)
        return (real, Expr(:block, new_args...))
    elseif Meta.isexpr(real, :macrocall) && length(real.args) >= 4 &&
            _is_doc_macro_head(real.args[1]) && real.args[3] isa AbstractString
        # Julia auto-wraps `"""..."""` followed by an expr as
        # `Core.@doc(lnn, "...", expr)`. Peel the docstring, restore the
        # bare expr in its slot.
        doc = real.args[3]
        next_stmt = real.args[4]
        new_args = copy(model.args)
        new_args[real_idx] = next_stmt
        return (doc, Expr(:block, new_args...))
    end
    return ("", model)
else
    ("", model)
end
