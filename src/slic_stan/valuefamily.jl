# Value-based UDFs and distribution families: introduce new callable
# distributions WITHOUT defining Julia functions, methods, or types.
#
# A `@deffun` + `@lpxf` + `@lhs` family is method registration: every density
# overload, dispatch hook (`lpxf_expr`/`rng_expr`/`likelihood_expr`), support
# hook (`autokwargs`), and LHS-inference rule (`tracetype`) is a Julia method
# on a freshly-eval'd function singleton, so a *generated* family needs
# `Core.eval` (and a fresh module, and `invokelatest`, and a lock). A
# `ValueFamily` carries the same information as DATA — density/pointwise/RNG
# bodies as owned ASTs, overload signatures as shape patterns, support bounds
# as plain values, the defining module as an explicit field — and every
# trace-time hook dispatches on these two PREDECLARED types. Constructing one
# is pure Julia (no eval, no methods, no globals); tracing through one uses
# only per-trace state. Independent concurrent builds just work.
#
# There are two precedents in-repo: closure values (`inline_body`/`fundef` on
# `StanExpr2{<:types.closure}`, closures.jl) and spliced sub-model values
# (`_forward_head` passthrough for `SlicModel`/`SubmodelFn`, forward.jl). Value
# families follow the sub-model shape: the family (or companion) value sits
# RAW in call-head position and every hook reads the record off the value.
#
# v1 SCOPE (anything else fails loudly with a pointer here):
# - companions lower like non-`@inline` `@deffun`s (lifted Stan functions,
#   one Stan name per label, Stan-native overloading across shapes).
# - overload selection replicates `@deffun` dispatch: per-arg
#   center-type + ndim (+ `tokenof` for sized-RNG leading tokens,
#   `typeof(f)` function tags), fixed beats trailing-`args...`, ambiguity
#   errors. Kwarg formals, positional defaults, and signature stubs are
#   rejected (BRM's generated defs use none of them).
# - density bodies may call sibling companions (`density`/`pointwise`/`rng`)
#   and the family by label; the constructor splices the companion VALUES
#   (self-contained families, no module bindings).
# - NOT in v1 (each names its follow-up in the `ValueFamily` docstring):
#   HOF-wrapping (`truncated`/`censored`/`weighted`/… over a value family),
#   explicit user bounds at a value `~` site (put bounds in intrinsic
#   `support=`), selector/HOF-argument positions, data-dict carriage, ragged
#   observations, `_lcdf`/`_lccdf` kinds, and `:void` companions.

# --- Representation ------------------------------------------------------

# One formal of one overload: the `@deffun` dispatch key (`xsig_type`, with
# the `anything`-scalar escape hatch and the `tokenof` sized-token form).
struct ValueUDFPattern
    ct::Type
    ndim::Int
    istoken::Bool
    isfunc::Bool
end

struct ValueUDFOverload
    label::Symbol
    # Formal names (`anontok__i` for bare type tokens, `arg__i` for anonymous
    # formals — the same synthetic names `@deffun` uses) and their xref'd
    # `T[dims...]` ASTs (for size-name binding at trace time).
    formal_names::Vector{Symbol}
    formal_types::Vector{Expr}
    patterns::Vector{ValueUDFPattern}
    vararg::Union{Nothing,Symbol}
    # `:anything` (unannotated or explicitly `::anything`, including computed
    # `typeof(...)` returns) means infer-from-body via `forward_return!`;
    # otherwise the xref'd return AST plus its precomputed Stan skeleton.
    rettype::Any
    sig_rv::Union{Nothing,StanType}
    # Owned, macroexpanded, `~`/`target`-free, label-rewritten body.
    body::Expr
    body_key::UInt
    # Precomputed `_udf_size_analysis` outputs (preamble/checks/aliases).
    fun_sizes::OrderedDict
    fun_checks::Vector{String}
    size_aliases::NamedTuple
    mod::Module
    source::LineNumberNode
end

struct ValueUDF
    label::Symbol
    overloads::Tuple{Vararg{ValueUDFOverload}}
    mod::Module
    # Sibling/self links (label => companion/family value), one Dict shared by
    # reference across a family's companions. Populated once at construction,
    # read-only after — the `@deffun` module scope's value analogue. Treat as
    # immutable: mutating it after publication is undefined behavior.
    links::Dict{Symbol,Any}
end

struct ValueFamily
    label::Symbol
    kind::Symbol  # :lpdf or :lpmf
    density::ValueUDF
    pointwise::ValueUDF
    rng::ValueUDF
    support::NamedTuple
    mod::Module
end

Base.show(io::IO, v::ValueFamily) =
    print(io, "ValueFamily(", v.label, ")")
Base.show(io::IO, v::ValueUDF) =
    print(io, "ValueUDF(", v.label, ")")

_value_label(v::ValueFamily) = v.label
_value_label(v::ValueUDF) = v.label

# --- Def parsing (mirrors the `@deffun` single-definition grammar) ---------

# Peel `@deffun` marker macrocalls off one value def. Markers are
# method-registration annotations; a value's ROLE already implies them, so the
# role-vacuous ones are accepted-and-dropped (documented, not silent) and the
# rest are rejected. Returns `(stripped_def, source)`.
_value_strip_markers(def, role::Symbol, label::Symbol, source) = begin
    while Meta.isexpr(def, :macrocall)
        is_lhs = _is_lhs_macrocall(def)
        is_lpxf = _is_lpxf_macrocall(def)
        is_inline = _is_at_inline_macrocall(def)
        is_julia = _is_juliacompat_macrocall(def)
        is_stanonly = _is_stanonly_macrocall(def)
        is_doc = _is_doc_macrocall(def)
        (is_lhs || is_lpxf || is_inline || is_julia || is_stanonly || is_doc) || error(
            "ValueUDF `", label, "`: unexpected marker `", def.args[1], "` — value ",
            "definitions accept only `@lhs`/`@lpxf` (on a family density, where the ",
            "role implies them), `@stanonly` (values are Stan-only), and no `@doc`.",
        )
        is_doc && error(
            "ValueUDF `", label, "`: `@doc` is not supported on value definitions — ",
            "document the family at the construction site instead.",
        )
        is_inline && error(
            "ValueUDF `", label, "`: `@inline` has no value meaning in v1 — value ",
            "companions always lower to lifted Stan functions like non-`@inline` ",
            "`@deffun`s. Drop the marker.",
        )
        is_julia && error(
            "ValueUDF `", label, "`: `@juliacompat` has no value meaning in v1 — ",
            "values are Stan-only (probability families are outside the Julia ",
            "target by construction, like `@deffun`). Drop the marker.",
        )
        if is_lhs || is_lpxf
            role === :density || error(
                "ValueUDF `", label, "`: `@lhs`/`@lpxf` annotate only a family's ",
                "density definition (role `", role, "` here) — the same rule as ",
                "`@deffun`, whose `@lpxf` derives the family from the `_lpdf` name.",
            )
        end
        length(def.args) >= 3 || error(
            "ValueUDF `", label, "`: malformed marker `", def.args[1], "` — expected ",
            "one annotated definition.",
        )
        source = _macrocall_source(def.args[2], source)
        def = def.args[3]
    end
    (def, source)
end

# Resolve one formal's center-type AST to a SLIC type. `typeof(name)` function
# tags resolve ONE bare module binding via `getproperty` (no eval); anything
# else that is not a `types.*` name is rejected (in `@deffun` it would splice
# raw into a method signature and never match — dead, but silent).
_value_resolve_ct(ct, label::Symbol, mod::Module) = begin
    ct isa Symbol && isdefined(types, ct) && return getproperty(types, ct)
    if Meta.isexpr(ct, :call) && length(ct.args) == 2 && ct.args[1] === :typeof &&
            ct.args[2] isa Symbol && isdefined(mod, ct.args[2])
        return types.func{typeof(getproperty(mod, ct.args[2]))}
    end
    error(
        "ValueUDF `", label, "`: formal center type `", ct, "` is not a SLIC type ",
        "(`types.*`) or a resolvable `typeof(name)` in `", mod, "` — in `@deffun` ",
        "such a formal splices raw into the dispatch signature and can never ",
        "match. Spell a SLIC type (`real`, `vector`, `int`, …) or bind the ",
        "function to a name in `", mod, "` first.",
    )
end

_value_parse_formals!(label::Symbol, all_args, mod::Module) = begin
    if !isempty(all_args) && Meta.isexpr(all_args[1], :parameters)
        error(
            "ValueUDF `", label, "`: keyword formals (`f(x; k=…)` ) are not ",
            "supported in v1 — `@deffun` lowers those through a `kwcall` ",
            "canonical method plus an `@inline` shim, which has no value form ",
            "yet. Drop the kwargs or file a follow-up.",
        )
    end
    any(a -> Meta.isexpr(a, :kw), all_args) && error(
        "ValueUDF `", label, "`: default positional args (`f(x, y=…)`) are not ",
        "supported in v1 — `@deffun` desugars those to `@inline` trampolines. ",
        "Spell every arity out as its own overload instead.",
    )
    args, vararg = if hasvararg(all_args)
        all_args[1:end-1], all_args[end]
    else
        all_args, nothing
    end
    vararg_name = if vararg === nothing
        nothing
    else
        vararg.args[1] isa Symbol || error(
            "ValueUDF `", label, "`: only untyped trailing varargs (`args...`) ",
            "are supported, got `", vararg, "` (matching `@deffun`, whose ",
            "vararg handling names the pack from a bare Symbol).",
        )
        vararg.args[1]
    end
    is_token = Bool[_is_type_token(arg) for arg in args]
    typed = map(zip(args, is_token, eachindex(args))) do (arg, tok, i)
        if tok
            xtyped(Symbol("anontok__", i), _type_token_ref(arg))
        else
            ensure_xtyped(arg, Symbol("arg__", i))
        end
    end
    formal_names = map(arg -> arg.args[1], typed)
    for name in formal_names
        name isa Symbol || error(
            "ValueUDF `", label, "`: unsupported formal `", name, "` — use ",
            "`name`, `name::T[dims...]`, `::T[dims...]`, or a bare sized-token ",
            "`T[dims...]`, like `@deffun`.",
        )
    end
    formal_types = map(arg -> ensure_xref(arg.args[2]), typed)
    patterns = map(zip(formal_types, is_token)) do (at, tok)
        ct_ast, dims... = at.args
        ct = _value_resolve_ct(ct_ast, label, mod)
        ndim = length(dims)
        isfunc = ct <: types.func
        ValueUDFPattern(ct, ndim, tok, isfunc)
    end
    (formal_names, formal_types, patterns, is_token, vararg_name)
end

# Parse one `@deffun`-shaped `sig = body` definition into an overload.
# `role` is `:density`, `:pointwise`, `:rng`, or `:udf`.
_value_parse_def(label::Symbol, role::Symbol, raw_def, mod::Module) = begin
    def = deepcopy(raw_def)
    def isa Expr || error(
        "ValueUDF `", label, "`: each overload must be one `@deffun`-shaped ",
        "`sig = body` expression, got `", def, "` (`", typeof(def), "`).",
    )
    try
        def = lower_string_interp(slic_macroexpand(mod, def))
    catch e
        error(
            "ValueUDF `", label, "`: macro expansion against `", mod, "` failed: ",
            sprint(showerror, e),
        )
    end
    source = LineNumberNode(0, :none)
    (def, source) = _value_strip_markers(def, role, label, source)
    fsig, body = ensure_xassign(def).args
    ismissing(body) && error(
        "ValueUDF `", label, "`: signature stubs (no body) have no value form — ",
        "a stub registers dispatch for a NATIVE Stan function, and native ",
        "families are already plain Julia functions. Give every overload a body.",
    )
    definition_lnn = something(_last_source_lnn(body), source)
    fcall, rv = ensure_xtyped(fsig).args
    Meta.isexpr(fcall, :call) || error(
        "ValueUDF `", label, "`: the signature must be a `:call` like ",
        "`", label, "(args...)::T`, got `", fcall, "`.",
    )
    f, all_args... = fcall.args
    f === label || error(
        "ValueUDF `", label, "`: definition names `", f, "`, not `", label, "` — ",
        "every overload of one companion must carry the companion label.",
    )
    (formal_names, formal_types, patterns, is_token, vararg_name) =
        _value_parse_formals!(label, all_args, mod)
    # Only the density needs an observation first formal (the `@lhs` rule drops
    # it from sampling dispatch and types the sample from it). Pointwise/RNG
    # companions match whole calls, so vararg-only shapes (like the mixture
    # `pointwise(args...) = density(args...)` delegation) are meaningful.
    role === :density && isempty(patterns) && error(
        "ValueUDF `", label, "`: a density companion needs an observation ",
        "first formal — got a nullary/vararg-only definition (mirroring ",
        "`@deffun`, whose `@lhs` requires an explicit observation argument).",
    )
    role === :density && !isempty(patterns) && patterns[1].istoken && error(
        "ValueUDF `", label, "`: a family density's first formal is the ",
        "observation — it cannot be a sized type token (tokens lead `_rng` ",
        "overloads, mirroring `@deffun`).",
    )
    body isa Expr || error(
        "ValueUDF `", label, "`: the body must be an expression, got `", body, "`.",
    )
    # Surface syntax wraps short-form bodies in `:block` at parse time;
    # programmatic ASTs may spell them bare — wrap, matching the surface.
    body = Meta.isexpr(body, :block) ? body : Expr(:block, body)
    _reject_udf_forms!(body, label)
    body = ensure_xreturn(body)
    rv isa Symbol || Meta.isexpr(rv, :ref) || _is_computed_ret_type(rv) || error(
        "ValueUDF `", label, "`: unsupported return annotation `", rv, "` — use ",
        "`::T[dims...]`, `::anything`, or a computed `typeof(...)` return, like ",
        "`@deffun`.",
    )
    rv === :void && error(
        "ValueUDF `", label, "`: `:void` companions are not supported in v1 — ",
        "density/pointwise/RNG companions always return values.",
    )
    analysis = _udf_size_analysis(label, formal_names, formal_types, is_token, body, rv)
    sig_rv = (rv === :anything || _is_computed_ret_type(rv)) ? nothing :
        make_stan_type(rv)
    (
        formal_names, formal_types, patterns, vararg_name, rv, sig_rv, body,
        analysis.fun_sizes, analysis.fun_checks,
        (; analysis.fun_size_alias_names...), definition_lnn,
    )
end

# --- Construction ----------------------------------------------------------

# Structural body hash with spliced values normalized to their labels, so two
# separately-constructed identical families agree (plain `==`/`hash` on the
# structs would compare object identity).
_value_body_key!(h::UInt, x::ValueUDF) = hash((:valueudf, x.label), h)
_value_body_key!(h::UInt, x::ValueFamily) = hash((:valuefamily, x.label), h)
_value_body_key!(h::UInt, x::Expr) = begin
    h = hash(x.head, h)
    for arg in x.args
        h = _value_body_key!(h, arg)
    end
    h
end
_value_body_key!(h::UInt, x::QuoteNode) = hash(x.value, h)
_value_body_key!(h::UInt, ::LineNumberNode) = h
_value_body_key!(h::UInt, x) = hash(x, h)
_value_body_key(body::Expr) = _value_body_key!(UInt(0), body)

_value_overload_key(patterns, vararg_name) = (
    Tuple((p.ct, p.ndim, p.istoken, p.isfunc) for p in patterns),
    vararg_name === nothing,
)

# Sibling/self references inside companion bodies resolve through one Dict
# shared by reference across the family's companions (populated once at
# construction, read-only after — the `@deffun` module scope's value analogue).
# A shared Dict (rather than splicing values into the ASTs) keeps consumer
# ASTs byte-identical to their `@deffun` form: overload recursion and
# companion cross-calls keep spelling bare labels, and shadowing follows
# ordinary Julia scope rules (formals/locals bind over the links).
_value_build_udf(label::Symbol, role::Symbol, defs, mod::Module, links) = begin
    overloads = map(enumerate(defs)) do (i, raw_def)
        (formal_names, formal_types, patterns, vararg_name, rv, sig_rv, body,
            fun_sizes, fun_checks, size_aliases, definition_lnn) =
            _value_parse_def(label, role, raw_def, mod)
        ValueUDFOverload(
            label, formal_names, formal_types, patterns, vararg_name, rv, sig_rv,
            body, _value_body_key(body), fun_sizes, fun_checks, size_aliases,
            mod, definition_lnn,
        )
    end
    seen = Set()
    for (i, o) in enumerate(overloads)
        key = _value_overload_key(o.patterns, o.vararg)
        key in seen && error(
            "ValueUDF `", label, "`: overload ", i, " duplicates an earlier ",
            "overload's signature — `@deffun` would method-redefine (last wins, ",
            "with a warning); programmatic defs duplicate only by bug, so this ",
            "is an error. Drop or distinguish the duplicate.",
        )
        push!(seen, key)
    end
    ValueUDF(label, Tuple(overloads), mod, links)
end

"""
    ValueUDF(label::Symbol, def, mod::Module)
    ValueUDF(label::Symbol, defs::AbstractVector, mod::Module)

A value-based Stan function: one `label` with one or more `@deffun`-shaped
`sig = body` overloads, traced in `mod`'s context. No Julia function, method,
or module binding is created — the value splices directly into generated
`Expr`s as a call head, or binds in a module and resolves by name.

Each def mirrors one `@deffun`-block definition (`name(args...)::ret = body`,
`args...` trailing vararg allowed, sized-token leading formals allowed).
v1 rejects kwarg formals, positional defaults, signature stubs, `:void`
returns, and every marker except the vacuous `@stanonly` (`@lhs`/`@lpxf` claim
a family density role a bare UDF does not have; `@inline`/`@juliacompat` have
no value meaning — companions always lift like non-`@inline` `@deffun`s).
Bodies may reference the companion's own `label` (overload recursion).
"""
ValueUDF(label::Symbol, def::Expr, mod::Module) = ValueUDF(label, Any[def], mod)
ValueUDF(label::Symbol, defs::AbstractVector, mod::Module) = begin
    label isa Symbol || error("ValueUDF: the label must be a Symbol, got `", label, "`.")
    isempty(defs) && error("ValueUDF `", label, "`: supply at least one overload definition.")
    links = Dict{Symbol,Any}()
    udf = _value_build_udf(label, :udf, defs, mod, links)
    links[label] = udf
    udf
end

_value_check_family_label(label::Symbol) = begin
    s = string(label)
    occursin(r"^[A-Za-z][A-Za-z0-9_]*$", s) || error(
        "ValueFamily: the label must be a Stan identifier ",
        "(`[A-Za-z][A-Za-z0-9_]*`), got `", label, "`.",
    )
    for suffix in ("_lpdf", "_lpmf", "_lpdfs", "_lpmfs", "_rng", "_lcdf", "_lccdf")
        endswith(s, suffix) && error(
            "ValueFamily: the label `", label, "` ends in `", suffix, "` — family ",
            "labels name the bare distribution (`y ~ ", label, "(…)` is emitted ",
            "verbatim); the companions derive their suffixed labels from it.",
        )
    end
    nothing
end

"""
    ValueFamily(label::Symbol, kind::Symbol, density_defs, pointwise_defs, rng_defs, mod::Module; support=(;))

A value-based sampling distribution: the `_lpdf`/`_lpmf` + pointwise + `_rng`
triad as DATA, usable in `y ~ family(…)` with no `Core.eval`, no Julia
methods, and no module bindings. `kind` is `:lpdf` or `:lpmf`. Each `*_defs`
is one `@deffun`-shaped `sig = body` expression or a vector of them (scalar
AND whole-vector overloads, as usual); companion labels derive from `label`
exactly like `@lpxf` derives them (`<label>_<kind>`, `<label>_<kind>s`,
`<label>_rng`). `support` is the intrinsic-support NamedTuple (the
`autokwargs` contract — scalars, vectors, `adjoint`/`transpose` wrappers —
kept on ONE native parameter declaration). `mod` is the defining module
(actual identity, not a printed name): companion bodies trace in its context.

Companion bodies spell sibling labels bare (`density` calling `pointwise`,
RNG recursing across overloads, …) exactly as in `@deffun` source; the
constructor links them to the companion values, so `@deffun`-shaped ASTs
migrate by swapping the `Core.eval` for this constructor. Splice the family
into a generated body (`Expr(:call, family, args…)`) or bind it in a module
and reference it by name.

v1 FOLLOW-UPS (each fails loudly today): HOF-wrapping a value family
(`truncated`/`censored`/`weighted`/…), explicit user bounds at a value `~`
site (put bounds in `support=`), selector/HOF-argument positions,
data-dict carriage, ragged observations, `_lcdf`/`_lccdf` kinds.
"""
ValueFamily(
    label::Symbol, kind::Symbol, density_defs, pointwise_defs, rng_defs,
    mod::Module; support::NamedTuple=(;),
) = begin
    _value_check_family_label(label)
    (kind === :lpdf || kind === :lpmf) || error(
        "ValueFamily `", label, "`: kind must be `:lpdf` or `:lpmf`, got `", kind,
        "` (`:lcdf`/`:lccdf` companions are a v1 follow-up).",
    )
    density_label = Symbol(label, :_, kind)
    pointwise_label = Symbol(density_label, :s)
    rng_label = Symbol(label, :_rng)
    links = Dict{Symbol,Any}()
    density = _value_build_udf(density_label, :density, _value_defs(density_defs), mod, links)
    pointwise = _value_build_udf(pointwise_label, :pointwise, _value_defs(pointwise_defs), mod, links)
    rng = _value_build_udf(rng_label, :rng, _value_defs(rng_defs), mod, links)
    family = ValueFamily(label, kind, density, pointwise, rng, support, mod)
    links[label] = family
    links[density_label] = density
    links[pointwise_label] = pointwise
    links[rng_label] = rng
    family
end

_value_defs(def::Expr) = Any[def]
_value_defs(defs::AbstractVector) = defs
_value_defs(defs) = error(
    "ValueFamily: each companion's defs must be one `sig = body` expression ",
    "or a vector of them, got `", defs, "` (`", typeof(defs), "`).",
)

# --- Overload matching (replicates `@deffun` dispatch) ----------------------

_value_pattern_matches(p::ValueUDFPattern, actual) = begin
    actual isa StanExpr || return false
    ct = center_type(actual)
    nd = stan_ndim(actual)
    if p.istoken
        return ct <: types.tokenof{<:p.ct} && nd == p.ndim
    end
    # The `anything`-scalar escape hatch (`xsig_type`): an untyped formal
    # matches any shape.
    p.ct === types.anything && p.ndim == 0 && return true
    ct <: p.ct && nd == p.ndim
end

_value_arity_ok(o::ValueUDFOverload, pats, nargs::Int) =
    o.vararg === nothing ? nargs == length(pats) : nargs >= length(pats)

_value_overload_matches(o::ValueUDFOverload, pats, actuals) =
    _value_arity_ok(o, pats, length(actuals)) &&
    all(_value_pattern_matches(p, a) for (p, a) in zip(pats, actuals))

# Julia dispatch order among matching patterns: `p1` is at least as specific
# as `p2` at equal ndim/tokenness/funcness with a subtype center type.
_value_pattern_le(p1::ValueUDFPattern, p2::ValueUDFPattern) =
    p1.ndim == p2.ndim && p1.istoken == p2.istoken && p1.isfunc == p2.isfunc &&
    p1.ct <: p2.ct

_value_overload_le(o1::ValueUDFOverload, o2::ValueUDFOverload) = begin
    pats1, pats2 = o1.patterns, o2.patterns
    v1, v2 = o1.vararg !== nothing, o2.vararg !== nothing
    # Fixed beats trailing-`args...` (Julia: the fixed tuple is a subtype of
    # the vararg one).
    v1 != v2 && return v2
    n1, n2 = length(pats1), length(pats2)
    if v1
        # Both vararg: the longer fixed prefix is more specific (a 2+-tuple
        # is a 1+-tuple).
        n1 != n2 && return n1 > n2
    else
        n1 == n2 || return false
    end
    all(_value_pattern_le(p1, p2) for (p1, p2) in zip(pats1, pats2))
end

_value_pattern_str(at::Expr) = sprint(show, at)
_value_actual_str(a) = begin
    a isa StanExpr || return string(typeof(a))
    ct = center_type(a)
    base = ct <: types.tokenof ? string("tokenof(", ct.parameters[1], ")") : string(nameof(ct))
    string(base, "[", stan_ndim(a), "]")
end

_value_match_shapes(udf::ValueUDF, actuals, drop_first::Bool) = begin
    actual_str = isempty(actuals) ? "(no args)" : join(map(_value_actual_str, actuals), ", ")
    overload_strs = map(udf.overloads) do o
        fixed = drop_first ? o.formal_types[2:end] : o.formal_types
        inner = join(map(_value_pattern_str, fixed), ", ")
        o.vararg === nothing ? string("(", inner, ")") :
            string("(", inner, isempty(inner) ? "" : ", ", o.vararg, "...)")
    end
    (actual_str, overload_strs)
end

# Match a call's actuals against one companion's overloads. `drop_first`
# drops the observation formal (the `@lhs` rule: the sampling call carries
# the density's args minus the observation).
_value_match_overload(udf::ValueUDF, actuals, drop_first::Bool) = begin
    matches = ValueUDFOverload[]
    for o in udf.overloads
        pats = drop_first ? o.patterns[2:end] : o.patterns
        _value_overload_matches(o, pats, actuals) && push!(matches, o)
    end
    if isempty(matches)
        (actual_str, overload_strs) = _value_match_shapes(udf, actuals, drop_first)
        role = drop_first ? "sampling" : "call"
        error(
            "ValueUDF `", udf.label, "`: no overload matches this ", role, " (",
            actual_str, "). Overloads:", isempty(overload_strs) ? " (none)" : "",
            map(s -> string("\n    ", udf.label, s), overload_strs)...,
        )
    end
    length(matches) == 1 && return first(matches)
    for m in matches
        all(o -> m === o || _value_overload_le(m, o), matches) && return m
    end
    (actual_str, _) = _value_match_shapes(udf, actuals, drop_first)
    role = drop_first ? "sampling" : "call"
    error(
        "ValueUDF `", udf.label, "`: the ", role, " (", actual_str, ") matches ",
        length(matches), " overloads ambiguously (Julia would reject the ",
        "equivalent `@deffun` the same way). Distinguish the overloads' formal ",
        "shapes.",
    )
end

# --- Size binding + return types (mirrors `@deffun` deconstruction) ----------

# Bind formal size names from the call's actual sizes (later formals overwrite,
# mirroring the generated deconstruct's sequential `xassign`s; the emitted
# Stan `reject`s — not the trace — enforce multi-site agreement).
_value_bind_sizes!(bindings, o::ValueUDFOverload, actuals, drop_first::Bool) = begin
    formals = drop_first ? o.formal_names[2:end] : o.formal_names
    types = drop_first ? o.formal_types[2:end] : o.formal_types
    for (name, at, actual) in zip(formals, types, actuals)
        actual isa StanExpr || continue
        sizes = stan_size(actual)
        for (i, dim) in enumerate(at.args[2:end])
            dim isa Symbol || continue
            dim === :_ && continue
            dim in o.formal_names && continue
            i <= length(sizes) && (bindings[dim] = sizes[i])
        end
    end
    bindings
end

_value_resolve_size(dim, bindings, scope, label::Symbol) = begin
    dim isa Integer && return stan_expr(dim)
    if dim isa Symbol
        haskey(bindings, dim) || error(
            "ValueUDF `", label, "`: size `", dim, "` is unbound — it names no ",
            "formal dimension of the matched overload (the `@deffun` hidden-size ",
            "rule: every size in the return type must come from the signature).",
        )
        return bindings[dim]
    end
    # Computed sizes (`vector[n + 1]`) forward against the size scope, exactly
    # like `@deffun`'s `xsig_expr` forwards them against its deconstruction.
    forward!(canonical(dim); info=scope)
end

_value_size_scope(o::ValueUDFOverload, bindings, context) = begin
    scope = OrderedDict{Symbol,Any}(pairs(bindings))
    scope[:__mod__] = o.mod
    _attach_trace_context!(scope, context)
    scope
end

# The declared return/observation type with sizes resolved from the call
# (`xsig_expr` semantics, at runtime).
_value_declared_type(o::ValueUDFOverload, rettype, bindings, context) = begin
    xref = ensure_xref(rettype)
    ct_ast, dims... = xref.args
    ct = _value_resolve_ct(ct_ast, o.label, o.mod)
    scope = _value_size_scope(o, bindings, context)
    sizes = map(d -> _value_resolve_size(d, bindings, scope, o.label), dims)
    StanType(ct, Tuple(sizes))
end

# The argument scope one lift/inference runs in: sibling links first (so
# bodies spell bare labels, like `@deffun` source), formals over them
# (ordinary shadowing), then sizes.
_value_lift_scope(udf::ValueUDF, o::ValueUDFOverload, actuals, context) = begin
    scope = OrderedDict{Symbol,Any}()
    for (name, target) in pairs(udf.links)
        scope[name] = target
    end
    nfixed = length(o.patterns)
    for (name, actual) in zip(o.formal_names, actuals[1:nfixed])
        scope[name] = actual
    end
    o.vararg !== nothing && (scope[o.vararg] = Tuple(actuals[nfixed+1:end]))
    bindings = _value_bind_sizes!(OrderedDict{Symbol,Any}(), o, actuals, false)
    for (name, size) in pairs(bindings)
        scope[name] = size
    end
    scope[:__mod__] = o.mod
    _attach_trace_context!(scope, context)
    scope
end

# --- Trace methods -----------------------------------------------------------

# Values are already resolved: raw passthrough everywhere (heads via the
# generic `_forward_head`, module bindings via `_forward_module_value` below,
# `Mod.name` refs via the `GlobalRef` path). Non-head positions that need a
# Stan representation fail loudly at their own hook (`stan_expr`/`stan_type`,
# selectors, HOF tokens — each names the v1 boundary).
forward!(v::ValueFamily; info) = v
forward!(v::ValueUDF; info) = v

_forward_module_value(v::ValueFamily, info) = v
_forward_module_value(v::ValueUDF, info) = v
_resolve_module_value(v::ValueFamily, info) = v
_resolve_module_value(v::ValueUDF, info) = v

_value_position_error(v, where::String) = error(
    _value_kind_str(v), " `", _value_label(v), "` ", where, " — v1 supports ",
    "value families/companions in call-head and `~` positions (spliced or ",
    "module-bound) only. Selector/HOF-argument positions, explicit user bounds, ",
    "data-dict carriage, and ragged generated-quantities are documented ",
    "follow-ups (see the `ValueFamily` docstring).",
)

_value_kind_str(::ValueFamily) = "ValueFamily"
_value_kind_str(::ValueUDF) = "ValueUDF"

# Conversion positions (user-bounds redraw's `stan_expr(head)`, `stan_call`
# args, `_trace_stan_arg`) fail closed with the v1 boundary.
stan_expr(v::ValueFamily; kwargs...) =
    _value_position_error(v, "reached a Julia-value position")
stan_expr(v::ValueUDF; kwargs...) =
    _value_position_error(v, "reached a Julia-value position")
stan_type(expr, v::ValueFamily; kwargs...) =
    _value_position_error(v, "cannot ride the data dict")
stan_type(expr, v::ValueUDF; kwargs...) =
    _value_position_error(v, "cannot ride the data dict")

# `type()`/`expr()` backstops: a raw value reaching representation queries is
# outside the supported positions (heads never query them — only exotic
# arg/data flows like HOF family splices do).
type(v::ValueFamily) = _value_position_error(v, "has no Stan type outside call-head position")
type(v::ValueUDF) = _value_position_error(v, "has no Stan type outside call-head position")
expr(v::ValueFamily) = _value_position_error(v, "has no Stan expression outside call-head position")
expr(v::ValueUDF) = _value_position_error(v, "has no Stan expression outside call-head position")

# Direct assignment of a bare value (`x = myfamily`) is meaningless in v1
# (values are call heads, not first-class Stan data); name the boundary.
forward!(x::CanonicalExprV{:(=),Tuple{Symbol,ValueFamily}}; info) =
    _value_position_error(x.args[2], "cannot be assigned to a variable")
forward!(x::CanonicalExprV{:(=),Tuple{Symbol,ValueUDF}}; info) =
    _value_position_error(x.args[2], "cannot be assigned to a variable")

# Selectors (`density`/`pointwise`/`predictive`) take function families; a
# value in the family slot names the boundary instead of falling through to
# a confusing `tracetype` failure. (Function-first calls keep hitting the
# more specific method above this one in `lpxf_builtin.jl`.)
expand_inline_or_trace(x::CanonicalExpr{<:DistributionFamilySelector}; info) = begin
    if !isempty(x.args) && first(x.args) isa Union{ValueFamily,ValueUDF}
        _value_position_error(first(x.args), "cannot select a companion from")
    end
    invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info)
end

# HOF token positions (`truncated(family, …)`, ragged-GQ markers) take
# function tokens; a value names the boundary.
_family_function(v::ValueFamily) = _value_position_error(v, "cannot serve as an HOF family")
_family_function(v::ValueUDF) = _value_position_error(v, "cannot serve as an HOF family")

# One companion call's result type: match, bind sizes, resolve the return.
_tracetype(x::CanonicalExpr{<:ValueUDF}, context) = begin
    udf = head(x)
    o = _value_match_overload(udf, x.args, false)
    bindings = _value_bind_sizes!(OrderedDict{Symbol,Any}(), o, x.args, false)
    if o.sig_rv !== nothing
        return _value_declared_type(o, o.rettype, bindings, context)
    end
    scope = _value_lift_scope(udf, o, x.args, context)
    forward_return!(canonical(o.body), string(o.label); info=scope).type
end
tracetype(x::CanonicalExpr{<:ValueUDF}) = _tracetype(x, nothing)

# The `@lhs` rule, at runtime: the sampling call carries the density's args
# minus the observation, and its type is the observation's type with sizes
# bound from the call.
_tracetype(x::CanonicalExpr{<:ValueFamily}, context) = begin
    family = head(x)
    o = _value_match_overload(family.density, x.args, true)
    bindings = _value_bind_sizes!(OrderedDict{Symbol,Any}(), o, x.args, true)
    obs_ast = o.formal_types[1]
    # `anything[]` observations mean scalar `real` (the `@lhs` normalization).
    obs_ast = obs_ast == :(anything[]) ? :(real[]) : obs_ast
    _value_declared_type(o, obs_ast, bindings, context)
end
tracetype(x::CanonicalExpr{<:ValueFamily}) = _tracetype(x, nothing)

# --- Lifting (mirrors the `@deffun` non-`@inline` `fundef`) ------------------

# One lifted Stan function per (label, arg-shape): same scope shape as the
# generated `@deffun` lift (links + anonymized formals + sizes + aliases +
# module + context), same preamble/checks/body assembly.
_value_lift(udf::ValueUDF, o::ValueUDFOverload, x::CanonicalExpr, context) = begin
    actuals = x.args
    scope = OrderedDict{Symbol,Any}()
    for (name, target) in pairs(udf.links)
        scope[name] = target
    end
    nfixed = length(o.patterns)
    named = NamedTuple(
        name => actual for (name, actual) in zip(o.formal_names, actuals[1:nfixed])
    )
    if o.vararg !== nothing
        named = merge(named, (; Symbol(o.vararg) => Tuple(actuals[nfixed+1:end])))
    end
    for (name, value) in pairs(anon_info(named))
        scope[name] = value
    end
    # Size names bind as FRESH `StanExpr`s keyed by name — exactly what the
    # generated deconstruct-then-`anon_info` produces (`anon_expr(key, size)`
    # is `StanExpr(key, int)` whichever formal carried the size). Binding a
    # formal's `dims(name)[i]` placeholder instead would leak the LAST
    # formal's access into body references (`for i in 1:n` emitting
    # `1:dims(b)[1]`). Only the analysis's canonical `fun_sizes` keys reach
    # the body scope (the analysis already drops `_`, formal-name, and
    # unreferenced dims); hidden/dropped dims never do.
    for dim in keys(o.fun_sizes)
        scope[dim] = StanExpr(dim, StanType(types.int))
    end
    scope[:__size_alias_names__] = o.size_aliases
    scope[:__mod__] = o.mod
    _attach_trace_context!(scope, context)
    body_block = forward!(canonical(o.body); info=scope)
    sig_names = o.vararg === nothing ? o.formal_names : vcat(o.formal_names, o.vararg)
    args_nt = (; [name => scope[name] for name in sig_names]...)
    if o.sig_rv !== nothing
        sig_rv = o.sig_rv
    else
        sig_rv = forward_return!(canonical(o.body), string(o.label); info=scope).type
    end
    StanFunction3(
        string("// value UDF ", o.label, "\n"),
        sig_rv, udf, args_nt,
        vcat(collect(values(o.fun_sizes)), o.fun_checks, body_block),
        _deref_lnn(_current_lnn(scope)),
    )
end

_fundef(x::CanonicalExpr{<:ValueUDF}, context) = begin
    udf = head(x)
    o = _value_match_overload(udf, x.args, false)
    _with_slic_diagnostic(
        :slic_lowering_error, "function", context, o.source,
    ) do
        _value_lift(udf, o, x, context)
    end
end
fundef(x::CanonicalExpr{<:ValueUDF}) = _fundef(x, nothing)

# Fetch with a same-key divergence check: two value UDFs that share a
# (label, arg-shape) key in ONE trace must carry the same body — `@deffun`
# would silently first-win here (method redefinition across surfaces), but
# programmatic labels collide only by bug. The stored definition's parent is
# the first-seen UDF value, so the comparison needs no extra table.
fetch_functions!(x::CanonicalExpr{<:ValueUDF}; info) = begin
    sx = sig_expr(x)
    if sx in keys(info)
        prior = info[sx]
        if prior isa StanFunction3 && prior.parent isa ValueUDF
            now = _value_match_overload(head(x), x.args, false)
            for o in prior.parent.overloads
                _value_overload_key(o.patterns, o.vararg) ==
                    _value_overload_key(now.patterns, now.vararg) || continue
                o.body_key == now.body_key || error(
                    "ValueUDF `", head(x).label, "`: two different bodies share ",
                    "one (label, signature) in this trace — labels must be unique ",
                    "per definition (BRM fingerprints derive them from content). ",
                    "Rename one family.",
                )
            end
        end
        return
    end
    info[sx] = _fundef(x, _trace_context(info))
    isnothing(info[sx]) && return
    fetch_subfunctions!(info[sx].body; info)
end

# --- Family hooks (the `@lpxf` triad + `@lhs` support, as values) -------------

# The one-arg hooks return the companion VALUES; the two-arg call builders in
# `passes.jl`/`lpxf_builtin.jl` then trace `CanonicalExpr(companion, …)`
# through the methods above. No per-family methods are defined, ever.
lpxf_expr(v::ValueFamily) = v.density
rng_expr(v::ValueFamily) = v.rng
likelihood_expr(v::ValueFamily) = v.pointwise

# Intrinsic support bounds (the `autokwargs` contract): the stored values flow
# through the same downstream normalization as method-returned bounds.
autokwargs(x::CanonicalExpr{<:ValueFamily}) = head(x).support

# The declared kind drives ragged-carrier checks and HOF-kind dispatch that
# would otherwise `nameof` a companion function.
_probability_kind(v::ValueFamily) = v.kind

# --- Naming, dedup keys, display ---------------------------------------------

# One-arg `func_name` feeds the generic two-arg name mangler: ordinary args
# contribute nothing, so overloads share one Stan name and Stan overloads
# natively — exactly like `@deffun`.
func_name(v::ValueFamily) = string(v.label)
func_name(v::ValueUDF) = string(v.label)

# Dedup keys live in their own head namespace (`:valueudf`) so a value UDF
# never dedups against a same-labeled `@deffun` — that would be silent
# first-wins across definition surfaces. Keys are write-only dedup tags
# (consumers read only the stored definitions).
sig_expr(x::CanonicalExpr{<:ValueUDF}) =
    CanonicalExpr(:valueudf, head(x).label, sig_expr(x.args)...)
