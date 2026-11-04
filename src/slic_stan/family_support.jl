"""
    family_support(f, args...) -> NamedTuple

Inspect the registered observation type and intrinsic support of an ordinary
`@lpxf` family, builtin family, or `ValueFamily` for a typed signature.
Pass symbolic declarations such as `:(mu::real)`, `:(sigma::real)`, or
`:(mu::vector[n])`. Undeclared size symbols are symbolic data integers.
Numeric literals are also accepted; they remain literals, not representative
values substituted for symbolic arguments.

The result has fields `kind` (`:continuous` or `:discrete`), `type` (the Stan
center-type symbol), `size` (a tuple of dimension expressions), and `support`
(a named tuple of intrinsic `lower`/`upper` bounds). Bounds and dimensions
retain symbolic expressions; compound expressions are Stan source strings.
For example, `family_support(StanBlocks.stan.uniform, :(lo::real), :(hi::real))`
reports `support = (lower = :lo, upper = :hi)`. Literal `lower=-Inf` and
`upper=Inf` are omitted, as they impose no restriction.

A whole-real scalar family has `kind === :continuous`, `type === :real`,
`isempty(size)` and `isempty(support)`. This inspects SLIC's registration,
including typed `autokwargs` hooks; it does not prove support from a density's
mathematical body. Extension authors must register nontrivial support.
Unknown families, unsupported signatures and untyped arguments raise errors;
they never become an empty-support result. This query does not test companion
implementations or add call-site truncation bounds.
"""
family_support(f, args...) = _with_slic_read() do
    _family_support(f, args)
end

function _family_support(f, args)
    f isa Union{Function,ValueFamily} || throw(ArgumentError(
        "family_support expects a registered family function or ValueFamily, got $(typeof(f))."))
    kind = _probability_kind(f) === :lpdf ? :continuous : :discrete
    context = TraceContext()
    scope = OrderedDict{Symbol,Any}()
    _attach_trace_context!(scope, context)
    # Keep the caller's AST untouched. Declare every formal first so sizes may
    # refer to a later integer argument as well as an implicit size symbol.
    declarations = Dict{Symbol,Any}()
    for arg in args
        arg isa Number && continue
        Meta.isexpr(arg, :(::), 2) && arg.args[1] isa Symbol || throw(ArgumentError(
            "family_support arguments must be numeric literals or typed declarations such as :(mu::real); got $arg."))
        name = arg.args[1]
        haskey(declarations, name) && throw(ArgumentError("family_support: duplicate argument `$name`."))
        declarations[name] = arg
    end
    for (name, arg) in declarations
        type_ast = ensure_xref(arg.args[2])
        ct = type_ast.args[1]
        ct isa Symbol && isdefined(types, ct) && getproperty(types, ct) <: types.anything ||
            throw(ArgumentError("family_support: `$ct` is not a builtin Stan type."))
        ct === :anything && throw(ArgumentError(
            "family_support: argument `$name` needs a concrete Stan type."))
        # Placeholders bind named scalar integer dimensions before forwarding
        # the complete declaration through SLIC's ordinary type parser.
        if ct === :int && length(type_ast.args) == 1
            scope[name] = StanExpr(name, StanType(types.int; qual=:data))
        end
        for dim in type_ast.args[2:end]
            symbols = Set{Symbol}()
            _size_symbols(canonical(dim), symbols)
            for sym in symbols
                if haskey(declarations, sym)
                    declarations[sym].args[2] === :int || throw(ArgumentError(
                        "family_support: dimension `$sym` must be declared as a scalar int."))
                    continue
                end
                scope[sym] = StanExpr(sym, StanType(types.int; qual=:data))
            end
        end
    end
    actuals = map(args) do arg
        arg isa Number && return stan_expr(arg)
        forward!(canonical(deepcopy(arg)); info=scope)
        name = arg.args[1]
        typed = scope[name]
        center_type(typed) === types.anything && throw(ArgumentError(
            "family_support: argument `$name` needs a concrete Stan type."))
        scope[name] = remake(typed; qual=center_type(typed) <: types.int ? :data : :parameter)
    end
    call = CanonicalExpr(f, actuals...)
    rhs = _stan_expr(call, context)
    rt = type(rhs)
    support = autokwargs(call)
    support isa NamedTuple || throw(ArgumentError("family_support: autokwargs must return a NamedTuple."))
    # Many native families infer their variate from their RNG, rather than an
    # @lhs signature. Unlike autotype's permissive model fallback, an unknown
    # result must not become a scalar-real certificate here.
    if center_type(rt) === types.anything && !haskey(support, :type)
        rt = type(_stan_expr(CanonicalExpr(rng_expr(f), actuals...), context))
    end
    ct = _unwrap_type_kwarg(get(support, :type, center_type(rt)))
    if ct === types.anything
        sized = _sized_predictive_type(StanType(types.real), rhs)
        if sized !== nothing
            rt = sized
            ct = center_type(rt)
        end
    end
    ct === types.anything && throw(ArgumentError(
        "family_support: no registered observation or RNG type matches $f(" *
        _return_type_signature(actuals) * "). Register a typed @lhs or RNG signature."))
    rt = _broadcast_fresh_sample_type(autotype(rt; support...), rhs)
    any(_unknown_size, stan_size(rt)) && throw(ArgumentError(
        "family_support: the observation shape of $f is unresolved for this signature."))
    (; kind, type=_descriptor_type(rt), size=map(_support_expression, stan_size(rt)),
       support=(; (key => _support_expression(info(rt)[key]) for key in BOUND_KEYS
           if haskey(info(rt), key) && !_unbounded_support(key, info(rt)[key]))...))
end

_support_expression(x) = x
_support_expression(x::StanExpr) = _support_expression(expr(x))
_support_expression(x::CanonicalExpr) = sprint(io -> show(StanIO(io), x))
_support_expression(x::AbstractArray) = map(_support_expression, x)
_unbounded_support(key, x) = x isa Real && x == (key === :lower ? -Inf : Inf)
_unbounded_support(key, x::StanExpr) = _unbounded_support(key, expr(x))
