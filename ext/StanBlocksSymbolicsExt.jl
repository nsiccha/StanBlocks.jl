module StanBlocksSymbolicsExt

# `ode_rk45(f, jacobian(f), …)`: the analytic Jacobian UDF of an ODE right-hand
# side, derived with Symbolics.jl from `f`'s traced Stan definition.
#
# 1. `f`'s traced body is evaluated on symbolic values: one variable per state
#    and per real scalar of the trailing arguments, at the sizes of the solver
#    call (`ODEJacobian.sizes`).
# 2. `Symbolics.jacobian` differentiates the returned state derivative with
#    respect to those variables, in Stan's column order `[df/dy | df/dargs]`.
# 3. The nonzero entries, after common-subexpression elimination, become the
#    body of an ordinary SLIC function, which StanBlocks traces like any other.
#
# Stan builtins with a Julia counterpart that has a DiffRules rule stay opaque
# (`inv_logit`, `log1p_exp`, …), so the emitted Jacobian calls the same
# numerically careful Stan functions instead of their textbook formulas.

using StanBlocks
import Symbolics
using OrderedCollections: OrderedDict

const SB = StanBlocks
const SU = Symbolics.SymbolicUtils
const DR = Symbolics.DiffRules
const SF = StanBlocks.SpecialFunctions
const builtin = StanBlocks.builtin
const types = StanBlocks.types

struct JacobianRefusal <: Exception
    msg::String
end
refuse(args...) = throw(JacobianRefusal(string(args...)))
Base.showerror(io::IO, e::JacobianRefusal) = print(io, e.msg)

# ── Opaque Stan builtins with DiffRules derivatives ─────────────────────────
#
# Each entry is a Stan builtin and the Julia function DiffRules knows by
# `(module, name)`. Its symbolic stand-in `_sym_<name>` is registered with
# Symbolics, and its derivative is DiffRules' rule with every Julia function
# that is itself one of these builtins spelled as that builtin's stand-in.
const _OPAQUE = [
    # Stan builtin        DiffRules module   function       arity
    (:inv_logit,          :LogExpFunctions, :logistic,      1),
    (:log_inv_logit,      :LogExpFunctions, :loglogistic,   1),
    (:log1m_inv_logit,    :LogExpFunctions, :log1mlogistic, 1),
    (:logit,              :LogExpFunctions, :logit,         1),
    (:log1p_exp,          :LogExpFunctions, :log1pexp,      1),
    (:log1m_exp,          :LogExpFunctions, :log1mexp,      1),
    (:lgamma,             :SpecialFunctions, :loggamma,     1),
    (:digamma,            :SpecialFunctions, :digamma,      1),
    (:erf,                :SpecialFunctions, :erf,          1),
    (:erfc,               :SpecialFunctions, :erfc,         1),
    (:tgamma,             :SpecialFunctions, :gamma,        1),
]
# Stan builtin → its symbolic stand-in, and back for the emitted body.
const _STAND_IN = Dict{Function,Function}()
const _STAN_OF = Dict{Function,Function}()
# DiffRules' spelling → the stand-in it is rewritten to inside a rule.
const _RULE_NAME = Dict{Tuple{Symbol,Symbol},Symbol}()
filter!(e -> isdefined(builtin, e[1]), _OPAQUE)
for (stan, mod, fun, _) in _OPAQUE
    _RULE_NAME[(mod, fun)] = Symbol(:_sym_, stan)
end
_rewrite_rule(x) = x
_rewrite_rule(x::Expr) = begin
    if Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[1] isa Symbol
        key = (x.args[1], x.args[2].value)
        haskey(_RULE_NAME, key) && return _RULE_NAME[key]
        x.args[1] === :SpecialFunctions && return :(SF.$(x.args[2].value))
    end
    Expr(x.head, map(_rewrite_rule, x.args)...)
end
for (stan, mod, fun, arity) in _OPAQUE
    sym = Symbol(:_sym_, stan)
    argnames = [Symbol(:x, i) for i in 1:arity]
    @eval begin
        function $sym end
        # Plain numbers take StanBlocks' Julia twin of the builtin.
        $sym($(argnames...)) = SB.jcall(Val($(QuoteNode(stan))), SB, $(argnames...))
        Symbolics.@register_symbolic $sym($(argnames...))
        _STAND_IN[builtin.$stan] = $sym
        _STAN_OF[$sym] = builtin.$stan
    end
    for i in 1:arity
        rule = arity == 1 ? DR.diffrule(mod, fun, :(args[1])) :
            DR.diffrule(mod, fun, ntuple(k -> :(args[$k]), arity)...)[i]
        @eval Symbolics.derivative_rule(::typeof($sym), ::Val{$arity},
                args::SU.ROArgsT{Symbolics.VartypeT}, ::Val{$i}) =
            Symbolics.SConst($(_rewrite_rule(rule)))
    end
end

# Stan builtins that are exactly a Julia expression Symbolics differentiates
# itself. The derivative then reads in those Julia functions, which the
# emitted body maps back to Stan.
const _EXPANDED = Dict{Function,Function}()
for (name, g) in (
    :square => x -> x^2,
    :log1m => x -> log1p(-x),
    :inv_square => x -> inv(x^2),
    :inv_sqrt => x -> inv(sqrt(x)),
    :fma => (x, y, z) -> x * y + z,
    :fmin => min,
    :fmax => max,
    :Phi => x -> SF.erfc(-x / sqrt(2)) / 2,
    :inv => inv, :exp2 => exp2, :log2 => log2, :log10 => log10, :cbrt => cbrt,
    :hypot => hypot, :fabs => abs, :pow => ^,
)
    isdefined(builtin, name) && (_EXPANDED[getfield(builtin, name)] = g)
end

# ── Symbolic values for the arguments ───────────────────────────────────────

struct Unset end
const UNSET = Unset()

# A fresh symbolic scalar and the SLIC expression it stands for.
mutable struct Vars
    k::Int
    ast::Dict{Symbol,Any}
end
Vars() = Vars(0, Dict{Symbol,Any}())
fresh!(v::Vars, ast) = begin
    v.k += 1
    name = Symbol(:sbjac_, v.k)
    v.ast[name] = ast
    Symbolics.variable(name)
end

# The array dimensions a value of type `t` has before its Stan container.
_array_ndim(t) = SB.stan_ndim(t) - SB.r_ndim(SB.center_type(t))

# Symbolic value of an argument `name` of type `t` at literal `sizes`, built
# in Stan's column order: arrays element by element, then each element's own
# entries (matrices column-major). Integers are concrete when the call passes
# a literal, and otherwise symbolic integers, which serve arithmetic but not
# indexing or loop bounds.
function symbolic_value(vars::Vars, name, t, sizes, literal; reals=Any[])
    ct = SB.center_type(t)
    adims = _array_ndim(t)
    isint = ct <: types.int
    elem(path) = begin
        ast = isempty(path) ? name : foldl((a, i) -> Expr(:ref, a, i...), path; init=name)
        isint ? Symbolics.variable(Symbol(:sbint_, vars.k += 1); T=Int) :
            (x = fresh!(vars, ast); push!(reals, x); x)
    end
    isint && adims == 0 && SB.stan_ndim(t) == 0 && literal isa Integer && return literal
    build(path, dims) = if length(path) < adims
        n = dims[length(path) + 1]
        Any[build([path..., (i,)], dims) for i in 1:n]
    else
        rest = dims[(adims + 1):end]
        if isempty(rest)
            elem(path)
        elseif length(rest) == 1
            Any[elem([path..., (i,)]) for i in 1:rest[1]]
        else
            m, n = rest
            M = Matrix{Any}(undef, m, n)
            for j in 1:n, i in 1:m
                M[i, j] = elem([path..., (i, j)])
            end
            M
        end
    end
    build(Tuple[], sizes)
end

# ── Evaluating a traced Stan function body on symbolic values ───────────────

struct Returned
    value
end

mutable struct Env
    vals::Dict{Symbol,Any}
    rhs::Any              # the function being differentiated, for messages
    context::Any
end

lookup(env::Env, s::Symbol) = haskey(env.vals, s) ? env.vals[s] :
    refuse("`", env.rhs, "` reads `", s, "`, which is neither one of its arguments nor a local it assigned.")

_copy(x::AbstractArray) = (y = similar(x, Any); map!(_copy, y, x))
_copy(x) = x

ev(x::LineNumberNode, env) = nothing
ev(x::Nothing, env) = nothing
ev(x::Number, env) = x
ev(x::Symbol, env) = lookup(env, x)
ev(x::AbstractVector, env) = begin
    for s in x
        r = ev(s, env)
        r isa Returned && return r
    end
    nothing
end
# The size preamble of a function with sized arguments: `int n = dims(y)[1];`
# binds a size, and `if (dims(A)[1] != n) reject(…);` checks that another
# argument agrees with it. At the call's literal sizes the check either holds
# or would reject every evaluation of `f`.
ev(x::AbstractString, env) = begin
    m = match(r"^\s*int\s+(\w+)\s*=\s*dims\((\w+)\)\[(\d+)\];\s*$", x)
    if !isnothing(m)
        env.vals[Symbol(m[1])] = _dims(lookup(env, Symbol(m[2])))[parse(Int, m[3])]
        return nothing
    end
    m = match(r"^\s*if \(dims\((\w+)\)\[(\d+)\] != (\w+)\) reject\(", x)
    isnothing(m) && refuse("`jacobian(f)` cannot evaluate the generated statement `", x, "` of `", env.rhs, "`.")
    have = _dims(lookup(env, Symbol(m[1])))[parse(Int, m[2])]
    want = lookup(env, Symbol(m[3]))
    have == want || refuse("`jacobian(f)`: `", env.rhs, "` needs dimension ", m[2], " of `", m[1],
        "` to equal `", m[3], "` (", want, "), but the solver passes ", have, ".")
    nothing
end
_dims(x::AbstractMatrix) = size(x)
_dims(x::AbstractVector) = isempty(x) ? (0,) : (length(x), _dims(first(x))...)
_dims(x) = ()

ev(x::SB.StanExpr, env) = ev_expr(SB.expr(x), x, env)
ev_expr(e::Symbol, x, env) = lookup(env, e)
ev_expr(e::Number, x, env) = e
ev_expr(e::SB.CanonicalExpr, x, env) = ev(e, env)
ev_expr(e, x, env) = refuse("`jacobian(f)` cannot evaluate `", SB.short_expr(x), "` in `", env.rhs, "`.")

ev(x::SB.CanonicalExpr, env) = ev_head(SB.head(x), x, env)

ev_head(h::Symbol, x, env) = ev_head(Val(h), x, env)
ev_head(::Val{:block}, x, env) = ev(collect(x.args), env)
ev_head(::Val{:return}, x, env) = Returned(ev(only(x.args), env))
ev_head(::Val{:(::)}, x, env) = begin
    target = only(x.args)
    name = SB.expr(target)
    name isa Symbol || refuse("`jacobian(f)` cannot evaluate the declaration `", SB.short_expr(x), "`.")
    sizes = map(s -> concrete_int(ev(s, env), "the size of `$name`"), SB.stan_size(SB.type(target)))
    env.vals[name] = unset_value(SB.type(target), Tuple(sizes))
    nothing
end
unset_value(t, sizes) = begin
    isempty(sizes) && return UNSET
    adims = _array_ndim(t)
    if adims == 0 && length(sizes) == 2
        return fill!(Matrix{Any}(undef, sizes...), UNSET)
    end
    Any[unset_value_inner(t, sizes[2:end], adims - 1) for _ in 1:sizes[1]]
end
unset_value_inner(t, sizes, adims) = isempty(sizes) ? UNSET :
    adims <= 0 && length(sizes) == 2 ? fill!(Matrix{Any}(undef, sizes...), UNSET) :
    Any[unset_value_inner(t, sizes[2:end], adims - 1) for _ in 1:sizes[1]]

concrete_int(x::Integer, what) = Int(x)
concrete_int(x, what) = refuse(
    "`jacobian(f)` needs ", what, " as a number when the program is emitted; it is `", x, "`."
)

ev_head(::Val{:(=)}, x, env) = (assign!(env, x.args[1], ev(x.args[2], env)); nothing)
for (op, f) in ((:(+=), +), (:(-=), -), (:(*=), *), (:(/=), /))
    @eval ev_head(::Val{$(QuoteNode(op))}, x, env) =
        (assign!(env, x.args[1], $f(ev(x.args[1], env), ev(x.args[2], env))); nothing)
end
assign!(env, lhs::SB.StanExpr, value) = assign!(env, SB.expr(lhs), value)
assign!(env, lhs::Symbol, value) = (env.vals[lhs] = _copy(value); nothing)
assign!(env, lhs::SB.CanonicalExpr, value) = begin
    h = SB.head(lhs)
    if h === :(::)
        return assign!(env, only(lhs.args), value)
    elseif h === getindex
        target = lhs.args[1]
        container = ev(target, env)
        idx = map(i -> index_value(ev(i, env)), lhs.args[2:end])
        setindex!(container, _copy(value), idx...)
        return nothing
    end
    refuse("`jacobian(f)` cannot evaluate the assignment to `", SB.short_expr(lhs), "` in `", env.rhs, "`.")
end
index_value(i::Integer) = Int(i)
index_value(i::AbstractRange) = i
index_value(i::Colon) = i
index_value(i) = refuse("`jacobian(f)` needs every index as a number when the program is emitted; one is `", i, "`.")

ev_head(::Val{:for}, x, env) = begin
    spec, body = x.args
    var, range = loop_spec(spec, env)
    for i in range
        env.vals[var] = i
        r = ev(body, env)
        r isa Returned && return r
    end
    nothing
end
loop_spec(spec::SB.StanExpr, env) = loop_spec(SB.expr(spec), env)
loop_spec(spec::SB.CanonicalExpr, env) = begin
    SB.head(spec) === :(=) || refuse("`jacobian(f)` cannot evaluate the loop `", SB.short_expr(spec), "`.")
    var = spec.args[1] isa Symbol ? spec.args[1] : SB.expr(spec.args[1])
    range = ev(spec.args[2], env)
    range isa AbstractRange{<:Integer} || range isa AbstractVector{<:Integer} ||
        refuse("`jacobian(f)` needs the bounds of the loop over `", var, "` as numbers when the program is emitted; they are `", range, "`.")
    var, range
end
ev_head(::Val{:while}, x, env) = begin
    cond, body = x.args
    while concrete_bool(ev(cond, env), "a `while` condition")
        r = ev(body, env)
        r isa Returned && return r
    end
    nothing
end
concrete_bool(c::Bool, what) = c
concrete_bool(c, what) = refuse(
    "`jacobian(f)` needs ", what, " as true or false when the program is emitted; it depends on `", c, "`."
)

# `cond ? a : b`, and an `if` statement whose condition depends on the state,
# the arguments or the time: both sides are evaluated, and every value they
# leave different is the elementwise `ifelse` of the two.
ev_head(::Val{:ternary}, x, env) = begin
    c = ev(x.args[1], env)
    c isa Bool && return ev(x.args[c ? 2 : 3], env)
    branch_merge(c, ev(x.args[2], env), ev(x.args[3], env))
end
ev_head(::Val{:if}, x, env) = begin
    c = ev(x.args[1], env)
    if c isa Bool
        c && return ev(x.args[2], env)
        return length(x.args) >= 3 ? ev(x.args[3], env) : nothing
    end
    before = Dict(k => _copy(v) for (k, v) in env.vals)
    then_env = Env(Dict(k => _copy(v) for (k, v) in before), env.rhs, env.context)
    else_env = Env(Dict(k => _copy(v) for (k, v) in before), env.rhs, env.context)
    r1 = ev(x.args[2], then_env)
    r2 = length(x.args) >= 3 ? ev(x.args[3], else_env) : nothing
    if r1 isa Returned || r2 isa Returned
        r1 isa Returned && r2 isa Returned && return Returned(branch_merge(c, r1.value, r2.value))
        refuse("`jacobian(f)`: in `", env.rhs, "`, only one side of an `if` on `", c,
            "` returns. Return from both sides, or from neither.")
    end
    for k in union(keys(then_env.vals), keys(else_env.vals))
        haskey(then_env.vals, k) && haskey(else_env.vals, k) || continue
        env.vals[k] = branch_merge(c, then_env.vals[k], else_env.vals[k])
    end
    nothing
end
ev_head(::Val{:elseif}, x, env) = ev_head(Val(:if), x, env)
branch_merge(c, a::AbstractArray, b::AbstractArray) = begin
    size(a) == size(b) || refuse("`jacobian(f)`: the two sides of a condition on `", c, "` give values of different sizes.")
    map((u, v) -> branch_merge(c, u, v), a, b)
end
branch_merge(c, a::Unset, b::Unset) = UNSET
branch_merge(c, a, b) = isequal(a, b) ? a : Symbolics.ifelse(c, a, b)

ev_head(::Val{:&&}, x, env) = foldl((a, b) -> a & b, map(a -> ev(a, env), x.args))
ev_head(::Val{:||}, x, env) = foldl((a, b) -> a | b, map(a -> ev(a, env), x.args))
ev_head(::Val{:vect}, x, env) = Any[ev(a, env) for a in x.args]
ev_head(::Val{:tuple}, x, env) = Tuple(ev(a, env) for a in x.args)
ev_head(::Val{:(:)}, x, env) = Colon()
ev_head(::Val{h}, x, env) where {h} =
    refuse("`jacobian(f)` cannot evaluate the `", h, "` form `", SB.short_expr(x), "` in `", env.rhs, "`.")

# A call: Julia arithmetic and math functions apply to the symbolic values
# directly; Stan builtins go through their stand-in or exact expansion; a
# user-defined function is evaluated from its own traced definition.
ev_head(f::Function, x, env) = begin
    args = map(a -> ev(a, env), x.args)
    any(a -> a isa Unset || (a isa AbstractArray && any(e -> e isa Unset, a)), args) &&
        refuse("`jacobian(f)`: `", env.rhs, "` reads an entry of a local before assigning it, in `", SB.short_expr(x), "`.")
    apply(f, x, args, env)
end
ev_head(f, x, env) =
    refuse("`jacobian(f)` cannot evaluate the call `", SB.short_expr(x), "` in `", env.rhs, "`.")

apply(::typeof(getindex), x, args, env) = getindex(args[1], map(index_value, args[2:end])...)
# Stan applies its math functions elementwise to a container; Julia's methods
# of the same names do not (`exp` of a matrix is the matrix exponential).
const _ELEMENTWISE = Set{Any}([exp, log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh,
    tanh, asinh, acosh, atanh, log1p, expm1, abs, inv, cbrt, log2, log10, exp2])
elementwise(g, args) = any(a -> a isa AbstractArray, args) ? broadcast(g, args...) : g(args...)
# Containers of plain and symbolic numbers become `Num` arrays, so Julia's
# linear algebra applies (`zero(Any)` does not exist).
narrow(x::AbstractArray) = all(e -> e isa Number, x) ? Symbolics.Num.(x) : map(narrow, x)
narrow(x) = x
apply(f::Function, x, args, env) = begin
    haskey(_STAND_IN, f) && return elementwise(_STAND_IN[f], args)
    haskey(_EXPANDED, f) && return elementwise(_EXPANDED[f], args)
    def = SB._fundef(x, env.context)
    isnothing(def) || return evaluate_function(def, args, env.context)
    parentmodule(f) === builtin && return builtin_twin(f, x, args, env)
    args = map(narrow, args)
    f in _ELEMENTWISE && length(args) == 1 && return elementwise(f, args)
    try
        f(args...)
    catch err
        err isa JacobianRefusal && rethrow()
        refuse("`jacobian(f)` cannot apply `", f, "` to symbolic values in `", SB.short_expr(x),
            "` (", sprint(showerror, err), ").")
    end
end
# Any other builtin: its Julia twin, if that is plain arithmetic on these
# values (`rep_vector`, `rows`, `dot_product`, …).
builtin_twin(f, x, args, env) = try
    SB.jcall(Val(nameof(f)), SB, map(narrow, args)...)
catch err
    refuse("`jacobian(f)` has no derivative for the Stan builtin `", nameof(f), "`, called in `",
        SB.short_expr(x), "` (", sprint(showerror, err), ").")
end

# Evaluate a traced Stan function definition at symbolic argument values.
evaluate_function(def::SB.StanFunction3, args, context) = begin
    names = collect(keys(def.args))
    length(names) == length(args) || refuse("`jacobian(f)`: `", def.parent, "` takes ", length(names),
        " arguments, called with ", length(args), ".")
    env = Env(Dict{Symbol,Any}(zip(names, map(_copy, args))), def.parent, context)
    r = ev(def.body, env)
    r isa Returned || refuse("`jacobian(f)`: `", def.parent, "` returns no value.")
    r.value
end

# ── Assembling the Jacobian function ────────────────────────────────────────

function SB._symbolic_ode_jacobian(J::SB.ODEJacobian, rhs::SB.StanFunction3, x::SB.CanonicalExpr, context)
    try
        _derive(J, rhs, x, context)
    catch err
        err isa JacobianRefusal || rethrow()
        error(err.msg)
    end
end

function _derive(J, rhs, x, context)
    names = collect(keys(rhs.args))
    callargs = x.args
    N, argsizes = J.sizes[1], J.sizes[2:end]
    length(names) == length(callargs) == 2 + length(argsizes) || refuse(
        "`jacobian(f)`: `", rhs.parent, "` takes ", length(names), " arguments, the solver passes ", length(callargs), ".")
    vars = Vars()
    t = fresh!(vars, names[1])
    yreals = Any[]
    y = symbolic_value(vars, names[2], SB.type(callargs[2]), (N,), nothing; reals=yreals)
    argreals = Any[]
    argvals = [
        symbolic_value(vars, names[2 + i], SB.type(callargs[2 + i]), argsizes[i], SB.expr(callargs[2 + i]); reals=argreals)
        for i in eachindex(argsizes)
    ]
    out = evaluate_function(rhs, Any[t, y, argvals...], context)
    out isa AbstractVector && length(out) == N || refuse(
        "`jacobian(f)`: `", rhs.parent, "` returns `", out, "`, not a vector of the ", N, " state derivatives.")
    any(o -> o isa Unset, out) && refuse(
        "`jacobian(f)`: `", rhs.parent, "` leaves an entry of its returned vector unassigned.")
    wrt = Any[yreals..., argreals...]
    jac = Symbolics.jacobian(Symbolics.Num[Symbolics.Num(o) for o in out], Symbolics.Num[w for w in wrt])
    body = jacobian_body(jac, vars, Set(names))
    info = OrderedDict{Symbol,Any}(pairs(rhs.args))
    SB._attach_trace_context!(info, context)
    info[:__mod__] = StanBlocks
    traced = SB.forward!(SB.canonical(SB.ensure_xreturn(body)); info)
    rhs_name = join(vcat(SB.func_name(J.f)), "_")
    SB.StanFunction3(
        SB.maybedoc(string("The Jacobian of `", rhs_name, "` with respect to its state and to the ",
            "real scalars of its trailing arguments, derived symbolically by StanBlocks (`jacobian(",
            rhs_name, ")`).")),
        SB.StanType(types.matrix, (SB.stan_expr(N), SB.stan_expr(size(jac, 2)))),
        J, rhs.args, [traced],
    )
end

# The SLIC body returning `jac`: locals for the shared subexpressions, a zero
# matrix, and one assignment per structurally nonzero entry.
function jacobian_body(jac, vars::Vars, taken)
    entries = [(i, j, Symbolics.unwrap(jac[i, j])) for j in axes(jac, 2), i in axes(jac, 1)
               if !SU._iszero(Symbolics.unwrap(jac[i, j]))]
    for (_, _, e) in entries
        check_differentiated(e)
    end
    state = SU.Code.CSEState()
    reduced = SU.Code.cse!(Any[e for (_, _, e) in entries], state)
    assigns = [(a.lhs, a.rhs) for a in state.sorted_exprs]
    pick(base) = base in taken ? (k = 1; while Symbol(base, :_, k += 1) in taken end; Symbol(base, :_, k)) : base
    out = pick(:jac)
    lhs_names = Dict{Symbol,Symbol}()
    stmts = Any[]
    exprs = [(Symbol(SU.Code.toexpr(lhs)), stan_ast(SU.Code.toexpr(rhs), vars)) for (lhs, rhs) in assigns]
    uses = Dict{Symbol,Int}()
    count_uses!(uses, ex) = ex isa Symbol ? (uses[ex] = get(uses, ex, 0) + 1) :
        ex isa Expr ? foreach(a -> count_uses!(uses, a), ex.args) : nothing
    final = [stan_ast(SU.Code.toexpr(e), vars) for e in reduced]
    foreach(((_, e),) -> count_uses!(uses, e), exprs)
    foreach(e -> count_uses!(uses, e), final)
    # A temporary used once, or naming a single variable or number, is inlined.
    inline = Dict{Symbol,Any}()
    subst(ex) = ex isa Symbol ? get(inline, ex, get(lhs_names, ex, ex)) :
        ex isa Expr ? Expr(ex.head, map(subst, ex.args)...) : ex
    k = 0
    for (lhs, rhs) in exprs
        rhs = subst(rhs)
        if get(uses, lhs, 0) <= 1 || cheap(rhs)
            inline[lhs] = rhs
        else
            name = pick(Symbol(:jac_t, k += 1))
            push!(taken, name)
            lhs_names[lhs] = name
            push!(stmts, :($name = $(tidy(rhs))))
        end
    end
    m, n = size(jac)
    push!(stmts, :($out = $(builtin.rep_matrix)(0.0, $m, $n)))
    for ((i, j, _), e) in zip(entries, final)
        push!(stmts, :($out[$i, $j] = $(tidy(subst(e)))))
    end
    push!(stmts, out)
    Expr(:block, stmts...)
end

# A value as cheap to recompute as to read: a variable, a number, an entry,
# or the negation of one.
cheap(x) = !(x isa Expr) || Meta.isexpr(x, :ref) ||
    (Meta.isexpr(x, :call, 2) && x.args[1] === (-) && cheap(x.args[2]))

# A derivative Symbolics could not take stays a `Differential`.
check_differentiated(e) = SU.iscall(e) ? begin
    op = SU.operation(e)
    op isa Symbolics.Differential && refuse(
        "`jacobian(f)` has no derivative rule for `", SU.arguments(e)[1], "`.")
    foreach(check_differentiated, SU.arguments(e))
end : nothing

# ── From Symbolics' Julia expressions to SLIC ───────────────────────────────

const _STAN_CALLABLE = Set{Any}([+, -, *, /, ^, exp, log, sqrt, sin, cos, tan, asin, acos,
    atan, sinh, cosh, tanh, asinh, acosh, atanh, log1p, expm1, abs, min, max, inv, cbrt,
    hypot, log2, log10, exp2, <, <=, >, >=, ==, !=, !, &, |, SF.erfc, SF.erf, SF.digamma,
    SF.trigamma, SF.loggamma, SF.gamma])
const _IRRATIONAL = Dict{Symbol,Float64}(:invsqrtπ => 1 / sqrt(π), :sqrtπ => sqrt(π),
    :invπ => 1 / π, :sqrt2 => sqrt(2), :invsqrt2 => 1 / sqrt(2), :logπ => log(π), :log2π => log(2π))

stan_ast(x::Symbol, vars) = haskey(vars.ast, x) ? vars.ast[x] :
    haskey(_IRRATIONAL, x) ? _IRRATIONAL[x] : x
stan_ast(x::Rational, vars) = Float64(x)
stan_ast(x::Irrational, vars) = Float64(x)
stan_ast(x::Bool, vars) = x
stan_ast(x::Integer, vars) = x
stan_ast(x::AbstractFloat, vars) = Float64(x)
stan_ast(x, vars) = x
stan_ast(x::Expr, vars) = begin
    args = map(a -> stan_ast(a, vars), x.args)
    x.head === :call || return Expr(x.head, args...)
    f = args[1]
    if f === Base.literal_pow
        return Expr(:call, ^, args[3], literal_val(args[4]))
    elseif f === ifelse || f === Symbolics.ifelse || f === SU.ifelse_branching
        return Expr(:if, args[2], args[3], args[4])
    elseif f === signbit
        return Expr(:call, <, args[2], 0)
    elseif f === sign
        return Expr(:if, Expr(:call, >, args[2], 0), 1.0,
            Expr(:if, Expr(:call, <, args[2], 0), -1.0, 0.0))
    elseif f === sech || f === csch || f === sec || f === csc || f === cot || f === coth
        g = Dict(sech => cosh, csch => sinh, sec => cos, csc => sin, cot => tan, coth => tanh)[f]
        return Expr(:call, inv, Expr(:call, g, args[2]))
    elseif f === abs2
        return Expr(:call, ^, args[2], 2)
    elseif haskey(_STAN_OF, f)
        return Expr(:call, _STAN_OF[f], args[2:end]...)
    elseif f === (*) && length(args) == 3 && isequal(args[2], -1)
        return Expr(:call, -, args[3])
    elseif (f === (+) || f === (*)) && length(args) > 3
        return foldl((a, b) -> Expr(:call, f, a, b), args[2:end])
    elseif f isa Function && !(f in _STAN_CALLABLE)
        refuse("`jacobian(f)`: the derivative uses `", f, "`, which StanBlocks cannot emit as Stan.")
    end
    Expr(:call, args...)
end
# Read `a + (-b)` as `a - b` once temporaries are inlined.
_negation(x) = Meta.isexpr(x, :call, 2) && x.args[1] === (-)
tidy(x) = x
tidy(x::Expr) = begin
    args = map(tidy, x.args)
    if Meta.isexpr(x, :call, 3) && args[1] === (+)
        _negation(args[3]) && return Expr(:call, -, args[2], args[3].args[2])
        _negation(args[2]) && return Expr(:call, -, args[3], args[2].args[2])
    end
    Expr(x.head, args...)
end

literal_val(::Val{p}) where {p} = p
literal_val(x::Expr) = Meta.isexpr(x, :call) && x.args[1] in (Val, :Val) ? x.args[2] : x
literal_val(x) = x

end
