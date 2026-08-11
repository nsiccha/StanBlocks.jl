const RV_NAME = gensym("RV")
"""
The AST and the data, pre-tracing. Can be instantiated via `stan_instantiate`.

The `mod` field stores the defining module (set automatically by `@slic`), used for
symbol resolution during tracing — functions defined via `@deffun` in package extensions
are found by checking `mod` before falling back to `Main`.

**Warning:**

Repeatedly instantiating `SlicModel`s is inefficient, as the tracing is redone for every instantiation.
Instead, get the `StanModel` first (via `model = stan_model(slic_model)`) and update its data (via `new_model = model(;x=new_x)`).
"""
struct SlicModel#{M,D}
    model#::M
    data#::D
    mod::Module
end
SlicModel(model, data) = SlicModel(model, data, Main)
"The inferred Stan model, post-tracing. Can be instantiated via `stan_instantiate`."
struct StanModel#{M,V,B}
    meta#::M
    vars#::V
    blocks#::B
end
struct SubModel#{P,N,L}
    parent#::P
    name#::N
    locals#::L
end

# All mutable compiler scratch belongs to one explicit transpilation. Keeping it
# on the model / UDF `info` graph makes nested tracing and re-entrancy visible in
# the call graph; no state is inherited implicitly from the current Julia Task.
mutable struct TraceContext
    inline_counter::Int
    closure_counter::Int
    anon_counter::Int
    expr_stack::Vector{Any}
    current_lnn::Base.RefValue{Any}
    inline_pending::Union{Nothing,Vector{Any}}
    ragged_density_targets::Tuple
    plate_context::Any
end
TraceContext() = TraceContext(0, 0, 0, Any[], Ref{Any}(nothing), Any[], (), nothing)
_context_or_new(context) = context === nothing ? TraceContext() : context
"""
A named sub-model function, produced by `@slic f(args...) = body`. The singleton
`SubmodelFn{:f}()` is bound to `f`; each `@slic f(...) = ...` adds a call method
`(::SubmodelFn{:f})(args...; kwargs...) = SlicModel(body, data, mod)` that binds the
positional args by name into the sub-model's data. Multiple definitions of the same
`f` add methods → native multiple-dispatch. A call to a `SubmodelFn` is embedded as a
sub-model by `stan_expr(::CanonicalExpr{<:SubmodelFn})` (the value it returns is a
`SlicModel`, which flows through the existing embedding path). Contrast the anonymous
`SlicModel` value built by `@slic begin ... end`.
"""
struct SubmodelFn{name} end
abstract type AbstractStanType end
struct StanExpr{E,T<:AbstractStanType}
    expr::E
    type::T
end
struct StringStanType <: AbstractStanType
    val::AbstractString
end
Base.show(io::IO, x::StringStanType) = print(io, x.val)
struct StanType{T,S} <: AbstractStanType
    size::NTuple{S,StanExpr}
    info#::I
    StanType(T,size=tuple(), info=(;); kwargs...) = new{T,length(size)}(size, merge(info, kwargs))
end
StanExpr2{T,S,E} = StanExpr{E,StanType{T,S}}
struct StanBlock{N}
    content#::C
    StanBlock(N,content=[]) = new{N}(content)
end

struct CanonicalExpr{H,A}
    head::H
    args::A
    kwargs#::K
    # Maybe a bit dangerous
    CanonicalExpr(head::Symbol, args...; kwargs...) = CanonicalExpr(Val(head), args...; kwargs...)
    CanonicalExpr(head, args...; kwargs...) = canonical(new{typeof(head),typeof(args)}(head, args, (;kwargs...)))
    CanonicalExpr(head::Val{:block}, args...; kwargs...) = new{Val{:block},typeof(collect(args))}(head, collect(args), (;kwargs...))
    # CanonicalExpr(head::Val{:tuple})
end
CanonicalExprV{H,A} = CanonicalExpr{Val{H},A}
BlockExpr{A} = CanonicalExprV{:block,A}
AssignmentExpr{L,R} = CanonicalExprV{:(=),Tuple{L,R}}
SamplingExpr{L,R} = CanonicalExprV{:(~),Tuple{L,R}} 
Colon2Expr{L,T} = CanonicalExpr{Colon,T} 
ReturnExpr{V} = CanonicalExprV{:return,Tuple{V}} 
DocumentExpr{L,R} = CanonicalExprV{:document,Tuple{L,R}} 
QuoteExpr{T} = CanonicalExprV{:quote,T} 
TupleExpr{T} = CanonicalExprV{:tuple,T} 
KwExpr{T} = CanonicalExprV{:kw,T} 
NamedTupleExpr{T} = CanonicalExprV{:nt,T} 
GetPropertyExpr{T} = CanonicalExprV{:.,T} 
BracesExpr{T} = CanonicalExprV{:braces,T} 
VectExpr{T} = CanonicalExprV{:vect,T} 
DeclExpr{T} = CanonicalExprV{:(::),T} 
ComprehensionExpr{T} = CanonicalExprV{:comprehension,T}
GeneratorExpr{T} = CanonicalExprV{:generator,T}
FilterExpr{T} = CanonicalExprV{:filter,T}
ForExpr{T} = CanonicalExprV{:for,T}
WhileExpr{T} = CanonicalExprV{:while,T}
ColonExpr{T} = CanonicalExprV{:(:),T}
IfExpr{T} = CanonicalExprV{:if,T}
ElseIfExpr{T} = CanonicalExprV{:elseif,T}
# A ternary conditional EXPRESSION `cond ? a : b` — distinct head from the
# `if`-STATEMENT (`:if`). Julia parses both to `Expr(:if, …)`, but a ternary's
# branches are bare VALUES while a statement's are `:block`s; `canonical` splits
# them so the value form lowers to Stan's `cond ? a : b` operator (a real result
# type from the branches) instead of being (mis)handled as block-bodied control
# flow. See `canonical(::Expr)` (tracing.jl), `tracetype`/`forward!`/`show`.
TernaryExpr{T} = CanonicalExprV{:ternary,T}
BreakExpr{T} = CanonicalExprV{:break,T}
ContinueExpr{T} = CanonicalExprV{:continue,T}
StringExpr{T} = CanonicalExprV{:string,T}
SplatExpr{T} = CanonicalExprV{:...,T}


model(x::SlicModel) = x.model
data(x::SlicModel) = x.data
meta(x::StanModel) = x.meta
_trace_context(x::StanModel) = get(x.meta, :_trace_context, nothing)
_trace_context(x::SubModel) = _trace_context(parent(x))
_trace_context(x::AbstractDict) = get(x, :__trace_context__, nothing)
_trace_context(x::NamedTuple) = get(x, :__trace_context__, nothing)
_trace_context(_) = nothing
_attach_trace_context!(info::AbstractDict, context) = begin
    info[:__trace_context__] = _context_or_new(context)
    info
end
_with_trace_state(body::Function, info, field::Symbol, value) = begin
    context = _trace_context(info)
    context === nothing && return body()
    old = getfield(context, field)
    setfield!(context, field, value)
    try
        body()
    finally
        setfield!(context, field, old)
    end
end
_next_trace_id!(context::TraceContext, field::Symbol) = begin
    id = getfield(context, field) + 1
    setfield!(context, field, id)
    id
end
_next_trace_id!(info, field::Symbol) = begin
    context = _trace_context(info)
    context === nothing && error("internal: trace state is missing while allocating `$field`.")
    _next_trace_id!(context, field)
end
_expr_stack(x) = (context = _trace_context(x); context === nothing ? nothing : context.expr_stack)
_expr_stack(x::SubModel) = _expr_stack(parent(x))
_current_lnn(x) = (context = _trace_context(x); context === nothing ? nothing : context.current_lnn)
vars(x::StanModel) = x.vars
blocks(x::StanModel) = x.blocks
remake(x::StanModel; kwargs...) = StanModel((;x.meta..., kwargs...), x.vars, x.blocks)
block(x::StanModel, name) = blocks(x)[name]
Base.getindex(x::StanModel, name) = getindex(vars(x), name)
Base.setindex!(x::StanModel, value, name) = setindex!(vars(x), value, name)
Base.keys(x::StanModel) = keys(vars(x))
# Drop a name from model scope. `forward!(::ForExpr)`/`backward!(::ForExpr)` add a
# loop index then `pop!` it once the body is traced; this was only reachable with a
# plain-Dict UDF scope before compiler-injected `for`s (Feature-1 ragged-simplex,
# plating) landed loops in the top-level model body, where `info` is a StanModel.
Base.pop!(x::StanModel, name) = pop!(vars(x), name)
Base.parent(x::SubModel) = x.parent
name(x::SubModel) = x.name
locals(x::SubModel) = x.locals
remake(x::SubModel; kwargs...) = SubModel(remake(parent(x); kwargs...), name(x), locals(x))
Base.getindex(x::SubModel, name) = getindex(locals(x), name)
Base.setindex!(x::SubModel, value, name) = begin
    setindex!(parent(x), supvalue(x, value), supname(x, name))
    setindex!(locals(x), getindex(parent(x), supname(x, name)), name)
end
# A symbol-valued local is represented by its FLATTENED name in the parent
# model (`theta_x`, not `x`).  Replacing its type metadata later (qualifier,
# declaration role, likelihood reachability) must keep that one prefix rather
# than feeding the already-prefixed symbol through `supvalue` a second time.
Base.setindex!(x::SubModel, value::StanExpr{Symbol}, local_name) = begin
    parent_name = supname(x, local_name)
    setindex!(parent(x), StanExpr(parent_name, type(value)), parent_name)
    setindex!(locals(x), getindex(parent(x), parent_name), local_name)
end
Base.keys(x::SubModel) = keys(locals(x))
# Compiler-owned loops temporarily register their index in whichever scope is
# tracing the body.  Remove both views when that scope is a flattened submodel.
Base.pop!(x::SubModel, local_name) = begin
    value = pop!(locals(x), local_name)
    pop!(parent(x), supname(x, local_name))
    value
end
supname(x::SubModel, post) = Symbol(name(x), "_", post)
supvalue(x::SubModel, value) = value
supvalue(x::SubModel, value::StanExpr{Symbol}) = StanExpr(supname(x, expr(value)), type(value))
expr(x::StanExpr) = x.expr
type(x::StanExpr) = x.type
type(x::Function) = StanType(types.func{typeof(x)}; qual=:data)
remake(x::StanExpr; kwargs...) = StanExpr(expr(x), remake(type(x); kwargs...))
weak_remake(x::StanExpr; kwargs...) = StanExpr(expr(x), weak_remake(type(x); kwargs...))
center_type(x::StanExpr) = center_type(type(x))
center_type(::StanType{T}) where {T} = T
stan_size(x::StanExpr) = stan_size(type(x))
stan_size(x::StanType) = x.size
stan_size(x, i) = stan_size(x)[i]
stan_ndim(x) = length(stan_size(x))
info(x::StanType) = x.info
remake(x::StanType, args...; kwargs...) = StanType(center_type(x), args, info(x); kwargs...)
remake(x::StanType; kwargs...) = StanType(center_type(x), stan_size(x), info(x); kwargs...)
weak_remake(x::StanType; kwargs...) = StanType(center_type(x), stan_size(x), info(x); kwargs..., info(x)...)
name(::StanBlock{N}) where {N} = replace(string(N), "_"=>" ")
content(x::StanBlock) = x.content

FunctionsBlock = StanBlock{:functions}
DataBlock = StanBlock{:data}
TransformedDataBlock = StanBlock{:transformed_data}
ParametersBlock = StanBlock{:parameters}
TransformedParametersBlock = StanBlock{:transformed_parameters}
ModelBlock = StanBlock{:model}
GeneratedQuantitiesBlock = StanBlock{:generated_quantities}
remake(x::StanBlock{N}, c) where {N} = StanBlock(N, c)

head(x::CanonicalExpr) = x.head
head(::CanonicalExprV{H}) where {H} = H
remake(x::CanonicalExpr, args...; kwargs...) = CanonicalExpr(head(x), args...; kwargs...)

StanModel(name=gensym("stan_model")) = StanModel(
    (;name, _trace_context=TraceContext()),
    OrderedDict(),
    (;
        functions=StanBlock(:functions,OrderedDict()),
        data=StanBlock(:data,OrderedDict()),
        transformed_data=StanBlock(:transformed_data),
        parameters=StanBlock(:parameters,OrderedDict()),
        transformed_parameters=StanBlock(:transformed_parameters),
        model=StanBlock(:model),
        generated_quantities=StanBlock(:generated_quantities)
    ),
)
replace_name(x::Expr) = replace_name(canonical(x))
replace_name(x::Union{SamplingExpr,AssignmentExpr}) = _replace_key(x.args[1])
replace_name(::ReturnExpr) = RV_NAME
replace_name(::Any) = missing
# A statement's replacement KEY is what its LHS *names*, so a typed LHS keys on
# the bare name: `beta::vector[k] ~ normal(…)` must override the base's
# `beta ~ std_normal(…)`, which is the whole point of a `Base.merge` splice
# (swap one statement's distribution, keep the rest). Keying on the whole
# `DeclExpr` made those two look like different statements — the override was
# appended instead of replacing, and the leftover key then reached `usedin`,
# which is `Symbol`-only, as a bare `MethodError`.
# Snag `slicmodel-value-8e7afcdb`, reported by BayesianRegressionModels.
_replace_key(x) = x
_replace_key(x::DeclExpr) = x.args[1]
usedin(s::Symbol) = Base.Fix1(usedin, s)
usedin(s::Symbol, x::Expr) = any(usedin(s), x.args)
usedin(s::Symbol, x::Symbol) = s == x
usedin(s::Symbol, x::CanonicalExpr) = any(usedin(s), x.args)
usedin(s::Symbol, x) = false
# Where a statement's LHS sits in its RAW AST: `x ~ rhs` parses to
# `Expr(:call, :~, lhs, rhs)`, `x = rhs` to `Expr(:(=), lhs, rhs)`. `0` means
# "not a shape whose LHS we rewrite" (a `return`, or anything unrecognised).
_lhs_position(x::Expr) =
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ ? 2 :
    Meta.isexpr(x, :(=), 2) ? 1 : 0
_lhs_position(x) = 0
# An override's LHS declares; OMITTING a declaration means "unchanged", not
# "cleared". So when a spliced statement's LHS is a bare name and the base
# statement it replaces DECLARED that name (`rho :: vector[n_axes] ~ …`), the
# override supplies the RHS and the base supplies the declaration.
#
# Replacing the statement wholesale instead dropped the type and size, which
# does not fail — it silently RESCOPES the parameter (`rho` becomes a scalar)
# into a well-formed but different model, and the mismatch surfaces passes
# later in whatever first consumes the wrong shape, naming neither the
# parameter nor the merge. Swapping one statement's distribution while keeping
# its declaration is the whole point of the splice surface, so it must not
# require every caller to know and repeat each sub-model's declared form.
#
# An override that WANTS a different declaration still writes its own
# (`rho :: real ~ …`), so nothing becomes inexpressible.
# Snag `merge-plain-over-f228c5b2`, reported by BayesianRegressionModels.
_inherit_lhs_decl(base::Expr, override::Expr) = begin
    bi, oi = _lhs_position(base), _lhs_position(override)
    (bi == 0 || oi == 0) && return override
    Meta.isexpr(base.args[bi], :(::)) || return override
    Meta.isexpr(override.args[oi], :(::)) && return override
    out = copy(override)
    out.args[oi] = base.args[bi]
    out
end
_inherit_lhs_decl(_base, override) = override
top_replace_components(x::Expr; rep::OrderedDict) = begin
    @assert x.head == :block "top_replace_components expects a `begin ... end` block, got `$x` (head `$(x.head)`)."
    args = []
    for arg in x.args
        override = pop!(rep, replace_name(arg), nothing)
        push!(args, isnothing(override) ? arg : _inherit_lhs_decl(arg, override))
    end
    i = 1
    while i <= length(args)
        for key in keys(rep)
            usedin(key, args[i]) || continue
            insert!(args, i, pop!(rep, key))
            i -= 1
            break
        end
        i += 1
    end
    append!(args, values(rep))
    Expr(:block, args...)
end
# `Base.merge(submodel, stmts..., fixed::NamedTuple)` — model composition:
#
# - statement arguments override body statements whose LHS-name matches and
#   append the rest;
# - NamedTuple arguments FIX their names: matching sampling/assignment
#   statements are removed and the supplied values become model data.
#
# Fixed bindings are applied after all statement splices, independent of their
# argument position. This makes the mixed form unambiguous: in
# `Base.merge(model, :(x ~ prior()), (; x=value))`, the explicit value wins and
# `x ~ prior()` does not survive as a likelihood contribution.
#
# This REPLACES the old positional-call splice (`submodel(quote … end)`); a
# positional sub-model call now errors loudly (see the call operator below).
unblock(x::BlockExpr) = mapreduce(unblock, vcat, x.args; init=[])
unblock(x::Expr) = x.head === :block ? mapreduce(unblock, vcat, x.args; init=[]) : [x]
unblock(x::LineNumberNode) = []
unblock(x) = [x]
# Canonicalise ONLY to derive the replacement key and to validate the statement
# shape — splice the argument through UNCHANGED. A `@slic` body is raw Julia AST,
# so canonicalising the overrides used to leave `model(merged)` a MIXED tree, and
# `show` on a `CanonicalExpr` is the *Stan* emitter (§R8): it assumes every node
# has already been traced. Printing a spliced body to inspect it therefore died in
# the emitter (`MethodError: no method matching type(::Symbol)`, show.jl's
# `DeclExpr` method) — and making that emitter tolerate untraced nodes would hide
# a genuine compiler bug, so the body is kept raw instead. `forward!(::SlicModel)`
# canonicalises the whole body anyway, so nothing downstream loses information.
# Snag `slicmodel-value-8e7afcdb`, reported by BayesianRegressionModels.
_check_splice_stmt(raw, ::Union{SamplingExpr,AssignmentExpr,ReturnExpr}) = raw
_check_splice_stmt(raw, _canonical) = error(
    "Base.merge(submodel, …): every spliced statement must be a sampling (`x ~ …`), ",
    "an assignment (`x = …`) or a `return …`; got `", raw, "`."
)
_splice_body(x::SlicModel, args...) = begin
    rep = OrderedDict{Any,Any}()
    for raw in mapreduce(unblock, vcat, args; init=[])
        c = canonical(raw)
        _check_splice_stmt(raw, c)
        rep[replace_name(c)] = raw
    end
    top_replace_components(model(x); rep)
end
_without_fixed_components(x::Expr, fixed_names) = begin
    @assert x.head == :block "_without_fixed_components expects a `begin ... end` block, got `$x` (head `$(x.head)`)."
    Expr(:block, filter(x.args) do arg
        key = replace_name(arg)
        ismissing(key) || !(key in fixed_names)
    end...)
end
_merge_parts(args) = begin
    stmts = Any[]
    fixed = NamedTuple()
    for arg in args
        if arg isa NamedTuple
            fixed = merge(fixed, arg)
        else
            push!(stmts, arg)
        end
    end
    stmts, fixed
end
Base.merge(x::SlicModel, args...) = begin
    stmts, fixed = _merge_parts(args)
    body = _splice_body(x, stmts...)
    isempty(fixed) && return SlicModel(body, data(x), x.mod)
    body = _without_fixed_components(body, keys(fixed))
    SlicModel(body, merge(data(x), pairs(fixed)), x.mod)
end

_submodel_positional_error(args...) = error(
    "A @slic sub-model was called with positional argument(s). ",
    "Data inputs are KEYWORD arguments (`submodel(; X=X)`); statement-splice overrides ",
    "now use `Base.merge(submodel, stmts...)` (e.g. `Base.merge(base, quote … end)`); and ",
    "positional scalar inputs require a named sub-model function declared with `@slic f(args...) = …`."
)
(x::SlicModel)(; kwargs...) = SlicModel(model(x), merge(data(x), kwargs), x.mod)
(x::SlicModel)(arg, args...; kwargs...) = _submodel_positional_error(arg, args...)

qual(x) = :data
qual(x::StanExpr) = qual(type(x))
qual(x::StanType) = get(info(x), :qual, :undefined)
_is_fresh_decl(x::StanExpr) = get(info(type(x)), :fresh_decl, false)
_decl_role(x::StanExpr) = get(info(type(x)), :decl_role, :none)
lqual(x) = :undefined
lqual(x::StanExpr) = lqual(type(x))
lqual(x::StanType) = get(info(x), :lqual, :undefined) 
cqual(x) = qual(x) == :data ? :d : lqual(x) == :undefined ? :g : :p
getvalue(x::StanExpr) = getvalue(type(x))
getvalue(x::StanType) = get(info(x), :value, missing)
getvalue(x::DocumentExpr) = getvalue(x.args[2])
hasvalue(x::StanExpr) = !ismissing(getvalue(x))
hasvalue(x::StanType) = !ismissing(getvalue(x))
cv(x) = false
cv(x::StanExpr) = cv(type(x))
cv(x::StanType) = get(info(x), :cv, false) || any(cv, stan_size(x))

stan_type(expr, value; kwargs...) = error("Do not know how to handle `stan_type($expr, $value)`")
stan_type(expr, value::Integer; kwargs...) = StanType(types.int; value, kwargs..., qual=:data)
stan_type(expr, value::AbstractFloat; kwargs...) = StanType(types.real; value, kwargs...)
stan_type(expr, value::AbstractVector{<:Real}; kwargs...) = StanType(
    types.vector,
    stan_expr.((Symbol(expr, "_n"), ), size(value));
    value, kwargs...
)
stan_type(expr, value::AbstractMatrix{<:Real}; kwargs...) = StanType(
    types.matrix,
    stan_expr.((Symbol(expr, "_m"), Symbol(expr, "_n"), ), size(value));
    value, kwargs...
)
stan_type(expr, value::AbstractVector{<:Integer}; kwargs...) = StanType(
    types.int, 
    stan_expr.((Symbol(expr, "_n"), ), size(value)); 
    value, kwargs..., qual=:data
)
stan_type(expr, value::AbstractMatrix{<:Integer}; kwargs...) = StanType(
    types.int,
    stan_expr.((Symbol(expr, "_m"), Symbol(expr, "_n"), ), size(value));
    value, kwargs..., qual=:data
)
# String literals — used as messages for `reject` / `print`. The
# `value` carries the raw text; the rendered form quotes it.
stan_type(expr, value::AbstractString; kwargs...) = StanType(types.anything; value, kwargs..., qual=:data)
stan_type(expr, value::Function; kwargs...) = StanType(types.func{typeof(value)}; value, qual=:data, kwargs...)
stan_type(expr, value::Tuple; kwargs...) = StanType(
    types.tup, tuple();
    arg_types=ntuple(i->stan_type(Symbol(expr, "_", i), value[i]), length(value)), value, kwargs...
)
stan_type(expr, value::NamedTuple; kwargs...) = StanType(
    types.ntup, tuple();
    arg_types=(;[key=>stan_type(Symbol(expr, "_", key), val) for (key, val) in pairs(value)]...),
    value, kwargs...
)
"""
Encode a vector of real-valued subvectors as a ragged vector: a `NamedTuple` with
`mem::Vector` (concatenation of all subvectors) and `ends::Vector{Int}` (inclusive
1-based end indices of each subvector in `mem`).

Passing a `Vector{<:AbstractVector{<:Real}}` directly as a data kwarg to a `SlicModel`
applies this transformation automatically.
"""
to_ragged(x::AbstractVector{<:AbstractVector{T}}) where {T<:Real} = (;
    mem=reduce(vcat, x; init=T[]),
    ends=cumsum(length.(x)),
)
# `stan_type(expr, ::AbstractVector{<:AbstractVector{<:Real}})` — ragged DATA ingest —
# lives in `builtin.jl` (after `RaggedVector` is defined), minting a nominal
# `RaggedVector` so ragged data is a first-class indexable container everywhere
# (decision 2026-07-17T00-14-01-598-1g0cf6y, approach B). `to_ragged` stays here.
