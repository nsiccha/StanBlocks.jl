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
ForExpr{T} = CanonicalExprV{:for,T}
WhileExpr{T} = CanonicalExprV{:while,T}
ColonExpr{T} = CanonicalExprV{:(:),T}
IfExpr{T} = CanonicalExprV{:if,T}
ElseIfExpr{T} = CanonicalExprV{:elseif,T}
BreakExpr{T} = CanonicalExprV{:break,T}
ContinueExpr{T} = CanonicalExprV{:continue,T}
StringExpr{T} = CanonicalExprV{:string,T}
SplatExpr{T} = CanonicalExprV{:...,T}


model(x::SlicModel) = x.model
data(x::SlicModel) = x.data
meta(x::StanModel) = x.meta
_expr_stack(x::StanModel) = get(x.meta, :_expr_stack, nothing)
_expr_stack(x::SubModel) = _expr_stack(parent(x))
_expr_stack(x) = nothing
_current_lnn(x::StanModel) = get(x.meta, :_current_lnn, nothing)
_current_lnn(x::SubModel) = _current_lnn(parent(x))
_current_lnn(x) = nothing
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
Base.keys(x::SubModel) = keys(locals(x))
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
    (;name),
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
replace_name(x::Union{SamplingExpr,AssignmentExpr}) = x.args[1]
replace_name(::ReturnExpr) = RV_NAME
replace_name(::Any) = missing
usedin(s::Symbol) = Base.Fix1(usedin, s)
usedin(s::Symbol, x::Expr) = any(usedin(s), x.args)
usedin(s::Symbol, x::Symbol) = s == x
usedin(s::Symbol, x::CanonicalExpr) = any(usedin(s), x.args)
usedin(s::Symbol, x) = false
top_replace_components(x::Expr; rep::OrderedDict) = begin
    @assert x.head == :block "top_replace_components expects a `begin ... end` block, got `$x` (head `$(x.head)`)."
    args = []
    for arg in x.args
        push!(args, pop!(rep, replace_name(arg), arg))
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
model(x::SlicModel, args::Union{SamplingExpr,AssignmentExpr,ReturnExpr}...) = top_replace_components(model(x); rep=OrderedDict([
    replace_name(arg)=>arg for arg in args
]))
unblock(x::BlockExpr) = mapreduce(unblock, vcat, x.args)
unblock(x::LineNumberNode) = []
unblock(x) = [x]
model(x::SlicModel, args::Union{BlockExpr,SamplingExpr,AssignmentExpr,ReturnExpr}...) = model(x, mapreduce(unblock, vcat, args)...)
model(x::SlicModel, args::Expr...) = model(x, canonical.(args)...)
(x::SlicModel)(args...; kwargs...) = SlicModel(model(x, args...), merge(data(x), kwargs), x.mod)

qual(x) = :data
qual(x::StanExpr) = qual(type(x))
qual(x::StanType) = get(info(x), :qual, :undefined)
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
stan_type(expr, value::AbstractVector{<:AbstractVector{<:Real}}; kwargs...) = stan_type(expr, to_ragged(value); kwargs...)
