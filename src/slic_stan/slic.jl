module stan
using OrderedCollections
const RV_NAME = gensym("RV")
dumperror(x) = (dump(x); error(x))
struct SlicModel#{M,D}
    model#::M
    data#::D
end
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
# CanonicalExprT{H,A} = CanonicalExpr{typeof(H),A}
BlockExpr{A} = CanonicalExprV{:block,A} 
AssignmentExpr{L,R} = CanonicalExprV{:(=),Tuple{L,R}} 
# ReAssignmentExpr{L,R} = CanonicalExprV{:(reassign),Tuple{L,R}} 
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
IfThenExpr2{I,T<:BlockExpr} = CanonicalExprV{:if,Tuple{I,T}}
IfThenElseExpr{I,T<:BlockExpr,E<:BlockExpr} = CanonicalExprV{:if,Tuple{I,T,E}}
StringExpr{T} = CanonicalExprV{:string,T}
SplatExpr{T} = CanonicalExprV{:...,T}


model(x::SlicModel) = x.model
data(x::SlicModel) = x.data
meta(x::StanModel) = x.meta
vars(x::StanModel) = x.vars
blocks(x::StanModel) = x.blocks
var(x::StanModel, name) = error()#vars(x)[name]
block(x::StanModel, name) = blocks(x)[name]
Base.getindex(x::StanModel, name) = getindex(vars(x), name)
Base.setindex!(x::StanModel, value, name) = setindex!(vars(x), value, name)
Base.keys(x::StanModel) = keys(vars(x))
Base.parent(x::SubModel) = x.parent
name(x::SubModel) = x.name
locals(x::SubModel) = x.locals
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
replace_components(x; rep) = x
replace_components(x::Expr; rep::Dict) = if (
    x.head == :(=) && x.args[1] in keys(rep) || 
    x.head == :call && x.args[1] == :~ && x.args[2] in keys(rep)
)
    rep[x.head == :(=) ? x.args[1] : x.args[2]]
else
    Expr(x.head, replace_components.(x.args; rep)...)
end
model(x::SlicModel, args::SamplingExpr...) = replace_components(model(x); rep=Dict([
    arg.args[1]=>arg
    for arg in args
]))
model(x::SlicModel, args::Union{SamplingExpr,AssignmentExpr}...) = replace_components(model(x); rep=Dict([
    arg.args[1]=>arg
    for arg in args
]))
unblock(x::BlockExpr) = mapreduce(unblock, vcat, x.args)
unblock(x::LineNumberNode) = []
unblock(x) = [x]
model(x::SlicModel, args::Union{BlockExpr,SamplingExpr}...) = model(x, mapreduce(unblock, vcat, args)...)
model(x::SlicModel, args::Expr...) = model(x, canonical.(args)...)
(x::SlicModel)(args...; kwargs...) = SlicModel(model(x, args...), merge(data(x), kwargs))

qual(x) = :data
qual(x::StanExpr) = qual(type(x))
qual(x::StanType) = get(info(x), :qual, :undefined)
lqual(x) = :undefined
lqual(x::StanExpr) = lqual(type(x))
lqual(x::StanType) = get(info(x), :lqual, :undefined) 
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
stan_type(expr, value::AbstractVector{<:AbstractFloat}; kwargs...) = StanType(
    types.vector, 
    stan_expr.((Symbol(expr, "_n"), ), size(value)); 
    value, kwargs...
)
stan_type(expr, value::AbstractMatrix{<:AbstractFloat}; kwargs...) = StanType(
    types.matrix, 
    stan_expr.((Symbol(expr, "_m"), Symbol(expr, "_n"), ), size(value)); 
    value, kwargs...
)
stan_type(expr, value::AbstractVector{<:Integer}; kwargs...) = StanType(
    types.int, 
    stan_expr.((Symbol(expr, "_n"), ), size(value)); 
    value, kwargs..., qual=:data
)
stan_type(expr, value::Function; kwargs...) = StanType(types.func{typeof(value)}; value, qual=:data, kwargs...)
stan_call(f, args...) = stan_expr(CanonicalExpr(f, map(stan_expr, args)...))
stan_expr(x::StanExpr; kwargs...) = weak_remake(x; kwargs...)
stan_expr(x; kwargs...) = stan_expr(x, x; kwargs...)
stan_expr(expr, value; kwargs...) = StanExpr(expr, stan_type(expr, value; kwargs...))
stan_expr(x, value::StanExpr; kwargs...) = weak_remake(value; kwargs...)
maybedata(expr, value; kwargs...) = stan_expr(expr, value; qual=:data, kwargs...)
maybedata(expr, value::Function; kwargs...) = stan_expr(value, value; qual=:data, kwargs...)
maybecv(expr, value) = stan_expr(expr, value; cv=true)
stan_model(x::SlicModel; info=StanModel()) = begin 
    distribute!(backward!(forward!(x; info); info); info)
    info
end
maybedata!(x::StanModel, key, value) = x[key] = maybedata(key, value)
maybedata!(x::SubModel, key, value) = locals(x)[key] = maybedata(key, value)
forward!(x::SlicModel; info=StanModel()) = begin 
    for (key, value) in pairs(data(x))
        maybedata!(info, key, value)
    end
    forward!(canonical(model(x)); info)
end
isexpr(h) = Base.Fix2(isexpr, h)
isexpr(x, h) = false
isexpr(x::CanonicalExpr, h) = head(x) == h
canonical(x) = x
canonical(x::Expr) = CanonicalExpr(x.head, canonical.(x.args)...)
ensure_kw(x::CanonicalExprV{:kw}) = x
ensure_kw(x::Symbol) = CanonicalExpr(:kw, x, x)
ensure_kw(x::CanonicalExprV{:.}) = CanonicalExpr(:kw, kw_name(x), x)
kw_name(x::CanonicalExprV{:.}) = kw_name(x.args[2])
kw_name(x::QuoteNode) = x.value
canonical(x::CanonicalExprV{:parameters}) = if all(isexpr(:kw), x.args)
    x
else
    CanonicalExpr(x.head, ensure_kw.(x.args)...)
end
canonical(x::CanonicalExprV{:call}) = begin 
    f = x.args[1]
    args = []
    kwargs = []
    for arg in x.args[2:end]
        if isexpr(arg, :parameters)
            @assert all(isexpr(:kw), arg.args)
            for argi in arg.args
                push!(kwargs, argi.args[1]=>argi.args[2])
            end
        elseif isexpr(arg, :kw) 
            push!(kwargs, arg.args[1]=>arg.args[2])
        else
            push!(args, arg)
        end
    end
    # isa(f, StanExpr) && error()
    CanonicalExpr(f, args...; kwargs...)
end
canonical(x::CanonicalExprV{:tuple}) = begin 
    @assert length(x.args) > 0
    if any(Base.Fix2(isexpr, :parameters), x.args)
        @assert length(x.args) == 1
        @assert all(isexpr(:kw), x.args[1].args)
        CanonicalExpr(:nt, x.args[1].args...)
    else
        x
    end
end
canonical(x::CanonicalExprV{:macrocall}) = begin 
    @assert x.args[1] == GlobalRef(Core, Symbol("@doc"))
    CanonicalExpr(:document, x.args[3:4]...)
end
canonical(x::CanonicalExprV{Symbol("'")}) = CanonicalExpr(:adjoint, x.args...)
canonical(x::CanonicalExprV{:ref}) = CanonicalExpr(:getindex, x.args...)
canonical(x::CanonicalExprV{Symbol(".*")}) = CanonicalExpr(.*, x.args...)
canonical(x::CanonicalExprV{Symbol("./")}) = CanonicalExpr(./, x.args...)

forwards!(;info) = x->forwards!(x; info)
forwards!(x; info) = [forward!(x; info)]
forwards!(x::SplatExpr; info) = [forward!(x.args[1]; info)...]
forward!(x; info) = error(x)
forward!(;info) = x->forward!(x; info)
forward!(x::Tuple; info) = (mapreduce(forwards!(;info), vcat, x; init=[])...,)
forward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(forward!(;info), x)
forward!(x::Union{String,Number,LineNumberNode,Function,Nothing}; info) = x
forward!(x::QuoteNode; info) = x.value
forward!(x::Irrational; info) = error(x)
forward!(x::Irrational{:π}; info) = forward!(Float64(pi); info)
forward!(x::Number; info) = maybedata(x, x)
forward!(x::Symbol; info) = begin 
    x in keys(info) && return info[x]
    isdefined(builtin, x) && return forward!(getproperty(builtin, x); info)
    if isdefined(Main, x)
        Mx = getproperty(Main, x)
        isa(Mx, Function)  && return forward!(Mx; info)
        isa(Mx, SlicModel) && return Mx
        error("Found $x in Main, but is of type $(typeof(Mx))!")
    end
    error("Could not find $(x) in model, builtin or Main!")
end
forward!(x::Function; info) = stan_expr(x)
forward!(x::Colon; info) = x
forward!(x::StanExpr{Symbol}; info) = x
forward!(x::StanExpr; info) = x
forward!(x::CanonicalExpr; info) = begin
    stan_expr(CanonicalExpr(forward!(head(x); info), forward!(x.args; info)...; forward!(x.kwargs; info)...))
end
forward!(x::BlockExpr; info) = remake(x, forward!(x.args; info)...)
forward!(x::AssignmentExpr{Symbol}; info) = begin
    name, rhs = x.args 
    (name in keys(info) && isa(info, SubModel)) && return nothing 
    rhs = forward!(rhs; info)::Union{StanExpr}
    forward!(remake(x, name, rhs); info)
end
maybe_lazy_size(key::Symbol, i, sizei) = sizei
maybe_lazy_size(key::Symbol, i, ::StanExpr{<:CanonicalExpr}) = StanExpr("dims($key)[$i]", StanType(types.int))
forward!(x::AssignmentExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args 
    @assert name ∉ keys(info)
    info[name] = StanExpr(name, remake(type(rhs); value=missing))
    @assert center_type(rhs) != types.anything "tracetype not defined for $name = $(short_expr(rhs))!"
    rv = remake(x, info[name], rhs)
    info[name] = StanExpr(name, remake(type(rhs), [
        maybe_lazy_size(name, i, sizei)
        for (i, sizei) in enumerate(stan_size(type(rhs)))
    ]...; value=missing))
    # @info "$x \n=> $rv\n=> $(info[name])"
    rv 
end
forward!(x::AssignmentExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::SamplingExpr{Symbol}; info) = begin
    name, rhs = x.args 
    (name in keys(info) && isa(info, SubModel)) && return nothing
    rhs = forward!(rhs; info)::Union{StanExpr,SlicModel}
    forward!(remake(x, name, rhs); info)
end
forward!(x::SamplingExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args 
    if name in keys(info)
        @assert stan.qual(info[name]) == :data
        stan.cv(rhs) && (info[name] = remake(info[name]; cv=true))
    else
        # base = type(rhs)
        # @info expr(rhs).kwargs
        autotype = stan.autotype(rhs)
        cv = stan.cv(autotype) || stan.cv(rhs)
        qual = cv ? :quantities : :parameter
        info[name] = StanExpr(name, remake(autotype; qual, cv))
    end
    remake(x, info[name], rhs)
end
forward!(x::SamplingExpr{Symbol,<:SlicModel}; info) = begin
    name, rhs = x.args
    forward!(rhs; info=SubModel(info, name, Dict()))
end
forward!(x::SamplingExpr; info) = begin
    lhs, rhs = forward!(x.args; info)
    # rhs = forward!(rhs; info)::Union{StanExpr,SlicModel}
    forward!(remake(x, lhs, rhs::StanExpr); info)
end
forward!(x::SamplingExpr{<:Any,<:StanExpr}; info) = begin
    lhs, rhs = x.args 
    @assert stan.qual(lhs) == :data
    remake(x, lhs, rhs)
end
forward!(x::ReturnExpr; info) = if isa(info, SubModel)
    rhs = forward!(x.args[1]; info)
    forward!(CanonicalExpr(:(=),name(info),rhs); info=parent(info))
else
    rv = forward!(x.args[1]; info)
    info[RV_NAME] = rv
    remake(x, rv)
end
forward!(x::DocumentExpr; info) = remake(x, forward!(x.args; info)...)
forward!(x::TupleExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::KwExpr; info) = stan_expr(remake(x, x.args[1], forward!(x.args[2]; info)))
forward!(x::NamedTupleExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::GetPropertyExpr; info) = begin
    @assert length(x.args) == 2
    obj, name = forward!(x.args; info)
    names = keys(obj.type.info.arg_types)
    @assert name in names
    return forward!(CanonicalExpr(:getindex, x.args[1], findfirst(==(name), names)); info)
    error(dump((;obj, name)))
    stan_expr(remake(x, forward!(x.args; info)...))
end
forward!(x::BracesExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::VectExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::DeclExpr; info) = begin
    @assert length(x.args) == 2
    name, type = x.args
    @assert isa(name, Symbol)
    @assert isa(type, CanonicalExprV{:getindex})
    ct, s... = type.args
    @assert isa(ct, Symbol)
    ct = gettype(ct)
    t = StanType(ct, forward!.(s; info))
    info[name] = rv = StanExpr(name, t)
    stan_expr(remake(x, rv))
end
forward!(x::ForExpr; info) = begin 
    @assert length(x.args) == 2
    head, body = x.args
    @assert isa(head, CanonicalExprV{:(=)})
    @assert isa(body, CanonicalExprV{:block})
    idx = head.args[1]
    @assert isa(idx, Symbol)
    info[idx] = StanExpr(idx, StanType(types.int))
    body = forward!(body; info)
    pop!(info, idx)
    stan_expr(remake(x, head, body))
end
forward!(x::WhileExpr; info) = begin 
    @assert length(x.args) == 2
    head, body = x.args
    # @assert isa(head, CanonicalExprV{:(=)})
    @assert isa(body, CanonicalExprV{:block})
    # body = forward!(body; info)
    stan_expr(remake(x, forward!(x.args; info)...))
end
forward!(x::IfExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::ElseIfExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::BreakExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::ContinueExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::QuoteExpr; info) = x.args[1]
forward!(x::StringExpr; info) = join(map(stan_code2, forward!(x.args; info)))

stan_expr(x::CanonicalExpr) = StanExpr(x, remake(tracetype(x); qual=maximum(qual, x.args; init=:data), cv=any(cv, x.args) || cv(tracetype(x))))
stan_expr(x::CanonicalExpr{<:SlicModel}) = head(x)(x.args...;x.kwargs...)

backward!(x; info) = error(x)
backward!(;info) = x->backward!(x; info)
backward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(backward!(;info), x)
backward!(x::Union{String,Number,LineNumberNode,Symbol,Nothing,Colon}; info) = x
backward!(x::CanonicalExpr; info) = remake(x, backward!(x.args; info)...)
backward!(x::BlockExpr; info) = remake(x, reverse(backward!.(reverse(x.args); info))...)
backward!(x::AssignmentExpr; info) = if lqual(info[expr(x.args[1])]) == :affects_likelihood
    lhs, rhs = x.args
    remake(x, info[expr(x.args[1])], backward!(rhs; info))
elseif qual(x.args[1]) == :parameter
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
else
    x
end
backward!(x::SamplingExpr{<:StanExpr{Symbol}}; info) = if qual(x.args[1]) == :data || lqual(info[expr(x.args[1])]) == :affects_likelihood
    lhs, rhs = x.args
    remake(x, info[expr(x.args[1])], backward!(rhs; info))
else
    lhs, rhs = x.args
    remake(x, remake(lhs, qual=:quantities), rhs)
end
backward!(x::SamplingExpr; info) = begin 
    @assert qual(x.args[1]) == :data
    lhs, rhs = x.args
    remake(x, backward!(lhs; info), backward!(rhs; info))
end
backward!(x::ReturnExpr; info) = x
backward!(x::DocumentExpr; info) = remake(x, backward!.(x.args; info)...)
backward!(x::StanExpr; info) = StanExpr(backward!(expr(x); info), backward!(type(x); info))
backward!(x::StanExpr{Symbol}; info) = info[expr(x)] = remake(x; lqual=:affects_likelihood)
backward!(x::StanType; info) = remake(x; lqual=:affects_likelihood)

distribute!(x::BlockExpr; info) = distribute!.(x.args; info)
distribute!(x::Union{LineNumberNode,Nothing}; info) = nothing
distribute!(x::DocumentExpr{<:Any,<:BlockExpr}; info) = distribute!(x.args[2]; info)
distribute!(x; info) = for b in distribution_blocks(x; info)
    push!(block(info, b), x; info)
end
qual(x::AssignmentExpr) = qual(x.args[1])
qual(x::SamplingExpr) = qual(x.args[1])
distribution_blocks(x::AssignmentExpr; info) = if qual(x) == :data
    (:transformed_data, )
elseif qual(x) == :parameter
    (:transformed_parameters, )
else
    (:generated_quantities, )
end
distribution_blocks(x::SamplingExpr; info) = if qual(x) == :data
    if cv(x.args[1])
        (:generated_quantities,)
    else
        (:model, :generated_quantities)
    end
elseif qual(x) == :parameter
    (:parameters, :model)
else
    (:generated_quantities, )
end
distribution_blocks(x::ReturnExpr; info) = (:generated_quantities,)
distribution_blocks(x::DocumentExpr; info) = distribution_blocks(x.args[2]; info)
distribution_blocks(::Union{Nothing}; info) = tuple()
# I had removed this, I wonder why!
distribution_blocks(x::StanExpr{Symbol}; info) = hasvalue(x) ? (:data,) : tuple()

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
fetch_data!(x::Union{Tuple}; info) = map(fetch_data!(;info), x)
fetch_data!(x::Union{Function,String}; info) = nothing 
fetch_data!(x::StanExpr{<:Union{Number,String,Missing}}; info) = nothing 
fetch_data!(x::StanType; info) = fetch_data!(stan_size(x); info) 
fetch_data!(x::StanExpr{Symbol}; info) = begin
    # fetch_data!(type(x); info)
    hasvalue(x) && push!(block(info, :data), x; info)
end
fetch_data!(x::StanExpr{<:CanonicalExpr}; info) = fetch_data!((type(x), expr(x)); info)
fetch_data!(x::CanonicalExpr; info) = begin
    fetch_functions!(x; info=block(info, :functions).content)
    # fdef = fundef(x)
    # map(fundefs(x)) do fdef
    #     push!(block(info, :functions), fdef; info)
    # end
    # isnothing(fdef) || push!(block(info, :functions), fdef; info)
    fetch_data!(x.args; info)
end
fetch_data!(x::CanonicalExprV{:kw}; info) = fetch_data!(x.args[2]; info)

# fetch_data!(x::DocumentExpr; info) = fetch_data!(x.args[2]; info=remake(x, x.args[1], info))
fetch_data!(x; info) = error(x)

Base.get!(b::DocumentExpr{<:Any,<:DeclarativeBlock}, k, x) = get!(content(b.args[2]), k, remake(b, b.args[1], x))
# Base.get!(b::DocumentExpr{<:Any,<:DeclarativeBlock}, k, x) = content(b.args[2])[k] = remake(b, b.args[1], x)
Base.push!(b::DocumentExpr{<:Any,<:ImperativeBlock}, x) = push!(content(b.args[2]), remake(b, b.args[1], x))

Base.push!(b::StanBlock, x; info) = error("Block $(typeof(b)) does not know how to handle $(x)!")
# Base.push!(b::StanBlock, x::String; info) = push!()
Base.push!(b::StanBlock, x::DocumentExpr; info) = begin
    push!(remake(b, remake(x, x.args[1], b)), x.args[2]; info)
end
Base.push!(b::DeclarativeBlock, x::SamplingExpr; info) = push!(b, x.args[1]; info)
Base.push!(b::DeclarativeBlock, x::StanExpr{Symbol}; info) = begin
    fetch_data!(type(x); info)
    # @info name(b)=>expr(x)=>typeof(b)
    get!(content(b), expr(x), x)
end
Base.push!(b::ImperativeBlock, x; info) = begin 
    fetch_data!(x; info)
    push!(content(b), x)
end
Base.push!(b::ImperativeBlock, x::DocumentExpr; info) = begin
    push!(remake(b, remake(x, x.args[1], b)), x.args[2]; info)
    # push!(content(b), remake(x, x.args[1], "//"))
    # push!(b, x.args[2]; info)
    # fetch_data!(x.args[2]; info)
    # push!(content(b), x)
end
Base.push!(b::GeneratedQuantitiesBlock, x::SamplingExpr; info) = begin
    lhs, rhs = x.args
    # if hasvalue(lhs)
    if qual(lhs) == :data
        likelihood_rhs = likelihood_expr(lhs, rhs)
        push!(b, CanonicalExpr(
            :(=), 
            StanExpr(Symbol(expr(lhs), "_likelihood"), type(likelihood_rhs)), 
            likelihood_rhs
        ); info)
        lhs = StanExpr(Symbol(expr(lhs), "_gen"), remake(type(lhs); value=missing))
    end
    rng_rhs = rng_expr(lhs, rhs)
    lhs = StanExpr(expr(lhs), type(rng_rhs))
    push!(b, CanonicalExpr(:(=), lhs, rng_rhs); info)
end
    # if hasvalue(x.args[1])
    # push!(b, CanonicalExpr(:(=), rng_lhs(x.args[1]), rng_expr(x.args...)); info)
# end
# likelihood_expr(lhs, rhs) = likelihood_expr(rhs)

function rng_expr end
function likelihood_expr end
include("functions.jl")
include("builtin.jl")
function dummy_likelihood end
function dummy_rng end
likelihood_expr(lhs, rhs::StanExpr) = likelihood_expr(lhs, expr(rhs))
likelihood_expr(lhs, rhs::CanonicalExpr) = stan_call(likelihood_expr(head(rhs)), lhs, rhs.args...)
likelihood_expr(rhs) = dummy_likelihood
rng_expr(lhs, rhs) = rng_expr(rhs)
rng_expr(rhs::StanExpr) = rng_expr(expr(rhs))
rng_expr(rhs::CanonicalExpr) = stan_call(rng_expr(head(rhs)), rhs.args...)
rng_expr(x) = dummy_rng
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(std_normal)}}) = stan_call(vector_std_normal_rng, stan_size(lhs)...)
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(normal)}}) = stan_call(to_vector, stan_call(normal_rng, expr(rhs).args...))
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(normal),<:Tuple{<:StanExpr2{<:types.real, 0},<:StanExpr2{<:types.real, 0}}}}) = rng_expr(lhs, stan_call(normal, stan_call(rep_vector, expr(rhs).args[1], stan_size(lhs, 1)), expr(rhs).args[2]))
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(cauchy)}}) = stan_call(to_vector, stan_call(cauchy_rng, expr(rhs).args...))
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(cauchy),<:Tuple{<:StanExpr2{<:types.real, 0},<:StanExpr2{<:types.real, 0}}}}) = rng_expr(lhs, stan_call(cauchy, stan_call(rep_vector, expr(rhs).args[1], stan_size(lhs, 1)), expr(rhs).args[2]))

rng_expr(lhs::StanExpr2{types.real}, rhs::StanExpr{<:CanonicalExpr{typeof(exponential)}}) = stan_call(exponential_rng, expr(rhs).args...)
rng_expr(lhs::StanExpr2{types.vector}, rhs::StanExpr{<:CanonicalExpr{typeof(exponential)}}) = stan_call(vector_exponential_rng, expr(rhs).args..., stan_size(lhs)...)


struct Join
    iterator
    delim
end
Base.show(io::IO, x::Join) = join(io, x.iterator, x.delim)

abstract type WrappedIO <: IO end
Base.parent(io::WrappedIO) = io.parent
Base.write(io::WrappedIO, arg) = error(arg)#write(parent(io), arg)
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
current_indent(io) = ""
current_indent(io::StanIO) = repeat("    ", current_indent_level(io))
current_indent_level(io::StanIO) = get(io.info, :current_indent_level, 0)
indent(io) = StanIO(io; current_indent_level=1)
indent(io::StanIO) = remake(io; current_indent_level=1+current_indent_level(io))
maybe_indent(io, x::StanBlock) = indent(io)
maybe_indent(io, x::FunctionsBlock) = io
nobreak(io) = StanIO(io; maybreak=false)
nobreak(io::StanIO) = remake(io; maybreak=false)
maybreak(io) = true
maybreak(io::StanIO) = get(io.info, :maybreak, true)
line_limit(io) = 100


autoprint(io, args...) = if maybreak(io)
    buf = IOBuffer()
    print(remake(nobreak(io), buf), args...)
    rv = String(take!(buf))
    if length(rv) <= line_limit(io)
        print(io, rv)
    else
        idx = findfirst(x->isa(x, Join), args)
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
Base.show(io::StanIO, x::StanModel) = print(io, Join(blocks(x), "\n"))
Base.show(io::StanIO, x::StanExpr) = isa(type(x), StringStanType) ? print(io, expr(x), "::", type(x)) : print(io, expr(x))
Base.show(io::StanIO, ::Colon) = print(io, ":")
Base.show(io::IO, x::StanModel) = show(StanIO(io), x)

Base.show(io::IO, x::SlicModel; mayfail=true) = try
    print(io, stan_model(x))
catch e
    mayfail && return print(io, "SlicModel: Something went wrong: $e")
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
function maybetype end
maybetype(x::StanExpr) = center_type(x) == types.anything ? "// Disabled because type inference failed\n    // $(type(x))" : type(x)
Base.show(io::IO, x::AssignmentExpr{<:StanExpr{Symbol}}) = begin
    name, rhs = x.args
    @assert center_type(rhs) != types.anything "tracetype not defined for $name = $(short_expr(rhs))!"
    # @info "$(x.args[1]) = $(x.args[2])"
    print(io, type(rhs), " ", name, " = ", rhs)
end
Base.show(io::IO, x::AssignmentExpr) = print(io, x.args[1], " = ", x.args[2])

prettystring(f) = " $f "
prettystring(f::Base.BroadcastFunction) = " .$(f.f) "
Base.show(io::IO, x::CanonicalExpr) = begin
    fname = func_name(head(x), x.args)
    fargs = filter(!always_inline, x.args)
    is_lpxf = endswith(string(fname), r"_lp[md]f")
    if is_lpxf 
        autoprint(io, fname, "(", fargs[1], " | ", Join(fargs[2:end], ", "), ")")
    else
        autoprint(io, fname, "(", Join(fargs, ", "), ")")
    end
end
Base.show(io::IO, x::CanonicalExpr{<:ODESolver}) = autoprint(io, head(x), "(", Join(
    (func_name(x.args[1], x.args[2:end]), filter(!always_inline, x.args[2:end])...), ", "
), ")")
commentstring(x::String) = "// " * replace(x, "\n"=>"\n    // ") * "\n"
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
Base.show(io::IO, x::IfExpr) = begin
    print(io, "if(", x.args[1], ")", StanBlock(Symbol(), x.args[2].args))
    if length(x.args) == 3
        e = x.args[3]
        print(io, " else", if isa(e, BlockExpr)
            StanBlock(Symbol(), e.args)
        else 
            error(dump(x))#remake(x, e.args...)
        end)
    end
end
Base.show(io::IO, x::CanonicalExpr{typeof(adjoint)}) = print(io, "(", x.args[1], "')")
Base.show(io::IO, x::CanonicalExpr{typeof(range)}) = autoprint(io, "linspaced_vector(", Join((x.args[end], x.args[1], x.args[2]), ", "), ")")
Base.show(io::IO, x::CanonicalExpr{typeof(getindex)}) = autoprint(io, x.args[1], "[", Join(x.args[2:end], ", "), "]")
Base.show(io::IO, x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = print(io, x.args[1], ".", x.args[2])
for f in (-,+,*,\,/,^,.*,./,<,<=,==,!=,>=,>)
    @eval Base.show(io::IO, x::CanonicalExpr{typeof($f)}) = autoprint(io, "(", Join(x.args, prettystring($f)), ")")
    @eval Base.show(io::IO, x::CanonicalExpr{typeof($f),Tuple{A}}) where {A} = print(io, "(", string($f), x.args[1], ")")
end
Base.show(io::IO, x::CanonicalExpr{typeof(÷)}) = autoprint(io, "(", Join(x.args, " %/% "), ")")
for f in (Meta.quot(:(~)), Meta.quot(:(=)))
    @eval Base.show(io::IO, x::CanonicalExprV{$f}) = print(io, Join(x.args, prettystring($f)))
end
Base.show(io::IO, x::SamplingExpr) = print(io, Join(x.args, " ~ "))
# Base.show(io::IO, x::SamplingExpr{<:Any,<:StanExpr{<:CanonicalExpr{typeof(flat)}}}) = nothing

for f in (:+=,:-=,:*=)
    qf = Meta.quot(f)
    @eval forward!(x::CanonicalExprV{$qf}; info) = stan_expr(remake(x, forward!(x.args; info)...))
    @eval Base.show(io::IO, x::CanonicalExprV{$qf}) = print(io, Join(x.args, prettystring($qf)))
end
@eval forward!(x::CanonicalExprV{:(.=)}; info) = stan_expr(remake(x, forward!(x.args; info)...))
@eval Base.show(io::IO, x::CanonicalExprV{:(.=)}) = print(io, Join(x.args, " = "))
    

stan_code(x::SlicModel; mayfail=false) = begin 
    buf = IOBuffer()
    show(buf, x; mayfail)
    String(take!(buf))
end
stan_code2(x) = begin 
    buf = IOBuffer()
    print(StanIO(buf), x)
    String(take!(buf))
end
function bridgestan_data end
function instantiate end
debug_instantiate(x; kwargs...) = instantiate(x; nan_on_error=false, kwargs...)
passinstantiate(x; kwargs...) = (instantiate(x; kwargs...); x)
stan_data(x::SlicModel) = stan_data(stan_model(x))
stan_data(x::StanModel) = Dict([
    key=>getvalue(value) for (key, value) in pairs(content(block(x, :data)))
    if !always_inline(value)
])
slic_expr(x::Expr) = x

include("test.jl")

end
macro slic(model)
    stan.SlicModel(model, Dict())
end
macro slic(data, model)
    mod = @__MODULE__
    qmodel = Meta.quot(model)
    esc(:($mod.stan.SlicModel($qmodel, $data)))
end
macro defsig(x)
    esc(stan.defsig(x))
end
macro deffun(x)
    esc(stan.deffun(x))
end
stan_code(args...; kwargs...) = stan.stan_code(args...; kwargs...)
stan_instantiate(args...; kwargs...) = stan.instantiate(args...; kwargs...)