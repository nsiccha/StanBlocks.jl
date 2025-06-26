using OrderedCollections
dumperror(x) = (dump(x); error(x))

abstract type AbstractModel end

struct SlicModel{M,D} <: AbstractModel
    model::M
    data::D
end
# struct SubModel{P<:AbstractModel,N,L} <: AbstractModel
#     parent::P
#     name::N
#     locals::L
# end

# struct CanonicalExpr{H,A,K}
#     head::H
#     args::A
#     kwargs::K
#     # Maybe a bit dangerous
#     CanonicalExpr(head::Symbol, args...; kwargs...) = CanonicalExpr(Val(head), args...; kwargs...)
#     CanonicalExpr(head, args...; kwargs...) = canonical(new{typeof(head),typeof(args),typeof((;kwargs...))}(head, args, (;kwargs...)))
#     # CanonicalExpr(head::Val{:block}, args...; kwargs...) = new{Val{:block},typeof(collect(args)),typeof((;kwargs...))}(head, collect(args), (;kwargs...))
# end

# head(x::CanonicalExpr) = x.head


# model(x::SlicModel) = x.model
# data(x::SlicModel) = x.data
# Base.parent(x::SubModel) = x.parent
# name(x::SubModel) = x.name
# locals(x::SubModel) = x.locals
# Base.getindex(x::SubModel, name) = getindex(locals(x), name)
# Base.setindex!(x::SubModel, value, name) = begin
#     setindex!(parent(x), supvalue(x, value), supname(x, name))
#     setindex!(locals(x), getindex(parent(x), supname(x, name)), name)
# end
# Base.keys(x::SubModel) = keys(locals(x))
# supname(x::SubModel, post) = Symbol(name(x), "_", post)
# supvalue(x::SubModel, value) = value
struct DispatchableExpr{H<:Val,A<:Tuple,K<:NamedTuple}
    head::H
    args::A
    kwargs::K
    DispatchableExpr(head::Val, args...; kwargs...) = canonical(new{typeof(head), typeof(args), typeof((;kwargs...))}(head, args, (;kwargs...)))
    DispatchableExpr(head::Symbol, args...; kwargs...) = DispatchableExpr(Val(head), args...; kwargs...)
    DispatchableExpr(expr::Expr) = DispatchableExpr(expr.head, DispatchableExpr.(expr.args)...)
    DispatchableExpr(x::Symbol) = x
    DispatchableExpr(x) = x
end
DispatchableExprV2{H,A<:Tuple,K<:NamedTuple} = DispatchableExpr{Val{H},A,K} 
CallExpr{A<:Tuple,K<:NamedTuple} = DispatchableExprV2{:call,A,K}
AssignExpr{A<:Tuple,K<:NamedTuple} = DispatchableExprV2{:(=),A,K}
SampleExpr{A<:Tuple,K<:NamedTuple} = DispatchableExprV2{:(~),A,K}
remake(x::DispatchableExpr, args...; kwargs...) = DispatchableExpr(x.head, args...; kwargs...)

canonical(x) = x
# canonical(x::CallExpr) = x
canonical(x::DispatchableExprV2{:kw,Tuple{Symbol}}) = remake(x, x.args[1], x.args[1])
canonical(x::DispatchableExprV2{:parameters,<:Tuple{Vararg{Symbol}}}) = remake(x, [
    DispatchableExpr(:kw, arg) for arg in x.args
]...)
canonical(x::CallExpr{<:Tuple{<:Any, <:DispatchableExprV2{:parameters},Vararg{Any}}}) = remake(
    x, x.args[1], x.args[3:end]...; [
        kw.args[1]=>kw.args[2] for kw::DispatchableExprV2{:kw} in x.args[2].args
    ]...
)
canonical(x::CallExpr) = if x.args[1] == :~ 
    DispatchableExpr(:~, x.args[2:end]...)
else
    x
end

Base.show(io::IO, x::DispatchableExpr) = print(io, decanonicalize(x))
val(::Val{T}) where {T} = T
decanonicalize(x) = x
decanonicalize(x::DispatchableExpr) = Expr(val(x.head), decanonicalize.(x.args)...)
decanonicalize(x::CallExpr) = if length(x.kwargs) > 0
    Expr(val(x.head), decanonicalize(x.args[1]), Expr(:parameters, [
        Expr(:kw, key, decanonicalize(value)) for (key, value) in pairs(x.kwargs)
    ]...), decanonicalize.(x.args[2:end])...)
else
    Expr(val(x.head), decanonicalize.(x.args)...)
end
decanonicalize(x::SampleExpr) = Expr(:call, :~, x.args...)



jslic(x) = x
jslic_top(x::LineNumberNode) = x
jslic_top(x::Expr) = if x.head == :block
    Expr(x.head, jslic_top.(x.args)...)
else
    @assert x.head == :(=)
    lhs, rhs = x.args
    Expr(x.head, lhs, SlicModel(DispatchableExpr(rhs), OrderedDict()))
end 
macro jslic(x)
    esc(jslic_top(x))
end

transform(f, x::SlicModel; data=deepcopy(x.data)) = SlicModel(f(x.model; data), data)
# transform!(f, x::SlicModel) = SlicModel((transform!(f, x.model; x.data)), x.data)
# transform!(f, x; data) = (f(x; data); x)
# ftransform!(f, x::Expr; data) = error()#f(Expr(x.head, [ftransform!(f, arg; data) for arg in x.args]...); data)
# btransform!(f, x::SlicModel) = SlicModel(deaugment2(btransform!(f, x.model; x.data)), x.data)
# btransform!(f, x; data) = f(x; data) 
# btransform!(f, x::Expr; data) = error()f(Expr(x.head, reverse([btransform!(f, arg; data) for arg in reverse(x.args)])...); data)

struct AugmentedExpr2{E}
    expr::E
    meta::OrderedDict{Symbol,Any}
    AugmentedExpr2(expr) = new{typeof(expr)}(expr, OrderedDict{Symbol,Any}())
end
# AugmentedDispatchableExpr{H<:Val,A<:Tuple,K<:NamedTuple} = AugmentedExpr2{DispatchableExpr{H,A,K}}
Base.getproperty(x::AugmentedExpr2, k::Symbol) = hasfield(AugmentedExpr2, k) ? getfield(x, k) : x.meta[k]
Base.setproperty!(x::AugmentedExpr2, k::Symbol, v) = setindex!(x.meta, v, k)
# Base.show(io::IO, x::AugmentedExpr2) = print(io, deaugment(x), "::", last(collect(values(x.meta))))
Base.show(io::IO, x::AugmentedExpr2{Symbol}) = if length(x.meta) > 0 
    print(io, x.expr, "::", last(collect(values(x.meta))))
else
    print(io, x.expr, )
end
# deaugment(x) = x
deaugment2(x::AugmentedExpr2) = x.expr
# deaugment(x::AugmentedExpr2{Expr}) = Expr(x.expr.head, deaugment.(x.expr.args)...)
# deaugment(x::AugmentedExpr2{Symbol}) = x



resolve2(;kwargs...) = x->resolve2(x; kwargs...)
resolve2(x::SlicModel; data=deepcopy(x.data)) = SlicModel(resolve2(x.model; data), data)
resolve2(x; data) = x
resolve2(x::DispatchableExpr; data) = remake(x, map(resolve2(;data), x.args)...; map(resolve2(;data), x.kwargs)...)
resolve2(x::Symbol; data) = get!(data, x) do 
    isdefined(Main, x) ? AugmentedExpr2(getproperty(Main, x)) : AugmentedExpr2(x)
end
# inline2(;kwargs...) = x->inline2(x; kwargs...)
# inline2(x::SlicModel; data=deepcopy(x.data)) = SlicModel(inline2(x.model; data), data)
# inline2(x; data) = x
# inline2(x::DispatchableExpr; data) = (@info typeof(x); remake(x, map(inline2(;data), x.args)...; map(inline2(;data), x.kwargs)...))
# inline2(x::AssignExpr{<:Tuple{<:Any,<:CallExpr{<:Tuple{<:SlicModel}}}}; data) = error()

# inline(x::CallExpr) = remake(x, inline.(x.args...); map(inline, x.kwargs)...)
# inline(x::CallExpr) = remake(x, inline.(x.args...); map(inline, x.kwargs)...)
# inline(x::Exp)

# qual(x) = :Const
# qual(x::AugmentedExpr2) = x.qual
get2!(f, d, k1, k2) = getproperty(get!(f, d, k1), k2)
trace_qual(;data) = x->trace_qual(x; data)
trace_qual(x; data) = :Const#(@info typeof(x); x)
trace_qual(x::Symbol; data) = get2!(data, x, :qual) do 
    isdefined(Main, x) && return AugmentedExpr2(x, OrderedDict{Symbol,Any}(:qual=>:Const))
    AugmentedExpr2(x, OrderedDict{Symbol,Any}(:qual=>:Parameters))
end
trace_qual(x::DispatchableExpr; data) = maximum(trace_qual(;data), (x.args..., x.kwargs...))
trace_qual(x::AssignExpr; data) = begin 
    prelhs, rhs = trace_qual.(x.args; data)
    lhs = data[x.args[1]]
    lhs.qual = rhs
    lhs.qual
    # @assert lhs isa AugmentedExpr2{Symbol} lhs
    # lhs.qual = qual(rhs)
    
end

trace_likelihood(;kwargs...) = x->trace_likelihood(x; kwargs...)
trace_likelihood(x; kwargs...) = false
trace_likelihood(x::Symbol; data, value=false) = begin
    @assert x in keys(data)
    ax = data[x]
    get!(ax.meta, :likelihood, false)
    ax.likelihood |= value
    ax.likelihood
end
trace_likelihood(x::DispatchableExpr; data, value=false) = map(trace_likelihood(;data, value), (reverse(x.args)..., x.kwargs...))
trace_likelihood(x::AssignExpr; data, value=false) = trace_likelihood(x.args[2]; data, value=value || trace_likelihood(x.args[1]; data, value))
trace_likelihood(x::SampleExpr; data, value=false) = begin
    trace_likelihood(x.args[2]; data, value=(value || trace_qual(x.args[1]; data) <= :Data))
end