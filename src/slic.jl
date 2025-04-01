module stan
using OrderedCollections
include("slic/meta.jl")
struct StanModel{I,B}
    info::I
    blocks::B
end
struct SubModel{I,P}
    name::Symbol
    info::I
    parent::P
end
struct SlicModel{M,D}
    model::M
    data::D
end
struct StanExpr{E,T}
    expr::E
    type::T
end
struct StanType{Q,T,S,C}
    size::NTuple{S,StanExpr}
    cons::C
    StanType(Q,T,size=tuple(); kwargs...) = new{Q,T,length(size),typeof(kwargs)}(size, kwargs)
end
StanModel() = StanModel(
    Dict(),
    (;functions=OrderedSet(), data=OrderedSet(), transformed_data=[], parameters=OrderedSet(), transformed_parameters=[], model=[], generated_quantities=[])
)
Base.keys(x::StanModel) = keys(x.info)
Base.setindex!(x::StanModel, xname, name) = setindex!(x.info, xname, name)
Base.getindex(x::StanModel, name::Symbol) = x.info[name]
block(x::StanModel, key::Symbol) = x.blocks[key]
Base.show(io::IO, x::StanModel) = begin
    for key in (:functions, :data, :transformed_data, :parameters, :transformed_parameters, :model, :generated_quantities)
        length(block(x, key)) == 0 && continue
        print(io, replace(string(key), "_"=>" "), " {\n")
        for stmt in block(x, key)
            top_print(io, stmt)
        end
        print(io, "\n}\n")
    end
end 
Base.parent(x::SubModel) = x.parent
block(x::SubModel, key::Symbol) = block(parent(x), key)
supname(x::SubModel, key::Symbol) = key == RV_SYMBOL ? x.name : Symbol(x.name, "_", key)
supvalue(x::SubModel, v::StanExpr{Symbol}) = StanExpr(supname(x, expr(v)), type(v))
Base.keys(x::SubModel) = keys(x.info)
Base.setindex!(x::SubModel, xname, name) = begin
    setindex!(parent(x), supvalue(x, xname), supname(x, name))
    setindex!(x.info, parent(x)[supname(x, name)], name)
end
Base.getindex(x::SubModel, name::Symbol) = x.info[name]
expr(x::StanExpr) = x.expr
type(x::StanExpr) = x.type
qual(x::StanExpr) = qual(type(x))
qual(::StanType{Q}) where {Q} = Q
qual(args...) = all(arg->qual(arg) == :data, args) ? :data : :parameters
utype(x::StanExpr) = utype(type(x))
utype(x::StanType{<:Any,T}) where {T} = (T, x.size)
StanExpr2{T,S,E,Q,C} = StanExpr{E,StanType{Q,T,S,C}}
center_type(::StanType{<:Any,T}) where {T} = T
r_ndim(::StanType{<:Any,:int}) = 0
r_ndim(::StanType{<:Any,:real}) = 0
r_ndim(::StanType{<:Any,:vector}) = 1
r_ndim(::StanType{<:Any,:row_vector}) = 1
r_ndim(::StanType{<:Any,:matrix}) = 2
r_ndim(::StanType{<:Any,:cholesky_factor_corr}) = 1
l_ndim(x::StanType) = length(x.size) - r_ndim(x)
lr_size(x::StanType) = x.size[1:l_ndim(x)], x.size[1+l_ndim(x):end]

include("slic/print.jl")

comp(x::Function) = StanExpr(x, StanType(:comp, :function))
comp(x::Irrational) = StanExpr(Dict(pi=>:(pi()))[x], StanType(:data, :real))
comp(x::SlicModel) = x

data!(x::StanType; info) = begin 
    data!.(x.size; info)
end
data!(x::StanExpr; info) = begin 
    data!(x.type; info)
    info[x.expr] = x
    push!(block(info, :data), info[x.expr])
end
data!(name, x; info) = data!(data(name, x); info)

data(x; kwargs...) = data(x, x; kwargs...)
data(name, ::Int64) = StanExpr(name, StanType(:data, :int))
data(name, ::Float64) = StanExpr(name, StanType(:data, :real))
data(name, x::Vector{Int64}; n=Symbol(name, "_n")) = StanExpr(
    name, 
    StanType(:data, :int, (data(n, length(x)), ))
)
data(name, x::AbstractVector{Float64}; n=Symbol(name, "_n")) = StanExpr(
    name, 
    StanType(:data, :vector, (data(n, length(x)), ))
)
data(name, x::Matrix{Float64}; m=Symbol(name, "_m"), n=Symbol(name, "_n")) = StanExpr(
    name, 
    StanType(:data, :matrix, data.((m,n), size(x)))
)
data(name, x::StanExpr) = StanExpr(name, type(x))

(x::SlicModel)(df=(;); kwargs...) = SlicModel(
    x.model, merge(x.data, Dict(pairs(df)), Dict(pairs(kwargs)))
)
# (x::SlicModel)
function bridgestan_data end
function instantiate end
debug_instantiate(x; kwargs...) = instantiate(x; nan_on_error=false, kwargs...)
stan_data!(x::StanExpr, y::Number; stan_data) = stan_data[expr(x)] = y
stan_data!(x::StanExpr, y::AbstractVector; stan_data) = begin
    stan_data!.(type(x).size, size(y); stan_data)
    stan_data[expr(x)] = y
end
stan_code(x::SlicModel; code_info=code(x)) = begin 
    buf = IOBuffer()
    print(buf, code_info)
    String(take!(buf))
end
stan_data(x::SlicModel; code_info=code(x)) = begin 
    rv = Dict()
    for name in keys(code_info)
        name in keys(x.data) || continue
        stan_data!(code_info[name], x.data[name]; stan_data=rv)
    end
    rv
end
code(x::SlicModel) = begin 
    info = StanModel()
    for (name, value) in pairs(x.data)
        data!(name, value; info)
    end 
    code!(x.model; info)
    info
end
code!(::LineNumberNode; kwargs...) = nothing
code!(x::Expr; info) = if x.head == :block
    code!.(x.args; info)
elseif x.head == :(=)
    trace!(x.args...; info)
elseif x.head == :return
    trace!(RV_SYMBOL, x.args[1]; info)
else
    (;fargs) = xcanonical(x)
    @assert fargs[1] == :~
    sample!(fargs[2:end]...; info)
end
trace!(name, x; info, kwargs...) = begin
    x = trace(x; info)
    (name in keys(info) && !isa(info, StanModel)) && return
    info[name] = StanExpr(name, type(x))
    push!(
        block(info, get(kwargs, :block, Symbol("transformed_", qual(x)))),
        info[name],
        Expr(:(=), info[name], x)
    )
end
postfixname(x, p) = endswith(string(x), p) ? x : Symbol(x, p)
genname(x) = postfixname(x, "_gen")
rngname(x) = postfixname(x, "_rng")
sample!(name, x; info) = begin 
    (;fargs, kwargs) = trace(xcanonical(x); info)
    if name ∉ keys(info)
        samplecall!(fargs[1], name, fargs[2:end]...; info, kwargs...)
    elseif isa(info, StanModel)
        if isa(fargs[1], SlicModel) 
            @warn "Skipping known param ~ ::SlicModel statement (not implemented yet)"
            return
        end
        f = x.args[1]
        f == rngname(f) || push!(block(info, :model), xcall(:~, info[name], x))
        # gen_expr = Expr(:(=), Symbol(name, "_gen"), Expr(:call, Symbol(x.args[1], "_rng"), x.args[2:end]...))
        (f == :normal || f == rngname(f)) && trace!(
            Symbol(name, "_gen"), 
            Expr(:call, rngname(f), x.args[2:end]...); 
            info, block=:generated_quantities
        )
        # push!(block(info, :generated_quantities), StanExpr(Symbol(name, "_generated"), :real))
    end
end
samplecall!(f::StanExpr, lhs::Expr, args...; info, kwargs...) = begin 
    push!(block(info, :model), xcall(:~, trace(lhs; info), xcall(Symbol(expr(f)), args...)))

end
function flat end
samplecall!(f::StanExpr{typeof(flat)}, name::Symbol, args...; info, kwargs...) = samplecall!(f, name, :real, args...; info, kwargs...)
samplecall!(f::StanExpr{typeof(flat)}, name::Symbol, type::Symbol, size...; info, kwargs...) = begin 
    info[name] = StanExpr(name, StanType(:parameters, type, size))
    push!(block(info, :parameters), info[name])
end
samplecall!(f::StanExpr, name::Symbol, args...; info, kwargs...) = begin 
    type = get(kwargs, :type, autotype(f, args...))
    size = get(
        kwargs, :size, 
        (vcat(get.(Ref(kwargs), (:m,:n,:o), Ref([]))...)...,)
    )
    if ismissing(type)
        type = get(kwargs, :type, [:real, :vector, :matrix][1+length(size)])
    end
    cons = autocons(f, args...)#Symbol(expr(f)) == :lognormal ? (;lower=0.) : (;)
    info[name] = StanExpr(name, StanType(:parameters, type, size; cons...))
    push!(block(info, :parameters), info[name])
    push!(block(info, :model), xcall(:~, info[name], xcall(Symbol(expr(f)), args...)))
end
samplecall!(f::SlicModel, name::Symbol; info, kwargs...) = begin 
    code!(f.model; info=SubModel(name, Dict{Symbol,Any}(pairs(kwargs)), info))
end
trace(;info) = x->trace(x; info)
trace(x::Union{Tuple,NamedTuple,Vector}; info) = map(trace(;info), x)
trace(x::Symbol; info) = begin 
    startswith(string(x), ".") && return comp(Base.BroadcastFunction(getproperty(stan, Symbol(string(x)[2:end]))))
    x in keys(info) && return info[x]
    isdefined(stan, x) && return comp(getproperty(stan, x))
    isdefined(Main, x) && isa(getproperty(Main, x), Union{Function,SlicModel}) && return comp(getproperty(Main, x)) 
    # @error "Could not find $x in info $((keys(info)...,)), in stan module, or in Main module."
    error((;x, info))
end
trace(x::QuoteNode; info) = x.value
trace(x::Number; info) = data(x)
trace(x::Expr; info) = begin
    (;fargs, kwargs) = trace(xcanonical(x); info)
    udef = fundef(fargs...; kwargs...)
    !isnothing(udef) && push!(block(info, :functions), udef)
    tracecall(fargs...; kwargs...)
end
tracecall(f, args...) = StanExpr(traceexpr(f, args...), tracetype(f, args...))
# traceexpr(f, args...) = xcall(Symbol(expr(f)), args...)
traceexpr(f, args...) = xcall(f, args...)
tracetype(f, args...; kwargs...) = StanType(qual(args...), traceutype(f, args...; kwargs...)...)
traceutype(f, args...; kwargs...) = error("traceutype missing for $(expr(f))$((type.(args)...,))")
traceutype(f::StanExpr{<:Union{typeof(*),typeof(+)}}, x, y, z, args...) = traceutype(f, x, tracecall(f, y, z, args...))
tracecall(::StanExpr{Colon}, x::StanExpr2{:int, 0}, y::StanExpr2{:int, 0}) = begin
    n = expr(x) == 1 ? y : StanExpr(:($y-$x+1), StanType(:data, :int))
    StanExpr(
        xcall(:linspaced_vector, n, x, y),
        StanType(:data, :vector, (n,))
    )
end

include("slic/functions.jl")

end


macro slic(model)
    stan.SlicModel(model, Dict())
end