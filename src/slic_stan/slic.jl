module stan
using OrderedCollections, JSON, StanLogDensityProblems
const RV_NAME = gensym("RV")
dumperror(x) = (dump(x); error("dumperror (see dump above): value of type `$(typeof(x))` hit an unhandled code path."))
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
IfThenExpr2{I,T<:BlockExpr} = CanonicalExprV{:if,Tuple{I,T}}
IfThenElseExpr{I,T<:BlockExpr,E<:BlockExpr} = CanonicalExprV{:if,Tuple{I,T,E}}
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
stan_call(f, args...) = stan_expr(CanonicalExpr(f, map(stan_expr, args)...))
"""
"""
function stan_expr end
stan_expr(x::StanExpr; kwargs...) = weak_remake(x; kwargs...)
stan_expr(x; kwargs...) = stan_expr(x, x; kwargs...)
stan_expr(expr, value; kwargs...) = StanExpr(expr, stan_type(expr, value; kwargs...))
stan_expr(x, value::StanExpr; kwargs...) = weak_remake(value; kwargs...)
maybedata(expr, value; kwargs...) = stan_expr(expr, value; qual=:data, kwargs...)
maybedata(expr, value::Function; kwargs...) = stan_expr(value, value; qual=:data, kwargs...)
maybecv(expr, value) = stan_expr(expr, value; cv=true)

"Traces through its first argument (a `StanBlocks.SlicModel`) and returns the inferred `StanBlocks.StanModel`."
function stan_model end
const _StanBlocksError = parentmodule(@__MODULE__).StanBlocksError
stan_model(x::SlicModel; info=StanModel()) = begin
    _expr_stack = Any[]
    _current_lnn = Ref{Any}(nothing)
    info = remake(info; _expr_stack, _current_lnn)
    task_local_storage(:_slic_expr_stack, _expr_stack) do
        task_local_storage(:_slic_current_lnn, _current_lnn) do
            try
                distribute!(backward!(forward!(x; info); info); info)
                remake(info; docstring=get(x.data, :docstring, ""))
            catch e
                e isa _StanBlocksError && rethrow()
                bt = catch_backtrace()
                # Keep the raw (exception, backtrace, expr_stack) tuple so the
                # display layer can show the Julia traceback alongside the
                # SLIC-level expression trace.
                throw(_StanBlocksError(:transpile, "model", (e, bt, copy(_expr_stack))))
            end
        end
    end
end
maybedata!(x::StanModel, key, value) = x[key] = maybedata(key, value)
maybedata!(x::StanModel, key, value::AbstractString) = nothing
maybedata!(x::SubModel, key, value) = locals(x)[key] = maybedata(key, value)
_control_flow_kind(::ForExpr) = "for"
_control_flow_kind(::WhileExpr) = "while"
_control_flow_kind(::IfExpr) = "if"
_control_flow_kind(::ElseIfExpr) = "elseif"
_control_flow_kind(::BreakExpr) = "break"
_control_flow_kind(::ContinueExpr) = "continue"
_reject_model_control_flow(x) = x
_reject_model_control_flow(x::CanonicalExpr) = (foreach(_reject_model_control_flow, x.args); x)
_reject_model_control_flow(x::Union{ForExpr,WhileExpr,IfExpr,ElseIfExpr,BreakExpr,ContinueExpr}) = error(
    "`$(_control_flow_kind(x))` control flow is not supported in @slic model bodies — ",
    "move the logic into an @deffun function body, or use a vectorised form."
)
"""
Descends forwards through the expression tree. Basically _always_ returns a StanExpr.
"""
function forward! end
forward!(x::SlicModel; info=StanModel()) = begin
    info = remake(info; mod=x.mod)
    for (key, value) in pairs(data(x))
        maybedata!(info, key, value)
    end
    body = canonical(model(x))
    _reject_model_control_flow(body)
    forward!(body; info)
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
            @assert all(isexpr(:kw), arg.args) "canonical(:call): `:parameters` block must contain only `:kw` entries, got `$(arg.args)`."
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
    @assert length(x.args) > 0 "canonical(:tuple): empty tuple expression `$x`."
    if any(Base.Fix2(isexpr, :parameters), x.args)
        @assert length(x.args) == 1 "canonical(:tuple): a `:parameters`-bearing tuple must have exactly one arg (the parameters block), got $(length(x.args)) in `$x`."
        @assert all(isexpr(:kw), x.args[1].args) "canonical(:tuple): named-tuple parameters block must contain only `:kw` entries, got `$(x.args[1].args)`."
        CanonicalExpr(:nt, x.args[1].args...)
    else
        x
    end
end
canonical(x::CanonicalExprV{:macrocall}) = begin
    head = x.args[1]
    is_doc = head == GlobalRef(Core, Symbol("@doc")) || head == Symbol("@doc")
    @assert is_doc "canonical(:macrocall): only `@doc` macrocalls are supported in SLIC bodies, got `$(head)`."
    CanonicalExpr(:document, x.args[3:4]...)
end
canonical(x::CanonicalExprV{Symbol("'")}) = CanonicalExpr(:adjoint, x.args...)
canonical(x::CanonicalExprV{:ref}) = CanonicalExpr(:getindex, x.args...)
canonical(x::CanonicalExprV{Symbol(".*")}) = CanonicalExpr(.*, x.args...)
canonical(x::CanonicalExprV{Symbol("./")}) = CanonicalExpr(./, x.args...)
pretty_type_expr(T::Symbol) = string(T)
pretty_type_expr(ref::CanonicalExprV{:getindex}) = string(ref.args[1], "[", join(ref.args[2:end], ", "), "]")

# TODO: task-local storage is used here to propagate _expr_stack/_current_lnn through
# @deffun calls, which rebuild `info` from scratch. Cleaner alternatives:
# - Thread info through stan_expr (invasive: changes signature everywhere + codegen)
# - Carry refs as CanonicalExpr kwargs (requires codegen changes in functions.jl)
_get_expr_stack(info) = something(_expr_stack(info), get(task_local_storage(), :_slic_expr_stack, nothing), Some(nothing))
_get_lnn_ref(info) = something(_current_lnn(info), get(task_local_storage(), :_slic_current_lnn, nothing), Some(nothing))
_get_lnn(info) = (lnn = _get_lnn_ref(info); lnn isa Ref ? lnn[] : nothing)
_push_expr!(info, x) = (s = _get_expr_stack(info); isnothing(s) || push!(s, (x, _get_lnn(info))); nothing)
_pop_expr!(info) = (s = _get_expr_stack(info); isnothing(s) || pop!(s); nothing)

forwards!(;info) = x->forwards!(x; info)
forwards!(x; info) = [forward!(x; info)]
forwards!(x::SplatExpr; info) = [forward!(x.args[1]; info)...]
forward!(x; info) = error("forward! not defined for value `$x` of type `$(typeof(x))` — no method matches a more specific signature.")
forward!(;info) = x->forward!(x; info)
forward!(x::Tuple; info) = (mapreduce(forwards!(;info), vcat, x; init=[])...,)
forward!(x::Union{Tuple,NamedTuple,Vector,Base.Pairs}; info) = map(forward!(;info), x)
forward!(x::Union{String,Number,Function,Nothing}; info) = x
forward!(x::LineNumberNode; info) = (lnn = _get_lnn_ref(info); lnn isa Ref && (lnn[] = x); x)
forward!(x::QuoteNode; info) = x.value
forward!(x::Irrational; info) = error("forward! not defined for irrational `$x` — only `π` is handled; convert to a concrete numeric value first.")
forward!(x::Irrational{:π}; info) = forward!(Float64(pi); info)
forward!(x::Number; info) = maybedata(x, x)
get_module(info::StanModel) = get(info.meta, :mod, Main)
get_module(info::AbstractDict) = get(info, :__mod__, Main)
get_module(info::NamedTuple) = get(info, :__mod__, Main)
get_module(info::SubModel) = get_module(parent(info))
forward!(x::Symbol; info) = begin
    x in keys(info) && return info[x]
    isdefined(builtin, x) && return forward!(getproperty(builtin, x); info)
    mod = get_module(info)
    if isdefined(mod, x)
        Mx = getproperty(mod, x)
        isa(Mx, Function)  && return forward!(Mx; info)
        isa(Mx, SlicModel) && return Mx
        error("Found $x in $(mod), but is of type $(typeof(Mx))!")
    end
    if mod !== Main && isdefined(Main, x)
        Mx = getproperty(Main, x)
        isa(Mx, Function)  && return forward!(Mx; info)
        isa(Mx, SlicModel) && return Mx
        error("Found $x in Main, but is of type $(typeof(Mx))!")
    end
    error("Could not find $(x) in model, builtin, $(mod) or Main!")
end
forward!(x::Function; info) = stan_expr(x)
# `macroexpand` (used by `slic_macroexpand`) hygienically resolves all free
# names — including model-scope SLIC variables and SLIC builtins like
# `mean`/`length` — to `GlobalRef(mod, :name)`. Always try SLIC's own
# resolution chain (model → builtin → mod → Main) first so that e.g.
# `mean` resolves to `stan.builtin.mean` (with a registered tracetype),
# not `Statistics.mean` (which lacks one). Fall back to the GlobalRef's
# native module only when nothing else matches.
forward!(x::GlobalRef; info) = begin
    rv = _try_symbol_lookup(x.name; info)
    rv === nothing || return forward!(rv; info)
    isdefined(x.mod, x.name) || error(
        "Could not resolve $x — not in SLIC scope (model/builtin/mod/Main) and not defined in $(x.mod)."
    )
    forward!(getproperty(x.mod, x.name); info)
end
_try_symbol_lookup(x::Symbol; info) = begin
    x in keys(info) && return info[x]
    isdefined(builtin, x) && return getproperty(builtin, x)
    mod = get_module(info)
    if isdefined(mod, x)
        v = getproperty(mod, x)
        (isa(v, Function) || isa(v, SlicModel)) && return v
    end
    if mod !== Main && isdefined(Main, x)
        v = getproperty(Main, x)
        (isa(v, Function) || isa(v, SlicModel)) && return v
    end
    nothing
end
forward!(x::Colon; info) = x
forward!(x::StanExpr{Symbol}; info) = x
forward!(x::StanExpr; info) = x
forward!(x::CanonicalExpr; info) = begin
    _push_expr!(info, x)
    resolved = CanonicalExpr(forward!(head(x); info), forward!(x.args; info)...; forward!(x.kwargs; info)...)
    s = _get_expr_stack(info)
    isnothing(s) || (s[end] = (resolved, s[end][2]))
    rv = expand_inline_or_trace(resolved; info)
    _pop_expr!(info)
    rv
end
# `inline_body` is the dispatch hook populated by `@deffun @inline f(...)`
# and `@deffun f!(...)`. A non-`nothing` return signals the call site should
# substitute its args into the stored body AST and re-trace in the caller's
# scope rather than emitting a Stan-function call.
inline_body(::Any) = nothing
expand_inline_or_trace(x::CanonicalExpr; info) = begin
    meta = inline_body(x)
    isnothing(meta) && return fold_shape_query(stan_expr(x))
    expand_inline!(x, meta; info)
end
expand_inline!(x::CanonicalExpr, meta; info) = begin
    arg_names = meta.arg_names
    n_pos = length(arg_names)
    @assert length(x.args) >= n_pos "@inline: call to $(head(x)) has $(length(x.args)) args, expected ≥ $n_pos."
    subst = Dict{Symbol,Any}()
    for (i, name) in enumerate(arg_names)
        subst[name] = x.args[i]
    end
    if meta.vararg_name !== nothing
        subst[meta.vararg_name] = collect(x.args[n_pos+1:end])
    elseif length(x.args) > n_pos
        error("@inline: call to $(head(x)) has $(length(x.args)) args but the UDF takes exactly $n_pos.")
    end
    body = inline_unwrap_single(meta.body, head(x))
    substituted = inline_substitute(body, subst, meta.vararg_name)
    forward!(canonical(substituted); info)
end
# Phase A2: only single-expression bodies. A `:block` body must contain
# exactly one non-`LineNumberNode` arg; multi-statement support comes next.
inline_unwrap_single(body, fname) = body
inline_unwrap_single(body::Expr, fname) = if body.head === :block
    real = filter(a -> !isa(a, LineNumberNode), body.args)
    length(real) == 1 || error(
        "@inline / `!` UDF $fname: only single-expression bodies are supported in this version (multi-statement support coming next)."
    )
    inline_unwrap_single(real[1], fname)
elseif body.head === :return
    inline_unwrap_single(body.args[1], fname)
else
    body
end
# Walk the (un-canonicalised) body AST replacing parameter Symbols with
# their call-site StanExprs. Splat positions referencing the vararg name
# (`args...`) expand into separate arguments at the parent call.
inline_substitute(x, subst, vararg_name) = x
inline_substitute(x::Symbol, subst, vararg_name) = get(subst, x, x)
inline_substitute(x::Expr, subst, vararg_name) = begin
    new_args = []
    for arg in x.args
        if vararg_name !== nothing && Meta.isexpr(arg, :...) &&
            length(arg.args) == 1 && arg.args[1] === vararg_name
            for v in subst[vararg_name]
                push!(new_args, v)
            end
        else
            push!(new_args, inline_substitute(arg, subst, vararg_name))
        end
    end
    Expr(x.head, new_args...)
end
fold_shape_query(x) = x
fold_shape_query(x::StanExpr) = x
forward!(x::BlockExpr; info) = remake(x, forward!(x.args; info)...)
forward!(x::AssignmentExpr{Symbol}; info) = begin
    name, rhs = x.args 
    (name in keys(info) && isa(info, SubModel)) && return nothing 
    rhs = forward!(rhs; info)::Union{StanExpr}
    forward!(remake(x, name, rhs); info)
end
maybe_lazy_size(key::Symbol, i, sizei; info) = sizei
is_simple_size(x::StanExpr) = is_simple_size(expr(x))
is_simple_size(x::CanonicalExpr{<:Union{typeof.((+,-,*,÷))...}}) = all(is_simple_size, x.args)
is_simple_size(x::CanonicalExpr) = false
is_simple_size(x::Symbol) = true
is_simple_size(x::Number) = true
is_simple_size(x) = false#error(typeof(x))
maybe_lazy_size(key::Symbol, i, sizei::StanExpr{<:CanonicalExpr}; info) = if is_simple_size(sizei) || qual(sizei) == :data
    sizei
else
    forward!(canonical(:(dims($key)[$i])); info)
end
forward!(x::AssignmentExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args 
    @assert name ∉ keys(info)
    info[name] = StanExpr(name, remake(type(rhs); value=missing))
    @assert center_type(rhs) != types.anything "tracetype not defined for $name = $(short_expr(rhs))!"
    rv = remake(x, info[name], rhs)
    info[name] = StanExpr(name, remake(type(rhs), [
        maybe_lazy_size(name, i, sizei; info)
        for (i, sizei) in enumerate(stan_size(type(rhs)))
    ]...; value=missing))
    # @info "$x \n=> $rv\n=> $(info[name])"
    rv 
end
forward!(x::AssignmentExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::SamplingExpr{Symbol}; info) = begin
    name, rhs = x.args
    rhs = forward!(rhs; info)::Union{StanExpr,SlicModel}
    forward!(remake(x, name, rhs); info)
end
forward!(x::SamplingExpr{Symbol,<:StanExpr}; info) = begin
    name, rhs = x.args
    if name in keys(info)
        q = stan.qual(info[name])
        q == :data || error("Sampling statement `$name ~ ...` has LHS bound to a $q-qualified value — only data-qualified LHS is supported here (submodel kwargs typically refer to caller-provided data).")
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
forward!(x::SamplingExpr{<:DeclExpr}; info) = begin
    decl, rhs_raw = x.args
    name = decl.args[1]
    isa(name, Symbol) || error(
        "Typed-LHS sampling currently requires a Symbol LHS, got `$name`. ",
        "For more complex LHS shapes, use a bare assignment + sampling pair."
    )
    type_expr = decl.args[2]
    ct, sizes... = if isa(type_expr, CanonicalExprV{:getindex})
        type_expr.args
    else
        (type_expr,)
    end
    isa(ct, Symbol) || error("Typed-LHS sampling: type center must be a Symbol, got `$ct`")
    ct_resolved = gettype(ct)
    sizes_forwarded = isempty(sizes) ? () : Tuple(forward!(collect(sizes); info))
    base_lhs_type = StanType(ct_resolved, sizes_forwarded)
    isa(rhs_raw, CanonicalExpr) || error(
        "Typed-LHS sampling `$name::$(pretty_type_expr(type_expr)) ~ rhs` requires rhs to be a distribution call"
    )
    head_resolved = forward!(head(rhs_raw); info)
    args_resolved = collect(forward!(rhs_raw.args; info))
    kwargs_resolved = forward!(rhs_raw.kwargs; info)
    rhs_canonical = CanonicalExpr(head_resolved, args_resolved...; kwargs_resolved...)
    cv_args = any(stan.cv, args_resolved)
    qual = cv_args ? :quantities : :parameter
    info[name] = StanExpr(name, remake(base_lhs_type; qual, cv=cv_args))
    rhs_stan = StanExpr(rhs_canonical, info[name].type)
    remake(x, info[name], rhs_stan)
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
elseif isa(info, StanModel)
    rhs = forward!(x.args[1]; info)
    forward!(CanonicalExpr(:(=), :MODEL_RV, rhs); info)
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
    @assert isa(obj, StanExpr2{<:types.ntup}) "Trying to access property `$name` of object of type without named properties ($(type(obj)))!"
    # @assert isa(obj, )
    names = keys(obj.type.info.arg_types)
    @assert name in names
    return forward!(CanonicalExpr(:getindex, x.args[1], findfirst(==(name), names)); info)
    error("getproperty fallback: unable to resolve `$obj.$name` (type $(typeof(obj))).")
    stan_expr(remake(x, forward!(x.args; info)...))
end
forward!(x::BracesExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::VectExpr; info) = stan_expr(remake(x, forward!(x.args; info)...))
forward!(x::DeclExpr; info) = begin
    @assert length(x.args) == 2
    lhs, type = x.args
    # @assert isa(lhs, Symbol)
    ct, s... = if isa(type, CanonicalExprV{:getindex})
        type.args
    else 
        (type, )
    end
    # @assert isa(type, CanonicalExprV{:getindex})
    @assert isa(ct, Symbol)
    ct = gettype(ct)
    t = StanType(ct, forward!.(s; info))
    rv = StanExpr(lhs, t)
    isa(lhs, Symbol) || return StanExpr(expr(forward!(lhs; info)), t)
    info[lhs] = rv 
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
    idx_range = forward!(head.args[2]; info)
    body = forward!(body; info)
    pop!(info, idx)
    stan_expr(remake(x, remake(head, idx, idx_range), body))
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
forward!(x::StringExpr; info) = join(map(stan_code, forward!(x.args; info)))

deanon_size(s, x) = s
deanon_size(s::StanExpr, x::CanonicalExpr) = begin
    e = expr(s)
    if isa(e, Symbol)
        m = match(r"^_arg(\d+)$", string(e))
        isnothing(m) && return s
        i = parse(Int, m[1])
        return i <= length(x.args) ? x.args[i] : s
    elseif isa(e, CanonicalExpr)
        new_args = map(a -> deanon_size(a, x), e.args)
        new_args == e.args && return s
        return StanExpr(remake(e, new_args...), type(s))
    end
    s
end
deanon_type(tt::StanType, x::CanonicalExpr) = begin
    sz = stan_size(tt)
    nsz = map(s -> deanon_size(s, x), sz)
    sz == nsz ? tt : StanType(center_type(tt), nsz; [k => v for (k, v) in pairs(info(tt)) if k != :size]...)
end
stan_expr(x::CanonicalExpr) = begin
    tt = deanon_type(tracetype(anon_canonical(x)), x)
    StanExpr(x, remake(tt; qual=maximum(qual, x.args; init=:data), cv=any(cv, x.args) || cv(tt)))
end
stan_expr(x::CanonicalExpr{<:SlicModel}) = head(x)(x.args...;x.kwargs...)

backward!(x; info) = error("backward! not defined for value `$x` of type `$(typeof(x))` — no method matches a more specific signature.")
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
distribute!(x; info) = begin
    _push_expr!(info, x)
    for b in distribution_blocks(x; info)
        push!(block(info, b), x; info)
    end
    _pop_expr!(info)
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
distribution_blocks(x::StanExpr{Symbol}; info) = hasvalue(x) ? (:data,) : tuple()

DeclarativeBlock = Union{DataBlock,ParametersBlock}
ImperativeBlock = Union{FunctionsBlock,TransformedDataBlock,TransformedParametersBlock,ModelBlock,GeneratedQuantitiesBlock}
fetch_data!(;info) = x->fetch_data!(x; info)
fetch_data!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_data!(;info), x)
fetch_data!(x::Union{Function,String}; info) = nothing 
fetch_data!(x::StanExpr{<:Union{Number,String,Missing}}; info) = nothing 
fetch_data!(x::StanType; info) = fetch_data!(stan_size(x); info)
fetch_data!(x::StanExpr{Symbol}; info) = begin
    # fetch_data!(type(x); info)
    # @info x => hasvalue(x) => getvalue(x)
    hasvalue(x) && push!(block(info, :data), x; info)
end
fetch_data!(x::StanExpr{<:Function}; info) = nothing
fetch_data!(x::StanExpr{<:DataType}; info) = nothing
fetch_data!(x::StanExpr{<:CanonicalExpr}; info) = fetch_data!((type(x), expr(x)); info)
fetch_data!(x::CanonicalExpr; info) = begin
    fetch_functions!(x; info=block(info, :functions).content)
    fetch_data!(x.args; info)
end
fetch_data!(x::CanonicalExprV{:kw}; info) = fetch_data!(x.args[2]; info)
fetch_data!(x; info) = error("fetch_data! not defined for value `$x` of type `$(typeof(x))`.")

Base.get!(b::DocumentExpr{<:Any,<:DeclarativeBlock}, k, x) = get!(content(b.args[2]), k, remake(b, b.args[1], x))
Base.push!(b::DocumentExpr{<:Any,<:ImperativeBlock}, x) = push!(content(b.args[2]), remake(b, b.args[1], x))

Base.push!(b::StanBlock, x; info) = error("Block $(typeof(b)) does not know how to handle $(x)!")
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
            StanExpr(Symbol(expr(lhs), "_likelihood"), remake(type(likelihood_rhs); value=missing)),
            likelihood_rhs
        ); info)
        lhs = StanExpr(Symbol(expr(lhs), "_gen"), remake(type(lhs); value=missing))
    end
    # Build a type token carrying the wanted output shape (from lhs, which is
    # either explicitly declared via typed-LHS or inferred by autotype). This
    # token becomes the leading arg to the rng call, letting each `*_rng`
    # @deffun dispatch on the shape.
    lhs_ct = center_type(lhs)
    token = StanExpr(lhs_ct, StanType(types.tokenof{lhs_ct}, stan_size(lhs); value=lhs_ct, qual=:data))
    rng_rhs = rng_expr(token, rhs)
    lhs = StanExpr(expr(lhs), remake(type(rng_rhs); value=missing))
    push!(b, CanonicalExpr(:(=), lhs, rng_rhs); info)
end
    # if hasvalue(x.args[1])
    # push!(b, CanonicalExpr(:(=), rng_lhs(x.args[1]), rng_expr(x.args...)); info)
# end
# likelihood_expr(lhs, rhs) = likelihood_expr(rhs)

function lpxf_expr end
function rng_expr end
function likelihood_expr end

const _LPXF_SUFFIXES = ("_lpdf", "_lpmf", "_lcdf", "_lccdf")
_lpxf_base(name::Symbol) = begin
    s = string(name)
    suffix_idx = findfirst(suf -> endswith(s, suf), _LPXF_SUFFIXES)
    isnothing(suffix_idx) && error(
        "@lpxf/@lhs: `$name` does not end in one of $(_LPXF_SUFFIXES). ",
        "Pass the `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf` function name itself."
    )
    Symbol(s[1:end-length(_LPXF_SUFFIXES[suffix_idx])])
end

lpxf_register(x::LineNumberNode; source=x) = x
lpxf_register(x::Expr; source=LineNumberNode(0, :none)) = if x.head === :block
    Expr(:block, [lpxf_register(arg; source) for arg in x.args]...)
else
    error("@lpxf expects a bare symbol or a `begin … end` block of bare symbols, got `$x`")
end
lpxf_register(x; source=LineNumberNode(0, :none)) = error(
    "@lpxf expects a bare symbol or a `begin … end` block of bare symbols, got `$x`"
)
lpxf_register(name::Symbol; source=LineNumberNode(0, :none)) = begin
    base = _lpxf_base(name)
    rng = Symbol(base, "_rng")
    lpxfs = Symbol(name, "s")
    M = @__MODULE__
    quote
        $source
        function $base end
        function $rng end
        function $lpxfs end
        $M.lpxf_expr(::typeof($base)) = $name
        $M.rng_expr(::typeof($base)) = $rng
        $M.likelihood_expr(::typeof($base)) = $lpxfs
    end
end

include("functions.jl")
include("builtin.jl")
fold_shape_query(x::StanExpr{<:CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:CanonicalExpr{typeof(builtin.dims)}},<:StanExpr{<:Integer}}}}) = begin
    inner = expr(expr(x).args[1])
    arg = inner.args[1]::StanExpr
    i = expr(expr(x).args[2])::Integer
    sz = stan_size(arg)
    if 1 <= i <= length(sz)
        candidate = sz[i]
        qual(candidate) == :data ? fold_shape_query(candidate) : x
    else
        x
    end
end
fetch_data!(x::StanType{<:types.tup}; info) = fetch_data!((stan_size(x), x.info.arg_types); info)
function dummy_likelihood end
function dummy_rng end
lpxf_expr(lhs, rhs::StanExpr) = lpxf_expr(lhs, expr(rhs))
lpxf_expr(lhs, rhs::CanonicalExpr) = stan_call(lpxf_expr(head(rhs)), lhs, rhs.args...)
for lpxf_rhs in (
    :flat_lpdf, :std_normal_lpdf, :normal_lpdf, :student_t_lpdf, :cauchy_lpdf,
    :beta_lpdf, :beta_proportion_lpdf, :lognormal_lpdf, :exponential_lpdf, :gamma_lpdf,
    :inv_gamma_lpdf, :weibull_lpdf, :uniform_lpdf,
    :chi_square_lpdf, :inv_chi_square_lpdf, :scaled_inv_chi_square_lpdf,
    :frechet_lpdf, :rayleigh_lpdf, :loglogistic_lpdf, :von_mises_lpdf,
    :double_exponential_lpdf, :logistic_lpdf, :gumbel_lpdf,
    :skew_normal_lpdf, :exp_mod_normal_lpdf, :skew_double_exponential_lpdf,
    :pareto_lpdf, :pareto_type_2_lpdf, :wiener_lpdf,
    :dirichlet_lpdf, :multi_normal_lpdf, :multi_normal_prec_lpdf, :multi_normal_cholesky_lpdf,
    :multi_gp_lpdf, :multi_gp_cholesky_lpdf,
    :multi_student_t_lpdf, :multi_student_t_cholesky_lpdf,
    :gaussian_dlm_obs_lpdf,
    :lkj_corr_lpdf, :lkj_corr_cholesky_lpdf,
    :wishart_lpdf, :inv_wishart_lpdf, :inv_wishart_cholesky_lpdf, :wishart_cholesky_lpdf,
    :normal_id_glm_lpdf,
    :bernoulli_lpmf, :bernoulli_logit_lpmf, :bernoulli_logit_glm_lpmf,
    :binomial_lpmf, :binomial_logit_lpmf, :beta_binomial_lpmf,
    :neg_binomial_lpdf, :neg_binomial_2_lpmf, :neg_binomial_2_log_lpdf,
    :neg_binomial_2_log_glm_lpmf,
    :poisson_lpmf, :poisson_log_lpmf, :poisson_log_glm_lpmf,
    :discrete_range_lpmf, :hypergeometric_lpmf, :multinomial_lpmf,
    :categorical_lpmf, :categorical_logit_lpmf,
    :ordered_logistic_lpmf,
)
    base_rhs = Symbol(string(lpxf_rhs)[1:end-5])
    rng_rhs = Symbol(base_rhs, "_rng")
    lpxfs_rhs = Symbol(lpxf_rhs, "s")

    @eval lpxf_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxf_rhs
    @eval rng_expr(::typeof(builtin.$base_rhs)) = builtin.$rng_rhs
    @eval likelihood_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxfs_rhs
end

# lpxf_expr(x::CanonicalExpr) = lpxf_expr(head(x))
lpxf_expr(x) = error("$x is missing `lpxf_expr`")
likelihood_expr(lhs, rhs::StanExpr) = likelihood_expr(lhs, expr(rhs))
likelihood_expr(lhs, rhs::CanonicalExpr) = stan_call(likelihood_expr(head(rhs)), lhs, rhs.args...)
likelihood_expr(rhs) = error("$rhs is missing `likelihood_expr`")#dummy_likelihood
# gq `~` synthesis: `rng_expr(token, rhs)` builds either `rng_fn(args...)` (for
# scalar tokens — matches Stan's native rng signatures) or `rng_fn(token, args...)`
# (for sized tokens — dispatched into per-shape `*_rng` @deffun overloads).
# The token is a `tokenof{T}` StanExpr carrying the wanted output shape.
rng_expr(token, rhs::StanExpr) = rng_expr(token, expr(rhs))
# Scalar token path: native Stan rng, no token forwarding.
rng_expr(token::StanExpr2{<:types.tokenof,0}, rhs::CanonicalExpr) = stan_call(rng_expr(head(rhs)), rhs.args...)
# Sized token path: prepend token so per-shape @deffun overloads dispatch.
rng_expr(token::StanExpr2{<:types.tokenof}, rhs::CanonicalExpr) = stan_call(rng_expr(head(rhs)), token, rhs.args...)
rng_expr(x) = error("$x is missing `rng_expr`")#dummy_likelihood


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
autoprint(io, args...) = if maybreak(io)
    bufio, buf = _autoprint_buf(io)
    print(bufio, args...)
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
Base.show(io::StanIO, x::StanModel) = begin
    print(io, maybedoc(get(x.meta, :docstring, "")))
    print(io, Join(blocks(x), "\n"))
end
Base.show(io::StanIO, x::StanExpr) = isa(type(x), StringStanType) ? print(io, expr(x), "::", type(x)) : print(io, expr(x))
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
    is_lxxf = endswith(string(fname), r"_lp[md]f|_l?c?cdf")
    if is_lxxf && length(fargs) > 1
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
            error("if/elseif rendering: else branch is not a BlockExpr (got `$(typeof(e))` from `$x`).")#remake(x, e.args...)
        end)
    end
end
Base.show(io::IO, x::CanonicalExpr{typeof(adjoint)}) = print(io, "(", x.args[1], "')")
Base.show(io::IO, x::CanonicalExpr{typeof(range)}) = autoprint(io, "linspaced_vector(", Join((x.args[end], x.args[1], x.args[2]), ", "), ")")
Base.show(io::IO, x::CanonicalExpr{typeof(getindex)}) = autoprint(io, x.args[1], "[", Join(x.args[2:end], ", "), "]")
Base.show(io::IO, x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = print(io, x.args[1], ".", x.args[2])
for f in (-,+,*,\,/,^,.*,./,<,<=,==,!=,>=,>,&,|)
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
    
"Return the stan code of its first argument (a `StanBlocks.SlicModel` or a `StanBlocks.StanModel`) as a string."
function stan_code end

stan_code(x::StanModel) = begin 
    buf = IOBuffer()
    show(buf, x)
    String(take!(buf))
end
stan_code(x::SlicModel) = stan_code(stan_model(x))
prepare_for_stan(x::Dict) = Dict(key => prepare_for_stan(value) for (key, value) in x)
prepare_for_stan(x::Number) = x
prepare_for_stan(x::AbstractVector{<:Number}) = x
prepare_for_stan(x::AbstractMatrix{<:Number}) = x'
prepare_for_stan(x::NamedTuple) = prepare_for_stan(values(x))
prepare_for_stan(x::Tuple) = prepare_for_stan(Dict(enumerate(x)))
bridgestan_data(x::Dict) = JSON.json(prepare_for_stan(x))
"""
Returns the StanLogDensityProblem (a compiled posterior).
"""
instantiate(x::Union{SlicModel,StanModel}; nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false, kwargs...) = begin
    sc = stan_code(x)
    stan_path = get(kwargs, :path, joinpath("tmp", string(hash(sc)) * ".stan"))
    mkpath(dirname(stan_path))
    if !isfile(stan_path)
        open(stan_path, "w") do fd
            write(fd, sc)
        end
    end
    StanLogDensityProblems.StanProblem(
        stan_path,
        bridgestan_data(stan_data(x));
        nan_on_error,
        make_args,
        warn
    )
end
debug_instantiate(x; kwargs...) = instantiate(x; nan_on_error=false, kwargs...)
passinstantiate(x; kwargs...) = (instantiate(x; kwargs...); x)
stan_data(x::SlicModel) = stan_data(stan_model(x))
stan_data(x::StanModel) = Dict([
    key=>getvalue(value) for (key, value) in pairs(content(block(x, :data)))
    if !always_inline(value)
])

"StanModels can update the associated data (via `new_model = model(;x=new_x)`)."
(x::StanModel)(;kwargs...) = begin
    xkwargs = Dict{Symbol,Any}(pairs(kwargs))
    for (key, value) in pairs(kwargs)
        if isa(value, AbstractVector) 
            xkwargs[Symbol(key, "_n")] = length(value)
        elseif isa(value, AbstractMatrix) 
            xkwargs[Symbol(key, "_m")], xkwargs[Symbol(key, "_n")] = size(value)
        end
    end
    StanModel(x.meta, x.vars, merge(x.blocks, (;data=StanBlock(:data,OrderedDict([
        key=>StanExpr(expr(x), remake(type(x); value=get(xkwargs, key, getvalue(x))))
        for (key, x) in pairs(block(x, :data).content)
    ])))))
end
slic_expr(x::Expr) = x

# Macro names handled structurally by SLIC's own parser. Macrocalls with these
# heads are preserved verbatim during `slic_macroexpand`; everything else is
# expanded against the user's module so standard Julia macros (`@views`, `@.`,
# `@inbounds`, user-defined macros) work transparently inside `@slic` /
# `@deffun` bodies.
const _SLIC_RESERVED_MACROS = (Symbol("@doc"), Symbol("@lpxf"), Symbol("@lhs"), Symbol("@inline"))

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

include("test.jl")

end
"""
Defines `SlicModel`s (see `test/slic.jl` for usage examples).

The defining module is captured automatically via `__module__`, so that `@deffun` functions
defined in the same module (e.g. a package extension) are found during symbol resolution.
"""
macro slic(model)
    stan.SlicModel(stan.slic_macroexpand(__module__, model), Dict(), __module__)
end
macro slic(data, model)
    mod = @__MODULE__
    qmodel = Meta.quot(stan.slic_macroexpand(__module__, model))
    esc(:($mod.stan.SlicModel($qmodel, $data, $(__module__))))
end
"""
Utility macro to define function signatures (see `src/slic_stan/builtin.jl` for usage examples).

**Note:**

This macro is mainly useful for bulk built-in function signature definitions. 
StanBlocks.jl users should generally prefer using @deffun.
"""
macro defsig(x)
    esc(stan.defsig(x; source=__source__))
end
"""
    @deffun function_definition

Define a Stan-compatible function with type inference and code generation.

Parses a Julia-style function definition (with type-annotated arguments and return type),
generates the corresponding Stan function, and registers type-inference signatures so the
transpiler can propagate types through calls to this function.

For functions ending in `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf`, the return type is automatically
set to `real` and companion `_lpdfs`/`_rng` stubs are generated for use in `generated_quantities`.

# Example

```julia
@deffun garch11_lpdf(y::vector[T], mu::real, alpha0::real, alpha1::real, beta1::real)::real = begin
    sigma2 = alpha0
    rv = 0.
    for t in 1:T
        rv += normal_lpdf(y[t], mu, sqrt(sigma2))
        sigma2 = alpha0 + alpha1 * square(y[t] - mu) + beta1 * sigma2
    end
    return rv
end
```

See `src/slic_stan/builtin.jl` for many more examples.
"""
macro deffun(x)
    esc(stan.deffun(stan.slic_macroexpand(__module__, x); source=__source__))
end
"""
    @lpxf foo_lpdf
    @lpxf begin foo_lpdf; bar_lpmf end

Register the three SLIC dispatch hooks (`lpxf_expr`, `rng_expr`, `likelihood_expr`)
for one or more user-defined log-probability functions.

The argument(s) must be bare symbols ending in `_lpdf`, `_lpmf`, `_lcdf`, or `_lccdf`.
For each `foo_lpdf` (or `_lpmf`/etc.), the macro emits the registrations:

    StanBlocks.stan.lpxf_expr(::typeof(foo))       = foo_lpdf
    StanBlocks.stan.rng_expr(::typeof(foo))        = foo_rng
    StanBlocks.stan.likelihood_expr(::typeof(foo)) = foo_lpdfs

The companion `foo_rng` and `foo_lpdfs` (resp. `_lpmfs`/`_lcdfs`/`_lccdfs`) names
must already exist when the registrations execute. This macro does not parse
function bodies and does not wrap `@deffun`.
"""
macro lpxf(x)
    stan.lpxf_register(x; source=__source__)
end

"""
    @lhs foo_lpdf(y::T, args...) = body

Inside a `@deffun` block, opt this method into base-level LHS inference. Without
`@lhs`, only the `_lpdf`-keyed tracetype is registered (so the method dispatches
when called explicitly), but `lhs ~ foo(args...)` cannot trace because the base
`foo` has no tracetype keyed on its argument signature. `@lhs` registers
`tracetype(::CanonicalExpr{<:typeof(foo), <:Tuple{lhs_type[2:end]...}})` so the
sampling form works.

Compose with `@lpxf` (any order — `@lhs @lpxf …` or `@lpxf @lhs …`) to also
register the dispatch hooks for `foo`/`foo_rng`/`foo_lpdfs`.

Standalone `@lhs` (outside `@deffun`) is not supported and errors immediately.
"""
macro lhs(x)
    error("@lhs may only appear inside a @deffun block")
end
const stan_model = stan.stan_model
const stan_code = stan.stan_code
const stan_data = stan.stan_data
const stan_instantiate = stan.instantiate
const StanModel = stan.StanModel
const SlicModel = stan.SlicModel