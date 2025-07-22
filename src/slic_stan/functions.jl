module types
    abstract type anything end
    abstract type matrix <: anything end
    abstract type square_matrix <: matrix end
    abstract type cov_matrix <: square_matrix end
    abstract type corr_matrix <: square_matrix end
    abstract type cholesky_factor_cov <: square_matrix end
    abstract type cholesky_factor_corr <: square_matrix end
    abstract type any_vector <: anything end
    abstract type vector <: any_vector end
    abstract type row_vector <: any_vector end
    abstract type simplex <: vector end
    abstract type complex <: anything end
    abstract type real <: complex end
    abstract type int <: real end
    abstract type func{T} <: anything end 
    abstract type tup <: anything end
    abstract type ntup <: tup end
end
function stan_code end
Base.show(io::IO, ::Type{T}) where {T<:types.anything} = print(io, T.name.name)
Base.show(io::IO, ::Type{T}) where {T<:types.func} = print(io, "func")#.parameters[1].name.name)
Base.show(io::IO, ::Type{<:types.tup}) = print(io, "tuple(...)")
r_ndim(::Type{types.anything}) = 0
r_ndim(::Type{types.matrix}) = 2
r_ndim(::Type{<:types.square_matrix}) = 1
r_ndim(::Type{<:types.any_vector}) = 1
r_ndim(::Type{<:types.complex}) = 0
r_ndim(::Type{<:types.func}) = 0
r_ndim(::Type{<:types.tup}) = 0
r_ndim(::StanType{T}) where {T} = r_ndim(T)
l_ndim(x::StanType) = stan_ndim(x) - r_ndim(x)
lr_size(x::StanType) = stan_size(x, 1:l_ndim(x)), stan_size(x, 1+l_ndim(x):stan_ndim(x))
canonical(x::CanonicalExpr{<:StanExpr2{<:types.func}}) = CanonicalExpr(type(x.head).info.value, x.args...; x.kwargs...)
backward!(x::StanExpr2{<:types.func}; info) = x
# fetch_data!(::StanExpr2{<:types.func}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{<:types.func}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{types.func}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{types.func, 0}}; info) = nothing
# fetch_data!(::StanExpr{Symbol, StanType{<:types.func, 0}}; info) = nothing

short_expr(x::Symbol) = x
short_expr(x::StanExpr2{types.anything}) = StanExpr(short_expr(expr(x)), type(x))
short_expr(x::StanExpr) = StanExpr("", StringStanType(sigtype(x)))
short_expr(x::CanonicalExpr) = CanonicalExpr(head(x), short_expr.(x.args)...)
tracetype(x::CanonicalExpr) = begin
    map(x.args) do arg 
        # tracetype not defined for $(head(expr(arg)))$(typeof.(type.(expr(arg).args))) (nargs = $(length(expr(arg).args))).
        center_type(type(arg)) == types.anything && error("""
            `tracetype` not defined for $(short_expr(arg))!
            This is only allowed if this return value does not get used in another expression,
            but it is used in $(short_expr(x)) (nargs = $(length(x.args))).
            """)
                # but needed in $(head(x))$(typeof.(type.(x.args))) (nargs = $(length(x.args))).
    end
    StanType(types.anything)
end
tracetype(x::CanonicalExpr{typeof(==)}) = StanType(types.int)
tracetype(x::CanonicalExpr{<:Union{typeof.((+, -, ^, *, /))...}}) = if length(x.args) > 2
    f = head(x)
    tracetype(CanonicalExpr(f, x.args[1], stan_expr(CanonicalExpr(f, x.args[2:end]...))))
else 
    error("tracetype not defined for $(short_expr(x))!")
    StanType(types.anything)
end
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 1),))))
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 1),))), x.args[3])
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], x.args[2], StanExpr(missing, StanType(types.int, (stan_size(x.args[1], 2),))))
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:types.tup}, <:StanExpr2{<:types.int}}}) = x.args[1].type.info.arg_types[x.args[2].type.info.value]

tracetype(x::CanonicalExpr{Colon}) = StanType(types.int, (stan_call(+,stan_expr(1,1),stan_call(-,x.args[2],x.args[1])), ))
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{<:types.matrix}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.vector, (stan_size(x.args[1], 1),))
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{types.real,2}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.real, (stan_size(x.args[1], 1),))
tracetype(x::BracesExpr) = StanType(types.real, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::VectExpr) = StanType(types.vector, (stan_expr(length(x.args),length(x.args)),))
tracetype(x::TupleExpr) = StanType(types.tup; arg_types=map(type, x.args))
tracetype(x::KwExpr) = type(x.args[2])
# tracetype(x::NamedTupleExpr) = StanType(types.ntup; arg_types=map(type, x.args))
tracetype(x::NamedTupleExpr) = StanType(types.ntup; arg_types=(;[
    kw.args[1]=>type(kw.args[2]) for kw in map(expr, x.args)
]...))
tracetype(x::DeclExpr) = x.args[1].type
tracetype(x::ForExpr) = StanType(types.anything)
tracetype(x::WhileExpr) = StanType(types.anything)
tracetype(x::IfExpr) = StanType(types.anything)
tracetype(x::ElseIfExpr) = StanType(types.anything)
tracetype(x::BlockExpr) = error(dump(x))#tracetype(expr(x.args[end]))

autokwargs(::CanonicalExpr) = (;)
autotype(x::StanExpr) = autotype(type(x); merge(autokwargs(expr(x)), expr(x).kwargs)...)
autotype(x::StanType; kwargs...) = begin 
    ct = get(kwargs, :type, center_type(x))
    nsize = [
        getindex(kwargs, key)
        for key in (:m, :n, :o) if key in keys(kwargs)
    ]
    size = length(nsize) > 0 ? (nsize..., ) : get(kwargs, :size, stan_size(x))
    (ct == types.anything) && (ct = [types.real, types.vector, types.matrix][1+length(size)])
    cons = (;[
        key=>getindex(kwargs, key)
        for key in (:lower, :upper, :offset, :multiplier) if key in keys(kwargs)
    ]...)
    StanType(ct, size; cons...)
end

struct StanFunction3
    docstring::AbstractString
    rv_type::StanType
    parent::Function
    args::NamedTuple
    body::Vector
end
Base.show(io::IO, f::StanFunction3) = autoprint(
    io,
    f.docstring,
    sigtype(f.rv_type), " ", func_name(f.parent, f.args), "(", func_args(f.args), ")",
    StanBlock(Symbol(), f.body)
)

begin
    xiscall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
    xassign(args...) = Expr(:(=), args...)
    xtuple(args...) = Expr(:tuple, args...)
    xref(args...) = Expr(:ref, args...)
    xtyped(args...) = Expr(:(::), args...)
    xpair(args...) = Expr(:call, :(=>), args...)
    xvect(args...) = Expr(:vect, args...)
    xstring(args...) = Expr(:string, args...)
    ensure_xassign(x, default=missing) = Meta.isexpr(x, :(=)) ? x : xassign(x, default)
    ensure_xtuple(x) = Meta.isexpr(x, :tuple) ? x : xtuple(x)
    ensure_xref(x) = Meta.isexpr(x, :ref) ? x : xref(x)
    ensure_xtyped(x, default=:anything) = if Meta.isexpr(x, :(::))
        if length(x.args) == 2
            x
        else
            xtyped(gensym("_"), x.args[1])
        end
    else
        xtyped(x, default)
    end
    ensure_xpair(x, default) = xiscall(x, :(=>)) ? x : xpair(x, default)
    ensure_xvect(x) = Meta.isexpr(x, :vect) ? x : xvect(x)
    ensure_xreturn(x::Expr) = if x.head in (:block, :macrocall)
        Expr(x.head, x.args[1:end-1]..., ensure_xreturn(x.args[end]))
    elseif x.head == :if
        Expr(x.head, x.args[1], ensure_xreturn.(x.args[2:end])...)
    elseif x.head == :return
        x
    else
        Expr(:return, x)
    end
    ensure_xreturn(x) = Expr(:return, x)

    gettype(ct::Symbol) = getproperty(types, ct)
    gettype(ct::Expr) = begin
        return :($types.func{$ct})
    end

    xsig_type(x::Expr) = begin 
        @assert x.head == :ref
        ct, size... = x.args
        ct = gettype(ct)
        ndims = length(size)
        if ct == types.anything && ndims == 0
            :(<:$StanExpr2{<:$ct})
        else
            :(<:$StanExpr2{<:$ct, $ndims})
        end
    end
    xsig_expr(x::Expr) = begin 
        @assert x.head == :ref x
        ct, size... = x.args
        ct = gettype(ct)
        size = xtuple([:($forward!($arg; info)) for arg in canonical.(size)]...)
        :($StanType($ct, $size))
    end

    defsig(x::LineNumberNode) = x
    defsig(x::Expr) = if x.head == :block
        Expr(:block, defsig.(x.args)...)
    else
        @assert xiscall(x, :(=>))
        _, ftype, rhs = x.args
        @assert Meta.isexpr(rhs, :block)
        Expr(:block, map(sig->defsig(ftype, sig), rhs.args)...)
    end
    defsig(ftype, x::LineNumberNode) = x
    defsig(ftype, sig::Expr) = begin 
        @assert xiscall(sig, :(=>))
        _, lhs, rv = sig.args
        lhs = ensure_xref.(ensure_xtuple(lhs).args)
        rv = ensure_xref(rv)
        arg_types = lhs
        lhs_type = xsig_type.(lhs)
        dim_names = OrderedSet()
        for arg_type in arg_types
            for dim_name in arg_type.args[2:end]
                isa(dim_name, Symbol) || continue
                push!(dim_names, dim_name)
            end
        end

        xexpr = :(x::$CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, [
            xassign(xtuple(ensure_xlhs.(lhsi.args[2:end])...), :(stan_size(x.args[$i])))
            for (i, lhsi) in enumerate(lhs)
        ]..., :(info = (;$(dim_names...),)), xsig_expr(rv))
        :($stan.tracetype($xexpr) = $xbody)
    end
    funbody(x::Expr) = begin 
        @assert x.head == :block x
        funbody(x.args)
    end
    funbody(x::AbstractVector) = join(map(funbody, x), "\n")
    funbody(x::LineNumberNode) = ""
    funbody(x::String) = strip(x)
    # funbody_expr(x) = if x.head
    make_stan_type(x::Symbol) = make_stan_type(xref(x))
    make_stan_type(x::Expr) = begin 
        @assert x.head == :ref x
        ct, size... = x.args
        ct = getproperty(types, ct)
        StanType(ct, StanExpr.((size..., ), Ref(StanType(types.int))))
    end
    sigtype(x::Symbol) = sigtype(xref(x))
    sigtype(x::Expr) = begin 
        @assert x.head == :ref x
        ct, size... = x.args
        ct = getproperty(types, ct)
        l = length(size) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array[", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigtype(x::Type) = x
    sigtype(x::Type{types.cholesky_factor_corr}) = types.matrix
    sigtype(x::Type{<:types.vector}) = types.vector
    sigtype(x::StanExpr) = sigtype(x.type)
    sigtype(x::StanType) = begin 
        ct = center_type(x)
        ct == types.anything && @error("Stan compilation will fail: `sigtype($x)` == anything")
        # @assert ct != types.anything
        l = stan_ndim(x) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array[", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigtype(x::StanType{<:types.tup}) = begin 
        io = IOBuffer()
        stan_ndim(x) > 0 && print(io, "array[", join(fill("", stan_ndim(x)), ", "), "] ")
        print(io, "tuple(", join(map(sigtype, x.info.arg_types), ", "), ")")
        String(take!(io))
    end
    sigarg(x, name::Symbol) = error()#sigtype(x) * " $name"
    sigarg(::StanExpr2{<:types.func}, ::Symbol) = error()#nothing
    sigarg(x::Tuple, name::Symbol) = error()#join(ntuple(i->sigarg(x[i], Symbol(name, i)), length(x)), ", ")
    always_inline(x) = false
    always_inline(::StanExpr2{<:types.func}) = true
    # sigargs(x::Tuple) = filter(!isnothing, map(sigargs, x))
    sigarg(x::Expr) = begin 
        @assert x.head == :(::)
        "$(sigtype(x.args[2])) $(x.args[1])"
    end
    # fname(x) = string(x)
    # fname(::typeof(>=)) = "gte" 
    # stan_call(;kwargs...) = x->stan_call(x, ;kwargs...)
    # stan_call(x::Expr; kwargs...) = stan_expr()
    expr_replace(x; kwargs...) = get(kwargs, x, x) 
    expr_replace(x::Expr; kwargs...) = Expr(x.head, expr_replace.(x.args; kwargs...)...)

    ensure_xlhs(arg::Symbol) = arg
    ensure_xlhs(::Expr) = Symbol("_")

    hasvararg(args) = length(args) > 0 && Meta.isexpr(args[end], :(...))
    maybedoc(x::AbstractString) = length(strip(x)) == 0 ? "" : strip(replace("\n" * strip(x), "\n"=>"\n// ")) * "\n"
    forward_return!(x; info) = begin
        info = OrderedDict{Symbol,Any}(pairs(info))
        forward!(x; info)
        info[RV_NAME]
    end
    deffun(x::LineNumberNode; kwargs...) = x
    deffun(x::Expr; docstring="") = if x.head == :block
        Expr(:block, deffun.(x.args; docstring)...)
    elseif x.head == :macrocall
        @assert x.args[1] == GlobalRef(Core, Symbol("@doc"))
        # @assert x.args[3] isa String
        deffun(x.args[4]; docstring=:($maybedoc($(x.args[3]))))
    else
        # @assert x.head == :(=)
        fsig, body = ensure_xassign(x).args
        fcall, rv = ensure_xtyped(fsig).args
        @assert Meta.isexpr(fcall, :call)
        f, args... = fcall.args
        is_lpxf = endswith(string(f), r"_lp[md]f")
        is_lpxf && (rv = :real)
        ftype = :(typeof($f))
        args, vararg = if hasvararg(args)
            args[1:end-1], args[end]
        else
            args, nothing
        end
        args = ensure_xtyped.(args, :anything)
        arg_names = map(arg->arg.args[1], args)
        sig_names = copy(arg_names)
        arg_types = map(arg->ensure_xref(arg.args[2]), args)
        lhs_type = xsig_type.(arg_types)
        if !isnothing(vararg)
            push!(sig_names, vararg.args[1])
            # push!(arg_types, vararg.args[1]) 
            push!(lhs_type, :(Vararg{Any}))
        end

        fun_sizes = OrderedDict()
        for (arg_name, arg_type) in zip(arg_names, arg_types)
            for (i, dim_name) in enumerate(arg_type.args[2:end])
                isa(dim_name, Symbol) || continue
                dim_name == :(_) && continue
                fun_sizes[dim_name] = "int $dim_name = dims($arg_name)[$i];"
            end
        end
        deconstruct = Expr(:block, 
            xassign(xtuple(arg_names..., (isnothing(vararg) ? () : (vararg,))...), :(x.args)), 
            [
                xassign(xtuple(ensure_xlhs.(args_type.args[2:end])...), :($stan_size($args_name)))
                for (args_name, args_type) in zip(arg_names, arg_types)
            ]..., 
            :(info = (;$(sig_names...), $(keys(fun_sizes)...),))
        )
        anon_deconstruct = Expr(
            :block, 
            deconstruct.args..., 
            :(info = $anon_info(info)),
            :((;$(sig_names...), $(keys(fun_sizes)...),) = info),
            :(info = $OrderedDict{Symbol,Any}(pairs(info)))
        )

        stmts = []
        rv_expr = xsig_expr(ensure_xref(rv))
        stan_fundef = nothing
        if !ismissing(body)
            @assert Meta.isexpr(body, :block)
            body = ensure_xreturn(body)
            sig_rv = if rv == :anything
                rv_expr = :($forward_return!($(canonical(body)); info).type)
                :($rv_expr)
            else
                make_stan_type(rv)
            end
            stan_fundef = :($StanFunction3(
                $docstring,
                $sig_rv,
                $f,
                (;$(sig_names...), ),vcat(
                    $(collect(values(fun_sizes))),
                    $forward!($(canonical(body)); info)
                )
            ))
        end

        xexpr = :(x::$CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        isa(f, Symbol) && push!(stmts, :(function $f end))
        push!(stmts, quote
            $stan.tracetype($xexpr) = $(Expr(:block, deconstruct, rv_expr))
        end)
        if !ismissing(body)
            push!(stmts, :($stan.fundef($xexpr) = $(Expr(:block, anon_deconstruct, stan_fundef))))
        end
        isa(f, Symbol) || return Expr(:block, stmts...)
        if is_lpxf
            base_f = Symbol(string(f)[1:end-length("_lpdf")])
            rng_f = Symbol(base_f, "_rng")
            lpdfs_f = Symbol(f, "s")
            base_ftype = :(typeof($base_f))
            base_xexpr = :(_x::$CanonicalExpr{<:$base_ftype,<:Tuple{$(lhs_type[2:end]...)}})
            dummy1 = StanExpr(missing, StanType(getproperty(types, arg_types[1].args[1]), ntuple(i->StanExpr(missing, StanType(types.int)), length(arg_types[1].args)-1)))
            reconstruct = :(x = $CanonicalExpr($f, $dummy1, _x.args...))
            push!(stmts, quote
                function $base_f end
                function $rng_f end
                function $lpdfs_f end
                $stan.tracetype($base_xexpr) = $(Expr(:block, reconstruct, deconstruct, xsig_expr(ensure_xref(arg_types[1]))))
                $stan.lpxf_expr(::typeof($base_f)) = $f
                $stan.rng_expr(::typeof($base_f)) = $rng_f
                $stan.likelihood_expr(::typeof($base_f)) = $lpdfs_f
                $stan.fundef($base_xexpr) = nothing
            end)
            # if !ismissing(body)
            #     push!(stmts, :($stan.fundef($base_xexpr) = nothing))#$(Expr(:block, reconstruct, anon_deconstruct, stan_fundef))))
            # end
        end
        Expr(:block, stmts...)
    end
    
end

macro defsig(x)
    esc(defsig(x))
end
macro deffun(x)
    esc(deffun(x))
end

fundef(x) = begin
    # @assert isa(x, CanonicalExpr)
    # if head(x) isa Function && parentmodule(head(x)) ∉ (builtin, Base)
    #     @error "Stan compilation will fail: no function definition found for $x."
    # end
    nothing
end
sig_expr(x) = x
sig_expr(x::Union{Tuple,NamedTuple,Vector}) = map(sig_expr, x)
sig_expr(x::CanonicalExpr) = remake(x, sig_expr(x.args)...)
sig_expr(x::StanExpr) = StanExpr(:_, sig_expr(type(x)))
sig_expr(x::StanType) = StanType(center_type(x), sig_expr(stan_size(x)))
sig_expr(x::StanType{<:types.tup}) = StanType(center_type(x), sig_expr(stan_size(x)); arg_types=sig_expr(info(x).arg_types))
sig_expr(x::StanType{<:types.func}) = StanType(center_type(x), sig_expr(stan_size(x)); value=sig_expr(info(x).value))
fetch_functions!(x::CanonicalExpr; info) = begin 
    sx = sig_expr(x)
    sx in keys(info) && return
    info[sx] = fundef(sx)
    isnothing(info[sx]) && return
    fetch_subfunctions!(info[sx].body; info)
end
fetch_functions!(x::SamplingExpr; info) = begin 
    lhs, rhs = x.args
    fetch_functions!(expr(lpxf_expr(lhs, rhs)); info)
    if qual(lhs) == :data || lqual(lhs) == :undefined
        fetch_functions!(expr(likelihood_expr(lhs, rhs)); info)
        fetch_functions!(expr(rng_expr(lhs, rhs)); info)
    end
end
fetch_subfunctions!(;info) = x->fetch_subfunctions!(x; info)
fetch_subfunctions!(x; info) = nothing
fetch_subfunctions!(x::Union{Tuple,NamedTuple,Vector}; info) = map(fetch_subfunctions!(;info), x)
fetch_subfunctions!(x::StanExpr; info) = fetch_subfunctions!((expr(x), type(x)); info)
fetch_subfunctions!(x::StanType; info) = fetch_subfunctions!((stan_size(x), x.info); info)
fetch_subfunctions!(x::CanonicalExpr; info) = begin 
    fetch_functions!(x; info)
    fetch_subfunctions!((x.args, x.kwargs); info)
end
anon_info(x::NamedTuple) = (;[
    key=>anon_expr(key, value)
    for (key, value) in pairs(x)
]...)
anon_expr(key, x) = error(typeof(x))
anon_expr(key, x::Tuple) = begin
    idxs = cumsum(map(!always_inline, x))
    ([
        anon_expr(Symbol(key, idx), xi)
        for (idx, xi) in zip(idxs, x)
    ]...,)
end
anon_expr(key, x::StanExpr) = StanExpr(key, StanType(center_type(x), ([
    StanExpr("dims($key)[$i]", StanType(types.int))
    for (i, s) in enumerate(stan_size(x))
]...,)))
anon_expr(key, x::StanExpr2{<:types.func}) = StanExpr(type(x).info.value, type(x))
anon_expr(key, x::StanExpr2{<:types.tup}) = begin
    StanExpr(key, StanType(center_type(x); arg_types=([
        anon_expr(Symbol(key, ".", i), StanExpr(:_, arg_type)).type
        for (i, arg_type) in enumerate(x.type.info.arg_types)
    ]...,)))
end
anon_expr(key, x::StanExpr2{<:types.ntup}) = begin
    # arg_types = x.type.info.arg_types
    StanExpr(key, StanType(center_type(x); arg_types=(;[
        name=>anon_expr(Symbol(key, ".", i), StanExpr(:_, arg_type)).type
        for (i, (name, arg_type)) in enumerate(pairs(x.type.info.arg_types))
    ]...,)))
end
func_name(x::Symbol) = x
func_name(x::QuoteNode) = func_name(x.value)
func_name(x::Expr) = if x.head == :.
    func_name(x.args[end])
else
    error(dump(x))
end
func_name(f, args) = begin
    rv = join(vcat(func_name(f), func_name(args)...), "_")
    suffix_idxs = findfirst(r"_(rng|u?lp(m|d)fs?)_", rv)
    if isnothing(suffix_idxs) 
        rv
    else
        suffix = rv[suffix_idxs[1]:suffix_idxs[end]-1]
        base = rv[1:suffix_idxs[1]-1] * rv[suffix_idxs[end]:end]
        base * suffix
        # if isnothing(match(r"u?lp(m|d)f$", suffix))
        #     base * suffix
        # else
        #     replace(base, r"(_u?lp(m|d)f)+$"=>"") * suffix
        # end
    end
end
func_name(args::NamedTuple) = func_name(values(args))
func_name(args::Tuple) = mapreduce(func_name, vcat, args; init=[])
func_name(x) = []
func_name(x::StanExpr) = always_inline(x) ? [func_name(type(x).info.value)] : []
func_name(x::Function) = string(x)
func_name(::typeof(>=)) = "gte"
func_name(::typeof(>)) = "gt"
func_name(::typeof(==)) = "eq"
func_name(::typeof(<=)) = "lte"
func_name(::typeof(<)) = "lt"
func_name(::typeof(+)) = "add"
func_name(::typeof(-)) = "sub"
func_name(::typeof(*)) = "mul"
func_name(::typeof(/)) = "div"
func_args(args::NamedTuple) = Join(mapreduce(func_args, vcat, pairs(args); init=[]), ", ")
func_args(arg::Pair) = func_args(arg...)
func_args(name, ::StanExpr2{<:types.func}) = []
func_args(name, value::StanExpr2) = sigtype(value) * " $name"
func_args(name, value::Tuple) = reduce(vcat, [
    func_args(Symbol(name, i), vali)
    for (i, vali) in enumerate(filter(!always_inline, value))
]; init=[])