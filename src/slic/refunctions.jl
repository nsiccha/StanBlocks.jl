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
end
Base.show(io::IO, ::Type{T}) where {T<:types.anything} = print(io, T.name.name)#.parameters[1].name.name)
# Base.show(io::IO, ::Type{types.anything}) = print(io, "anything")
# Base.show(io::IO, ::Type{types.matrix}) = print(io, "matrix")
# Base.show(io::IO, ::Type{types.cov_matrix}) = print(io, "cov_matrix")
# Base.show(io::IO, ::Type{types.corr_matrix}) = print(io, "corr_matrix")
# Base.show(io::IO, ::Type{types.cholesky_factor_cov}) = print(io, "cholesky_factor_cov")
# Base.show(io::IO, ::Type{types.cholesky_factor_corr}) = print(io, "cholesky_factor_corr")
# Base.show(io::IO, ::Type{types.vector}) = print(io, "vector")
# Base.show(io::IO, ::Type{types.simplex}) = print(io, "simplex")
# Base.show(io::IO, ::Type{types.row_vector}) = print(io, "row_vector")
# Base.show(io::IO, ::Type{types.int}) = print(io, "int")
# Base.show(io::IO, ::Type{types.real}) = print(io, "real")
# Base.show(io::IO, ::Type{types.complex}) = print(io, "complex")
r_ndim(::Type{types.anything}) = 0
r_ndim(::Type{types.matrix}) = 2
r_ndim(::Type{<:types.square_matrix}) = 1
r_ndim(::Type{<:types.any_vector}) = 1
r_ndim(::Type{<:types.complex}) = 0
r_ndim(::StanType{T}) where {T} = r_ndim(T)
l_ndim(x::StanType) = length(x.size) - r_ndim(x)
lr_size(x::StanType) = x.size[1:l_ndim(x)], x.size[1+l_ndim(x):end]


tracetype(x::CanonicalExpr) = begin
    map(x.args) do arg 
        # tracetype not defined for $(head(expr(arg)))$(typeof.(type.(expr(arg).args))) (nargs = $(length(expr(arg).args))).
        center_type(type(arg)) == types.anything && error("""
            `tracetype` not defined for $(arg)!
            This is only allowed if this return value does not get used in another expression,
            but it is used in $x (nargs = $(length(x.args))).
            """)
                # but needed in $(head(x))$(typeof.(type.(x.args))) (nargs = $(length(x.args))).
    end
    StanType(types.anything)
end
tracetype(x::CanonicalExpr{<:Union{typeof.((+, -, ^, *, /))...}}) = if length(x.args) > 2
    f = head(x)
    tracetype(CanonicalExpr(f, x.args[1], stan_expr(CanonicalExpr(f, x.args[2:end]...))))
else 
    error("tracetype not defined for $(x)!")
    StanType(types.anything)
end
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (type(x.args[1]).size[1],))))
)
tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:Any,<:Colon,<:Any}}) = tracetype(
    CanonicalExpr(head(x), x.args[1], StanExpr(missing, StanType(types.int, (type(x.args[1]).size[1],))), x.args[3])
)
tracetype(x::CanonicalExpr{Colon}) = StanType(types.int, (stan_call(+,stan_expr(1,1),stan_call(-,x.args[2],x.args[1])), ))
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{<:types.matrix}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.vector, (type(x.args[1]).size[1],))
# tracetype(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:Any,<:StanType{types.real,2}},<:Colon,<:StanExpr{<:Any,<:StanType{<:types.int,0}}}}) = StanType(types.real, (type(x.args[1]).size[1],))
tracetype(x::BracesExpr) = StanType(types.real, (stan_expr(length(x.args),length(x.args)),))


autokwargs(::CanonicalExpr) = (;)
autotype(x::StanExpr) = autotype(type(x); merge(autokwargs(expr(x)), expr(x).kwargs)...)
autotype(x::StanType; kwargs...) = begin 
    ct = get(kwargs, :type, center_type(x))
    nsize = [
        getindex(kwargs, key)
        for key in (:m, :n, :o) if key in keys(kwargs)
    ]
    size = length(nsize) > 0 ? (nsize..., ) : get(kwargs, :size, x.size)
    (ct == types.anything) && (ct = [types.real, types.vector, types.matrix][1+length(size)])
    cons = (;[
        key=>getindex(kwargs, key)
        for key in (:lower, :upper, :offset, :multiplier) if key in keys(kwargs)
    ]...)
    StanType(ct, size; cons...)
end
# abstract type stan_typ
# module SlicMeta
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
    ensure_xtyped(x, default=:anything) = Meta.isexpr(x, :(::)) ? x : xtyped(x, default)
    ensure_xpair(x, default) = xiscall(x, :(=>)) ? x : xpair(x, default)
    ensure_xvect(x) = Meta.isexpr(x, :vect) ? x : xvect(x)

    xsig_type(x::Expr; mod=@__MODULE__) = begin 
        @assert x.head == :ref
        ct, size... = x.args
        ct = getproperty(types, ct)
        ndims = length(size)
        if ct == types.anything && ndims == 0
            :(<:$mod.StanExpr2{<:$ct})
        else
            :(<:$mod.StanExpr2{<:$ct, $ndims})
        end
    end
    xsig_expr(x::Expr; mod=@__MODULE__) = begin 
        @assert x.head == :ref x
        ct, size... = x.args
        ct = getproperty(types, ct)
        size = xtuple([:($mod.forward!($arg; info)) for arg in canonical.(size)]...)
        :($mod.StanType($ct, $size))
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
    defsig(ftype, sig::Expr; mod=@__MODULE__) = begin 
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

        xexpr = :(x::$mod.CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, [
            xassign(xtuple(ensure_xlhs.(lhsi.args[2:end])...), :(x.args[$i].type.size))
            for (i, lhsi) in enumerate(lhs)
        ]..., :(info = (;$(dim_names...),)), xsig_expr(rv))
        :($mod.tracetype($xexpr) = $xbody)
    end
    funbody(x::Expr) = begin 
        @assert x.head == :block x
        funbody(x.args)
    end
    funbody(x::AbstractVector) = join(map(funbody, x), "\n")
    funbody(x::LineNumberNode) = ""
    funbody(x::String) = x
    sigtype(x::Symbol) = sigtype(xref(x))
    sigtype(x::Expr; mod=@__MODULE__) = begin 
        @assert x.head == :ref x
        ct, size... = x.args
        ct = getproperty(types, ct)
        l = length(size) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array [", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigtype(x::Type) = x
    sigtype(x::Type{types.cholesky_factor_corr}) = types.matrix
    sigtype(x::Type{<:types.vector}) = types.vector
    sigtype(x::StanExpr) = sigtype(x.type)
    sigtype(x::StanType) = begin 
        ct = center_type(x)
        @assert ct != types.anything
        l = length(x.size) - r_ndim(ct)
        io = IOBuffer()
        l > 0 && print(io, "array [", join(fill("", l), ", "), "] ")
        print(io, sigtype(ct))
        String(take!(io))
    end
    sigarg(x::Expr) = begin 
        @assert x.head == :(::)
        "$(sigtype(x.args[2])) $(x.args[1])"
    end
    # stan_call(;kwargs...) = x->stan_call(x, ;kwargs...)
    # stan_call(x::Expr; kwargs...) = stan_expr()
    expr_replace(x; kwargs...) = get(kwargs, x, x) 
    expr_replace(x::Expr; kwargs...) = Expr(x.head, expr_replace.(x.args; kwargs...)...)

    ensure_xlhs(arg::Symbol) = arg
    ensure_xlhs(::Expr) = Symbol("_")
    deffun(x::LineNumberNode) = x
    deffun(x::Expr; mod=@__MODULE__) = if x.head == :block
        Expr(:block, deffun.(x.args)...)
    else
        # @assert x.head == :(=)
        fsig, body = ensure_xassign(x).args
        fcall, rv = ensure_xtyped(fsig).args
        @assert Meta.isexpr(fcall, :call)
        f, args... = fcall.args
        is_lpxf = endswith(string(f), r"_lp[md]f")
        is_lpxf && (rv = :real)
        ftype = :(typeof($f))
        args = ensure_xtyped.(args, :anything)
        arg_names = map(arg->arg.args[1], args)
        arg_types = map(arg->ensure_xref(arg.args[2]), args)
        lhs_type = xsig_type.(arg_types)

        fun_sizes = OrderedDict()
        for (arg_name, arg_type) in zip(arg_names, arg_types)
            for (i, dim_name) in enumerate(arg_type.args[2:end])
                isa(dim_name, Symbol) || continue
                fun_sizes[dim_name] = "int $dim_name = dims($arg_name)[$i];"
            end
        end
        deconstruct = Expr(:block, xassign(xtuple(arg_names...), :(x.args)), [
            xassign(xtuple(ensure_xlhs.(args_type.args[2:end])...), :($args_name.type.size))
            for (args_name, args_type) in zip(arg_names, arg_types)
        ]..., :(info = (;$(arg_names...), $(keys(fun_sizes)...),)))

        stmts = []
        # scope_names = vcat(arg_names)

        stan_fundef, subexprs = if ismissing(body) 
             "", nothing
        else
            @assert Meta.isexpr(body, :block)
            _, body = body.args
            _, body, subexprs = ensure_xpair(body, nothing).args
            if !isnothing(subexprs)
                # subexprs = ensure_vect(subexprs).args
                # @assert Meta.isexpr(subexprs, (:vect, :vcat))
                subexprs = Expr(:block, deconstruct, Expr(:vect, [
                    # :($mod.expr($mod.stan_call($arg; $(arg_names...))))
                    :($mod.expr($mod.forward!($arg; info)))
                    for arg in canonical.(ensure_xvect(subexprs).args)
                ]...))
            end
            sig_rv = sigtype(rv)
            sig_args = join(sigarg.(args), ", ")
            

            # """$sig_rv $f($sig_args){
            #     $(funbody(collect(values(fun_sizes))))
            #     $(funbody(body))
            # }
            # """
            sig_args_expr = :(join(map((x, name)->$mod.sigtype(x) * " $name", ($(arg_names...),), ($(Meta.quot.(arg_names)...),)), ", "))
            xstring("$sig_rv $f(", sig_args_expr, """){
                $(funbody(collect(values(fun_sizes))))
                $(strip(funbody(body)))
            }
            """), subexprs
        end

        xexpr = :(x::$mod.CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, deconstruct, xsig_expr(ensure_xref(rv)))
        isa(f, Symbol) && push!(stmts, :(function $f end))
        push!(stmts, quote
            $mod.tracetype($xexpr) = $xbody
        end)
        if !ismissing(body)
            push!(stmts, :($mod.fundef($xexpr) = $(Expr(:block, deconstruct, stan_fundef))))
        end
        if !isnothing(subexprs)
            push!(stmts, :($mod.fundefexprs($xexpr) = $subexprs))
        end
        isa(f, Symbol) || return Expr(:block, stmts...)
        if is_lpxf
            base_f = Symbol(string(f)[1:end-length("_lpdf")])
            rng_f = Symbol(base_f, "_rng")
            lpdfs_f = Symbol(f, "s")
            base_ftype = :(typeof($base_f))
            base_xexpr = :(_x::$mod.CanonicalExpr{<:$base_ftype,<:Tuple{$(lhs_type[2:end]...)}})
            dummy1 = StanExpr(missing, StanType(getproperty(types, arg_types[1].args[1]), ntuple(i->StanExpr(missing, StanType(types.int)), length(arg_types[1].args)-1)))
            reconstruct = :(x = $mod.CanonicalExpr($f, $dummy1, _x.args...))
            base_xbody = Expr(:block, reconstruct, deconstruct, xsig_expr(ensure_xref(arg_types[1])))
            push!(stmts, quote
                function $base_f end
                function $rng_f end
                function $lpdfs_f end
                $mod.tracetype($base_xexpr) = $base_xbody
                $mod.rng_expr(::typeof($base_f)) = $rng_f
                $mod.likelihood_expr(::typeof($base_f)) = $lpdfs_f
            end)
            if !ismissing(body)
                push!(stmts, :($mod.fundef($base_xexpr) = $(Expr(:block, reconstruct, deconstruct, stan_fundef))))
            end
            if !isnothing(subexprs)
                base_subexprs = Expr(:block, reconstruct, subexprs)
                push!(stmts, :($mod.fundefexprs($base_xexpr) = $base_subexprs))
            end
        end
        Expr(:block, stmts...)
    end
# @macroexpand @deffun begin 
#     # log_dirichlet_lpdf(log_theta, alpha) = "return dot_product(alpha, log_theta) + lgamma(sum(alpha)) - sum(lgamma(alpha));"
#     dummy2_lpdf(y, x) = "return 0.;"
#     # dummy_rng(x::vector[n])::vector[n] = "return x;"
# end
    
end

macro defsig(x)
    esc(defsig(x))
end
macro deffun(x)
    esc(deffun(x))
end


fundef(x) = nothing
fundefexprs(x) = []
fundefs(x) = filter(!isnothing, vcat(fundef(x), mapreduce(fundefs, vcat, fundefexprs(x); init=[])))