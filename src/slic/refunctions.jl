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
    ensure_xassign(x, default=missing) = Meta.isexpr(x, :(=)) ? x : xassign(x, default)
    ensure_xtuple(x) = Meta.isexpr(x, :tuple) ? x : xtuple(x)
    ensure_xref(x) = Meta.isexpr(x, :ref) ? x : xref(x)
    ensure_xtyped(x, default=Any) = Meta.isexpr(x, :(::)) ? x : xtyped(x, default)
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
        size = xtuple(size...)
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
        lhs_type = xsig_type.(lhs)

        xexpr = :(x::$mod.CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, [
            xassign(xtuple(lhsi.args[2:end]...), :(x.args[$i].type.size))
            for (i, lhsi) in enumerate(lhs)
        ]..., xsig_expr(rv))
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
    sigarg(x::Expr) = begin 
        @assert x.head == :(::)
        "$(sigtype(x.args[2])) $(x.args[1])"
    end
    # stan_call(;kwargs...) = x->stan_call(x, ;kwargs...)
    # stan_call(x::Expr; kwargs...) = stan_expr()
    expr_replace(x; kwargs...) = get(kwargs, x, x) 
    expr_replace(x::Expr; kwargs...) = Expr(x.head, expr_replace.(x.args; kwargs...)...)

    deffun(x::LineNumberNode) = x
    deffun(x::Expr; mod=@__MODULE__) = if x.head == :block
        Expr(:block, deffun.(x.args)...)
    else
        # @assert x.head == :(=)
        fsig, body = ensure_xassign(x).args
        fcall, rv = ensure_xtyped(fsig).args
        @assert Meta.isexpr(fcall, :call)
        f, args... = fcall.args
        ftype = :(typeof($f))
        args = ensure_xtyped.(args, :anything)
        arg_names = map(arg->arg.args[1], args)
        arg_types = map(arg->ensure_xref(arg.args[2]), args)
        lhs_type = xsig_type.(arg_types)
        # rv = ensure_xref(rv)
        # xrv = xsig_expr(rv)

        stmts = []
        deconstruct = Expr(:block, xassign(xtuple(arg_names...), :(x.args)), [
            xassign(xtuple(args_type.args[2:end]...), :($args_name.type.size))
            for (args_name, args_type) in zip(arg_names, arg_types)
        ]...)
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
                subexprs = Expr(:block, deconstruct, :(info = (;$(arg_names...))), Expr(:vect, [
                    # :($mod.expr($mod.stan_call($arg; $(arg_names...))))
                    :($mod.expr($mod.forward!($arg; info)))
                    for arg in canonical.(ensure_xvect(subexprs).args)
                ]...))
            end
            sig_rv = sigtype(rv)
            sig_args = join(sigarg.(args), ", ")
            fun_sizes = OrderedDict()
            for (arg_name, arg_type) in zip(arg_names, arg_types)
                for (i, dim_name) in enumerate(arg_type.args[2:end])
                    @assert isa(dim_name, Symbol)
                    fun_sizes[dim_name] = "int $dim_name = dims($arg_name)[$i];"
                end
            end
             """$sig_rv $f($sig_args){
                $(funbody(collect(values(fun_sizes))))
                $(funbody(body))
            }
            """, subexprs
        end

        xexpr = :(x::$mod.CanonicalExpr{<:$ftype,<:Tuple{$(lhs_type...)}})
        xbody = Expr(:block, deconstruct, xsig_expr(ensure_xref(rv)))
        isa(f, Symbol) && push!(stmts, :(function $f end))
        push!(stmts, quote
            $mod.tracetype($xexpr) = $xbody
        end)
        if !ismissing(body)
            push!(stmts, :($mod.fundef($xexpr) = $stan_fundef))
        end
        if !isnothing(subexprs)
            push!(stmts, :($mod.fundefexprs($xexpr) = $subexprs))
        end
        isa(f, Symbol) || return Expr(:block, stmts...)
        if endswith(string(f), r"_lp[md]f")
            base_f = Symbol(string(f)[1:end-length("_lpdf")])
            rng_f = Symbol(base_f, "_rng")
            lpdfs_f = Symbol(f, "s")
            base_ftype = :(typeof($base_f))
            base_xexpr = :(_x::$mod.CanonicalExpr{<:$base_ftype,<:Tuple{$(lhs_type[2:end]...)}})
            # dummy1 = :($mod.StanExpr(missing, $mod.StanType($mod.types.anything)))
            dummy1 = StanExpr(missing, StanType(types.anything, ntuple(i->StanExpr(missing, StanType(types.int)), length(arg_types[1].args)-1)))
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
                push!(stmts, :($mod.fundef($base_xexpr) = $stan_fundef))
            end
            if !isnothing(subexprs)
                base_subexprs = Expr(:block, reconstruct, subexprs)
                push!(stmts, :($mod.fundefexprs($base_xexpr) = $base_subexprs))
            end
        # elseif endswith(string(f), r"_lp[md]fs")
        #     base_f = Symbol(string(f)[1:end-length("_lpdfs")])
        #     push!(stmts, quote
        #         $mod.likelihood_expr(::typeof($base_f)) = $f
        #     end)
        # elseif endswith(string(f), r"_rng")
        #     base_f = Symbol(string(f)[1:end-length("_rng")])
        #     push!(stmts, quote
        #         $mod.rng_expr(::typeof($base_f)) = $f
        #     end)
        end
        Expr(:block, stmts...)
        # (;f, arg_names, arg_types, rv, body)
    end
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
# fundef(x::CanonicalExpr{typeof(vector_std_normal_rng)}) = """
# vector[] vector_std_normal_rng(int n){
#     vector[n] rv;
#     for(i in 1:n){
#         rv[i] = std_normal_rng();
#     }
#     return rv;
# }
# """
@macroexpand @deffun begin 
    truncated_normal_lpdfs(obs::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] = """
        vector[n] rv;
        for(i in 1:n){
            rv[i] = truncated_normal_lpdf(obs[i] | loc[i], scale[i], lloq[i], uloq[i])
        }
        return rv;
    """ => [
        truncated_normal_lpdf(loc[1], loc[1], scale[1], lloq[1], uloq[1])
    ]
    # truncated_normal_lpdf(obs::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::real = """
    #     return sum(truncated_normal_lpdfs(obs, loc, scale, lloq, uloq));
    # """ => [
    #     truncated_normal_lpdfs(loc, loc, scale, lloq, uloq)
    # ]
    # vector_std_normal_rng(n::int)::vector[n] = """
    #     vector[n] rv;
    #     for(i in 1:n){
    #         rv[i] = std_normal_rng();
    #     }
    #     return rv;
    # """
end
# fundefexprs(CanonicalExpr(truncated_normal_lpdfs, ))