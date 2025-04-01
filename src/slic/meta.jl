const RV_SYMBOL = :RETURN_VALUE
dumperror(x) = (dump(x); error(x))
xcanonical(x::Expr) = if x.head == :call
    fargs = []
    kwargs = []
    for arg in x.args
        if Meta.isexpr(arg, :parameters)
            for argi in arg.args
                if Meta.isexpr(argi, :kw)
                    push!(kwargs, argi.args[1]=>argi.args[2])
                elseif isa(argi, Symbol)
                    push!(kwargs, argi=>argi)
                else
                    dumperror(argi)
                end
            end
        elseif Meta.isexpr(arg, :kw) 
            push!(kwargs, arg.args[1]=>arg.args[2])
        else
            push!(fargs, arg)
        end
    end
    kwargs = (;kwargs...)
    (;fargs, kwargs)
else
    xcanonical(Expr(:call, Dict(
        Symbol("'")=>:adjoint,
        :ref=>:getindex
    )[x.head], x.args...))
end

xcall(fargs...) = Expr(:call, fargs...)