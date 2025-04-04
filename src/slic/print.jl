Base.show(io::IO, x::StanType) = begin 
    l, r = lr_size(x)
    length(l) > 0 && print(io, "array [", join(l, ", "), "] ")
    print(io, center_type(x))
    length(x.cons) > 0 && print(io, "<", join(map((k,v)->"$k=$v", keys(x.cons), values(x.cons)), ", "), ">")
    length(r) > 0 && print(io, "[", join(r, ", "), "]")
end
sigtype(x::StanType) = begin 
    l = length(lr_size(x)[1])
    io = IOBuffer()
    l > 0 && print(io, "array [", join(fill("", l), ", "), "] ")
    print(io, center_type(x))
    String(take!(io))
end
top_print(io::IO, x::StanExpr) = print(io, type(x), " ", x, ";\n")
top_print(io::IO, x::Expr) = print(io, x, ";\n")
top_print(io::IO, x::String) = print(io, x, "\n")
Base.show(io::IO, x::StanExpr) = error(typeof(x))#print(io, x.expr)
Base.show(io::IO, x::StanExpr{<:Union{Symbol,Number,String}}) = print(io, x.expr)
Base.show(io::IO, x::StanExpr{<:Function}) = print(io, Symbol(x.expr))
Base.show(io::IO, x::StanExpr{Colon}) = print(io, ":")
Base.show(io::IO, x::StanExpr{<:Base.BroadcastFunction}) = print(io, ".", Symbol(expr(x).f))
Base.show(io::IO, x::StanExpr{Expr}) = print_expr(io, Val(expr(x).head), expr(x).args...)
print_expr(io::IO, h::Val, args...) = error(h)
print_expr(io::IO, ::Val{:(=)}, lhs, rhs) = print(io, lhs, " = ", rhs)
print_expr(io::IO, ::Val{:call}, f, args...) = print(io, f, "(", join(args, ", "), ")")
print_expr(io::IO, ::Val{:call}, f::StanExpr{<:Union{Base.BroadcastFunction,typeof.((+,-,*,/,^))...}}, args...) = print(io, "(", join(args, " $f "), ")")
print_expr(io::IO, ::Val{:call}, f::StanExpr{<:Union{Base.BroadcastFunction,typeof.((+,-,*,/,^))...}}, x) = print(io, "(", f, x, ")")
print_expr(io::IO, ::Val{:call}, f::StanExpr{typeof(adjoint)}, x) = print(io, "(", x, "')")
print_expr(io::IO, ::Val{:call}, f::StanExpr{typeof(getindex)}, x, args...) = print(io, x, "[", join(args, ", "), "]")
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
