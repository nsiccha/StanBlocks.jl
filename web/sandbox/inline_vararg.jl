# Variadic inline UDFs: `args...` splats through to the inner call site.
# `wrap3(a,b,c)` inlines to `three(a, b, c)`.
@deffun three(a::real, b::real, c::real)::real = a + b + c
@deffun @inline wrap3(args...)::real = three(args...)

@slic (;y=0.) begin
    a ~ std_normal()
    b ~ std_normal()
    c ~ std_normal()
    y ~ normal(wrap3(a, b, c), 1.)
end
