# Demo: user-defined macros expand inside `@deffun` bodies as well.
# `@sumprod(a, b)` becomes `a*a + b*b`.
macro sumprod(args...)
    products = [Expr(:call, :*, a, a) for a in args]
    foldl((a,b) -> Expr(:call, :+, a, b), products)
end

@deffun mymetric(x::vector[n], y::vector[n])::real = @sumprod(x, y)

@slic (;y=randn(5), x=randn(5)) begin
    mu ~ std_normal()
    obs = mymetric(x + mu, y)
end
