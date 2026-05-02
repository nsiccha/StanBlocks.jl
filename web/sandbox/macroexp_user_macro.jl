# Demo: user-defined macros now expand inside `@slic` bodies.
# `@center` rewrites `x` to `x - mean(x)`. SLIC sees the post-expansion AST.
macro center(x); :($x - mean($x)); end

@slic (;y=randn(20), x=randn(20)) begin
    alpha ~ std_normal()
    beta  ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y ~ normal(alpha + beta * @center(x), sigma)
end
