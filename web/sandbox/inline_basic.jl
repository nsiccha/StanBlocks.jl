# `@inline` inlines a UDF at the call site — no entry in `functions {}`,
# the body is substituted directly into the caller's code.
@deffun @inline scale(x::vector[n], s::real)::vector[n] = x * s

@slic (;y=randn(5)) begin
    mu  ~ std_normal(;n=5)
    obs = scale(mu, 2.0)
    y   ~ normal(obs, 1.)
end