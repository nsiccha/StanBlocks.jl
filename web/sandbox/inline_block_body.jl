# Multi-statement inline UDF body. Locals are renamed with a per-callsite
# `__il_<n>` suffix so two calls don't collide; pre-statements are hoisted
# into the enclosing block (here, transformed_parameters).
@deffun @inline polished(x::vector[n])::vector[n] = begin
    tmp = x * 2
    return tmp + 1
end

@slic (;y=randn(5)) begin
    mu  ~ std_normal(;n=5)
    nu  ~ std_normal(;n=5)
    obs = polished(mu) + polished(nu)
    y   ~ normal(obs, 1.)
end