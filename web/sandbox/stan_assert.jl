# `@stan_assert cond [msg]` expands to `if !cond reject(msg) end` — a
# Stan-native runtime assertion. Use inside `@deffun` bodies (control
# flow is not allowed in `@slic` model bodies). Without an explicit
# message a default `"assertion failed: <cond>"` is used.
@deffun safe_log(x::real)::real = begin
    @stan_assert x > 0 "safe_log: argument must be positive"
    return log(x)
end

@deffun safe_div(a::real, b::real)::real = begin
    @stan_assert b != 0
    return a / b
end

@slic (;y=randn(5)) begin
    mu ~ std_normal(;lower=0.)
    nu ~ std_normal()
    foo = safe_log(mu)
    bar = safe_div(mu, nu)
    y ~ normal(foo + bar, 1.)
end
