# Void UDF: side-effect-only function. Calls are statements; binding the
# result is rejected at trace time. Stan-side, the function emits as
# `void foo(...)` with no `return`.
@deffun debug_print(label::real, x::real)::void = begin
    print(label)
    print(x)
end

@deffun shape_check(x::vector[n], y::vector[n])::void = begin
    @stan_assert n > 0 "shape_check: empty input"
end

@slic (;y=randn(4)) begin
    mu    ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    debug_print(mu, sigma)              # statement-position call: OK
    shape_check(rep_vector(0., 4), y)
    y ~ normal(mu, sigma)
end
