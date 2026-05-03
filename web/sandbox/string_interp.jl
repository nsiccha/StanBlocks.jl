# Julia-style string interpolation in `reject` / `print`. The interpolated
# string `"got x=$x"` is lowered to Stan's variadic form
# `reject("got x=", x)` at AST-rewrite time, so users can write the
# natural `"$x"` syntax without thinking about Stan's calling convention.
@deffun safe_log(x::real)::real = begin
    @stan_assert x > 0 "safe_log: argument must be positive, got x=$x"
    return log(x)
end

@deffun debug_print(x::real, label::int)::void = begin
    print("debug[$label] = $x")
end

@slic (;y=randn(3)) begin
    mu ~ std_normal(;lower=0.)
    z  = safe_log(mu)
    debug_print(z, 7)
    y ~ normal(z, 1.)
end