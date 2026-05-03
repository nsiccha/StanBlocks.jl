# Default positional arguments to `@deffun`. Lowered to multiple method
# definitions Julia-style: each call-arity gets its own method, and the
# short-arity ones are `@inline` trampolines that fill in the defaults
# before delegating to the full method. Stan-side, only the full method
# emits a `functions {}` entry — short calls inline directly to it.
@deffun lin(x::real, slope::real = 1.0, intercept::real = 0.0)::real = slope * x + intercept

@slic (;y=randn(5)) begin
    mu ~ std_normal()
    a = lin(mu)             # → lin(mu, 1.0, 0.0)
    b = lin(mu, 2.0)        # → lin(mu, 2.0, 0.0)
    c = lin(mu, 2.0, 0.5)   # full call
    y ~ normal(a + b + c, 1.)
end