# Auto-emitted runtime shape checks. When a dim name appears in multiple
# arg type annotations (e.g. `n` in both `x::vector[n]` and `y::vector[n]`),
# the first occurrence binds it (`int n = dims(x)[1];`) and each later
# occurrence is a `if (... != n) reject(...)` check at function entry.
# Run-time, the sampler aborts with a clear message instead of silently
# misbehaving on shape-mismatched inputs.
@deffun mydot(x::vector[n], y::vector[n])::real = dot_product(x, y)

@deffun rowdot(A::matrix[m, n], B::matrix[m, n])::vector[m] = rows_dot_product(A, B)

# Independent dims: NO check between m and n (no shared name).
@deffun mvmul(A::matrix[m, n], v::vector[n])::vector[m] = A * v

@slic (;y=randn(5), z=randn(5), Y=randn(3, 5), Z=randn(3, 5)) begin
    mu ~ std_normal(;n=5)
    s = mydot(mu, y)              # n=5 in both, OK
    rd = rowdot(Y, Z)             # m=3, n=5 in both, OK
    p = mvmul(Y, mu)              # m=3, n=5 — independent, no check
end
