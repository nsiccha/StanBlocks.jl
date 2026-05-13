# Smoke test for gq typed-LHS `~` rng synthesis via the tokenof machinery.
#
# The gq push wraps lhs shape into a `tokenof{T}` StanExpr and hands it to the
# rng call. Scalar (S=0) tokens drop through to Stan's native rng; sized (S>=1)
# tokens dispatch into per-shape `*_rng` @deffun overloads that know how to
# shape the output (e.g. `std_normal_rng(real[n]) = to_vector(normal_rng(...))`).

@slic begin
    x::real ~ std_normal()              # scalar: std_normal_rng()
    y::real[5] ~ std_normal()           # sized:  std_normal_rng(real[5])
    z::vector[3] ~ normal(0.0, 1.0)     # sized:  normal_rng(vector[3], 0.0, 1.0)
    a::real[4] ~ cauchy(0.0, 1.0)
    b::vector[4] ~ lognormal(0.0, 1.0)
    c::real[4] ~ gamma(2.0, 1.0)
    d::vector[4] ~ exponential(1.0)
    e::real[4] ~ student_t(3.0, 0.0, 1.0)
    f::vector[4] ~ beta(2.0, 2.0)
    g::real[4] ~ uniform(0.0, 1.0)
    h::vector[4] ~ chi_square(3.0)
end
