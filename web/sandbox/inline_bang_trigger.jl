# Trailing `!` is a synonym for `@inline` (Julia mutation convention).
# No `negate!` function is emitted — `negate!(mu)` becomes `(-mu)` inline.
@deffun negate!(x::real)::real = -x

@slic (;y=0.) begin
    mu ~ std_normal()
    y  ~ normal(negate!(mu), 1.)
end
