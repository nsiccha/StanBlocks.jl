# Baseline: `simple_reduce_sum` with a NAMED function (no closures involved).
# Confirms whether the `vector`/`T[]` stanc rejection is intrinsic to
# `simple_reduce_sum` or specific to the closure-lifting path.
@deffun mysq(y::real)::real = y * y

@slic (;n=5) begin
    mu ~ std_normal(;n=n)
    s  = simple_reduce_sum(mysq, mu)
end
