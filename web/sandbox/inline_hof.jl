# Higher-order inline UDFs: function arguments substitute through cleanly.
# `twice(mysq, mu)` inlines to `mysq(mysq(mu))`.
@deffun @inline twice(f, x::real)::real = f(f(x))
@deffun mysq(x::real)::real = x * x

@slic (;y=0.) begin
    mu ~ std_normal()
    y  ~ normal(twice(mysq, mu), 1.)
end