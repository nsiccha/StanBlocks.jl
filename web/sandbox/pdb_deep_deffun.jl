@deffun begin
    scale(x, s) = x * s
    shift(x, m) = x + m
    shift_and_scale(x, m, s) = shift(scale(x, s), m)
end

@slic (;N=20, y=randn(20), x=randn(20)) begin
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    y ~ normal(shift_and_scale(x, alpha, beta), sigma)
end
