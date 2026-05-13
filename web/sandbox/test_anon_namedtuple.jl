@deffun begin
    sum_and_scale(x, y, s) = (x + y) * s
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(sum_and_scale(mu, 1.0, sigma), 1)
end
