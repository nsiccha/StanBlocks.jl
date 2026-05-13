@deffun shift_scale(x, mu, sigma) = (x - mu) / sigma

@slic (;y=randn(10)) begin
    mu ~ normal(0, 5)
    sigma ~ gamma(2, 1)
    y ~ normal(shift_scale(mu, 0.0, sigma), 1)
end
