@slic (;N=25, y=randn(25)) begin
    mu ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    alpha ~ normal(0, 5)
    y ~ skew_normal(mu, sigma, alpha)
end
