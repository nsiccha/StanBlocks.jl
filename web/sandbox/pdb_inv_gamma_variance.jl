@slic (;N=20, y=randn(20)) begin
    mu ~ normal(0, 10)
    sigma_sq ~ inv_gamma(2, 1)
    y ~ normal(mu, sqrt(sigma_sq))
end
