@slic (;N=20, y=randn(20)) begin
    mu ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    y ~ double_exponential(mu, sigma)
end
