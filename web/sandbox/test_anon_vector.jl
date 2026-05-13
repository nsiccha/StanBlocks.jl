@slic (;y=randn(5), x=randn(5)) begin
    beta ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(x * beta, sigma)
end
