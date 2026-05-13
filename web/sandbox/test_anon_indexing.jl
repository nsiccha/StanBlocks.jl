@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(2, 1)
    tau ~ gamma(1, 1)
    y ~ normal(mu * sigma, tau)
end
