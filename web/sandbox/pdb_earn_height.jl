@slic (;N=20, earn=randn(20), height=randn(20)) begin
    beta ~ normal(0, 10)
    beta2 ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    earn ~ normal(beta + beta2 * height, sigma)
end
