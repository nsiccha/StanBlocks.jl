@slic (;N=20, log_earn=randn(20), height=randn(20).+170.0, male=Float64.(rand(0:1, 20))) begin
    beta ~ normal(0, 10)
    beta2 ~ normal(0, 10)
    beta3 ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    log_earn ~ normal(beta + beta2 * height + beta3 * male, sigma)
end
