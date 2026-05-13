@slic (;N=20, floor_measure=Float64.(rand(0:1, 20)), log_radon=randn(20)) begin
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma_y ~ normal(0, 1)
    log_radon ~ normal(alpha + beta * floor_measure, sigma_y)
end
