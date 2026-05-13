@slic (;N=20, y=randn(20), x=randn(20)) begin
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    nu ~ gamma(2, 0.1)
    y ~ student_t(nu, alpha + beta * x, sigma)
end
