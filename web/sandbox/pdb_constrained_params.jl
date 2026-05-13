@slic (;y=randn(10)) begin
    mu ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    nu ~ gamma(2, 0.1)
    y ~ student_t(nu, mu, sigma)
end
