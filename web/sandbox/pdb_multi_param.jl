@slic (;N=25, y=randn(25), x1=randn(25), x2=randn(25), x3=randn(25)) begin
    alpha ~ normal(0, 10)
    b1 ~ normal(0, 10)
    b2 ~ normal(0, 10)
    b3 ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    y ~ normal(alpha + b1 * x1 + b2 * x2 + b3 * x3, sigma)
end
