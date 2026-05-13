@slic (;N=20, y=randn(20), x=randn(20)) begin
    alpha ~ cauchy(0, 5)
    beta ~ double_exponential(0, 1)
    sigma ~ exponential(1)
    y ~ normal(alpha + beta * x, sigma)
end
