@slic (;N=20, y=rand(0:10, 20), x=randn(20)) begin
    alpha ~ normal(0, 5)
    beta ~ normal(0, 5)
    phi ~ gamma(2, 1)
    y ~ neg_binomial_2(exp(alpha + beta * x), phi)
end
