@deffun begin
    shifted(x, mu, sigma) = (x - mu) / sigma
end

@slic (;N=30, y=randn(30), x=randn(30)) begin
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    y ~ normal(shifted(alpha, 0, 1) + beta * x, sigma)
end
