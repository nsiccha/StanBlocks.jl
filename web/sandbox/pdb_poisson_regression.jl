@slic (;N=20, y=rand(0:10, 20), x=randn(20)) begin
    alpha ~ normal(0, 5)
    beta ~ normal(0, 5)
    y ~ poisson_log(alpha + beta * x)
end
