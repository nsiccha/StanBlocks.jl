@slic (;N=30, y=Float64.(rand(0:1, 30)), x1=randn(30), x2=randn(30)) begin
    alpha ~ normal(0, 5)
    beta1 ~ normal(0, 5)
    beta2 ~ normal(0, 5)
    y ~ bernoulli_logit(alpha + beta1 * x1 + beta2 * x2)
end
