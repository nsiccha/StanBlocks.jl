@slic (;N=20, switched=rand(0:1, 20), dist=randn(20)) begin
    beta ~ normal(0, 10)
    beta2 ~ normal(0, 10)
    switched ~ bernoulli_logit(beta + beta2 * dist)
end
