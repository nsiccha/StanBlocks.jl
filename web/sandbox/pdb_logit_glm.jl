@slic (;N=30, K=2, y=rand(0:1, 30), X=randn(30, 2)) begin
    alpha ~ normal(0, 5)
    beta ~ normal(0, 5; m=K)
    y ~ bernoulli_logit_glm(X, alpha, beta)
end
