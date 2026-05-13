@slic (;N=20, K=2, y=rand(0:5, 20), X=randn(20, 2)) begin
    alpha ~ normal(0, 5)
    beta ~ normal(0, 5; m=K)
    y ~ poisson_log_glm(X, alpha, beta)
end
