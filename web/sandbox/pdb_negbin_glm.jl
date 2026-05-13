@slic (;N=20, K=2, y=rand(0:10, 20), X=randn(20, 2)) begin
    alpha ~ normal(0, 5)
    beta ~ normal(0, 5; m=K)
    phi ~ gamma(2, 1)
    y ~ neg_binomial_2_log_glm(X, alpha, beta, phi)
end
