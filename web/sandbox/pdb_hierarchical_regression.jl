@deffun begin
    hier_mean(mu, tau, offset) = mu + tau * offset
end

@slic (;N=20, y=randn(20), x=randn(20)) begin
    mu_alpha ~ normal(0, 10)
    mu_beta ~ normal(0, 10)
    tau_alpha ~ gamma(2, 1)
    tau_beta ~ gamma(2, 1)
    alpha ~ normal(mu_alpha, tau_alpha)
    beta ~ normal(mu_beta, tau_beta)
    sigma ~ gamma(2, 1)
    y ~ normal(hier_mean(alpha, beta, x), sigma)
end
