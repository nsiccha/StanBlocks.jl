@slic (;y=randn(10), priors=(mu_mean=0.0, mu_sd=5.0)) begin
    mu ~ normal(priors.mu_mean, priors.mu_sd)
    y ~ normal(mu, 1)
end
