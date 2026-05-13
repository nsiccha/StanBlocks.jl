@slic (;J=8, y=randn(8), sigma=abs.(randn(8)).+0.1) begin
    mu ~ normal(0, 5)
    tau ~ cauchy(0, 5)
    theta ~ normal(mu, tau)
    y ~ normal(theta, sigma)
end
