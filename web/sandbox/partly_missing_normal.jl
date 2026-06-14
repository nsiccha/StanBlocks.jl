@slic (;y=[1.0, missing, 3.0, missing, 5.0]) begin
    mu ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    y ~ normal(mu, sigma)
end
