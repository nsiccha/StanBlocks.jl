@deffun my_add(f, g, h) = f + g * h

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(my_add(mu, sigma, mu), sigma)
end
