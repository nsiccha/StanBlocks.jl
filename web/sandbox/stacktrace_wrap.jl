@deffun begin
    many_args(a, b, c, d, e, f, g, h) = a + b + c + d + e + f + g + h + undefined_thing
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(many_args(mu, sigma, mu, sigma, mu, sigma, mu, sigma), sigma)
end