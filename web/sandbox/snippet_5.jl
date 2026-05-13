@deffun begin
    inner_bad(x, y) = x + y * undefined_thing
    outer_bad(p, q) = inner_bad(p, q)
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(outer_bad(mu, sigma), sigma)
end