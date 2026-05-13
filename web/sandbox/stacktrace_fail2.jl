@deffun begin
    typed_fn(x, y::vector[n]) = x + y
    untyped_wrapper(a, b) = typed_fn(a, b)
    top_level_fn(p, q) = untyped_wrapper(p, q)
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(top_level_fn(mu, sigma), sigma)
end
