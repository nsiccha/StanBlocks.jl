@deffun begin
    inner(x, y::vector[n]) = x + y
    middle(a, b::vector[n]) = inner(a, b)
    outer(p, q::vector[n]) = middle(p, q)
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(outer(mu, y), sigma)
end
