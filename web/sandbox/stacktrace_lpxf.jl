@deffun begin
    fake_dist(x) = x
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    y ~ fake_dist(mu)
end
