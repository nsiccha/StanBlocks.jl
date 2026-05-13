@slic (;y=randn(3)) begin
    c::ordered[3] ~ normal(0.0, 1.0)
    y ~ normal(c, 1.0)
end