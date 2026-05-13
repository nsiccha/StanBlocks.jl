@deffun begin
    ordered_normal_lpdf(y::ordered[n], loc, scale, n) = normal_lpdf(y, loc, scale)
end

@slic (;y=randn(3)) begin
    log_inv_rates ~ ordered_normal(0.0, 1.0, 3)
    y ~ normal(log_inv_rates, 1.0)
end