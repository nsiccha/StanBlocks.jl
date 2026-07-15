# ISOLATION PROBE (non-plate): typed-scalar LHS with lower= + a likelihood to keep it a param.
@slic (; y = 1.0) begin
    s::real ~ normal(0.0, 1.0; lower = 0.0)
    y ~ normal(s, 1.0)
end
