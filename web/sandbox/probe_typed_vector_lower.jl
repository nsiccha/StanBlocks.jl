# ISOLATION PROBE (non-plate): typed-vector LHS with lower= + a likelihood to keep it a param.
@slic (; y = 1.0) begin
    v::vector[3] ~ normal(0.0, 1.0; lower = 0.0)
    y ~ normal(sum(v), 1.0)
end
