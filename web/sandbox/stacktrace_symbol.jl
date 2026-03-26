@deffun begin
    uses_missing_symbol(x) = x + dosings
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    y ~ normal(uses_missing_symbol(mu), 1)
end
