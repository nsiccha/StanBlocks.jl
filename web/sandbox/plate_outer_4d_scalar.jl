# Four outer axes are deeper than the finite builtin getindex signatures and
# therefore exercise plate's generic array-prefix selection rule.
@slic (; y = randn(2, 2)) begin
    theta ~ plate(; outer = (2, 2, 2, 2)) do i, j, k, l
        z ~ normal(0.0, 1.0)
        y[i, j] ~ normal(z, 1.0)
        z + 0.0 * k + 0.0 * l
    end
end
