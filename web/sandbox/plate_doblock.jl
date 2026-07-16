@slic (; y = randn(6), mu0 = 0.5) begin
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    theta ~ plate(y; outer = (6,)) do yi
        t ~ normal(mu0, 1.0)          # fresh per-cell param (mu0 shared capture)
        yi ~ normal(t, sigma)         # observation: yi = y[i] sliced (sigma shared capture)
        t                             # cell output → theta
    end
end
