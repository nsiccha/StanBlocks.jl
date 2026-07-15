# AUTO-INFERENCE regression: bare scalar cell output must still infer vector[N].
@slic (; y = randn(6), mu0 = 0.5) begin
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    theta ~ plate(y; outer = (6,)) do yi
        t ~ normal(mu0, 1.0); yi ~ normal(t, sigma); t    # scalar cell output → vector[6]
    end
end
