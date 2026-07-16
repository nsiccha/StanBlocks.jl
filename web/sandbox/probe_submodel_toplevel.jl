# Control: top-level submodel call binds `t` (needs explicit `return`).
@slic ncp(mu::real, sig::real) = begin z ~ std_normal(); return mu + sig * z end
@slic (; y = 0.5) begin
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    t ~ ncp(0.0, sigma)
    y ~ normal(t, sigma)
end
