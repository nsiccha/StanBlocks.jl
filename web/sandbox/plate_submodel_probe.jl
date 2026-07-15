# PROBE: does a plate body inline a CALLED @slic submodel with its own ~ param?
@slic ncp(mu::real, sig::real) = begin z ~ std_normal(); return mu + sig * z end
@slic (; y = randn(6), mu0 = 0.5) begin
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    theta ~ plate(y; outer = (6,)) do yi
        t ~ ncp(mu0, sigma)
        yi ~ normal(t, sigma)
        t
    end
end
