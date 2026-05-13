@slic (;N=20, kid_score=randn(20), mom_iq=randn(20), mom_hs=randn(20)) begin
    beta ~ normal(0, 10)
    beta2 ~ normal(0, 10)
    beta3 ~ normal(0, 10)
    beta4 ~ normal(0, 10)
    sigma ~ cauchy(0, 2.5)
    kid_score ~ normal(beta + beta2 * mom_hs + beta3 * mom_iq + beta4 * mom_hs * mom_iq, sigma)
end
