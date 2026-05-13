@slic (;N=15, y1=randn(15), y2=randn(15)) begin
    mu ~ normal(0, 10)
    sigma1 ~ gamma(2, 1)
    sigma2 ~ gamma(2, 1)
    y1 ~ normal(mu, sigma1)
    y2 ~ normal(mu, sigma2)
end
