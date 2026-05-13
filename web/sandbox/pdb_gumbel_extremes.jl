@slic (;N=20, y=randn(20).+5.0) begin
    mu ~ normal(0, 10)
    beta ~ gamma(1, 1)
    y ~ gumbel(mu, beta)
end
