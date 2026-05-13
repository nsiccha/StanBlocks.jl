@slic (;N=15, t=abs.(randn(15)).+0.1) begin
    alpha ~ gamma(2, 1)
    sigma ~ gamma(2, 1)
    t ~ weibull(alpha, sigma)
end
