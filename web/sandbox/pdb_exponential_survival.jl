@slic (;N=15, t=abs.(randn(15)).+0.1) begin
    lambda ~ gamma(1, 1)
    t ~ exponential(lambda)
end
