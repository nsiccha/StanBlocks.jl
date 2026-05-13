@slic (;n=10, k=3) begin
    theta ~ beta(1, 1)
    k ~ binomial(n, theta)
end
