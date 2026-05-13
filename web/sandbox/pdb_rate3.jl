@slic (;n1=10, n2=15, k1=3, k2=7) begin
    theta ~ beta(1, 1)
    k1 ~ binomial(n1, theta)
    k2 ~ binomial(n2, theta)
end
