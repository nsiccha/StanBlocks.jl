@slic (;n1=10, n2=15, k1=3, k2=7) begin
    theta1 ~ beta(1, 1)
    theta2 ~ beta(1, 1)
    k1 ~ binomial(n1, theta1)
    k2 ~ binomial(n2, theta2)
end
