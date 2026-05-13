@slic (;a=randn(3), b=randn(5), c=randn(3)) begin
    X1 = hcat(a, c)
    X2 = hcat(a, c)
    y = X1
end