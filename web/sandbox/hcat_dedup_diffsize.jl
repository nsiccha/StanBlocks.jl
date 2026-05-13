@slic (;a=randn(3), b=randn(5), c=randn(3), d=randn(5)) begin
    X1 = hcat(a, c)
    X2 = hcat(b, d)
    y = X1
end