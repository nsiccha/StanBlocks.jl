@slic (;y=randn(3)) begin
    c ~ std_normal(; type=ordered, n=3)
    y ~ normal(c, 1.0)
end
