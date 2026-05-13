@slic (;y=randn(5)) begin
    x ~ std_normal(; type=vector, n=5)
    y ~ normal(x, 1.0)
end
