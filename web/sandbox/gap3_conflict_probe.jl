@slic (;y=randn(5)) begin
    x::vector[5] ~ std_normal(; n=5)
    y ~ normal(x, 1.0)
end