@slic (;y=randn(5)) begin
    x::vector[5] ~ std_normal()
    y ~ normal(x, 1.0)
end
