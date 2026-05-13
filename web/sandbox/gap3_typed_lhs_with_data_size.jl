@slic (;y=randn(7), n_x=5) begin
    x::vector[n_x] ~ std_normal()
    y ~ normal(x[1], 1.0)
end
