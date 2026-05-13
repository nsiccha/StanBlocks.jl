@slic (;y=randn(3)) begin
    c::ordered[3] ~ std_normal()
    y ~ normal(c, 1.0)
end