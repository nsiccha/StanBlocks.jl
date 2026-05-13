@slic (;y=randn(1)[1]) begin
    mu::real ~ std_normal()
    y ~ normal(mu, 1.0)
end
