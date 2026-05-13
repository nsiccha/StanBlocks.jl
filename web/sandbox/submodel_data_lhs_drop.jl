sb_me_drop = StanBlocks.@slic begin
    x_true ~ std_normal(; n=num_elements(x_obs))
    x_obs ~ normal(x_true, sd_x)
    return x_true
end
@slic (;a=randn(10), sd_a=0.5) begin
    me_a ~ sb_me_drop(; x_obs=a, sd_x=sd_a)
    y ~ normal(me_a, 1.0)
end