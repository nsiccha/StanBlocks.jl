@slic (;y=rand(1:3, 20), n_cuts=2) begin
    eta ~ std_normal(; n=num_elements(y))
    c_base ~ std_normal()
    c_log_incr ~ std_normal(; n=n_cuts-1)
    cuts = c_base + cumulative_sum(append_row(0.0, exp(c_log_incr)))
    y ~ ordered_logistic(eta, cuts)
end