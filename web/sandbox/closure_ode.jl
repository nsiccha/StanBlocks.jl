# Closure passed to Stan's `ode_rk45` — exponential decay `dy/dt = -lambda*y`
# where `lambda` is captured. Phase-2 lifting threads `lambda` through as a
# Stan-side arg of the receiver.
@slic (;n=5, ts=collect(1.0:5.0)) begin
    lambda ~ std_normal(;lower=0.)
    y0     = [1.0]
    y      = ode_rk45((t, y_state) -> -lambda * y_state, y0, 0.0, to_array_1d(ts))
end
