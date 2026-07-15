# Feature 1 increment 1 — FULL verification (transpile + stanc, Stan 2.37).
# Observed data `y` forces SB into inference mode (real parameters/model split,
# not RNG forward-sim), so `simplex_jacobian` lands in `transformed parameters`
# where Stan's jacobian accumulator is valid. Drives:
#   /sandbox/snippet/verify_simplex_stanc/stanc  -> stanc3 v2.37.0 accept/reject.
@slic (;y=0.3) begin
    free::vector[2] ~ std_normal()      # 2 free params -> parameters block
    theta = simplex_jacobian(free)      # vector[3] simplex + implicit Jacobian
    y ~ normal(theta[1], 0.1)           # likelihood ties theta to observed y
end
