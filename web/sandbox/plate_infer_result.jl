# AUTO-INFERENCE (goal §5): BARE `b ~ plate` — no `b::vector[6]` annotation.
# The result shape (matrix[6, n_series]) is inferred from the trailing expr.
@slic (; n_series = 8) begin
    L::cholesky_factor_corr[6] ~ lkj_corr_cholesky(2.0)
    tau::vector[6] ~ normal(0.0, 1.0; lower = 0.0)
    b ~ plate(; outer = (n_series,)) do s                 # BARE b — no ::vector[6]
        z::vector[6] ~ std_normal()
        diag_pre_multiply(tau, L) * z                     # vector[6] cell output → inferred matrix
    end
end
