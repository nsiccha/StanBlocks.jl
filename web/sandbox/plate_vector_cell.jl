# Vector-per-cell plate (correlated per-series random effects):
# shared L + tau captured, per-series z::vector[6] ~ std_normal(), return
# diag_pre_multiply(tau, L) * z — the K-vector cell output collected as
# matrix[6, n_series]. The plate form of a BRM `ranef_correlated_draws` block.
@slic (; n_series = 8) begin
    L::cholesky_factor_corr[6] ~ lkj_corr_cholesky(2.0)
    tau::vector[6] ~ normal(0.0, 1.0; lower = 0.0)
    b::matrix[6, n_series] ~ plate(; outer = (n_series,)) do s
        z::vector[6] ~ std_normal()              # fresh per-cell vector param → matrix[6, n_series]
        diag_pre_multiply(tau, L) * z            # vector[6] cell output → b[:, s]
    end
end
