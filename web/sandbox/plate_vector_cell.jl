# Vector-per-cell plate (BRM #1, biomarker_hierarchical_parametric correlated floor):
# shared L + tau captured, per-series z::vector[6] ~ std_normal(), return
# diag_pre_multiply(tau, L) * z — the K-vector cell output collected as
# matrix[6, n_series]. Replaces the ranef_correlated_draws n_series×6 floor.
@slic (; n_series = 8) begin
    L::cholesky_factor_corr[6] ~ lkj_corr_cholesky(2.0)
    tau::vector[6] ~ normal(0.0, 1.0; lower = 0.0)
    b::matrix[6, n_series] ~ plate(; outer = (n_series,)) do s
        z::vector[6] ~ std_normal()              # fresh per-cell vector param → matrix[6, n_series]
        diag_pre_multiply(tau, L) * z            # vector[6] cell output → b[:, s]
    end
end
