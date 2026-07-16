# Variable-size Cholesky factors of correlation matrices. The constrained K×K
# matrices share flat memory; L[g] reconstructs the selected group on demand.
@slic (; K = [2, 3, 4], y = 0.3) begin
    L::cholesky_factor_corr[K] ~ flat()
    y ~ normal(sum(to_vector(L[1])), 0.1)
end
