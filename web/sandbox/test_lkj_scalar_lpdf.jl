@slic (;K=3) begin
    L ~ lkj_corr_cholesky(1.; n=K)
    lp = lkj_corr_cholesky_lpdf(L, 2.0)
end