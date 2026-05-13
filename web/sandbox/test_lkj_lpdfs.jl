@slic (;K=3, S=4, y=randn(4)) begin
    L ~ lkj_corr_cholesky(1.; n=K, m=S)
    lp = lkj_corr_cholesky_lpdfs(L, 2.0)
    y ~ normal(sum(L) + sum(lp), 1.)
end