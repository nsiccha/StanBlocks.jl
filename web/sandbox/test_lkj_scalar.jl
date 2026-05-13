@slic (;K=3, y=randn(3)) begin
    L ~ lkj_corr_cholesky(1.; n=K)
    y ~ normal(sum(L), 1.)
end