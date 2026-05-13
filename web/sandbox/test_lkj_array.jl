@slic (;K=3, S=4, y=randn(4)) begin
    L ~ lkj_corr_cholesky(1.; n=K, m=S)
    y ~ normal(sum(L), 1.)
end