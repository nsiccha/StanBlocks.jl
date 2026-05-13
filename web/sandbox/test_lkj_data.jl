using LinearAlgebra
@slic (;K=3, S=4, L=[Matrix{Float64}(I, 3, 3) for _ in 1:4]) begin
    eta ~ gamma(1, 1)
    L ~ lkj_corr_cholesky(eta; n=K, m=S)
end