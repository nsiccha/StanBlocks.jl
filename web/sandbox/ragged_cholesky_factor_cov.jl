# Variable-size square Cholesky factors of covariance matrices. Their free
# dimensions are K(K+1)/2 even though each reconstructed value has K² cells.
@slic (; K = [2, 3, 4], y = 0.3) begin
    L::cholesky_factor_cov[K] ~ flat()
    y ~ normal(sum(to_vector(L[1])), 0.1)
end
