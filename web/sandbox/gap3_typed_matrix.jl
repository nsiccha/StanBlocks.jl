@slic (;y=randn(6)) begin
    A::matrix[2,3] ~ normal(0,1)
    y ~ normal(to_vector(A), 1.0)
end