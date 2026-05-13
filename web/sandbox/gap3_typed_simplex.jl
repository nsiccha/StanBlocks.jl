@slic (;y=randn(3)) begin
    p::simplex[3] ~ dirichlet(rep_vector(1.0, 3))
    y ~ normal(p, 1.0)
end