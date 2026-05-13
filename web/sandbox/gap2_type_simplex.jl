@slic (;y=randn(3)) begin
    p ~ dirichlet(rep_vector(1.0, 3); type=simplex)
    y ~ normal(p, 1.0)
end
