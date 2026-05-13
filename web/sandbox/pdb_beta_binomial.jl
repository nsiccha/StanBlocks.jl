@slic (;N=10, k=rand(0:10, 10), n=fill(10, 10)) begin
    alpha ~ gamma(2, 1)
    beta_param ~ gamma(2, 1)
    k ~ beta_binomial(n, alpha, beta_param)
end
