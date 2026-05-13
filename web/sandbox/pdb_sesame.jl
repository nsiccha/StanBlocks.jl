@slic (;N=30, encouraged=Float64.(rand(0:1, 30)), watched=randn(30)) begin
    beta ~ normal(0, 10)
    beta2 ~ normal(0, 10)
    sigma ~ gamma(2, 1)
    watched ~ normal(beta + beta2 * encouraged, sigma)
end
