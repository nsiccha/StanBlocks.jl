@deffun my_add_d3(f, g, h) = f + g * h

import StanBlocks.stan
m = @slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y::real = my_add_d3(mu, sigma, mu)
end
sm = stan_model(m)
