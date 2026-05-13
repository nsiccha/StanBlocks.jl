@deffun my_add_debug(f, g, h) = f + g * h

# Check what methods exist
ms = methods(StanBlocks.stan.tracetype)
matching = filter(m -> occursin("my_add_debug", string(m.sig)), collect(ms))
@info "tracetype methods for my_add_debug" matching

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(my_add_debug(mu, sigma, mu), sigma)
end
