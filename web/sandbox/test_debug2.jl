@deffun my_add_d2(f, g, h) = f + g * h

ms = methods(StanBlocks.stan.tracetype)
matching = filter(m -> occursin("my_add_d2", string(m.sig)), collect(ms))
error("Found $(length(matching)) methods: $(matching)")
