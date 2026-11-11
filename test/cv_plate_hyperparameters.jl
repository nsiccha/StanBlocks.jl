"""
A `maybecv` mark re-draws what it taints and leaves everything else informed by
the fit. A plate cell sized by a marked count is tainted, so it is re-drawn in
generated quantities, but an untainted hyperparameter that only its prior reads
must stay a parameter, as it does in the flat spelling (todo 00m5uxb). The plate
path used to skip the cell prior's arguments, so `s` was re-drawn from its own
prior and the population draw ignored the fit.
"""
@testitem "slic: a cv-tainted plate cell keeps the hyperparameter its prior reads" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    d = (; group_idx = StanBlocks.stan.maybecv(:group_idx, [1, 2, 2]), y = [0.1, 0.2, 0.3])
    flat = @slic (; d...) begin
        n_g = maximum(group_idx)
        s ~ exponential(1)
        c ~ normal(0.0, s; n = n_g)
        y ~ normal(c[group_idx], 1.0)
    end
    scalar_cell = @slic (; d...) begin
        n_g = maximum(group_idx)
        s ~ exponential(1)
        p ~ plate(; outer = (n_g,)) do g
            c ~ normal(0.0, s)
            c
        end
        y ~ normal(p[group_idx], 1.0)
    end
    vector_cell = @slic (; d...) begin
        n_g = maximum(group_idx)
        s ~ exponential(1)
        p ~ plate(; outer = (n_g,)) do g
            c ~ normal(0.0, s; n = 2)
            sum(c)
        end
        y ~ normal(p[group_idx], 1.0)
    end
    for m in (flat, scalar_cell, vector_cell)
        code = stan_code(m)
        @test occursin(r"real<lower=0(\.0)?> s;", stan_block(code, "parameters"))
        gq = stan_block(code, "generated quantities")
        @test !occursin("exponential_rng", gq)
        # The mark still takes: the held-out likelihood leaves the model block.
        @test !occursin(r"y ~ normal", stan_block(code, "model"))
        @test stanc_compiles(m)
    end
    # Only the cells are re-drawn, from the fitted `s`.
    for m in (scalar_cell, vector_cell)
        gq = stan_block(stan_code(m), "generated quantities")
        @test occursin(r"normal_(vector_)?rng\([^;]*s\)", gq)
    end
end
