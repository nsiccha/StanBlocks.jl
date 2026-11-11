"""
A `lower=`/`upper=` bound on a family is the `truncated` law (`stanblocks-use`
§8.1), and a `maybecv` mark taints whatever reads it. Through `truncated` the
bound is a positional argument of the lowered call, so a marked input in it
re-drew the parameter in generated quantities. Through the keyword spelling, or
`restricted`, it stayed a keyword, which the taint did not read: the parameter
stayed fitted, and replaying its draws with new marked values could break its
new bound (todo 1k3hvkc). A call's keyword values taint it like its arguments.
"""
@testitem "slic: a maybecv mark taints a prior through a keyword bound" tags=[:slic, :cv, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    data = (; y = [0.1, 0.2], s = StanBlocks.stan.maybecv(:s, 0.3))
    keyword = @slic data begin
        a ~ normal(0.0, 1.0); x ~ normal(0.0, 1.0; lower = a + s); y ~ normal(a + x, 1.0)
    end
    truncated_ = @slic data begin
        a ~ normal(0.0, 1.0); x ~ truncated(normal(0.0, 1.0); lower = a + s); y ~ normal(a + x, 1.0)
    end
    restricted_ = @slic data begin
        a ~ normal(0.0, 1.0); x ~ restricted(normal(0.0, 1.0); lower = a + s); y ~ normal(a + x, 1.0)
    end
    typed = @slic data begin
        a ~ normal(0.0, 1.0); x::real ~ normal(0.0, 1.0; lower = a + s); y ~ normal(a + x, 1.0)
    end
    for m in (keyword, truncated_, restricted_, typed)
        code = stan_code(m)
        params = stan_block(code, "parameters")
        @test occursin("real a;", params)
        @test !occursin(" x;", params)
        @test occursin("real x = ", stan_block(code, "generated quantities"))
        @test stanc_compiles(m)
    end
    @test stan_code(keyword) == stan_code(truncated_)
    @test stan_code(typed) == stan_code(truncated_)

    # A plate cell's prior: the keyword spelling emits what `truncated` emits
    # (it used to keep `p_x` a parameter).
    cells(rhs) = StanBlocks.stan.SlicModel(quote
        a ~ normal(0.0, 1.0)
        p ~ plate(; outer = (2,)) do g
            x ~ $rhs; x
        end
        y ~ normal(a .+ p, 1.0)
    end, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    keyword_cells = cells(:(normal(0.0, 1.0; lower = s)))
    truncated_cells = cells(:(truncated(normal(0.0, 1.0); lower = s)))
    @test !occursin("p_x", stan_block(stan_code(truncated_cells), "parameters"))
    @test stan_code(keyword_cells) == stan_code(truncated_cells)
    @test stanc_compiles(keyword_cells)

    # An unmarked bound keeps the parameter fitted.
    plain = @slic (; y = [0.1, 0.2], s = 0.3) begin
        a ~ normal(0.0, 1.0); x ~ normal(0.0, 1.0; lower = a + s); y ~ normal(a + x, 1.0)
    end
    @test occursin("> x;", stan_block(stan_code(plain), "parameters"))
end
