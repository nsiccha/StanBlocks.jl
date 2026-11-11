"""
A higher-order density specialised on a function-valued argument that is itself
named with a Stan suffix kept that suffix inside its name:
`hop_lpdf(y, base_lpdf, mu)` became `hop_base_lpdf_lpdf`, and its `~` statement
`y ~ hop_base_lpdf(mu)` still ended in `_lpdf`, which stanc rejects ("~
statement should refer to a distribution without its _lpdf suffix"; todo
06lvabf, all ten generated simplex programs of the old docs). Such a fragment
now gets `_arg`, so only the receiver's own suffix ends the name, in the
definition and at every call site alike.
"""
@testitem "slic: a function argument's Stan suffix does not end a specialised name" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    @deffun begin
        sfa_base_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)
        @lhs @lpxf sfa_hop_lpdf(y::real, f, mu::real)::real = f(y, mu)
        sfa_hop_lpdfs(y::real, f, mu::real)::real = f(y, mu)
        sfa_hop_rng(f, mu::real)::real = normal_rng(mu, 1.0)
    end
    m = @slic (; y = 0.3) begin
        mu ~ normal(0.0, 1.0)
        y ~ sfa_hop(sfa_base_lpdf, mu)
    end
    code = stan_code(m)
    model = stan_block(code, "model")
    @test occursin("y ~ sfa_hop_sfa_base_lpdf_arg(mu);", model)
    @test occursin("real sfa_hop_sfa_base_lpdf_arg_lpdf(", stan_block(code, "functions"))
    @test !occursin(r"_lpdf_lpdf\b", code)
    @test stanc_compiles(m)
end
