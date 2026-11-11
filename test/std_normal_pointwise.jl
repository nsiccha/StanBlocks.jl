"""
`std_normal` registers `std_normal_lpdfs` as its pointwise companion, but no
method defined it: an observed `y ~ std_normal()` failed at trace time
(`tracetype not defined for y_likelihood = std_normal_lpdfs(::vector)`; todo
0s2pan0), and a matrix observation of it had no matrix-shaped twin either.
It is now an elementwise family like `normal`, with both twins.
"""
@testitem "slic: an observed std_normal has its pointwise twin" tags=[:slic, :regression, :stanc, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    vec_obs = @slic (; y = [0.1, -0.4, 0.7, 1.2]) begin
        y ~ std_normal()
    end
    gq = stan_block(stan_code(vec_obs), "generated quantities")
    @test occursin("y_likelihood = std_normal_lpdfs(y);", gq)
    @test stanc_compiles(vec_obs)
    outs = Dict(o.name => o for o in stan_descriptor(vec_obs).outputs)
    @test outs[:y_likelihood].generative == :pointwise_loglik

    mat_obs = @slic (; Y = [0.1 -0.4; 0.7 1.2; 0.3 0.0]) begin
        Y ~ std_normal()
    end
    gq = stan_block(stan_code(mat_obs), "generated quantities")
    @test occursin(r"matrix\[[^\]]+\] Y_likelihood = std_normal_lpdfs\(Y\);", gq)
    @test stanc_compiles(mat_obs)
end
