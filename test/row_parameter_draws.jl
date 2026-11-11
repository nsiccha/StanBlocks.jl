@testitem "slic: two-argument RNGs accept either vector orientation" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    for family in (:normal, :cauchy, :lognormal, :gamma, :inv_gamma, :beta,
                   :uniform, :weibull, :frechet, :loglogistic, :von_mises,
                   :double_exponential, :logistic, :gumbel, :pareto,
                   :scaled_inv_chi_square)
        rng = Symbol(family, :_rng)
        choices = (1.0, :v, :(D[1, :]))
        draws = [Expr(:(=), Symbol(:draw_, i), Expr(:call, rng, args...))
                 for (i, args) in enumerate(Iterators.product(choices, choices))]
        m = StanBlocks.SlicModel(quote
            z ~ std_normal()
            $(draws...)
            y1 ~ $family(D[1, :], D[1, :])
            y2 ~ $family(D[1, :], v)
            y3 ~ $family(v, D[1, :])
            y4 ~ $family(D[1, :], 1.0)
        end, Dict{Symbol,Any}(:v => [0.5, 0.7, 0.9], :D => [0.5 0.7 0.9],
            :y1 => [1.0, 2.0, 3.0], :y2 => [1.0, 2.0, 3.0],
            :y3 => [1.0, 2.0, 3.0], :y4 => [1.0, 2.0, 3.0]), @__MODULE__)
        @test stanc_compiles(m)
    end
end

@testitem "slic: row-parameter normal predictions draw the observation shape" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    y = [0.2, -0.4, 0.7]
    m = @slic (; y) begin
        M::matrix[2, 3] ~ normal(0.0, 1.0)
        y ~ normal(M[1, :], exp(M[2, :]))
    end
    p = instantiate(stan_model(m))
    @test LogDensityProblems.dimension(p) == 6
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    draw = BridgeStan.param_constrain(p.model, zeros(6); include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 3))
    gen = draw[findall(n -> startswith(n, "y_gen"), names)]
    ll = draw[findall(n -> startswith(n, "y_likelihood"), names)]
    @test length(gen) == 3 && all(isfinite, gen)
    @test ll ≈ @. -0.5 * (log(2pi) + y^2)
end
