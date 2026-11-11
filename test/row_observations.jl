"""
A plate cell observing one row of a data matrix, `y2[i, :] ~ lognormal(…)`, has
a `row_vector` `_gen` twin. The sized RNG companions draw vectors and no family
defines a `row_vector` one, so the twin called an undeclared
`lognormal_row_vector_rng` and stanc rejected the program (todo 0uc7rmf). A
`row_vector` draw is now the transpose of the family's `vector` draw, for every
family at once.
"""
@testitem "slic: a matrix-row observation draws its twin as a transposed vector" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    y2 = [1.0 2.0 3.0; 1.5 2.5 3.5]
    for family in (:lognormal, :normal)
        m = StanBlocks.stan.SlicModel(quote
            mu ~ normal(0.0, 1.0)
            pred ~ plate(; outer = 2) do i
                m_i = mu + i
                y2[i, :] ~ $family(rep_vector(m_i, 3), 1.0)
                m_i
            end
        end, Dict{Symbol,Any}(:y2 => y2), @__MODULE__)
        code = stan_code(m)
        gq = stan_block(code, "generated quantities")
        @test !occursin("row_vector_rng", code)
        @test occursin("$(family)_vector_rng(", gq)
        @test occursin("')", gq) || occursin("'\n", gq) || occursin("';", gq)
        @test stanc_compiles(m)
    end
end

@testitem "slic: a matrix-row twin draws a row of the right shape" tags=[:slic, :plate, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    m = @slic (; y2 = [1.0 2.0 3.0; 1.5 2.5 3.5]) begin
        mu ~ normal(0.0, 1.0)
        pred ~ plate(; outer = 2) do i
            m_i = mu + i
            y2[i, :] ~ lognormal(rep_vector(m_i, 3), 1.0)
            m_i
        end
    end
    model = stan_model(m)
    problem = instantiate(model)
    @test LogDensityProblems.dimension(problem) == 1
    @test isfinite(LogDensityProblems.logdensity(problem, [0.2]))
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    gen = filter(n -> startswith(n, "y2_gen"), names)
    @test length(gen) == 6
    draw = BridgeStan.param_constrain(problem.model, [0.2]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(problem.model, 1))
    @test all(>(0), draw[[findfirst(==(n), names) for n in gen]])
end
