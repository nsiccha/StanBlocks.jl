@testitem "slic: untyped elementwise matrix samples infer their argument shape" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    model(body) = StanBlocks.SlicModel(body,
        Dict{Symbol,Any}(:M => [0.1 0.2; 0.3 0.4; 0.5 0.6], :y => [0.1, 0.2]), @__MODULE__)
    for rhs in (:(normal(M, 1.0)), :(normal(0.0, M)), :(normal(M, M)),
                :(student_t(4.0, 0.0, M)), :(exponential(M)))
        fitted = model(:(begin z ~ $rhs; y ~ normal(z[1, 1], 1.0) end))
        @test occursin("matrix", stan_block(stan_code(fitted), "parameters"))
        @test occursin("[M_m, M_n] z;", stan_block(stan_code(fitted), "parameters"))
        @test stanc_compiles(fitted)

        prior = model(:(begin z ~ $rhs end))
        @test occursin(r"^\{\s*\}$", strip(stan_block(stan_code(prior), "parameters")))
        @test occursin("matrix[M_m, M_n] z", stan_block(stan_code(prior), "generated quantities"))
        @test stanc_compiles(prior)
    end
    parameters = @slic (; y=[0.1, 0.2]) begin
        M::matrix[3, 2] ~ normal(0.0, 1.0)
        z ~ normal(M, 1.0)
        y ~ normal(z[1, 1], 1.0)
    end
    @test occursin("matrix[3, 2] z;", stan_block(stan_code(parameters), "parameters"))
    @test stanc_compiles(parameters)

    cells = model(:(begin
        zs ~ plate(; outer=2) do i
            z ~ normal(M, 1.0)
            z
        end
        y ~ normal(zs[1][1, 1], 1.0)
    end))
    @test occursin("array[2] matrix[M_m, M_n] zs_z;", stan_block(stan_code(cells), "parameters"))
    @test stanc_compiles(cells)
end

@testitem "slic: observed LKJ matrices have sized predictive twins" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    for (family, center, value) in (
        (:lkj_corr_cholesky, :cholesky_factor_corr, [1.0 0.0; 0.4 sqrt(0.84)]),
        (:lkj_corr, :corr_matrix, [1.0 0.4; 0.4 1.0]),
    ), typed in (false, true)
        lhs = typed ? :(Omega::$center[2]) : :Omega
        fitted = StanBlocks.SlicModel(:(begin $lhs ~ $family(2.0) end),
            Dict{Symbol,Any}(:Omega => value), @__MODULE__)
        code = stan_code(fitted)
        @test occursin("Omega ~ $family(2.0);", stan_block(code, "model"))
        @test occursin("Omega_gen", stan_block(code, "generated quantities"))
        @test occursin("real Omega_likelihood", stan_block(code, "generated quantities"))
        @test stanc_compiles(fitted)
    end
end
