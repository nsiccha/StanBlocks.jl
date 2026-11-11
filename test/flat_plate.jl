@testitem "slic: flat vector plate cells retain their declared support" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    fitted = @slic (; n_strata=3, y=[0.1, 0.2, 0.3]) begin
        tau_s ~ plate(; outer=(n_strata,)) do s
            tau::vector[2] ~ flat(; lower=0.0)
            tau
        end
        y ~ normal(tau_s[1, :], 1.0)
    end
    code = stan_code(fitted)
    @test occursin("matrix<lower=0.0>[2, n_strata] tau_s_tau;", stan_block(code, "parameters"))
    @test !occursin("flat", code)
    @test stanc_compiles(fitted)

    sliced = @slic (; n_strata=3, y=[0.1, 0.2, 0.3]) begin
        tau_s ~ plate(; outer=(n_strata,)) do s
            tau::vector[2] ~ flat(; lower=0.0, upper=5.0)
            tau[1] ~ exponential(1.0)
            tau[2] ~ normal(0.0, 1.0)
            tau
        end
        y ~ normal(tau_s[1, :], 1.0)
    end
    code = stan_code(sliced)
    @test occursin("matrix<lower=0.0, upper=5.0>[2, n_strata] tau_s_tau;", stan_block(code, "parameters"))
    @test !occursin("flat", code)
    @test occursin("~ exponential(1.0)", stan_block(code, "model"))
    @test occursin("~ normal(0.0, 1.0)", stan_block(code, "model"))
    @test stanc_compiles(sliced)
end
