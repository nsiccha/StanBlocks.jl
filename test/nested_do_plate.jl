@testitem "slic: a collecting do-plate nested in @plate for, and the multiple-axis form" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    nested = @slic (; J=2, S=3, y=[0.1, 0.2]) begin
        @plate for j in 1:J
            tau_s ~ plate(; outer=S) do s
                tau::vector[2] ~ normal(0.0, 1.0)
                tau
            end
            y[j] ~ normal(sum(tau_s), 1.0)
        end
    end
    # The nested plate stores its cells over the flattened index (v0.2.62; it was
    # refused as "collecting a `plate(...) do` result inside another
    # compiler-owned loop" since v0.2.37).
    code = stan_code(nested)
    @test occursin("matrix[2, sum(loop_j_tau_s__pl_n)] loop_j_tau_s_tau;", stan_block(code, "parameters"))
    @test stanc_compiles(nested)

    # Independent axes can share one annotated loop; each indexed binding
    # keeps both coordinates in its storage and observations.
    grid = @slic (; J=2, S=3, y=[0.1 0.2 0.3; 0.4 0.5 0.6]) begin
        @plate for j in 1:J, s in 1:S
            tau[j, s] ~ normal(0.0, 1.0)
            y[j, s] ~ normal(tau[j, s], 1.0)
        end
    end
    @test occursin("matrix[J, S] tau;", stan_block(stan_code(grid), "parameters"))
    @test stanc_compiles(grid)
end
