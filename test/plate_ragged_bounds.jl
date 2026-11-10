"""
A plate cell whose length varies by cell is stored as flat memory. A bound that
depends on the cell (`lower=L[g]`) used to be left off that memory, so a sampled
cell explored outside its support (todo 144j6sn). It is declared now through a
`transformed data` bound vector laid out like the memory.
"""
@testitem "slic: a varying-length plate cell keeps a bound that depends on the cell" tags=[:slic, :plate, :ragged, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    data = (; K = [2, 3], L = [0.5, -1.0], Lv = [[0.0, 0.1], [0.0, 0.1, 0.2]], y = [0.1, 0.2])
    scalar_bound = @slic data begin
        @plate for g in eachindex(K)
            zz::vector[K[g]] ~ normal(0, 1; lower = L[g])
            y[g] ~ normal(sum(zz), 1)
        end
    end
    element_bound = @slic data begin
        @plate for g in eachindex(K)
            zz::vector[K[g]] ~ normal(0, 1; lower = Lv[g])
            y[g] ~ normal(sum(zz), 1)
        end
    end
    for (m, fill) in ((scalar_bound, r"rep_vector\(L\[g\]"), (element_bound, r"= getindex_RaggedVector\(Lv, g\)|Lv"))
        code = stan_code(m)
        @test occursin(r"vector<lower=\w+>\[sum\(", stan_block(code, "parameters"))
        @test occursin(fill, stan_block(code, "transformed data"))
        @test stanc_compiles(m)
    end
end
