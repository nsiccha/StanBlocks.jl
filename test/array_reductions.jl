"""
Stan's `sum`, `prod`, `mean`, `max` and `min` take a one-dimensional array, a
vector or a matrix, never an `array[m, n] int`, so `sum(D)` of `Matrix{Int}`
data traced, transpiled and stanc rejected it ("Ill-typed arguments supplied to
function sum: (array[,] int)"; todo 03a19oe). Like Julia's reduction of a
matrix it covers every element, so it is the reduction of `to_array_1d(D)`.
"""
@testitem "slic: a reduction over a 2-D integer array covers every element" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    data = (; D = [1 2; 3 4])
    for (f, stanf) in ((:sum, "sum"), (:prod, "prod"), (:maximum, "max"), (:minimum, "min"))
        m = StanBlocks.stan.SlicModel(quote
            b ~ normal(0.0, 1.0)
            q ~ normal(b + $f(D), 1.0)
        end, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
        code = stan_code(m)
        @test occursin("$(stanf)(to_array_1d(D))", code)
        @test stanc_compiles(m)
    end
    # A one-dimensional array keeps the direct call.
    flat = @slic (; k = [1, 2, 3]) begin
        b ~ normal(0.0, 1.0)
        q ~ normal(b + sum(k), 1.0)
    end
    @test occursin("sum(k)", stan_code(flat))
    @test !occursin("to_array_1d", stan_code(flat))
end
