"""
A rank-3 or higher Julia array is data of the equal-rank container (snag
`stan-type-has-no-b2d832be`): a real `Array{<:Real,N}` is `matrix[d1, …, dN]`,
which Stan declares `array[d1, …, dN-2] matrix[dN-1, dN]`, and an integer one is
`array[d1, …, dN] int`. That is the reading a `@deffun` `matrix[K, N, J]`
annotation already has, so `A[i, j, k]` names the same element in the model,
the data file and Julia. Such data used to be refused with "Do not know how to
handle `stan_type(A, …)`".
"""
@testitem "slic: a rank-3+ array is equal-rank data" tags=[:slic, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    x = [0.2, 0.5, 0.9, 1.4]
    y = [0.3, 0.6, 1.1, 1.5]
    A = reshape(collect(1.0:8.0), 2, 2, 2) ./ 1.5

    # The snag shape: one element of a rank-3 real array.
    m = @slic (; x, y, A) begin
        b ~ normal(0, 1)
        sigma ~ exponential(1)
        mu = b * x + A[1, 2, 2]
        y ~ normal(mu, sigma)
    end
    code = stan_code(m)
    data = stan_block(code, "data")
    @test occursin("array[A_d1] matrix[A_d2, A_d3] A;", data)
    @test all(s -> occursin("int $s;", data), ("A_d1", "A_d2", "A_d3"))
    @test occursin("A[1, 2, 2]", stan_block(code, "transformed parameters"))
    @test stanc_compiles(m)

    # The data file nests the array by its first index, as Stan reads it.
    json = StanBlocks.JSON.parse(stan_data_json(m))
    @test (json["A_d1"], json["A_d2"], json["A_d3"]) == size(A)
    @test all(json["A"][i][j][k] == A[i, j, k] for i in 1:2, j in 1:2, k in 1:2)

    # The descriptor lists the container and its three sizes, which are derived.
    d = stan_descriptor(m)
    input = only(i for i in d.inputs if i.name === :A)
    @test input.type === :matrix
    @test input.size == (:A_d1, :A_d2, :A_d3)
    @test Set(StanBlocks.required_inputs(d)) == Set([:x, :y, :A])

    # Rebinding re-reads the sizes and the values.
    A2 = reshape(collect(1.0:24.0), 3, 2, 4)
    json2 = StanBlocks.JSON.parse(stan_data_json(m(; A=A2)))
    @test (json2["A_d1"], json2["A_d2"], json2["A_d3"]) == size(A2)
    @test all(json2["A"][i][j][k] == A2[i, j, k] for i in 1:3, j in 1:2, k in 1:4)

    # Integer arrays, a rank-4 real array, and a leading-index selection that
    # leaves the trailing matrix.
    K = reshape(collect(1:24), 2, 3, 4)
    B = reshape(collect(1.0:24.0), 2, 3, 2, 2)
    z = [0.1, 0.4]
    wide = @slic (; z, K, B, A) begin
        b ~ normal(0, 1)
        z ~ normal(b * (A[2] * [1.0, 1.0]) + K[2, 3, 4] + B[2, 3, 1, 2], 1.0)
    end
    code = stan_code(wide)
    data = stan_block(code, "data")
    @test occursin("array[K_d1, K_d2, K_d3] int K;", data)
    @test occursin("array[B_d1, B_d2] matrix[B_d3, B_d4] B;", data)
    @test stanc_compiles(wide)
    json = StanBlocks.JSON.parse(stan_data_json(wide))
    @test all(json["K"][i][j][k] == K[i, j, k] for i in 1:2, j in 1:3, k in 1:4)
    @test all(json["B"][i][j][k][l] == B[i, j, k, l] for i in 1:2, j in 1:3, k in 1:2, l in 1:2)

    # A zero-dimensional array names its scalar instead.
    @test_throws ArgumentError StanBlocks.stan.stan_type(:z, fill(1.0))
end

@testitem "slic: rank-3+ array data has the density of its elements" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems, Random
    rng = Xoshiro(20261026)
    x = randn(rng, 5)
    y = randn(rng, 5)
    A = randn(rng, 2, 3, 4)
    K = rand(rng, -5:5, 3, 2, 2)
    B = randn(rng, 2, 2, 3, 2)
    m = @slic (; x, y, A, K, B) begin
        b ~ normal(0, 1)
        y ~ normal(b * x + A[1, 3, 2] - 2 * A[2, 1, 4] + K[3, 1, 2] * B[1, 2, 3, 1], 1.0)
    end
    # An independent plain-Julia density, up to a constant: any transposed or
    # reversed layout reads other elements and moves the offset `c`.
    c = A[1, 3, 2] - 2 * A[2, 1, 4] + K[3, 1, 2] * B[1, 2, 3, 1]
    reference(b) = -b^2 / 2 - sum(abs2, y .- (b .* x .+ c)) / 2
    p = instantiate(m)
    lp(b) = LogDensityProblems.logdensity(p, [b])
    @test lp(0.7) - lp(0.0) ≈ reference(0.7) - reference(0.0) atol = 1e-10
    @test lp(-1.3) - lp(0.0) ≈ reference(-1.3) - reference(0.0) atol = 1e-10
end
