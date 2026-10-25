"""
A vector of matrices is ragged matrix data (snag `ragged-matrix-da-163255e9`),
as a vector of vectors is ragged vector data: `Vector{<:AbstractMatrix{<:Real}}`
ingests as a `RaggedMatrix`, one Stan data tuple `(mem, ends, rows, cols)` with
each matrix column-major in `mem`. `X[g]` is the g-th matrix wherever the model
reads it — a plate cell, an annotated loop, a plate positional, the model body, a
submodel keyword, a `@deffun` argument — and `length(X)` counts the matrices.
Per-group design matrices used to be refused with "Do not know how to handle
`stan_type(X, …)`", and had to be passed flattened with a per-group row count
and read back with `reshape(Xf[i], n[i], k)`.
"""
@testitem "slic: a vector of matrices is ragged matrix data" tags=[:slic, :ragged, :plate, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    X = [[0.5 -0.2; 0.1 0.3], [0.2 0.4; -0.6 0.1; 0.3 0.7]]
    y = [[0.1, -0.3], [0.4, 0.2, -0.1]]

    # Ingest: the encoding, real memory for integer matrices, an empty vector.
    @test StanBlocks.stan.to_ragged(X) == (;
        mem=[0.5, 0.1, -0.2, 0.3, 0.2, -0.6, 0.3, 0.4, 0.1, 0.7],
        ends=[4, 10], rows=[2, 3], cols=[2, 2])
    @test StanBlocks.stan.to_ragged([[1 0; 0 1]]).mem isa Vector{Float64}
    @test StanBlocks.stan.to_ragged(Matrix{Float64}[]) == (; mem=Float64[], ends=Int[], rows=Int[], cols=Int[])

    # The snag shape: a per-group design matrix read in an annotated plate.
    annotated = @slic (; y, X) begin
        b1 ~ normal(0.0, 1.0)
        b2 ~ normal(0.0, 1.0)
        @plate for i in eachindex(y)
            y[i] ~ normal(X[i] * [b1, b2], 1.0)
        end
    end
    code = stan_code(annotated)
    @test occursin("tuple(vector[X_mem_n], array[X_ends_n] int, array[X_rows_n] int, array[X_cols_n] int) X;",
        stan_block(code, "data"))
    @test occursin("return to_matrix(rv.1[ragged_start_RaggedMatrix(rv, i):ragged_end_RaggedMatrix(rv, i)], rv.3[i], rv.4[i]);", code)
    @test occursin("~ normal((getindex_RaggedMatrix(X, i) * [b1, b2]'), 1.0);", stan_block(code, "model"))
    @test stanc_compiles(annotated)

    # The data file and the descriptor carry each group's size.
    data = StanBlocks.JSON.parse(stan_data_json(annotated))
    @test data["X"] == Dict("1" => [0.5, 0.1, -0.2, 0.3, 0.2, -0.6, 0.3, 0.4, 0.1, 0.7],
        "2" => [4, 10], "3" => [2, 3], "4" => [2, 2])
    input = only(i for i in stan_descriptor(annotated).inputs if i.name === :X)
    @test input.type === :RaggedMatrix
    @test keys(input.fields) == (:mem, :ends, :rows, :cols)
    @test input.fields.rows.value == [2, 3]
    @test input.fields.cols.value == [2, 2]

    # In the model body: a literal and a data index, and the group count.
    body = @slic (; X, g=2) begin
        b ~ normal(0, 1)
        n = length(X)
        (sum(X[1] * rep_vector(1.0, 2)) + sum(X[g] * rep_vector(1.0, 2)) + n) ~ normal(b, 1.0)
    end
    code = stan_code(body)
    @test occursin("num_elements_RaggedMatrix(X)", code)
    @test occursin("getindex_RaggedMatrix(X, 1)", code)
    @test occursin("getindex_RaggedMatrix(X, g)", code)
    @test stanc_compiles(body)

    # A plate positional: each cell's result length is its matrix's row count.
    positional = @slic (; y, X) begin
        b1 ~ normal(0.0, 1.0)
        b2 ~ normal(0.0, 1.0)
        mu ~ plate(X) do Xi
            Xi * [b1, b2]
        end
        @plate for i in eachindex(y)
            y[i] ~ normal(mu[i], 1.0)
        end
    end
    code = stan_code(positional)
    @test occursin(r"mu__pl_len\[plate_i__pl_\d+\] = X\.3\[plate_i__pl_\d+\];", stan_block(code, "transformed data"))
    @test occursin(r"= \(getindex_RaggedMatrix\(X, plate_i__pl_\d+\) \* \[b1, b2\]'\);", stan_block(code, "transformed parameters"))
    @test stanc_compiles(positional)

    container = @slic (; X) begin
        b ~ normal(0.0, 1.0)
        @plate for Xi in X
            s = sum(Xi * rep_vector(b, 2))
        end
    end
    @test occursin(r"loop_Xi_s\[plate_i__pl_\d+\] = sum\(\(getindex_RaggedMatrix\(X, plate_i__pl_\d+\)", stan_code(container))
    @test stanc_compiles(container)

    # A positional whose group count differs from another positional's is refused.
    err = try
        stan_code(@slic (; y=[y; [[0.5]]], X) begin
            b ~ normal(0.0, 1.0)
            mu ~ plate(X, y) do Xi, yi
                yi ~ normal(Xi[:, 1] * b, 1.0)
                nothing
            end
        end)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("positional `X` (param `Xi`) has 2 groups, but positional `y` (param `yi`) has 3 groups", err)

    # A submodel keyword and a `@deffun` argument; sizes stay data.
    sub = @slic begin
        b ~ normal(0, 1)
        return Z[2] * rep_vector(b, 2)
    end
    composed = @slic (; X) begin
        r ~ sub(; Z=X)
        q ~ normal(sum(r), 1.0)
    end
    @test occursin("vector[X.3[2]] r = (getindex_RaggedMatrix(X, 2) * rep_vector(r_b, 2));", stan_code(composed))
    @test stanc_compiles(composed)

    @deffun ragged_matrix_rowsums(M::RaggedMatrix, g::int)::vector[M.rows[g]] = M[g] * rep_vector(1.0, M.cols[g])
    called = @slic (; X) begin
        b ~ normal(0, 1)
        q ~ normal(b + sum(ragged_matrix_rowsums(X, 2)), 1.0)
    end
    @test occursin("ragged_matrix_rowsums_RaggedMatrix(X, 2)", stan_code(called))
    @test stanc_compiles(called)
end

@testitem "slic: ragged matrix data has the density of its flattened spelling" tags=[:slic, :ragged, :plate, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems, Random
    rng = Xoshiro(20261025)
    X = [randn(rng, 2, 2), randn(rng, 3, 2), randn(rng, 1, 2)]
    y = [randn(rng, size(x, 1)) for x in X]
    matrices = @slic (; y, X) begin
        b1 ~ normal(0.0, 1.0)
        b2 ~ normal(0.0, 1.0)
        @plate for i in eachindex(y)
            y[i] ~ normal(X[i] * [b1, b2], 1.0)
        end
    end
    # The spelling a consumer had to write before: each matrix flattened
    # column-major, its row count as separate data.
    flattened = @slic (; y, Xf=[vec(x) for x in X], n=[size(x, 1) for x in X]) begin
        b1 ~ normal(0.0, 1.0)
        b2 ~ normal(0.0, 1.0)
        @plate for i in eachindex(y)
            y[i] ~ normal(to_matrix(Xf[i], n[i], 2) * [b1, b2], 1.0)
        end
    end
    # An independent plain-Julia density, up to a constant.
    reference(u, X, y) = -sum(abs2, u) / 2 - sum(sum(abs2, y[i] - X[i] * u) for i in eachindex(y)) / 2
    pm, pf = instantiate(matrices), instantiate(flattened)
    points = [[0.0, 0.0], [0.3, -0.7], [-1.1, 0.4]]
    lps = map(points) do u
        lpm, gm = LogDensityProblems.logdensity_and_gradient(pm, u)
        lpf, gf = LogDensityProblems.logdensity_and_gradient(pf, u)
        @test lpm == lpf
        @test gm == gf
        lpm
    end
    @test lps[2] - lps[1] ≈ reference(points[2], X, y) - reference(points[1], X, y) atol = 1e-10
    @test lps[3] - lps[1] ≈ reference(points[3], X, y) - reference(points[1], X, y) atol = 1e-10

    # Rebinding to other groups, with other sizes, re-reads them.
    X2 = [randn(rng, 4, 2), randn(rng, 2, 2)]
    y2 = [randn(rng, size(x, 1)) for x in X2]
    p2 = instantiate(matrices(; y=y2, X=X2))
    lp0, _ = LogDensityProblems.logdensity_and_gradient(p2, points[1])
    lp1, _ = LogDensityProblems.logdensity_and_gradient(p2, points[2])
    @test lp1 - lp0 ≈ reference(points[2], X2, y2) - reference(points[1], X2, y2) atol = 1e-10
end
