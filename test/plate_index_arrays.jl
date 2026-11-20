"""
A deterministic `array[] int` cell binding computed from data alone, such as a
permutation `sort_indices_asc(x)` or `findall(c .== 1)`, is stored once in
`transformed data` and read back per cell, like a data-only vector or matrix
cell binding (snag `use-a-native-sli-0c201703`). The binding used to be dropped
from promotion and inlined into its uses, which only reached a binding written
at the cell's top level: one inside a called `@slic` sub-model was left as a
loop local in the transformed-data copy of the loop, and stanc refused the
transformed-parameters copy that read it ("Identifier ... not in scope").
An int array that reads a parameter cannot be stored (Stan has no integer
parameters) and is still computed in the block that uses it, and a pure alias
(`c = cs[i]`) computes nothing and stays inline.
"""
@testitem "slic: a data-only int-array cell binding is stored once in transformed data" tags=[:slic, :plate, :ragged, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    using LogDensityProblems

    @slic sorted_scaled(x, b) = begin
        perm = sort_indices_asc(x)
        return (b * x)[perm]
    end
    xs = [[1.2, 0.4, 0.9], [1.0, 0.0], [0.3, -0.2, 0.8, 0.1]]
    y = [0.2, 0.5, -0.1]

    # The snag's shape: the sub-model's index array, in a ragged plate.
    called = @slic (; xs, y) begin
        b ~ normal(0.0, 1.0)
        z ~ plate(xs) do x
            p ~ sorted_scaled(x, b)
            sum(p .* p)
        end
        y ~ normal(z, 1.0)
    end
    code = stan_code(called)
    td = stan_block(code, "transformed data")
    tp = stan_block(code, "transformed parameters")
    @test occursin("array[sum(z_p_perm__pl_len)] int z_p_perm;", td)
    @test occursin("z_p_perm[ragged_start(z_p_perm__pl_end, plate_i__pl_1):ragged_end(z_p_perm__pl_end, plate_i__pl_1)] = sort_indices_asc(", td)
    @test occursin("z_p_perm[ragged_start(z_p_perm__pl_end, plate_i__pl_1):ragged_end(z_p_perm__pl_end, plate_i__pl_1)]", tp)
    @test !occursin("sort_indices_asc", tp)
    @test stanc_compiles(called)

    # The same body at the cell's top level is stored the same way.
    top = @slic (; xs, y) begin
        b ~ normal(0.0, 1.0)
        z ~ plate(xs) do x
            perm = sort_indices_asc(x)
            p = (b * x)[perm]
            sum(p .* p)
        end
        y ~ normal(z, 1.0)
    end
    code = stan_code(top)
    @test occursin("array[sum(z_perm__pl_len)] int z_perm;", stan_block(code, "transformed data"))
    @test !occursin("sort_indices_asc", stan_block(code, "transformed parameters"))
    @test stanc_compiles(top)

    # Both have the log density of a spelling with no index array.
    reference = @slic (; xs, y) begin
        b ~ normal(0.0, 1.0)
        z ~ plate(xs) do x
            p = b * sort_asc(x)
            sum(p .* p)
        end
        y ~ normal(z, 1.0)
    end
    pr = instantiate(reference)
    for m in (called, top)
        p = instantiate(m)
        @test LogDensityProblems.dimension(p) == 1
        for q in ([0.3], [-1.1], [2.0])
            lp, g = LogDensityProblems.logdensity_and_gradient(p, q)
            lr, gr = LogDensityProblems.logdensity_and_gradient(pr, q)
            @test lp ≈ lr
            @test g ≈ gr
        end
    end

    # Fixed-width cells: an `array[N, K] int` filled per cell.
    dense = @slic (; X = [1.2 0.4; 0.9 1.0; 0.0 0.3], y = [0.2, 0.5]) begin
        b ~ normal(0.0, 1.0)
        z ~ plate(EachCol(X)) do x
            p ~ sorted_scaled(x, b)
            sum(p .* p)
        end
        y ~ normal(z, 1.0)
    end
    code = stan_code(dense)
    @test occursin("array[cols(X), X_m] int z_p_perm;", stan_block(code, "transformed data"))
    @test occursin("[z_p_perm[plate_i__pl_1]]", stan_block(code, "transformed parameters"))
    @test stanc_compiles(dense)

    # Several plate axes.
    axes = @slic (; A = reshape(collect(1.0:24.0), 2, 3, 4), y = zeros(2, 3)) begin
        b ~ normal(0.0, 1.0)
        z ~ plate(; outer = (2, 3)) do i, j
            perm = sort_indices_desc(to_vector(A[i, j, :]))
            b * to_vector(A[i, j, :])[perm][1]
        end
        y ~ normal(z, 1.0)
    end
    code = stan_code(axes)
    @test occursin(r"array\[2, 3, \w+\] int z_perm;", stan_block(code, "transformed data"))
    @test !occursin("sort_indices_desc", stan_block(code, "transformed parameters"))
    @test stanc_compiles(axes)

    # An annotated loop's data-only index array.
    looped = @slic (; xs, ys = [[0.1, 0.2, 0.3], [0.5, 0.4], [0.0, 0.1, 0.2, 0.3]]) begin
        b ~ normal(0.0, 1.0)
        @plate for i in eachindex(ys)
            perm = sort_indices_asc(xs[i])
            ys[i] ~ normal((b * xs[i])[perm], 1.0)
        end
    end
    code = stan_code(looped)
    @test occursin("array[sum(loop_i_perm__pl_len)] int loop_i_perm;", stan_block(code, "transformed data"))
    @test !occursin("sort_indices_asc", stan_block(code, "model"))
    @test stanc_compiles(looped)

    # A pure alias of a data int array computes nothing: it stays inline, so the
    # program is the direct spelling's (no transformed-data copy).
    cs = [[3, 1, 2], [2, 1], [4, 2, 1, 3]]
    aliased = @slic (; xs, cs, ys = xs) begin
        b ~ normal(0.0, 1.0)
        @plate for i in eachindex(ys)
            c = cs[i]
            ys[i] ~ normal(b * xs[i][c], 1.0)
        end
    end
    plain = @slic (; xs, cs, ys = xs) begin
        b ~ normal(0.0, 1.0)
        @plate for i in eachindex(ys)
            ys[i] ~ normal(b * xs[i][cs[i]], 1.0)
        end
    end
    @test stan_code(aliased) == stan_code(plain)
    @test stanc_compiles(aliased)

    # An index array that reads a parameter stays where it is used.
    @slic param_sorted(x, b) = begin
        perm = sort_indices_asc(b .* x)
        return x[perm]
    end
    for m in (
        (@slic (; xs, y) begin
            b ~ normal(0.0, 1.0)
            z ~ plate(xs) do x
                p ~ param_sorted(x, b)
                b * p[1]
            end
            y ~ normal(z, 1.0)
        end),
        (@slic (; xs, y) begin
            b ~ normal(0.0, 1.0)
            z ~ plate(xs) do x
                perm = sort_indices_asc(b .* x)
                b * x[perm][1]
            end
            y ~ normal(z, 1.0)
        end),
    )
        code = stan_code(m)
        @test !occursin("sort_indices_asc", stan_block(code, "transformed data"))
        @test occursin("sort_indices_asc((b .* ", code)
        @test stanc_compiles(m)
    end

    # A ragged plate inside a called sub-model keeps the inline index: its
    # flat-memory carriers are not wired there yet.
    @slic sorted_cells(xs, b) = begin
        z ~ plate(xs) do x
            perm = sort_indices_asc(x)
            sum((b * x)[perm])
        end
        return z
    end
    nested = @slic (; xs, y) begin
        b ~ normal(0.0, 1.0)
        z ~ sorted_cells(xs, b)
        y ~ normal(z, 1.0)
    end
    code = stan_code(nested)
    @test occursin("sort_indices_asc(", stan_block(code, "transformed parameters"))
    @test stanc_compiles(nested)
end
