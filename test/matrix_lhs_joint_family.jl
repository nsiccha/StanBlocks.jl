"""
A joint vector family draws a vector, and Stan takes it on a vector or an array
of vectors, never on a `matrix`: `z::matrix[D, K] ~ multi_normal(…)`
transpiled to `z ~ multi_normal(…)` on `matrix[D, K] z`, which stanc rejects
(`Ill-typed arguments supplied to function "multi_normal"`; todo 0yeqj57). It
is refused at trace time, naming the array-of-vectors and plate spellings; both
of those, and an elementwise family on a matrix, compile.
"""
@testitem "slic: a matrix LHS refuses a joint vector family" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    data = (; D = 3, K = 2, y = [0.1, 0.2, 0.3, 0.4])
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end

    # refused: todo 0yeqj57.
    msg = errmsg(@slic(data, begin
        z::matrix[D, K] ~ multi_normal(rep_vector(0.0, D), diag_matrix(rep_vector(1.0, D)))
        y ~ normal(z[1, 1], 1.0)
    end))
    @test occursin("declares a `matrix`, but its distribution draws a `vector`", msg)
    @test occursin("`z::vector[K, D]`", msg)
    @test occursin("`plate(; outer = K) do k … end`", msg)

    # The spellings it names compile.
    columns = @slic data begin
        z::vector[K, D] ~ multi_normal(rep_vector(0.0, D), diag_matrix(rep_vector(1.0, D)))
        y ~ normal(z[1][1], 1.0)
    end
    @test occursin("array[K] vector[D] z;", stan_block(stan_code(columns), "parameters"))
    @test stanc_compiles(columns)
    plated = @slic data begin
        z ~ plate(; outer = K) do k
            zk::vector[D] ~ multi_normal(rep_vector(0.0, D), diag_matrix(rep_vector(1.0, D)))
            zk
        end
        y ~ normal(z[1, 1], 1.0)
    end
    @test stanc_compiles(plated)
    # An elementwise family keeps its matrix variate.
    elementwise = @slic data begin
        z::matrix[D, K] ~ normal(0.0, 1.0)
        y ~ normal(z[1, 1], 1.0)
    end
    @test stanc_compiles(elementwise)
end
