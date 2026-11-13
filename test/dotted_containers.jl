"""
A dotted call `f.(x)` is elementwise, as in Julia, over a `matrix` and a
`row_vector` too, not only over a `vector` or a scalar array (snag
`dotted-call-over-fc81a4d9`). It used to stop at trace time, naming the undotted
`exp(M)` as the fix; but a cell that also runs as plain Julia reads `exp(M)` as
the matrix exponential, so no spelling meant elementwise on both backends. A
Stan builtin that applies elementwise over native containers (builtin.jl
`_ELEMENTWISE_*`) now emits Stan's own vectorized call; any other function runs
the `jbroadcasted` element loop, a matrix flattened column-major and folded back.
"""
@testitem "slic: dotted calls over matrix and row_vector arguments" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # A downstream module's shape: a per-cell (K-1)×K matrix of logits in a plate,
    # turned elementwise into positive weights and their logs.
    plate_cells = @slic (; obs = [0.1, 0.2, 0.3], K = 3) begin
        th ~ plate(; outer = 3) do i
            logits::matrix[K - 1, K] ~ normal(0, 1)
            E = exp.(logits)
            G = E ./ sum(E)
            sum(log.(G))
        end
        obs ~ normal(th, 1)
    end
    code = stan_code(plate_cells)
    @test occursin(r"\bexp\(th_logits\[", code)
    @test occursin(r"\blog\(", code)
    @test !occursin(r"jbroadcasted_(exp|log)\b", code)
    @test stanc_compiles(plate_cells)

    # Unary and binary builtins over a matrix, a row vector and a vector emit
    # the vectorized Stan call, which keeps the container.
    native = @slic (; M = [1.0 -2.0; 3.0 4.0], w = [1.0, 2.0, 3.0]) begin
        G::matrix[2, 2] ~ normal(0, 1)
        v::vector[3] ~ normal(0, 1)
        rv = to_row_vector(w)
        A = inv_logit.(G)
        B = fmax.(M, 0.0)
        C = fmax.(M, G)
        D = log1p.(exp.(rv))
        F = exp.(v)
        z ~ normal(sum(A) + sum(B) + sum(C) + sum(D) + sum(F), 1)
    end
    code = stan_code(native)
    @test occursin("inv_logit(G)", code)
    @test occursin("fmax(M, 0.0)", code)
    @test occursin("fmax(M, G)", code)
    @test occursin("log1p(exp(", code)
    @test occursin("exp(v)", code)
    @test !occursin(r"jbroadcasted_(inv_logit|fmax|log1p|exp)\b", code)
    @test stanc_compiles(native)

    # Any other function runs the element loop: over a matrix every matrix
    # argument is flattened column-major and the result folded back to the
    # matrix shape; over a row vector the result is converted back to a row.
    looped = @slic (; M = [1.0 2.0; 3.0 4.0], w = [1.0, 2.0, 3.0]) begin
        a ~ normal(0, 1)
        rv = to_row_vector(w)
        X = log_sum_exp.(M, M)
        Y = (x -> x * a).(M)
        R = log_sum_exp.(rv, a)
        z ~ normal(sum(X) + sum(Y) + sum(R), 1)
    end
    code = stan_code(looped)
    @test occursin(r"to_matrix\(jbroadcasted_log_sum_exp\(to_vector\(M\), to_vector\(M\)\)", code)
    @test occursin(r"to_row_vector\(jbroadcasted_log_sum_exp\(", code)
    @test stanc_compiles(looped)

    # An int element over a matrix keeps the element type, as Julia's
    # `Matrix{Int}` result: a 2-D int array.
    @deffun dotted_positive_flag(x::real)::int = x > 0 ? 1 : 0
    int_cells = @slic (; M = [1.0 -2.0; 3.0 4.0]) begin
        S = dotted_positive_flag.(M)
        z ~ normal(sum(S), 1)
    end
    code = stan_code(int_cells)
    @test occursin(r"array\[M_m, M_n\] int S = reshape\(jbroadcasted_dotted_positive_flag\(to_vector\(M\)\), M_m, M_n\);", code)
    @test stanc_compiles(int_cells)

    # Arguments of different containers would expand in Julia; refused with the
    # tracking todo's shape named, not lowered with the wrong meaning.
    err = try
        stan_code(@slic (; M = [1.0 2.0; 3.0 4.0], v = [1.0, 2.0]) begin
            V::vector[2] ~ normal(0, 1)
            Y = fmax.(M, V)
            z ~ normal(sum(Y), 1)
        end)
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("mixes a `matrix", sprint(showerror, err))

    # A matrix row traces as a `row_vector`; it reshapes column-major like a
    # vector (the same snag's report).
    reshaped = @slic (; M = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]) begin
        R = reshape(M[1, :], 2, 2)
        z ~ normal(sum(exp.(R)), 1)
    end
    @test occursin(r"matrix\[2, 2\] R = reshape\(", stan_code(reshaped))
    @test stanc_compiles(reshaped)
end
