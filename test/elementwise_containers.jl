"""
Stan's elementwise operators `.*`, `./` and `.^` take two operands of one native
container type (`vector`, `row_vector`, `matrix`), or a container and a scalar,
and return that container. Only the `vector` forms and `row_vector .* row_vector`
used to trace: a matrix operand (`rain .* (ones_hours * g')`) or a row vector
with a scalar failed with `tracetype not defined for (::matrix .* ::matrix)::anything!`
(snag `matrix-elementwi-c69bbd9a`). `.÷` and the dotted comparisons share the
`Base.BroadcastFunction` head but have no Stan operator for any operand type;
they used to type like `.*` and render `Base.Broadcast.BroadcastFunction(div)(v, s)`
into the program, and now stop at trace time.
"""
@testitem "slic: elementwise .* ./ .^ on vector, row_vector and matrix operands" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles, transpiles

    # The reported shape: a data matrix times a rank-one parameter matrix, as the
    # location of a matrix observation.
    n_hours, n_probes = 4, 3
    reported = @slic (; n_hours, n_probes, rain=rand(n_hours, n_probes),
                       Y=rand(n_hours, n_probes), ones_hours=ones(n_hours)) begin
        g::vector[n_probes] ~ normal(0.0, 1.0)
        Y ~ normal(rain .* (ones_hours * g'), 1.0)
    end
    @test occursin("Y ~ normal((rain .* (ones_hours * (g'))), 1.0);",
                   stan_block(stan_code(reported), "model"))
    @test stanc_compiles(reported)

    # Every operand combination Stan types, for each operator. All operands are
    # parameters that reach the likelihood, so none is re-drawn.
    data = (; m=4, n=3, y=0.3, s=2.0, k=2, A=rand(4, 3))
    model(rhs) = StanBlocks.stan.SlicModel(quote
        v::vector[n] ~ normal(0.0, 1.0)
        w::vector[n] ~ normal(0.0, 1.0)
        rv::row_vector[n] ~ normal(0.0, 1.0)
        rw::row_vector[n] ~ normal(0.0, 1.0)
        X::matrix[m, n] ~ normal(0.0, 1.0)
        Z::matrix[m, n] ~ normal(0.0, 1.0)
        out = $rhs
        y ~ normal(sum(out) + sum(v) + sum(w) + sum(rv) + sum(rw) + sum(X) + sum(Z), 1.0)
    end, data, @__MODULE__)
    cases = [
        (:v, :w) => "vector[n]", (:v, :s) => "vector[n]", (:s, :v) => "vector[n]",
        (:rv, :rw) => "row_vector[n]", (:rv, :s) => "row_vector[n]", (:s, :rv) => "row_vector[n]",
        (:X, :Z) => "matrix[m, n]", (:X, :s) => "matrix[m, n]", (:s, :X) => "matrix[m, n]",
        (:A, :X) => "matrix[m, n]", (:X, :k) => "matrix[m, n]",
    ]
    for op in (:.*, :./, :.^), ((a, b), decl) in cases
        m = model(Expr(:call, op, a, b))
        tp = stan_block(stan_code(m), "transformed parameters")
        @test occursin("$decl out = ($a $op $b);", tp)
        @test stanc_compiles(m)
    end

    # Not built yet: Julia broadcasts these by expanding the lower-dimensional
    # operand, Stan has no such operator, and they stop at trace time
    # (todo `StanBlocks/todos/2026-10-21T10-01-20-936-18g5sfz`).
    for rhs in (:(X .* v), :(X ./ rv), :(v .* rv))
        @test_broken transpiles(model(rhs); re=false)
    end
    # Not built yet: no Stan operator for elementwise integer division or a
    # comparison; they stop at trace time instead of rendering
    # `BroadcastFunction(div)(…)` (todo `StanBlocks/todos/2026-10-21T10-01-22-097-1o15myu`).
    for rhs in (:(v .÷ s), :(X .÷ s), :(v .== w), :(s .< s))
        @test_broken transpiles(model(rhs); re=false)
    end
end
