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

"""
Stan vectorises its binary math builtins (`fmin`, `fmax`, `fdim`, `fmod`,
`atan2`, `lbeta`, `log_diff_exp`, `owens_t`, `gamma_p`, `gamma_q`,
`log_modified_bessel_first_kind`), the binaries whose first argument is an int
(`binary_log_loss` and the Bessel functions of an int order) and the ternary
`fma`. Only their scalar forms traced: `mg ./ fmax(nfit, 1.0)` with
`nfit::vector[k]` in a `@deffun` body failed with `tracetype not defined for
fmax(::vector, ::real)::anything!` (snag `in-a-deffun-body-17fa6434`), and a
container call was accepted only as a bare declaration bound. Each container
form stanc 2.39 accepts now types as that container.
"""
@testitem "slic: Stan's vectorised binary builtins trace over containers" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles, transpiles

    # The reported shape: per-group sums divided by a floored count, in a UDF.
    @deffun group_mean(f::vector[N], r::int[N], nfit::vector[k])::vector[k] = begin
        mg = rep_vector(0.0, k)
        for i in 1:N
            mg[r[i]] += f[i]
        end
        mg = mg ./ fmax(nfit, 1.0)
        mg
    end
    reported = @slic (; r = [1, 1, 2, 3], nfit = [2.0, 1.0, 0.0], y = [0.1, 0.2, 0.3]) begin
        f::vector[4] ~ normal(0.0, 1.0)
        y ~ normal(group_mean(f, r, nfit), 1.0)
    end
    @test occursin("mg = (mg ./ fmax(nfit, 1.0));", stan_block(stan_code(reported), "functions"))
    @test stanc_compiles(reported)

    # A container call read by another expression, in a bound and in the body.
    lo_data = (; lo = [0.1, -0.2], y = [1.0, 2.0])
    scaled = @slic lo_data begin
        rho::vector[2] ~ lognormal(0., 4.; lower = 2 * fmax(0.0, lo))
        y ~ normal(sum(rho), 1.)
    end
    @test occursin("vector<lower=(2 * fmax(0.0, lo))>[2] rho;", stan_block(stan_code(scaled), "parameters"))
    @test stanc_compiles(scaled)
    floored = @slic lo_data begin
        z = fmax(0.0, lo)
        y ~ normal(z, 1.)
    end
    @test occursin("vector[lo_n] z = fmax(0.0, lo);", stan_block(stan_code(floored), "transformed data"))
    @test stanc_compiles(floored)

    # Every container row, for every builtin. The real operands are parameters
    # that reach the likelihood, so each call is a transformed parameter; the
    # int operands are data.
    data = (; m = 4, n = 3, y = 0.3, k = 2, ks = [1, 2, 3])
    model(f, args) = StanBlocks.stan.SlicModel(quote
        s ~ normal(0.0, 1.0)
        v::vector[n] ~ normal(0.0, 1.0)
        rv::row_vector[n] ~ normal(0.0, 1.0)
        X::matrix[m, n] ~ normal(0.0, 1.0)
        a = to_array_1d(v)
        $((:($(Symbol(:out, i)) = $f($(arg...))) for (i, arg) in enumerate(args))...)
        y ~ normal($(Expr(:call, :+, :s, :(sum(a)), :(sum(rv)), :(sum(X)),
            (:(sum($(Symbol(:out, i)))) for i in eachindex(args))...)), 1.0)
    end, data, @__MODULE__)
    real_rows = [
        (:s, :v) => "vector[n]", (:v, :s) => "vector[n]", (:v, :v) => "vector[n]",
        (:s, :rv) => "row_vector[n]", (:rv, :s) => "row_vector[n]", (:rv, :rv) => "row_vector[n]",
        (:s, :X) => "matrix[m, n]", (:X, :s) => "matrix[m, n]", (:X, :X) => "matrix[m, n]",
        (:s, :a) => "array[n] real", (:a, :s) => "array[n] real", (:a, :a) => "array[n] real",
        # An int scalar, an int array (Stan promotes it to `array[] real`), the mixed pair.
        (:k, :v) => "vector[n]", (:ks, :s) => "array[ks_n] real", (:a, :ks) => "array[ks_n] real",
    ]
    int_rows = [
        (:k, :v) => "vector[n]", (:k, :rv) => "row_vector[n]", (:k, :X) => "matrix[m, n]",
        (:k, :a) => "array[n] real", (:ks, :s) => "array[ks_n] real", (:ks, :v) => "vector[n]",
        (:ks, :rv) => "row_vector[n]", (:ks, :a) => "array[n] real",
    ]
    fma_rows = [Tuple(slots) => decl
        for (c, decl) in ((:v, "vector[n]"), (:rv, "row_vector[n]"), (:X, "matrix[m, n]"))
        for slots in Iterators.product(ntuple(_ -> (:s, c), 3)...) if c in slots]
    groups = [
        (StanBlocks._ELEMENTWISE_BINARY, real_rows),
        (StanBlocks._ELEMENTWISE_INT_BINARY, int_rows),
        ((StanBlocks.fma,), fma_rows),
    ]
    for (fs, rows) in groups, f in fs
        name = nameof(f)
        m = model(name, first.(rows))
        tp = stan_block(stan_code(m), "transformed parameters")
        for (i, (args, decl)) in enumerate(rows)
            @test occursin("$decl out$i = $name($(join(args, ", ")));", tp)
        end
        @test stanc_compiles(m)
    end

    # refused: stanc 2.39 has no mixed container kinds (`fmax(vector, row_vector)`,
    # `fmax(array[] real, vector)`), no real Bessel order, and no array `fma`.
    for (f, args) in ((:fmax, (:v, :rv)), (:fmin, (:a, :v)), (:bessel_first_kind, (:s, :v)), (:fma, (:a, :s, :s)))
        @test_throws "tracetype" stan_code(model(f, [args]))
    end
    # Not built yet: nested arrays (`array[] vector`, which stanc vectorises too).
    @test_broken transpiles(StanBlocks.stan.SlicModel(quote
        vs::vector[2, n] ~ normal(0.0, 1.0)
        y ~ normal(sum(fmax(vs, 0.0)[1]), 1.0)
    end, data, @__MODULE__); re=false)
end
