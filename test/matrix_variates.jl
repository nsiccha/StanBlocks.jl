"""
Stan vectorises its elementwise families (`normal`, `lognormal`, `exponential`,
…) over a scalar, `vector`, `row_vector` or 1-D array variate, never a
`matrix`. A typed matrix prior `z::matrix[m, n] ~ normal(0, 1)` and a matrix
observation `Y ~ normal(M, s)` used to transpile to a statement stanc rejects
(`Ill-typed arguments supplied to function "normal": (matrix, real, real)`), and
their generated-quantities draws had no companion, so a prior-only program failed
at trace time (snag `matrix-parameter-94e1f96c`). Each continuous elementwise
family now has matrix overloads of its density, pointwise companion and sized
`_rng`; every argument is a scalar or a matrix of the variate's shape.
"""
@testitem "slic: a matrix variate of an elementwise family compiles, fitted and prior-only" tags=[:slic, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    D, K = 3, 2
    # A location prior on a size-expression matrix and a positive scale matrix, fitted.
    fitted = @slic (; D, K, y=[0.1, -0.2]) begin
        loc::matrix[D + 1, K] ~ normal(0.0, 2.0)
        scale::matrix[D, K] ~ lognormal(0.0, 0.5)
        y ~ normal(loc[1, 1] + scale[2, 2], 1.0)
    end
    code = stan_code(fitted)
    @test occursin("matrix[(D + 1), K] loc;", stan_block(code, "parameters"))
    @test occursin("matrix<lower=0.0>[D, K] scale;", stan_block(code, "parameters"))
    @test occursin("loc ~ normal(0.0, 2.0);", stan_block(code, "model"))
    @test occursin("scale ~ lognormal(0.0, 0.5);", stan_block(code, "model"))
    # The density overload defers to `_lupdf`, so a `~` drops constants as usual.
    @test occursin(
        "return normal_lupdf(to_vector(obs) | matrix_variate_arg(loc, m, n), matrix_variate_arg(scale, m, n));",
        code)
    @test stanc_compiles(fitted)

    # Prior-only: no sampler parameters, the matrix re-drawn with its own shape.
    prior = @slic (; D, K) begin
        loc::matrix[D + 1, K] ~ normal(0.0, 2.0)
    end
    code = stan_code(prior)
    @test !occursin("loc", stan_block(code, "parameters"))
    @test occursin("matrix[(D + 1), K] loc = normal_matrix_rng(((D + 1), K), 0.0, 2.0);",
        stan_block(code, "generated quantities"))
    @test stanc_compiles(prior)

    # Families of every arity, scalar and matrix arguments mixed, int literals.
    M = [0.5 1.0; 1.5 2.0; 2.5 3.0]
    families = @slic (; D, K, M, y=[0.1]) begin
        a::matrix[D, K] ~ std_normal()
        b::matrix[D, K] ~ exponential(1)
        c::matrix[D, K] ~ normal(M, 1.0)
        d::matrix[D, K] ~ student_t(3.0, 0.0, exp(M))
        e::matrix[D, K] ~ gamma(M, 2.0)
        y ~ normal(a[1, 1] + b[1, 1] + c[1, 1] + d[1, 1] + e[1, 1], 1.0)
    end
    @test stanc_compiles(families)
    # A cv-held-out matrix prior re-draws through the same companion.
    held = @slic (; D, K, y=[0.1], mu0=StanBlocks.stan.maybecv(:mu0, 0.0)) begin
        z::matrix[D, K] ~ normal(mu0, 1.0)
        y ~ normal(z[1, 1], 1.0)
    end
    @test occursin("matrix[D, K] z = normal_matrix_rng((D, K), mu0, 1.0);",
        stan_block(stan_code(held), "generated quantities"))
    @test stanc_compiles(held)
    # A bare override keeps the matrix declaration and swaps the family.
    swapped = Base.merge(held, :(z ~ cauchy(0.0, 2.0)))
    @test occursin("z ~ cauchy(0.0, 2.0);", stan_code(swapped))
    @test stanc_compiles(swapped)

    # A matrix observation: matrix-shaped twins, all three operations.
    obs = @slic (; Y=[0.1 0.2; 0.3 0.4; 0.5 0.6], M) begin
        s ~ exponential(1.0)
        Y ~ normal(M, s)
    end
    code = stan_code(obs)
    @test occursin("Y ~ normal(M, s);", stan_block(code, "model"))
    @test stanc_compiles(obs)
    d = stan_descriptor(obs)
    @test Set(op.name for op in d.operations) ⊇ Set([:fit, :predict, :pointwise_loglik])
    outputs = Dict(o.name => o for o in d.outputs)
    @test outputs[:Y_gen].generative === :draw && outputs[:Y_gen].dims == (3, 2)
    @test outputs[:Y_likelihood].generative === :pointwise_loglik &&
        outputs[:Y_likelihood].dims == (3, 2)
end

@testitem "slic: a matrix variate refuses a non-matrix container argument" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    D, K = 3, 2
    v = [0.1, 0.2, 0.3]
    # refused: Stan's elementwise families take each argument as a scalar or a
    # container of the variate's own shape; a vector names no axis of a matrix,
    # and `rep_matrix` states the intended one (Stan vectorisation rule).
    param = @slic (; D, K, v, y=[0.1]) begin
        z::matrix[D, K] ~ normal(v, 1.0)
        y ~ normal(z[1, 1], 1.0)
    end
    err = try stan_code(param); nothing catch e; e end
    @test err !== nothing
    @test occursin("argument 1 of `normal` is a `vector`", sprint(showerror, err))
    @test occursin("rep_matrix(mu, n)", sprint(showerror, err))
    observed = @slic (; Y=[0.1 0.2; 0.3 0.4; 0.5 0.6], v) begin
        s ~ exponential(1.0)
        Y ~ normal(v, s)
    end
    err = try stan_code(observed); nothing catch e; e end
    @test err !== nothing
    @test occursin("`Y ~ normal(…)` samples a `matrix`", sprint(showerror, err))
    # The suggested rewrite compiles.
    fixed = @slic (; D, K, v, y=[0.1]) begin
        z::matrix[D, K] ~ normal(rep_matrix(v, K), 1.0)
        y ~ normal(z[1, 1], 1.0)
    end
    @test transpiles(fixed)
end

@testitem "slic: a matrix variate has the density of its to_vector spelling" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    D, K = 3, 2
    y = [0.3, -0.4]
    as_matrix = @slic (; D, K, y) begin
        mu ~ normal(0.0, 1.0)
        s ~ exponential(1.0)
        z::matrix[D, K] ~ normal(mu, s)
        y ~ normal(z[2, 1] + z[3, 2], 1.0)
    end
    as_vector = @slic (; D, K, y) begin
        mu ~ normal(0.0, 1.0)
        s ~ exponential(1.0)
        zv::vector[D * K] ~ normal(mu, s)
        z = to_matrix(zv, D, K)
        y ~ normal(z[2, 1] + z[3, 2], 1.0)
    end
    Y = [0.1 -0.2; 0.3 0.4; -0.5 0.6]
    M = [0.5 1.0; 1.5 2.0; 2.5 3.0]
    obs_matrix = @slic (; Y, M) begin
        s ~ exponential(1.0)
        Y ~ normal(M, s)
    end
    obs_vector = @slic (; y=vec(Y), m=vec(M)) begin
        s ~ exponential(1.0)
        y ~ normal(m, s)
    end
    prior = @slic (; M) begin
        z::matrix[3, 2] ~ normal(100.0 * M, 0.001)
    end
    mktempdir() do dir
        cd(dir) do
            pm, pv = stan_instantiate(as_matrix), stan_instantiate(as_vector)
            @test LogDensityProblems.dimension(pm) == LogDensityProblems.dimension(pv) == 8
            rng = Random.Xoshiro(1)
            for _ in 1:5
                q = 0.3 .* randn(rng, 8)
                lm, gm = LogDensityProblems.logdensity_and_gradient(pm, q)
                lv, gv = LogDensityProblems.logdensity_and_gradient(pv, q)
                @test lm ≈ lv
                @test gm ≈ gv
            end
            qm, qv = stan_instantiate(obs_matrix), stan_instantiate(obs_vector)
            @test LogDensityProblems.logdensity(qm, [0.2]) ≈ LogDensityProblems.logdensity(qv, [0.2])
            d = stan_descriptor(obs_matrix)
            lik = stan_execute(d, :pointwise_loglik; problem=qm, draws=[0.2], seed=1).Y_likelihood
            ref = stan_execute(stan_descriptor(obs_vector), :pointwise_loglik; problem=qv, draws=[0.2], seed=1).y_likelihood
            @test lik ≈ ref
            # A prior-only draw lands element by element on its location.
            pp = stan_instantiate(prior)
            @test LogDensityProblems.dimension(pp) == 0
            draw = BridgeStan.param_constrain(pp.model, Float64[]; include_gq=true,
                rng=BridgeStan.StanRNG(pp.model, 7))
            @test maximum(abs.(reshape(draw, 3, 2) .- 100.0 .* M)) < 0.01
            # A matrix argument of the wrong shape is rejected at run time.
            bad = stan_instantiate(obs_matrix(; M=permutedims(M)))
            @test_throws ErrorException BridgeStan.log_density(bad.model, [0.2])
        end
    end
end
