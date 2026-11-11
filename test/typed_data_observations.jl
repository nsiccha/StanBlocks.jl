"""
Binding a sampled name conditions on it (`stanblocks-use` §11), and `y ~ dist`
on a data-bound `y` observes it. The typed spelling `y::T ~ dist` ignored the
binding: it overwrote the data with a fresh parameter, which prior-only lowering
turned into a draw, so the data was dropped silently and the program simulated
the prior (todo 1h0fnry). It observes the data now, declared with the type `T`
names (center, sizes, and the bounds that read data alone), so Stan checks the
data against `T` as it loads it.
"""
@testitem "slic: a typed sampling statement on a data-bound name observes the data" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end

    # The two reported repros.
    scalar = @slic (; sigma = 1.0) begin
        sigma::real ~ exponential(1.0)
    end
    code = stan_code(scalar)
    @test occursin("real<lower=0.0> sigma;", stan_block(code, "data"))
    @test occursin("sigma ~ exponential(1.0);", stan_block(code, "model"))
    @test !occursin("real sigma = ", stan_block(code, "generated quantities"))
    @test stanc_compiles(scalar)
    cholesky = @slic (; K = 2, Omega = [1.0 0.0; 0.6 0.8]) begin
        Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
    end
    code = stan_code(cholesky)
    @test occursin("cholesky_factor_corr[K] Omega;", stan_block(code, "data"))
    @test occursin("Omega ~ lkj_corr_cholesky(2.0);", stan_block(code, "model"))
    @test stanc_compiles(cholesky)

    # The typed and untyped spellings fit the same model.
    typed = @slic (; y = [0.3, 0.1]) begin
        mu ~ normal(0.0, 1.0); y::vector[2] ~ normal(mu, 1.0)
    end
    untyped = @slic (; y = [0.3, 0.1]) begin
        mu ~ normal(0.0, 1.0); y ~ normal(mu, 1.0)
    end
    @test stan_block(stan_code(typed), "model") == stan_block(stan_code(untyped), "model")
    @test strip(stan_block(stan_code(typed), "parameters")) == "{\n    real mu;\n}"
    # A use of the data before the typed `~` keeps the declared type.
    earlier = @slic (; y = [0.3, 0.1]) begin
        mu ~ normal(0.0, 1.0); m = sum(y); y::vector[2] ~ normal(mu + m, 1.0)
    end
    @test occursin("vector[2] y;", stan_block(stan_code(earlier), "data"))
    @test stanc_compiles(earlier)
    # An integer datum may be declared real.
    @test occursin("real k;", stan_block(stan_code(@slic((; k = 3), begin
        mu ~ normal(0.0, 1.0); k::real ~ normal(mu, 1.0)
    end)), "data"))
    # Data a sub-model receives by keyword is the caller's, observed under its name.
    inner = @slic begin
        m ~ normal(0.0, 1.0)
        obs::vector[2] ~ normal(m, 1.0)
    end
    caller = @slic (; y = [0.3, 0.1]) begin
        node ~ inner(; obs = y)
    end
    code = stan_code(caller)
    @test occursin("y ~ normal(node_m, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(caller)

    # Refused: a declared type that cannot hold the data, and a typed `~` on a
    # name bound to a parameter.
    @test occursin("which a `real` cannot hold", errmsg(@slic((; y = [0.3, 0.1]), begin
        mu ~ normal(0.0, 1.0); y::real ~ normal(mu, 1.0)
    end)))
    @test occursin("only data-qualified LHS", errmsg(@slic((; y = 0.3), begin
        mu ~ normal(0.0, 1.0); mu::real ~ normal(0.0, 2.0); y ~ normal(mu, 1.0)
    end)))
end

@testitem "slic: data observed through a typed statement is checked against its type on load" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    model(sigma) = @slic (; sigma) begin
        sigma::real ~ exponential(1.0)
    end
    msg = try
        instantiate(stan_model(model(-1.0)))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("sigma is -1, but must be greater than or equal to 0", msg)
    problem = instantiate(stan_model(model(0.5)))
    @test LogDensityProblems.dimension(problem) == 0
end
