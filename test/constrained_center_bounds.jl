"""
Stan declares no bound or affine transform on a `simplex` or on a covariance,
correlation or Cholesky-factor matrix, so `p::simplex[3] ~ beta(1, 1)` emitted
the stanc syntax error `simplex<lower=0, upper=1>[3] p;` while `transpiles()`
reported green (todo 09b3kz5). A bound every element already satisfies
(`[0, 1]` for a simplex, `[-1, 1]` for a correlation) restates the type and
leaves the declaration; any other constraint is refused at trace time, naming
the fix. Typed and bare declarations behave alike.
"""
@testitem "slic: a simplex or matrix center declares no bound" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    data = (; y = [0.5], a = [1.0, 1.0, 1.0])
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    params(m) = stan_block(stan_code(m), "parameters")

    # Bounds the type already implies leave the declaration (and the call).
    implied = @slic data begin
        p::simplex[3] ~ beta(1.0, 1.0); y ~ normal(p[1], 1.0)
    end
    @test occursin("simplex[3] p;", params(implied))
    @test stanc_compiles(implied)
    written = @slic data begin
        p::simplex[3] ~ dirichlet(a; lower = 0.0); y ~ normal(p[1], 1.0)
    end
    @test occursin("simplex[3] p;", params(written))
    @test occursin("p ~ dirichlet(a);", stan_block(stan_code(written), "model"))
    @test stanc_compiles(written)
    bare = @slic data begin
        p ~ dirichlet(a; lower = 0.0); y ~ normal(p[1], 1.0)
    end
    @test occursin(r"\n\s*simplex\[a_n\] p;", params(bare))
    @test stanc_compiles(bare)
    corr = @slic data begin
        L::cholesky_factor_corr[3] ~ lkj_corr_cholesky(2.0; lower = -1.0, upper = 1.0)
        y ~ normal(L[2, 1], 1.0)
    end
    @test occursin("cholesky_factor_corr[3] L;", params(corr))
    @test stanc_compiles(corr)

    # Any other constraint is refused by name.
    refused = "Stan declares no bound or affine transform on a `"
    @test occursin("`p::simplex[3]` has `upper=`, but $(refused)simplex`",
        errmsg(@slic(data, begin
            p::simplex[3] ~ dirichlet(a; upper = 0.5); y ~ normal(p[1], 1.0)
        end)))
    @test occursin("has `offset=`, but $(refused)simplex`",
        errmsg(@slic(data, begin
            p::simplex[3] ~ dirichlet(a; offset = 1.0); y ~ normal(p[1], 1.0)
        end)))
    @test occursin("`lower=` (implied by the distribution), but $(refused)cov_matrix`",
        errmsg(@slic(data, begin
            S::cov_matrix[2] ~ lognormal(0.0, 1.0); y ~ normal(S[2, 1], 1.0)
        end)))
    @test occursin("$(refused)cholesky_factor_cov`",
        errmsg(@slic(data, begin
            L::cholesky_factor_cov[2] ~ normal(0.0, 1.0; lower = 0.0); y ~ normal(L[2, 1], 1.0)
        end)))
    @test occursin("A `simplex` declaration has `upper=`, but $(refused)simplex`",
        errmsg(@slic(data, begin
            p ~ dirichlet(a; upper = 0.5); y ~ normal(p[1], 1.0)
        end)))
end
