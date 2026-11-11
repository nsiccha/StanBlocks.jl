"""
A sampling family's call takes the keywords its `~` statement consumes: the
bounds `lower`/`upper` (a bound truncates), the transform `offset`/`multiplier`
and the shape keywords `type`/`n`/`m`/`o`/`size`. Any other keyword used to be
dropped from the emitted statement without a word, so `y ~ normal(mu, s; uper =
hi)` lost the truncation its author wrote (todo 1o5urqm). The combinators already
refused theirs (snag in-cell-interval-c354f5f4); every family does now, wherever
its `~` stands.
"""
@testitem "slic: a sampling family refuses a keyword it does not take" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    const SB = StanBlocks
    trace_error(model) = try
        stan_code(model)
        ""
    catch err
        sprint(showerror, err)
    end
    refuses(model, family, key) = begin
        msg = trace_error(model)
        ok = occursin("$(family): no keyword argument `$(key)`", msg) &&
            occursin("`lower=`/`upper=`", msg)
        ok || println("unexpected outcome for `", family, "(…; ", key, ")`: ", first(msg, 600))
        ok
    end

    @deffun begin
        @lhs @lpxf kwfam_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)
    end
    value = ValueFamily(:kwvalue, :lpdf,
        :(kwvalue_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)),
        :(kwvalue_lpdfs(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)),
        :(kwvalue_rng(mu::real)::real = normal_rng(mu, 1.0)), @__MODULE__)

    # The reported spelling: a misspelt bound on an observation.
    @test refuses(@slic((; y = 0.3, hi = 2.0), begin
        mu ~ normal(0.0, 1.0); y ~ normal(mu, 0.8; uper = hi)
    end), :normal, :uper)
    # A parameter, bare and typed.
    @test refuses(@slic((; y = 0.3), begin
        mu ~ normal(0.0, 1.0; foo = 2.0); y ~ normal(mu, 0.8)
    end), :normal, :foo)
    @test refuses(@slic((; y = [0.3, 0.1]), begin
        mu::vector[2] ~ normal(0.0, 1.0; foo = 2.0); y ~ normal(mu, 0.8)
    end), :normal, :foo)
    # Families whose only arguments are keywords.
    @test refuses(@slic((; y = 0.3), begin
        mu ~ flat(; lower = 0.0, foo = 1.0); y ~ normal(mu, 0.8)
    end), :flat, :foo)
    @test refuses(@slic((; y = [0.3, 0.2]), begin
        mu ~ std_normal(; n = 2, foo = 1.0); y ~ normal(mu, 0.8)
    end), :std_normal, :foo)
    # A discrete family.
    @test refuses(@slic((; y = [1, 2]), begin
        lam ~ exponential(1.0); y ~ poisson(lam; foo = 1.0)
    end), :poisson, :foo)
    # A plate cell's prior.
    @test refuses(@slic((; y = [0.3, 0.1]), begin
        s ~ exponential(1.0)
        p ~ plate(; outer = (2,)) do g
            c ~ normal(0.0, s; foo = 1.0); c
        end
        y ~ normal(p, 1.0)
    end), :normal, :foo)
    # A custom `@lpxf` family and a value family.
    @test refuses(@slic((; y = 0.3), begin
        mu ~ normal(0.0, 1.0); y ~ kwfam(mu; foo = 1.0)
    end), :kwfam, :foo)
    value_rhs = Expr(:call, value, Expr(:parameters, Expr(:kw, :foo, 1.0)), 0.3)
    @test refuses(SB.SlicModel(Expr(:call, :~, :y, value_rhs),
        Dict{Symbol,Any}(:y => 0.5), @__MODULE__), :kwvalue, :foo)
    # The family call in a combinator's family slot.
    @test occursin("no keyword argument `foo`", trace_error(@slic((; y = 0.3, lo = 0.0), begin
        mu ~ normal(0.0, 1.0); y ~ censored(normal(mu, 1.0; foo = 1.0); lower = lo)
    end)))

    # Every keyword a family takes still passes, and the program is valid Stan.
    accepted = @slic((; y = [0.3, 0.2], hi = 2.0), begin
        z ~ std_normal(; n = 2)
        s ~ normal(0.0, 1.0; lower = 0.0)
        a ~ normal(1.0, 2.0; offset = 1.0, multiplier = 2.0)
        b ~ restricted(normal, 0.0, 1.0; lower = 0.0)
        w ~ flat(; lower = 0.0, upper = 1.0)
        y ~ normal(z .+ a .+ b .+ w, s; upper = hi)
    end)
    code = stan_code(accepted)
    @test occursin("vector[2] z", code)
    @test occursin("<offset=1.0, multiplier=2.0>", code)
    @test stanc_compiles(accepted)
end
