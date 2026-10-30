"""
`restricted(family, args...; lower, upper)` is Stan's `real<lower=lo> x; x ~
family(args);`: the bounds go on the parameter declaration and the model block
keeps the family's plain kernel, with no truncation normaliser, whatever the
bounds and arguments read. It is the law a hand-written Stan model gets without
`T[,]`, and the one BayesianRegressionModels' `restricted(D, lo, hi)` prior
lowers to (BRM decision `0q18o98`). `truncated(...)` and a family's
`lower=`/`upper=` keep the normalised law. Before, a bounded scalar whose
normaliser moved with a parameter had no plain-kernel spelling at all (snag
`in-slic-declare-996328e0`).
"""
@testitem "slic: restricted(...) bounds the declaration and keeps the plain kernel" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    @deffun begin
        @lpxf plainfam_lpdf(y::real, mu::real) = normal_lpdf(y, mu, 1.)
        plainfam_lpdfs(y::real, mu::real) = normal_lpdf(y, mu, 1.)
        plainfam_rng(mu::real) = normal_rng(mu, 1.)
    end
    # name => (model, parameter declaration, model statement)
    shapes = [
        "token, parameter bound" => (@slic((; y=0.3), begin
            a ~ normal(0., 1.); x ~ restricted(normal, 0., 1.; lower=a); y ~ normal(x, 1.)
        end), "real<lower=a> x;", "x ~ normal(0.0, 1.0);"),
        "call, parameter bound" => (@slic((; y=0.3), begin
            a ~ normal(0., 1.); x ~ restricted(normal(0., 1.); lower=a); y ~ normal(x, 1.)
        end), "real<lower=a> x;", "x ~ normal(0.0, 1.0);"),
        "two-sided, parameter argument" => (@slic((; y=0.3), begin
            m ~ normal(0., 1.); x ~ restricted(normal(m, 1.); lower=-0.5, upper=0.5); y ~ normal(x, 1.)
        end), "real<lower=-0.5, upper=0.5> x;", "x ~ normal(m, 1.0);"),
        "upper only" => (@slic((; y=0.3), begin
            b ~ normal(0., 1.); x ~ restricted(normal(0., 1.); upper=b); y ~ normal(x, 1.)
        end), "real<upper=b> x;", "x ~ normal(0.0, 1.0);"),
        "typed vector" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.); x::vector[2] ~ restricted(normal(mu, 1.); lower=0.); y ~ normal(x, 1.)
        end), "vector<lower=0.0>[2] x;", "x ~ normal(mu, 1.0);"),
        "plate cell" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.)
            theta ~ plate(y; outer=(2,)) do yi
                t ~ restricted(normal(mu, 1.); lower=0.); yi ~ normal(t, 1.); t
            end
        end), "vector<lower=0.0>[2] theta_t;", "theta_t[plate_i__pl_1] ~ normal(mu, 1.0);"),
        "@plate for" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.)
            @plate for i in 1:2
                x[i] ~ restricted(normal(mu, 1.); lower=0.); y[i] ~ normal(x[i], 1.)
            end
        end), "vector<lower=0.0>[2] x;", "x[i] ~ normal(mu, 1.0);"),
        "vcat block" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.); tau ~ vcat(exponential(1.), restricted(normal(mu, 1.); lower=0.)); y ~ normal(tau, 1.)
        end), "vector<lower=0.0>[2] tau;", "tau[2] ~ normal(mu, 1.0);"),
        "slice under a flat declaration" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.)
            tau::vector[2] ~ flat(; lower=0.)
            tau[1] ~ restricted(normal(mu, 1.); lower=0.); tau[2] ~ exponential(1.)
            y ~ normal(tau, 1.)
        end), "vector<lower=0.0>[2] tau;", "tau[1] ~ normal(mu, 1.0);"),
        # No `_lcdf`/`_lccdf` companion: the plain kernel needs none.
        "custom family" => (@slic((; y=0.3), begin
            a ~ normal(0., 1.); m ~ normal(0., 1.); x ~ restricted(plainfam(m); lower=a); y ~ normal(x, 1.)
        end), "real<lower=a> x;", "x ~ plainfam(m);"),
        "positive_ordered" => (@slic((; y=[0.3, 0.2]), begin
            mu ~ normal(0., 1.); x::positive_ordered[2] ~ restricted(normal(mu, 1.); upper=3.); y ~ normal(x, 1.)
        end), "positive_ordered[2] x = lower_upper_bound_jacobian(x_free, 0.0, 3.0);", "x ~ normal(mu, 1.0);"),
        # A bound that restates the family's support is no restriction either.
        "restated support" => (@slic((; y=0.3), begin
            r ~ exponential(1.); x ~ restricted(exponential(r); lower=0.); y ~ normal(x, 1.)
        end), "real<lower=0.0> x;", "x ~ exponential(r);"),
    ]
    for (name, (m, decl, statement)) in shapes
        @testset "$name" begin
            code = stan_code(m)
            @test occursin(decl, code)
            @test occursin(statement, stan_block(code, "model"))
            @test !occursin("conditioning", stan_block(code, "model"))
            @test stanc_compiles(m)
        end
    end

    # The same bound written on the family keeps truncating; `restricted` changes
    # only its own statements.
    kwarg = @slic (; y=0.3) begin
        a ~ normal(0., 1.); x ~ normal(0., 1.; lower=a); w ~ restricted(normal(0., 1.); lower=a); y ~ normal(x + w, 1.)
    end
    model = stan_block(stan_code(kwarg), "model")
    @test occursin("x ~ lower_conditioning_normal(a, 0.0, 1.0);", model)
    @test occursin("w ~ normal(0.0, 1.0);", model)
end

"""
A restricted prior's law, given its arguments, is the truncated law, so a
re-draw in generated quantities is the truncated draw. Over a whole prior-only
program that draw is exact only when the normaliser P(lo <= x <= hi | θ) is a
constant; otherwise the program's joint is the plain kernel, which drops it,
and the prior stays a sampled parameter with everything its bounds and
arguments read. This is how a non-exchangeable `ordered` prior is handled
(decision `StanBlocks:ordered-redraw/decisions/2026-10-20T15-03-22-468-1cht6zo`).
"""
@testitem "slic: a restricted prior is re-drawn only where its law allows" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # Constant normaliser: an exact fixed_param prior draw.
    constant = @slic (;) begin
        x ~ restricted(normal(0., 1.); lower=0.5)
    end
    code = stan_code(constant)
    @test strip(stan_block(code, "parameters")) == "{\n}"
    @test occursin("real x = lower_conditioning_normal_rng(0.5, 0.0, 1.0);", stan_block(code, "generated quantities"))
    @test stanc_compiles(constant)

    # A moving normaliser keeps the prior and its bound's source sampled; what
    # only reads them is still drawn.
    moving = @slic (;) begin
        a ~ normal(0., 1.); x ~ restricted(normal(0., 1.); lower=a); z ~ normal(a, 1.)
    end
    code = stan_code(moving)
    params = stan_block(code, "parameters")
    @test occursin("real a;", params)
    @test occursin("real<lower=a> x;", params)
    @test occursin("x ~ normal(0.0, 1.0);", stan_block(code, "model"))
    @test occursin("real z = normal_rng(a, 1.0);", stan_block(code, "generated quantities"))
    @test stanc_compiles(moving)
    moving_arg = @slic (;) begin
        m ~ normal(0., 1.); x ~ restricted(normal(m, 1.); lower=0.)
    end
    code = stan_code(moving_arg)
    @test occursin("real m;", stan_block(code, "parameters"))
    @test occursin("x ~ normal(m, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(moving_arg)
    moving_cell = @slic (;) begin
        a ~ normal(0., 1.)
        theta ~ plate(; outer=(2,)) do i
            t ~ restricted(normal(0., 1.); lower=a); t
        end
    end
    code = stan_code(moving_cell)
    @test occursin("vector<lower=a>[2] theta_t;", stan_block(code, "parameters"))
    @test occursin("theta_t[plate_i__pl_1] ~ normal(0.0, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(moving_cell)

    # Held out given a fit, it is re-drawn from its law given the fitted bound.
    base = @slic begin
        a ~ normal(0., 1.)
        x ~ restricted(normal(s, 1.); lower=a)
        y ~ normal(a, 1.)
    end
    held = base(; y=[0.1, 0.2], s=StanBlocks.stan.maybecv(:s, 0.3))
    code = stan_code(held)
    @test !occursin(" x;", stan_block(code, "parameters"))
    @test occursin("real x = lower_conditioning_normal_rng(a, s, 1.0);", stan_block(code, "generated quantities"))
    @test stanc_compiles(held)
end

@testitem "slic: restricted refuses an observation and malformed calls" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    message(f) = try
        stan_code(f())
        ""
    catch e
        sprint(showerror, e)
    end
    # refused: on an observation the plain kernel is the bare family, so the
    # bounds would change nothing but a support check; a truncated likelihood is
    # `truncated(...)` (snag `in-slic-declare-996328e0`).
    @test occursin("restricted: `y` is an observation", message(() -> @slic((; y=0.3), begin
        mu ~ normal(0., 1.); y ~ restricted(normal(mu, 1.); lower=0.)
    end)))
    # refused: a restriction without a bound is the bare family; say which.
    @test occursin("restricted: provide at least one of `lower=` or `upper=`", message(() -> @slic((; y=0.3), begin
        x ~ restricted(normal(0., 1.)); y ~ normal(x, 1.)
    end)))
    # refused: a misspelt bound silently dropped would change the model
    # (snag `in-cell-interval-c354f5f4`).
    @test occursin("restricted: no keyword argument `uper`", message(() -> @slic((; y=0.3), begin
        x ~ restricted(normal(0., 1.); uper=1.); y ~ normal(x, 1.)
    end)))
    # refused: a plain kernel of a normalised law has no single meaning; restrict
    # the base family, or truncate it.
    @test occursin("restricted: the family slot takes a base distribution family; `truncated` cannot be nested here",
        message(() -> @slic((; y=0.3), begin
            x ~ restricted(truncated(normal(0., 1.); lower=0.); upper=2.); y ~ normal(x, 1.)
        end)))
end

@testitem "slic: a restricted prior has Stan's plain-kernel log density" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    y = 0.3
    m = @slic (; y) begin
        a ~ normal(0., 1.)
        x ~ restricted(normal(0., 1.); lower=a)
        y ~ normal(x, 1.)
    end
    # Unconstrained (a, u) with x = a + exp(u); the Jacobian is u. No -log Φ(-a).
    hand(q) = begin
        a, x = q[1], q[1] + exp(q[2])
        -a^2 / 2 - x^2 / 2 - (y - x)^2 / 2 + q[2]
    end
    points = ([0.1, -0.3], [-1.5, 0.4], [2.0, -1.0])
    mktempdir() do dir
        cd(dir) do
            prob = stan_instantiate(m)
            for (p, q) in zip(points, points[2:end])
                @test LogDensityProblems.logdensity(prob, p) - LogDensityProblems.logdensity(prob, q) ≈
                    hand(p) - hand(q)
            end

            # The prior-only program samples the same plain-kernel joint.
            prior = @slic (;) begin
                a ~ normal(0., 1.)
                x ~ restricted(normal(0., 1.); lower=a)
            end
            pprob = stan_instantiate(prior)
            @test LogDensityProblems.dimension(pprob) == 2
            phand(q) = hand(q) + (y - (q[1] + exp(q[2])))^2 / 2
            for (p, q) in zip(points, points[2:end])
                @test LogDensityProblems.logdensity(pprob, p) - LogDensityProblems.logdensity(pprob, q) ≈
                    phand(p) - phand(q)
            end

            # Held out given a fit, it is drawn above the fitted bound.
            base = @slic begin
                a ~ normal(0., 1.)
                x ~ restricted(normal(s, 1.); lower=a)
                y ~ normal(a, 1.)
            end
            hprob = stan_instantiate(base(; y=[0.1, 0.2], s=StanBlocks.stan.maybecv(:s, 0.3)))
            rng = BridgeStan.StanRNG(hprob.model, 1234)
            names = BridgeStan.param_names(hprob.model; include_gq=true)
            ix = findfirst(==("x"), names)
            draws = [BridgeStan.param_constrain(hprob.model, [1.7]; include_gq=true, rng)[ix] for _ in 1:200]
            @test all(>=(1.7), draws)
        end
    end
end
