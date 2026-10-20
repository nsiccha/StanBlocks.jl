"""
A `lower=`/`upper=` bound on a parameter prior is a truncation of the family
(`stanblocks-use` §8). When the cut moves with a parameter — `x ~ normal(mu, 1;
lower=0)` — the model block must carry the normaliser, exactly as the
`truncated(normal, mu, 1; lower=0)` spelling does. Before this, the kwarg spelling
emitted the bare `x ~ normal(mu, 1.0)`, which gives `mu` the prior
`N(mu | 0, 1)·Φ(mu)`, while the generated-quantities re-draw already drew from
the truncated law (todo `1l1odrx`, found under snag `heterogeneous-pe-7668c283`).
A provably constant normaliser keeps the lean native statement. An observation's
bound used to be ignored outright (no support check, unnormalised likelihood,
untruncated `<obs>_gen`); it now samples from the `truncated` call whatever its
normaliser. Decision `2026-10-20T14-59-36-177-0w4wyro`.
"""
@testitem "slic: a kwarg bound with a parameter-dependent cut emits the truncated density" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # Each kwarg spelling emits exactly the Stan its `truncated` spelling emits.
    spellings = [
        "scalar" => (
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ normal(mu, 1.0; lower=0.); y ~ normal(x, 1.0)
            end),
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ truncated(normal, mu, 1.0; lower=0.); y ~ normal(x, 1.0)
            end)),
        "upper" => (
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ normal(mu, 1.0; upper=0.); y ~ normal(x, 1.0)
            end),
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ truncated(normal, mu, 1.0; upper=0.); y ~ normal(x, 1.0)
            end)),
        "two-sided" => (
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ normal(mu, 1.0; lower=-1., upper=1.); y ~ normal(x, 1.0)
            end),
            @slic((; y=0.3), begin
                mu ~ normal(0., 1.); x ~ truncated(normal, mu, 1.0; lower=-1., upper=1.); y ~ normal(x, 1.0)
            end)),
        # The cut point itself is a parameter.
        "parameter bound" => (
            @slic((; y=0.3), begin
                lo ~ normal(0., 1.); x ~ normal(0., 1.0; lower=lo); y ~ normal(x, 1.0)
            end),
            @slic((; y=0.3), begin
                lo ~ normal(0., 1.); x ~ truncated(normal, 0., 1.0; lower=lo); y ~ normal(x, 1.0)
            end)),
        # A bound inside a family's support with a parameter rate.
        "exponential lower=1" => (
            @slic((; y=0.3), begin
                r ~ exponential(1.); x ~ exponential(r; lower=1.); y ~ normal(x, 1.0)
            end),
            @slic((; y=0.3), begin
                r ~ exponential(1.); x ~ truncated(exponential, r; lower=1.); y ~ normal(x, 1.0)
            end)),
        "typed vector" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.); x::vector[2] ~ normal(mu, 1.0; lower=0.); y ~ normal(x, 1.0)
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.); x::vector[2] ~ truncated(normal, mu, 1.0; lower=0.); y ~ normal(x, 1.0)
            end)),
        "plate cell" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                theta ~ plate(y; outer=(2,)) do yi
                    t ~ normal(mu, 1.0; lower=0.); yi ~ normal(t, 1.0); t
                end
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                theta ~ plate(y; outer=(2,)) do yi
                    t ~ truncated(normal, mu, 1.0; lower=0.); yi ~ normal(t, 1.0); t
                end
            end)),
        "@plate for" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                @plate for i in 1:2
                    x[i] ~ normal(mu, 1.0; lower=0.); y[i] ~ normal(x[i], 1.0)
                end
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                @plate for i in 1:2
                    x[i] ~ truncated(normal, mu, 1.0; lower=0.); y[i] ~ normal(x[i], 1.0)
                end
            end)),
        # The bound may come from a `flat(; lower=…)` declaration, not a kwarg.
        "flat-declared slices" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                tau::vector[2] ~ flat(; lower=0.)
                tau[1] ~ normal(mu, 1.); tau[2] ~ normal(mu, 1.; lower=0.)
                y ~ normal(tau, 1.0)
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                tau::vector[2] ~ flat(; lower=0.)
                tau[1] ~ truncated(normal, mu, 1.; lower=0.); tau[2] ~ truncated(normal, mu, 1.; lower=0.)
                y ~ normal(tau, 1.0)
            end)),
        # `positive_ordered` IS the `lower=0` support.
        "positive_ordered" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.); x::positive_ordered[2] ~ normal(mu, 1.); y ~ normal(x, 1.0)
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.); x::positive_ordered[2] ~ truncated(normal, mu, 1.; lower=0.); y ~ normal(x, 1.0)
            end)),
    ]
    for (name, (kwarg, hof)) in spellings
        @testset "$name" begin
            @test stan_code(kwarg) == stan_code(hof)
            @test stanc_compiles(kwarg)
        end
    end
    @test occursin("x ~ lower_conditioning_normal(0.0, mu, 1.0);", stan_block(stan_code(first(last(spellings[1]))), "model"))
    vcat_prior = @slic (; y=[0.3, 0.2]) begin
        mu ~ normal(0., 1.); tau ~ vcat(exponential(1.), normal(mu, 1.; lower=0.)); y ~ normal(tau, 1.0)
    end
    model = stan_block(stan_code(vcat_prior), "model")
    @test occursin("tau[1] ~ exponential(1.0);", model)
    @test occursin("tau[2] ~ lower_conditioning_normal(0.0, mu, 1.0);", model)
    @test stanc_compiles(vcat_prior)

    # A constant normaliser keeps the lean native statement: every argument and
    # bound is data, the bound restates the family's own support, or a symmetric
    # family is cut once at its location (the half-normal / half-t for any scale).
    lean = [
        "normal(0., s; lower=0.)" => (@slic((; y=0.3), begin
            s ~ exponential(1.); x ~ normal(0., s; lower=0.); y ~ normal(x, 1.0)
        end), "x ~ normal(0.0, s);"),
        "student_t(3., 0., s; lower=0.)" => (@slic((; y=0.3), begin
            s ~ exponential(1.); x ~ student_t(3., 0., s; lower=0.); y ~ normal(x, 1.0)
        end), "x ~ student_t(3.0, 0.0, s);"),
        "cauchy(mu, s; upper=mu)" => (@slic((; y=0.3), begin
            mu ~ normal(0., 1.); s ~ exponential(1.); x ~ cauchy(mu, s; upper=mu); y ~ normal(x, 1.0)
        end), "x ~ cauchy(mu, s);"),
        "data bound" => (@slic((; y=0.3, lo=0.5), begin
            x ~ normal(0., 1.0; lower=lo); y ~ normal(x, 1.0)
        end), "x ~ normal(0.0, 1.0);"),
        "exponential(r; lower=0.)" => (@slic((; y=0.3), begin
            r ~ exponential(1.); x ~ exponential(r; lower=0.); y ~ normal(x, 1.0)
        end), "x ~ exponential(r);"),
        "positive_ordered ~ exponential(r)" => (@slic((; y=[0.3, 0.2]), begin
            r ~ exponential(1.); x::positive_ordered[2] ~ exponential(r); y ~ normal(x, 1.0)
        end), "x ~ exponential(r);"),
    ]
    for (name, (m, statement)) in lean
        @testset "lean: $name" begin
            code = stan_code(m)
            @test occursin(statement, stan_block(code, "model"))
            @test !occursin("conditioning", code)
            @test stanc_compiles(m)
        end
    end

    # Re-draws were already truncated and are unchanged; a prior-only
    # `positive_ordered` stays the exchangeable sorted re-draw.
    prior = @slic (;) begin
        mu ~ normal(0., 1.); x ~ normal(mu, 1.0; lower=0.)
    end
    @test occursin("real x = lower_conditioning_normal_rng(0.0, mu, 1.0);",
        stan_block(stan_code(prior), "generated quantities"))
    ordered_prior = @slic (;) begin
        mu ~ normal(0., 1.); x::positive_ordered[2] ~ normal(mu, 1.)
    end
    code = stan_code(ordered_prior)
    @test strip(stan_block(code, "parameters")) == "{\n}"
    @test occursin("positive_ordered[2] x = sort_asc(lower_conditioning_vector_normal_rng(2, 0.0, mu, 1.0));",
        stan_block(code, "generated quantities"))
    @test stanc_compiles(ordered_prior)
end

@testitem "slic: a kwarg bound on an observation truncates its likelihood and draw" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    spellings = [
        "scalar" => (
            @slic((; y=0.3), begin mu ~ normal(0., 1.); y ~ normal(mu, 1.0; lower=0.) end),
            @slic((; y=0.3), begin mu ~ normal(0., 1.); y ~ truncated(normal, mu, 1.0; lower=0.) end)),
        # Data carry no declared bound, so even a constant normaliser truncates:
        # the `truncated` call is what checks the support and bounds `y_gen`.
        "constant normaliser" => (
            @slic((; y=0.3), begin s ~ exponential(1.); y ~ normal(0., s; lower=0.) end),
            @slic((; y=0.3), begin s ~ exponential(1.); y ~ truncated(normal, 0., s; lower=0.) end)),
        "vector" => (
            @slic((; y=[0.3, 0.4]), begin mu ~ normal(0., 1.); y ~ normal(mu, 1.0; upper=1.) end),
            @slic((; y=[0.3, 0.4]), begin mu ~ normal(0., 1.); y ~ truncated(normal, mu, 1.0; upper=1.) end)),
        "discrete" => (
            @slic((; k=[1, 2]), begin l ~ exponential(1.); k ~ poisson(l; lower=1) end),
            @slic((; k=[1, 2]), begin l ~ exponential(1.); k ~ truncated(poisson, l; lower=1) end)),
        "@plate for" => (
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                @plate for i in 1:2
                    y[i] ~ normal(mu, 1.0; lower=0.)
                end
            end),
            @slic((; y=[0.3, 0.2]), begin
                mu ~ normal(0., 1.)
                @plate for i in 1:2
                    y[i] ~ truncated(normal, mu, 1.0; lower=0.)
                end
            end)),
        "slices" => (
            @slic((; y=[0.3, 0.2, 0.1]), begin
                mu ~ normal(0., 1.); y[1:2] ~ normal(mu, 1.0; lower=0.); y[3] ~ normal(mu, 1.0)
            end),
            @slic((; y=[0.3, 0.2, 0.1]), begin
                mu ~ normal(0., 1.); y[1:2] ~ truncated(normal, mu, 1.0; lower=0.); y[3] ~ normal(mu, 1.0)
            end)),
    ]
    for (name, (kwarg, hof)) in spellings
        @testset "$name" begin
            @test stan_code(kwarg) == stan_code(hof)
            @test stanc_compiles(kwarg)
        end
    end
    code = stan_code(first(last(spellings[1])))
    @test occursin("y ~ lower_conditioning_normal(0.0, mu, 1.0);", stan_block(code, "model"))
    @test occursin("real y_likelihood = lower_conditioning_normal_lpdfs(y, 0.0, mu, 1.0);", code)
    @test occursin("real y_gen = lower_conditioning_normal_rng(0.0, mu, 1.0);", code)

    # A bound restating the family's own support is not a truncation.
    restated = @slic (; y=[0.3, 0.4]) begin
        r ~ exponential(1.); y ~ exponential(r; lower=0.)
    end
    code = stan_code(restated)
    @test occursin("y ~ exponential(r);", stan_block(code, "model"))
    @test !occursin("conditioning", code)
end

@testitem "slic: a kwarg-truncated prior has the normalised log density" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    y = 0.3
    m = @slic (; y) begin
        mu ~ normal(0., 1.)
        x ~ normal(mu, 1.0; lower=0.)
        y ~ normal(x, 1.0)
    end
    # Unconstrained (mu, log x); the Jacobian of x = exp(u) is u.
    logPhi(z) = log(StanBlocks.SpecialFunctions.erfc(-z / sqrt(2)) / 2)
    hand(q) = begin
        mu, x = q[1], exp(q[2])
        -mu^2 / 2 - (x - mu)^2 / 2 - logPhi(mu) - (y - x)^2 / 2 + q[2]
    end
    mktempdir() do dir
        cd(dir) do
            prob = stan_instantiate(m)
            points = ([0.1, -0.3], [-1.5, 0.4], [2.0, -1.0])
            for (a, b) in zip(points, points[2:end])
                @test LogDensityProblems.logdensity(prob, a) - LogDensityProblems.logdensity(prob, b) ≈
                    hand(a) - hand(b)
            end

            # A bounded observation: normalised likelihood in `mu`, draws inside the bound.
            obs = @slic (; y) begin
                mu ~ normal(0., 1.)
                y ~ normal(mu, 1.0; lower=0.)
            end
            oprob = stan_instantiate(obs)
            ohand(q) = -q[1]^2 / 2 - (y - q[1])^2 / 2 - logPhi(q[1])
            for (a, b) in ((0.4, -1.2), (-1.2, 2.5))
                @test LogDensityProblems.logdensity(oprob, [a]) - LogDensityProblems.logdensity(oprob, [b]) ≈
                    ohand([a]) - ohand([b])
            end
            rng = BridgeStan.StanRNG(oprob.model, 1234)
            names = BridgeStan.param_names(oprob.model; include_gq=true)
            gen = findfirst(==("y_gen"), names)
            draws = [BridgeStan.param_constrain(oprob.model, [-2.0]; include_gq=true, rng)[gen] for _ in 1:200]
            @test all(>=(0), draws)
        end
    end
end
