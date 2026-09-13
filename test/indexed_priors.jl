using TestItemRunner

"""
Indexed splices replace an exact statement, never a whole vector's prior.
The supported heterogeneous-prior path retains a single constrained vector and
draws its elements through the custom family's sized RNG (snag indexed-prior-sp).
"""
@testitem "slic: indexed splice boundary and constrained heterogeneous vector prior" tags=[:slic, :regression, :stanc, :bridgestan] begin
    using StanBlocks, LogDensityProblems, Statistics
    import BridgeStan

    base = @slic begin
        tau::vector[n]
        y ~ normal(sum(tau), 1.)
        return tau
    end
    for bare in (base, @slic(begin tau::vector[n]; return tau end))
        for stmt in (:(tau[1] ~ normal(0., 1.; lower=0.)),
                     :(tau[1:2] ~ normal(0., 1.)),
                     :(tau[1] = 1.))
            err = try
                Base.merge(bare, stmt)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("exactly the same LHS", sprint(showerror, err))
            @test occursin("sized _rng", sprint(showerror, err))
        end
    end
    # An empty body used to bypass usedin entirely and append an indexed prior.
    @test_throws ArgumentError Base.merge(@slic(begin end), :(tau[1] ~ normal(0., 1.)))

    indexed = @slic begin
        tau::vector[2]
        tau[1] ~ normal(0., 1.)
        tau[2] ~ normal(3., 1.)
        y ~ normal(sum(tau), 1.)
        return tau
    end
    replaced = Base.merge(indexed, :(tau[1] ~ normal(1., 2.)))
    sc = stan_code(replaced(; y=0.5))
    @test occursin("tau[1] ~ normal(1.0, 2.0)", sc)
    @test !occursin("tau[1] ~ normal(0.0, 1.0)", sc)
    @test occursin("tau[2] ~ normal(3.0, 1.0)", sc)
    @test StanBlocks.stanc_check(sc; warn_pedantic=false).ok
    @test occursin("tau[1] ~ normal(0.0, 1.0)", stan_code(indexed(; y=0.5)))

    @deffun begin
        @lpxf hetero_prior_lpdf(tau::vector[n])::real = begin
            @stan_assert n == 2
            if tau[1] < 0.0
                return negative_infinity()
            end
            normal_lpdf(tau[1], 0., 1.) + log(2.) + exponential_lpdf(tau[2], 1.)
        end
        hetero_prior_lpdfs(tau::vector[n])::vector[n] = begin
            @stan_assert n == 2
            out::vector[n]
            out[1] = normal_lpdf(tau[1], 0., 1.) + log(2.)
            if tau[1] < 0.0
                out[1] = negative_infinity()
            end
            out[2] = exponential_lpdf(tau[2], 1.)
            out
        end
        hetero_prior_rng(vector[n])::vector[n] = begin
            @stan_assert n == 2
            out::vector[n]
            out[1] = abs(normal_rng(0., 1.))
            out[2] = exponential_rng(1.)
            out
        end
    end
    # Intrinsic support, as for the native positive families. The RNG itself
    # must honour this support; it is not an extra truncation of a base law.
    StanBlocks.autokwargs(::StanBlocks.CanonicalExpr{typeof(hetero_prior)}) = (; lower=0.)

    replacement = :(tau ~ hetero_prior())
    prior = Base.merge(base, replacement)(; n=2)
    fit = prior(; y=0.5)
    direct = @slic begin
        tau::vector[n] ~ hetero_prior()
        y ~ normal(sum(tau), 1.)
        return tau
    end
    @test stan_code(fit) == stan_code(direct(; n=2, y=0.5))
    @test stan_code(prior) == stan_code(direct(; n=2))
    @test replacement == :(tau ~ hetero_prior())
    @test !occursin("hetero_prior", sprint(print, StanBlocks.model(base)))
    fit_code, prior_code = stan_code(fit), stan_code(prior)
    @test occursin("vector<lower=0.0>[n] tau", fit_code)
    @test occursin("tau = hetero_prior_vector_rng(n)", prior_code)
    @test !occursin("lower_conditioning", prior_code)
    @test StanBlocks.stanc_check(fit_code; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok

    # Fixed values remove the bare declaration too. Explicit typed replacements
    # still override its shape, while the untyped replacement above inherits it.
    fixed = Base.merge(base, (; tau=[0.25, 0.5]))(; n=2, y=0.5)
    @test !occursin("tau::vector", sprint(print, StanBlocks.model(fixed)))
    @test StanBlocks.stanc_check(stan_code(fixed); warn_pedantic=false).ok
    scalar = Base.merge(@slic(begin tau::vector[n]; y ~ normal(tau[1], 1.) end),
        :(tau::real ~ normal(0., 1.)), :(y ~ normal(tau, 1.)))(; y=0.5)
    @test occursin("real tau;", stan_code(scalar))

    # A true observed vector exercises the pointwise and sized predictive twins.
    observed = @slic (; tau=[0.4, 0.8]) begin
        tau ~ hetero_prior()
    end
    obs_code = stan_code(observed)
    @test occursin("hetero_prior_lpdfs", obs_code)
    @test occursin("hetero_prior_vector_rng", obs_code)
    @test StanBlocks.stanc_check(obs_code; warn_pedantic=false).ok

    mktempdir() do dir
        cd(dir) do
            fitted = stan_instantiate(fit)
            @test LogDensityProblems.dimension(fitted) == 2
            @test BridgeStan.param_names(fitted.model) == ["tau.1", "tau.2"]
            # Native unconstrained coordinates are log(tau); include Stan's
            # Jacobian. Compare log-density differences to avoid constant terms
            # dropped by Stan's sampling-statement compilation.
            expected(q) = begin
                t = exp.(q)
                -t[1]^2 / 2 - t[2] - (0.5 - sum(t))^2 / 2 + sum(q)
            end
            q0 = [0., 0.]
            lp0 = LogDensityProblems.logdensity(fitted, q0)
            for q in ([0.1, -0.2], [-0.3, 0.4], [0.25, 0.5])
                @test LogDensityProblems.logdensity(fitted, q) - lp0 ≈ expected(q) - expected(q0) atol=1e-10
            end
            simulated = stan_instantiate(prior)
            @test LogDensityProblems.dimension(simulated) == 0
            names = BridgeStan.param_names(simulated.model; include_tp=true, include_gq=true)
            slots = [only(findall(==("tau.$i"), names)) for i in 1:2]
            rng = BridgeStan.StanRNG(simulated.model, 7301)
            draws = hcat([BridgeStan.param_constrain(simulated.model, Float64[];
                include_tp=true, include_gq=true, rng)[slots] for _ in 1:4000]...)
            @test all(isfinite, draws)
            @test all(>=(0.), draws)
            @test abs(mean(draws[1, :]) - sqrt(2 / pi)) < 0.04
            @test abs(mean(draws[2, :]) - 1.) < 0.06
            @test abs(cor(draws[1, :], draws[2, :])) < 0.06
        end
    end
end
