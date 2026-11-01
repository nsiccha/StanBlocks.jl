"""
`censored`/`truncated`/`interval_censored(normal, …)` take their tail masses
from `normal_lcdf_stable`/`normal_lccdf_stable`. Their erfc form underflows past
~37.6σ: the gradient turns ±Inf, then the value -Inf with a NaN gradient. A NaN
gradient survives a `log_sum_exp` whose other term is finite, so a censored
mixture rejected every start (snag `fit-a-left-censo-942540f9`). Past 35σ both
helpers switch to the asymptotic branch `std_normal_lcdf_tail`; within 35σ the
erfc form is unchanged. Value and gradient are checked against a 256-bit
reference in both tails and inside a mixture's `log_sum_exp`.
"""
@testitem "slic: normal censoring tail masses stay finite with exact gradients past 35σ" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    setprecision(BigFloat, 256) do
        logPhi(z) = log(StanBlocks.SpecialFunctions.erfc(-big(z) / sqrt(big(2))) / 2)
        dlogPhi(z) = exp(-big(z)^2 / 2) / sqrt(2 * big(pi)) / exp(logPhi(z))

        # Both atoms sit deep in opposite tails: the lower atom takes the lcdf,
        # the upper atom the lccdf, at standardized distances from 35σ to 65σ.
        lo, hi, s = -1.0, 1.0, 0.02
        two_sided = @slic (; y = [lo, hi], lo, hi, s) begin
            mu ~ normal(0.0, 100.0)
            y ~ censored(normal(mu, s); lower = lo, upper = hi)
        end
        code = stan_code(two_sided)
        functions = stan_block(code, "functions")
        @test occursin("std_normal_lcdf_tail(", functions)
        @test occursin("log(erfc(", functions)
        @test stanc_compiles(two_sided)
        problem = instantiate(stan_model(two_sided))
        hand(mu) = -big(mu)^2 / 2e4 + logPhi((lo - mu) / s) + logPhi((mu - hi) / s)
        dhand(mu) = -big(mu) / 1e4 - dlogPhi((lo - mu) / s) / s + dlogPhi((mu - hi) / s) / s
        reference = LogDensityProblems.logdensity(problem, [0.0])
        @test isfinite(reference)
        for mu in (-0.6, -0.3, 0.25, 0.3)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, [mu])
            @test isfinite(value) && all(isfinite, gradient)
            @test value - reference ≈ hand(mu) - hand(0.0) rtol = 1e-12
            @test gradient[1] ≈ dhand(mu) rtol = 1e-12
        end

        # The reported shape: a censored two-component mixture whose lcdf is a
        # `log_sum_exp` over two `normal_lcdf_stable` terms. The second
        # component sits 45 scales above the bound, so its term is deep in the
        # tail while the first keeps the density finite.
        dens = Any[
            :(tmix_lpdf(y::real, w::real, m1::real, m2::real, s::real)::real =
                log_sum_exp(log1m(w) + normal_lpdf(y, m1, s), log(w) + normal_lpdf(y, m2, s))),
            :(tmix_lpdf(y::vector[n], w::vector[n], m1::vector[n], m2::vector[n], s::vector[n])::real = begin
                rv = 0.0
                for i in 1:n
                    rv = rv + tmix_lpdf(y[i], w[i], m1[i], m2[i], s[i])::real
                end
                rv
            end),
        ]
        ptws = :(tmix_lpdfs(y::vector[n], w::vector[n], m1::vector[n], m2::vector[n], s::vector[n])::vector[n] = begin
            rv::vector[n]
            for i in 1:n
                rv[i] = tmix_lpdf(y[i], w[i], m1[i], m2[i], s[i])
            end
            rv
        end)
        rngs = Any[
            :(tmix_rng(w::real, m1::real, m2::real, s::real)::real =
                uniform_rng(0, 1) < w ? normal_rng(m2, s) : normal_rng(m1, s)),
            :(tmix_rng(vector[n], w::vector[n], m1::vector[n], m2::vector[n], s::vector[n])::vector[n] = begin
                rv::vector[n]
                for i in 1:n
                    rv[i] = tmix_rng(w[i], m1[i], m2[i], s[i])
                end
                rv
            end),
        ]
        cdf = :(tmix_lcdf(y::real, w::real, m1::real, m2::real, s::real)::real =
            log_sum_exp(log1m(w) + normal_lcdf_stable(y, m1, s), log(w) + normal_lcdf_stable(y, m2, s)))
        ccdf = :(tmix_lccdf(y::real, w::real, m1::real, m2::real, s::real)::real =
            log_sum_exp(log1m(w) + normal_lccdf_stable(y, m1, s), log(w) + normal_lccdf_stable(y, m2, s)))
        tmix = StanBlocks.ValueFamily(:tmix, :lpdf, dens, ptws, rngs, @__MODULE__; lcdf = cdf, lccdf = ccdf)
        data = Dict{Symbol,Any}(:y => [0.0, 0.0, 0.5], :w => fill(0.3, 3), :lo => fill(0.0, 3))
        rhs = Expr(:call, :censored, Expr(:parameters, Expr(:kw, :lower, :lo)),
            Expr(:call, tmix, :w, :(rep_vector(mu, 3)), :(rep_vector(mu + 45.0, 3)), :(rep_vector(1.0, 3))))
        mixture = StanBlocks.SlicModel(
            Expr(:block, :(mu ~ normal(0.0, 100.0)), Expr(:call, :~, :y, rhs)), data, @__MODULE__)
        @test stanc_compiles(mixture)
        mixture_problem = instantiate(stan_model(mixture))
        lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
        atom(mu) = lse(log(big(0.7)) + logPhi(-mu), log(big(0.3)) + logPhi(-mu - 45))
        point(mu) = lse(log(big(0.7)) - (0.5 - big(mu))^2 / 2, log(big(0.3)) - (big(mu) + 44.5)^2 / 2)
        mixture_hand(mu) = -big(mu)^2 / 2e4 + 2atom(mu) + point(mu)
        mixture_dhand(mu) = (h = big(1e-30); (mixture_hand(big(mu) + h) - mixture_hand(big(mu) - h)) / 2h)
        mixture_reference = LogDensityProblems.logdensity(mixture_problem, [0.0])
        for mu in (-1.0, 2.0, 10.0)
            value, gradient = LogDensityProblems.logdensity_and_gradient(mixture_problem, [mu])
            @test isfinite(value) && all(isfinite, gradient)
            @test value - mixture_reference ≈ mixture_hand(mu) - mixture_hand(0.0) rtol = 1e-12
            @test gradient[1] ≈ mixture_dhand(mu) rtol = 1e-12
        end
    end
end

"""
`normal_lcdf_stable` / `normal_lccdf_stable` have `@juliacompat` Julia methods
(snag `evaluate-the-unc-30e61f31`, which needed them for
`exp_mod_normal_lpdf_standardized`'s Julia method). They run the same branches
as the emitted Stan: the erfc form within 35 scales and the asymptotic series
beyond. Values and Mooncake gradients match a 256-bit reference in both tails.
"""
@testitem "slic: the stable normal log-CDF and log-CCDF evaluate in Julia and under Mooncake" tags=[:slic, :ad, :regression] setup=[StanBlocksImports] begin
    import Mooncake
    gradient(g, x...) = begin
        cache = Mooncake.prepare_gradient_cache(g, x...)
        value, grads = Mooncake.value_and_gradient!!(cache, g, x...)
        value, collect(grads[2:end])
    end
    reference(g, x...) = setprecision(BigFloat, 256) do
        logPhi(z) = log(StanBlocks.SpecialFunctions.erfc(-z / sqrt(big(2))) / 2)
        law(x, loc, scale) = g === StanBlocks.normal_lcdf_stable ?
            logPhi((x - loc) / scale) : logPhi((loc - x) / scale)
        h = big(1e-30)
        b = big.(x)
        value = law(b...)
        grad = map(1:3) do j
            e = ntuple(i -> i == j ? h : zero(h), 3)
            (law((b .+ e)...) - law((b .- e)...)) / 2h
        end
        value, grad
    end
    # Standardized distances -60, -36, -20, 0, 3, 20, 36, 60.
    points = ((-1.2, 0.0, 0.02), (-0.72, 0.0, 0.02), (-0.4, 0.0, 0.02), (0.1, 0.1, 1.0),
              (3.0, 0.0, 1.0), (0.4, 0.0, 0.02), (0.72, 0.0, 0.02), (1.2, 0.0, 0.02))
    for g in (StanBlocks.normal_lcdf_stable, StanBlocks.normal_lccdf_stable), p in points
        value, grad = gradient(g, p...)
        ref, dref = reference(g, p...)
        @test isfinite(value) && all(isfinite, grad)
        @test value ≈ ref rtol = 1e-13 atol = 1e-13
        @test all(isapprox.(grad, dref; rtol = 1e-11, atol = 1e-13))
    end
    @test Base.return_types(StanBlocks.normal_lccdf_stable, (Float64, Float64, Float64)) == [Float64]
end
