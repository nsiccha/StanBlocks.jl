"""
Stan Math 5.3.0/5.4.0 `exp_mod_normal_lpdf` and `skew_normal_lpdf` take
`log(erfc(r))` directly, so past r ≈ 26.5 the value is -Inf and the gradient
NaN or -Inf, also inside a `log_mix` whose other term is finite (snag
`ap3-native-skew-db722fb4`). `exp_mod_normal_lpdf_stable` and
`skew_normal_lpdf_stable` are the same normalized laws written through the
stable normal log-CDF. `exp_mod_normal_lpdf_stable` takes the scaled form
`log λ - ((y-μ)/σ)²/2 + [log Φ(z) + z²/2]` left of its mode, so a large λσ does
not cancel `λ(μ-y) + λ²σ²/2` against `log Φ(z)`. Values and gradients are
checked against a 256-bit reference in the body, deep in both tails and at
λσ = 1e4.
"""
@testitem "slic: stable exp_mod_normal and skew_normal densities keep exact values and gradients in deep tails" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    # One observation family evaluating either law or their mixture, selected
    # by data `k`, so each law is checked alone and inside `log_mix`.
    @deffun begin
        @lpxf tailpair_lpdf(y::vector[n], k::int, p::real, mu::real, s::real, l::real, xi::real, w::real, a::real)::real = begin
            lp = 0.0
            for i in 1:n
                e = exp_mod_normal_lpdf_stable(y[i], mu, s, l)
                q = skew_normal_lpdf_stable(y[i], xi, w, a)
                lp = lp + (k == 1 ? e : (k == 2 ? q : log_mix(p, e, q)))
            end
            lp
        end
        tailpair_lpdfs(y::vector[n], k::int, p::real, mu::real, s::real, l::real, xi::real, w::real, a::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = tailpair_lpdf(y[i:i], k, p, mu, s, l, xi, w, a)
            end
            out
        end
        tailpair_rng(vector[n], k::int, p::real, mu::real, s::real, l::real, xi::real, w::real, a::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = uniform_rng(0, 1) < p ? exp_mod_normal_rng(mu, s, l) : skew_normal_rng(xi, w, a)
            end
            out
        end
    end

    setprecision(BigFloat, 256) do
        logPhi(z) = log(StanBlocks.SpecialFunctions.erfc(-big(z) / sqrt(big(2))) / 2)
        emg(y, mu, s, l) = log(big(l)) + big(l) * (big(mu) - big(y)) + (big(l) * big(s))^2 / 2 +
            logPhi((big(y) - big(mu)) / big(s) - big(l) * big(s))
        sn(y, xi, w, a) = log(big(2)) - log(big(w)) - log(2 * big(pi)) / 2 - ((big(y) - big(xi)) / big(w))^2 / 2 +
            logPhi(big(a) * (big(y) - big(xi)) / big(w))
        lse(u, v) = max(u, v) + log1p(exp(-abs(u - v)))
        d(f, x) = (h = big(1e-30); (f(big(x) + h) - f(big(x) - h)) / 2h)

        # `theta` is the only parameter: it is the EMG location and the
        # skew-normal location, so the deep tails are reached by moving it.
        # The prior is improper-flat; values are compared as differences
        # against `theta = 0`, which removes any dropped constant.
        model = @slic (; y = [0.0], k = 1, p = 0.25, s = 0.01, l = 25.0, w = 0.01, a = 2.0) begin
            theta ~ flat()
            y ~ tailpair(k, p, theta, s, l, theta, w, a)
        end
        code = stan_code(model)
        functions = stan_block(code, "functions")
        @test occursin("std_normal_scaled_lcdf_tail(", functions)
        @test occursin("exp_mod_normal_lpdf_stable(", functions)
        @test occursin("skew_normal_lpdf_stable(", functions)
        @test stanc_compiles(model)

        # theta = 0.4 puts the EMG argument at r = 28.5 and the skew-normal
        # argument at αz = -80; theta = 1.0 and 3.0 go further. theta = -0.5
        # is the EMG's exponential (right) tail and the skew normal's
        # saturated side.
        thetas = (-0.5, -0.01, 0.01, 0.05, 0.4, 1.0, 3.0)
        cases = [
            1 => (t -> emg(0, t, 0.01, 25)),
            2 => (t -> sn(0, t, 0.01, 2)),
            3 => (t -> lse(log(big(0.25)) + emg(0, t, 0.01, 25), log(big(0.75)) + sn(0, t, 0.01, 2))),
        ]
        for (k, hand) in cases
            @testset "k = $k" begin
                problem = instantiate(stan_model(model(; k)))
                reference = LogDensityProblems.logdensity(problem, [0.0])
                @test isfinite(reference)
                for theta in thetas
                    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, [theta])
                    @test isfinite(value) && all(isfinite, gradient)
                    @test value - reference ≈ hand(theta) - hand(0.0) rtol = 1e-12
                    @test gradient[1] ≈ d(hand, theta) rtol = 1e-12
                end
            end
        end

        # λσ = 1e4: the unscaled form cancels λ(μ-y) + λ²σ²/2 ≈ 5e7 against
        # log Φ(z) ≈ -5e7; even with a correctly rounded log Φ(z) it is off by
        # about 1e-9 here (7e-6 at λσ = 1e6), while the scaled form stays
        # within 2e-15.
        wide = @slic (; y = [0.0], k = 1, p = 0.5, s = 1.0, l = 1e4, w = 1.0, a = 1.0) begin
            theta ~ flat()
            y ~ tailpair(k, p, theta, s, l, theta, w, a)
        end
        wide_problem = instantiate(stan_model(wide))
        wide_hand(t) = emg(0, t, 1, 1e4)
        wide_reference = LogDensityProblems.logdensity(wide_problem, [0.0])
        for theta in (-3.0, -0.5, 0.5, 3.0)
            value, gradient = LogDensityProblems.logdensity_and_gradient(wide_problem, [theta])
            @test value - wide_reference ≈ wide_hand(theta) - wide_hand(0.0) rtol = 1e-12
            @test gradient[1] ≈ d(wide_hand, theta) rtol = 1e-12
        end
    end
end
