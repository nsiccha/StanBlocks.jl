"""
References for the exponentially modified normal where λσ = τ underflows, in
standardized units: the law is U = Z + E, E ~ Exp(τ), with
F(u) = Φ(u) - exp(τ²/2 - τu)Φ(u - τ). `tiny_tau_lcdf` forms that difference at a
precision that absorbs its cancellation (F/Φ(u) is about τ·max(1, |u|)) while τ
is within reach (log τ ≥ -1000), and beyond it uses F = τ(uΦ(u) + φ(u)), whose
next term is below e^-300 of it at every double u. `tiny_tau_interval` does the
same for F(b) - F(a).
"""
@testsnippet TinyTauReference begin
    tt_Phi(x) = StanBlocks.SpecialFunctions.erfc(-x / sqrt(big(2))) / 2
    tt_G(u) = u * tt_Phi(u) + exp(-u^2 / 2) / sqrt(2 * big(pi))
    tt_F(u, tau) = tt_Phi(u) - exp(tau^2 / 2 - tau * u) * tt_Phi(u - tau)
    # Bits for the cancellation: -log τ while τ is formed, the interval's
    # relative width, and the center's magnitude (c ± h at c = 1e300).
    tt_bits(lt, lhw, c) = 512 + ceil(Int, ((lt >= -1000 ? -lt : 0.0) + max(-lhw, 0.0) + log1p(abs(c))) / log(2))
    tiny_tau_lcdf(u, lt) = setprecision(BigFloat, tt_bits(lt, 0.0, u)) do
        lt >= -1000 ? log(tt_F(big(u), exp(big(lt)))) : big(lt) + log(tt_G(big(u)))
    end
    tiny_tau_interval(c, lhw, lt) = setprecision(BigFloat, tt_bits(lt, lhw, c)) do
        a, b = big(c) - exp(big(lhw)), big(c) + exp(big(lhw))
        lt >= -1000 ? log(tt_F(b, exp(big(lt))) - tt_F(a, exp(big(lt)))) : big(lt) + log(tt_G(b) - tt_G(a))
    end
end

"""
The exponentially modified normal and skew-normal builtins return their limit,
never NaN, at an infinite or overflowing standardized residual, as a sampler's
extreme proposal produces (snag `emg-standardized-3a08ab62`, from a downstream
module's SMC): `exp_mod_normal_lpdf_standardized(Inf, log σ, log λ)` was
`0·(-∞) = NaN` once τ = exp(log σ + log λ) underflowed to 0, and the interval
forms subtracted two infinite log masses. Where τ underflows, the standardized
CDF, and the interval masses taken from it, keep their value from log τ
instead of reading D = 0: against the references above, value and Enzyme
gradient.
"""
@testitem "slic: exp_mod_normal and skew_normal builtins are total at infinite residuals and underflowing λσ" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EnzymeGradient, TinyTauReference] begin
    B = StanBlocks
    # The density's limit at u = ±∞, for log λσ from -∞ through τ's underflow
    # to beyond Float64, split between log σ and log λ both ways.
    for lt in (-Inf, -1e10, -800.0, -745.5, -50.0, 0.0, 25.0, 720.0), u in (Inf, -Inf)
        @test B.exp_mod_normal_lpdf_standardized(u, lt, -3.0) == -Inf
        @test B.exp_mod_normal_lpdf_standardized(u, -3.0, lt) == -Inf
    end
    # Positive control: a finite residual where τu is negligible is the
    # exponential's plateau, log λ + log Φ(u), unchanged.
    @test B.exp_mod_normal_lpdf_standardized(1e300, -1e10, -3.0) == -3.0
    @test B.exp_mod_normal_lpdf_standardized(0.5, -800.0, -3.0) ≈ -3.0 + log(Float64(tt_Phi(big(0.5)))) rtol = 1e-15
    # The native-scale density, and the skew-normal one at α = 0 (α·z = 0·∞).
    @test B.exp_mod_normal_lpdf_stable(Inf, 0.0, 1e-200, 1e-200) == -Inf
    @test B.exp_mod_normal_lpdf_stable(1.0, 0.0, 1e-320, 1e-10) == -Inf
    for z in (Inf, -Inf), al in (0.0, 4.0, -4.0)
        @test B.skew_normal_lpdf_stable(z, 0.0, 1.0, al) == -Inf
    end
    # An infinite center is an empty interval, whatever τ or α.
    for c in (Inf, -Inf), lhw in (-5.0, 0.0, 5.0)
        for lt in (-1e10, -800.0, 0.0, 720.0)
            @test B.exp_mod_normal_interval_lprob_standardized(c, lhw, lt, 0.0) == -Inf
        end
        for al in (0.0, 4.0, -4.0)
            @test B.skew_normal_interval_lprob_standardized(c, lhw, al) == -Inf
        end
    end
    # Both ends of an interval past Float64 on one side: -Inf, not -Inf - (-Inf).
    @test B.exp_mod_normal_interval_lprob_standardized(1e300, 0.0, 25.0, 0.0) == -Inf
    # (c - h, c + h] with h = e^700 around c = 1e300 under λσ = e^700 holds all
    # but a mass far below Float64: its left end's log CDF and right end's log
    # CCDF are both -Inf, and the mass between them is 1.
    @test B.exp_mod_normal_interval_lprob_standardized(1e300, 700.0, 700.0, 0.0) == 0.0
    @test B.skew_normal_interval_lprob_standardized(1e300, 0.0, 5.0) == -Inf
    @test B.skew_normal_interval_lprob_standardized(-1e300, 0.0, -5.0) == -Inf
    # The CDFs at the same extremes: limits, and log F = log Φ(u) once u ≥ τ/2
    # with log τ > 709 (it threw a log1p domain error once u > τ).
    @test B.exp_mod_normal_std_logcdf(Inf, -Inf) == 0.0
    @test B.exp_mod_normal_std_logcdf(-2e154, 400.0) == -Inf
    @test B.exp_mod_normal_std_logcdf(1.7e308, 709.2) == 0.0
    @test B.skew_normal_std_logccdf(1e200, 5.0) == -Inf
    @test B.skew_normal_std_logcdf(1e200, 0.0) == 0.0
    @test B.skew_normal_std_logcdf(-Inf, 0.0) == -Inf

    # τ underflowing: the log CDF from log τ, against the reference, and with
    # its gradient in (u, log τ): Φ(u)/(uΦ(u) + φ(u)) and 1 to first order.
    for lt in (-710.0, -720.0, -800.0, -5e3, -5e10), u in (-30.0, -3.0, 0.0, 0.5, 4.0, 40.0, 1e3, 1e100, 1e300)
        @testset "log F($u), log τ = $lt" begin
            r = Float64(tiny_tau_lcdf(u, lt))
            @test B.exp_mod_normal_std_logcdf(u, lt) ≈ r rtol = 1e-14
            # The log CCDF is log1m_exp of it, near 0.
            @test B.exp_mod_normal_std_logccdf(u, lt) ≈ log1p(-exp(r)) rtol = 1e-12 atol = 1e-300
            if lt < -1000
                v, g = gradient(B.exp_mod_normal_std_logcdf, u, lt)
                @test g[2] ≈ 1.0 rtol = 1e-15
                @test g[1] ≈ Float64(setprecision(() -> tt_Phi(big(u)) / tt_G(big(u)), BigFloat, 512)) rtol = 1e-12
            end
        end
    end
    # Interval masses from it: (c - h, c + h] at log λσ = lt, through the CDF
    # difference (wide) and the density rule (narrow), value and gradient in
    # (c, log h, log σ, log λ).
    f = (c, lhw, ls, ll) -> B.exp_mod_normal_interval_lprob_standardized(c, lhw, ls, ll)
    for lt in (-720.0, -800.0, -5e3, -5e10), (c, lhw) in ((0.5, 1.0), (-3.0, -0.5), (4.0, 2.0), (0.5, -8.0),
                                                           (1e3, 3.0), (1e300, 2.0))
        @testset "interval c = $c, log h = $lhw, log τ = $lt" begin
            r = Float64(tiny_tau_interval(c, lhw, lt))
            # The rule interval of a center beyond 2^53·h rounds to its center.
            @test f(c, lhw, lt + 3.0, -3.0) ≈ r rtol = 1e-14
            v, g = gradient(f, c, lhw, lt + 3.0, -3.0)
            @test v == f(c, lhw, lt + 3.0, -3.0)
            @test all(isfinite, g)
            @test g[3] ≈ 1.0 rtol = 1e-9
            @test g[4] ≈ 1.0 rtol = 1e-9
            if lt < -1000 && c < 1e4
                dref = setprecision(BigFloat, 512) do
                    a, b = big(c) - exp(big(lhw)), big(c) + exp(big(lhw))
                    m = tt_G(b) - tt_G(a)
                    Float64.([(tt_Phi(b) - tt_Phi(a)) / m, exp(big(lhw)) * (tt_Phi(b) + tt_Phi(a)) / m])
                end
                # The log masses are log λσ + O(1), resolved to ulp(log λσ), and the
                # gradient in (c, log h) reads their O(1) differences: good to
                # about 1e-16·|log λσ| relative (measured 9e-8 at -5e10).
                @test g[1:2] ≈ dref rtol = max(1e-10, 1e-16 * abs(lt))
            end
        end
    end

end

"""
The same in the emitted Stan, through BridgeStan: at an infinite residual the
exponentially modified normal density and both families' interval masses are
-Inf (data `Infinity`), and where λσ underflows the interval mass and its
gradient match the Julia methods.
"""
@testitem "slic: exp_mod_normal and skew_normal builtins are total at infinite residuals in Stan" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, TinyTauReference] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    B = StanBlocks
    @deffun begin
        @lpxf xtail_lpdf(y::vector[n], k::int, w::real, a::real, b::real, c::real)::real = begin
            lp = 0.0
            for i in 1:n
                lp = lp + (k == 1 ? exp_mod_normal_lpdf_standardized(y[i] + a, b, c) :
                          (k == 2 ? exp_mod_normal_interval_lprob_standardized(y[i] + a, w, b, c) :
                          (k == 3 ? skew_normal_interval_lprob_standardized(y[i] + a, w, c) :
                                    skew_normal_lpdf_stable(y[i] + a, 0., exp(b), c))))
            end
            lp
        end
        xtail_lpdfs(y::vector[n], k::int, w::real, a::real, b::real, c::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = xtail_lpdf(y[i:i], k, w, a, b, c)
            end
            out
        end
        xtail_rng(vector[n], k::int, w::real, a::real, b::real, c::real)::vector[n] = rep_vector(0., n)
    end
    model = @slic (; y = [0.0], k = 1, w = 0.0, b0 = 0.0, c0 = 0.0) begin
        a ~ flat()
        b ~ flat()
        c ~ flat()
        y ~ xtail(k, w, a, b0 + b, c0 + c)
    end
    @test stanc_compiles(model)
    evaluate(k, y, w, b, c) = begin
        problem = instantiate(stan_model(model(; y = [y], k, w, b0 = b, c0 = c)))
        LogDensityProblems.logdensity_and_gradient(problem, zeros(3))
    end
    # (k, y, w, b, c): log σ and log λ for k ≤ 2, α for k ≥ 3, log ω for k = 4.
    for p in ((1, Inf, 0.0, -800.0, -3.0), (1, Inf, 0.0, -1e10, -3.0), (1, -Inf, 0.0, -800.0, -3.0),
              (2, Inf, 1.0, -800.0, -3.0), (2, -Inf, 1.0, 0.0, 0.0), (3, Inf, 1.0, 0.0, 4.0),
              (3, -Inf, 1.0, 0.0, 0.0), (4, Inf, 0.0, 0.0, 0.0))
        @testset "$p" begin
            @test evaluate(p...)[1] == -Inf
        end
    end
    # λσ underflowing: the interval mass is finite, with a finite gradient, and
    # matches the Julia method and the reference.
    for (y, w, b, c) in ((0.5, 1.0, -800.0, -3.0), (4.0, 2.0, -5e10, -3.0), (0.5, -8.0, -800.0, -3.0))
        @testset "interval ($y, $w, $b, $c)" begin
            value, grad = evaluate(2, y, w, b, c)
            @test isfinite(value) && all(isfinite, grad)
            @test value ≈ B.exp_mod_normal_interval_lprob_standardized(y, w, b, c) rtol = 1e-14
            @test value ≈ Float64(tiny_tau_interval(y, w, b + c)) rtol = 1e-14
            @test grad[2] ≈ 1.0 rtol = 1e-9
            @test grad[3] ≈ 1.0 rtol = 1e-9
        end
    end
end
