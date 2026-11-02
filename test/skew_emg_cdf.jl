"""
256-bit references for the stable skew-normal and exponentially modified
normal log CDF, log CCDF and interval log mass (snag `odl-native-and-s-929bc7d0`).

- Skew-normal, standard law, z ≤ 0: for α > 0 the exact identity
  P(Z ≤ -h) = 2φ(H)/(k√(2π)) ∫_0^∞ exp(-bw - w²/2) M(H + δw) dw
  (k = √(1+α²), δ = α/k, H = kh, b = αh, M the Mills ratio), whose integrand is
  positive, cross-checked against the textbook Φ(z) - 2T(z, α) at a precision
  that absorbs its cancellation; for α ≤ 0, Φ(z) + 2T(|z|, |α|). z > 0 and the
  CCDF through 1 - F(z; α) = F(-z; -α).
- Exponentially modified normal, U = Z + E, E ~ Exp(τ): the closed forms
  1 - F(u) = Q(u) + exp(τ²/2 - τu)Φ(u - τ) and F(u) = Φ(u) - exp(τ²/2 - τu)Φ(u - τ)
  at a precision that absorbs the latter's cancellation, with
  exp(τ²/2 - τu)Φ(u - τ) = φ(u)M(τ - u) and, for u ≤ 0, F(u) = φ(u)(M(-u) - M(τ - u)).
  Past x = 1e4, M(x) is the integral ∫_0^∞ exp(-xt - t²/2) dt, so no reference
  forms exp(x²/2) or a vanishing erfc there: each fits a 32-bit MPFR exponent
  (±2^30 bits, Windows), where exp(5e9) at τ = 1e5 or φ(-1e6) do not.
- Interval masses: the density integrated over the interval at 256 bits.

References run at 192 bits with quadrature tolerance 1e-34; gradients are
central differences of the reference at h = 1e-12.
"""
@testsnippet SkewEMGCDFReference begin
    using QuadGK
    const SE_ERFC = StanBlocks.SpecialFunctions.erfc
    se_Phi(x) = SE_ERFC(-x / sqrt(big(2))) / 2
    se_mills(x) = SE_ERFC(x / sqrt(big(2))) / 2 * sqrt(2 * big(pi)) * exp(x^2 / 2)
    se_rtol() = big(1e-34)
    function se_quad(f, a, b; scale = big(1))
        bps = filter(x -> a < x < b, a .+ scale .* big.([0.1, 0.3, 1, 3, 10, 30, 100]))
        pts = [a; bps; b]
        sum(quadgk(f, pts[i], pts[i+1]; rtol = se_rtol(), order = 15)[1] for i in 1:length(pts)-1)
    end
    # The Mills ratio Q(x)/φ(x). se_mills needs x²/(2 log 2) exponent bits, past
    # a 32-bit MPFR exponent from x ≈ 3.9e4; past x = 1e4 the quadrature fits it
    # however large x is.
    se_R(x) = x <= 10_000 ? se_mills(x) :
        se_quad(t -> exp(-x * t - t^2 / 2), big(0), big(Inf); scale = 1 / x)
    se_logphi(x) = -x^2 / 2 - log(2 * big(pi)) / 2
    # F_SN(z; α) for z ≤ 0.
    function sn_lower_F(z, a)
        h = -z
        if a > 0
            k = sqrt(1 + a^2); d = a / k; H = k * h; b = a * h
            K = se_quad(w -> exp(-b * w - w^2 / 2) * se_mills(H + d * w), big(0), big(Inf);
                        scale = 1 / max(big(1), b))
            2 * exp(-H^2 / 2) / (2 * big(pi) * k) * K
        else
            T = quadgk(x -> exp(-h^2 * (1 + x^2) / 2) / (1 + x^2), big(0), -a;
                       rtol = se_rtol(), order = 15)[1] / (2 * big(pi))
            se_Phi(z) + 2T
        end
    end
    sn_lcdf_reference(z, a) = setprecision(BigFloat, 192) do
        z, a = big(z), big(a)
        z <= 0 ? log(sn_lower_F(z, a)) : log1p(-sn_lower_F(-z, -a))
    end
    sn_lccdf_reference(z, a) = sn_lcdf_reference(-big(z), -big(a))
    # Φ(z) - 2T(z, α) with enough bits for its cancellation (cross-check only).
    sn_owen_reference(z, a) = setprecision(BigFloat, 320 + ceil(Int, 1.1 * max(0.0, a^2 * z^2 / 2 / log(2)))) do
        z, a = big(z), big(a)
        T = quadgk(x -> exp(-z^2 * (1 + x^2) / 2) / (1 + x^2), big(0), a; rtol = big(2.0)^-150,
                   order = 15)[1] / (2 * big(pi))
        log(se_Phi(z) - 2T)
    end
    # 1 - F(u) = Q(u) + exp(τ²/2 - τu)Φ(u - τ), a positive sum, and
    # F(u) = Φ(u) - exp(τ²/2 - τu)Φ(u - τ) at a precision that absorbs its
    # cancellation; each log is taken from whichever of the two is below 1/2.
    emg_upper_S(u, tau) = setprecision(BigFloat, 512) do
        u, tau = big(u), big(tau)
        se_Phi(-u) + exp(se_logphi(u)) * se_R(tau - u)
    end
    # log F(u), in log space so that φ(u) at u = -1e6 never forms.
    emg_lower_logF(u, tau) = setprecision(BigFloat, 1024 + ceil(Int, 2 * log2(2 + abs(Float64(u)) / Float64(tau)) +
                                                        2 * log2(2 + 1 / Float64(tau)))) do
        u, tau = big(u), big(tau)
        u <= 0 ? se_logphi(u) + log(se_R(-u) - se_R(tau - u)) :
                 log(se_Phi(u) - exp(se_logphi(u)) * se_R(tau - u))
    end
    emg_lcdf_reference(u, tau) = setprecision(BigFloat, 512) do
        S = emg_upper_S(u, tau)
        S < 0.5 ? log1p(-S) : emg_lower_logF(u, tau)
    end
    emg_lccdf_reference(u, tau) = setprecision(BigFloat, 512) do
        S = emg_upper_S(u, tau)
        S < 0.5 ? log(S) : log1p(-exp(emg_lower_logF(u, tau)))
    end
    # Densities of the standard laws, and an interval's log mass by quadrature.
    sn_pdf(z, a) = 2 * exp(-z^2 / 2) / sqrt(2 * big(pi)) * se_Phi(a * z)
    emg_pdf(u, tau) = tau * exp(se_logphi(u)) * se_R(tau - u)
    interval_reference(pdf, lo, hi, p...) = setprecision(BigFloat, 192) do
        lo, hi = big(lo), big(hi)
        log(quadgk(x -> pdf(x, big.(p)...), lo, (lo + hi) / 2, hi; rtol = se_rtol(), order = 15)[1])
    end
    # (c - h, c + h], h = exp(lhw); below e^-100 the mass is 2h·pdf(c) to O(h²).
    interval_reference_ch(pdf, c, lhw, p...) = setprecision(BigFloat, 192) do
        lhw < -100 ? big(lhw) + log(big(2)) + log(pdf(big(c), big.(p)...)) :
            interval_reference(pdf, big(c) - exp(big(lhw)), big(c) + exp(big(lhw)), p...)
    end
    # A log CDF near 0 is log(1 - p) for a tail mass p, and carries p's relative
    # error, about 1e-16·|log p|: the tolerance a log probability can meet.
    cdf_rtol(r) = 1e-13 * max(1.0, -log(abs(Float64(r))))
    # Gradients below about 1e-154 that registered Enzyme 0.13.216 loses (Enzyme
    # snag `reverse-gradient-1717f5f9`); Stan's are right there. Dogfood's
    # libEnzyme build `df199913` (Enzyme `0be7fc2`) keeps them, so the items pin
    # them broken only where `!ENZYME_DOGFOOD`.
    const ENZYME_TINY_ADJOINT = Set([(:sn_lcdf, 30.0, 2.0), (:emg_lccdf, -30.0, 37.0)])
    # Central differences of a reference in each argument.
    reference_gradient(f, x...) = setprecision(BigFloat, 192) do
        h = big(1e-12)
        xb = big.(x)
        map(eachindex(xb)) do j
            e = ntuple(i -> i == j ? h : zero(h), length(xb))
            Float64((f((xb .+ e)...) - f((xb .- e)...)) / 2h)
        end
    end
end

"""
`skew_normal_lcdf_stable` / `skew_normal_lccdf_stable` (y, ξ, ω, α) and
`exp_mod_normal_lcdf_stable` / `exp_mod_normal_lccdf_stable` (y, μ, σ, λ) are
accurate relative to their own size in both tails, at λσ far beyond the 37.7
where Stan Math's `exp_mod_normal_lcdf` overflows and at the 8.3σ where it
collapses to Φ(z), and in the lower tail of a right-skewed law where Stan
Math's `skew_normal_lcdf` cancels to noise (snag `odl-native-and-s-929bc7d0`).
Their `@juliacompat` methods run the branches the emitted Stan runs: values and
Enzyme gradients in all four arguments match the 256-bit references.
"""
@testitem "slic: stable skew_normal and exp_mod_normal log CDF / CCDF in Julia and under Enzyme" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EnzymeGradient, SkewEMGCDFReference] begin
    # `const`: a testitem body runs at module top level, and Enzyme on Julia 1.12+
    # cannot differentiate a non-const global's `B.f` lookup (Enzyme snag
    # `reverse-gradient-dbf59c00`).
    const B = StanBlocks
    for f in (B.skew_normal_lcdf_stable, B.skew_normal_lccdf_stable,
              B.exp_mod_normal_lcdf_stable, B.exp_mod_normal_lccdf_stable)
        @test Base.return_types(f, NTuple{4,Float64}) == [Float64]
    end
    # The positive-integrand identity the skew-normal reference uses agrees with
    # the textbook Φ(z) - 2T(z, α) where that one's cancellation is affordable.
    for (z, a) in ((-0.5, 0.5), (-1.0, 2.0), (-2.0, 5.0), (-3.0, 1.0), (-0.1, 30.0))
        @test abs(sn_lcdf_reference(z, a) - sn_owen_reference(z, a)) < 1e-30
    end

    # Standard points (z, α), placed at ξ = 0.3, ω = 1.7 so every argument's
    # gradient is exercised: both tails, both signs of α, z = 0 and the
    # right-skewed lower tail (Stan Math: -41.4 at z = -3, α = 5).
    xi, om = 0.3, 1.7
    for (z, a) in ((-3.0, 5.0), (-2.0, 5.0), (-5.0, 10.0), (-30.0, 0.5), (-0.5, 100.0),
                   (-1e-3, 1e4), (0.0, 2.0), (1e-3, 50.0), (0.3, 2.0), (1.0, -5.0), (2.0, -5.0),
                   (5.0, 1.0), (30.0, 2.0), (-10.0, -3.0), (0.5, 0.0), (-4.0, 1e-3))
        @testset "skew_normal z = $z, α = $a" begin
            # References at the double y actually passed (z = (y - ξ)/ω rounds).
            y = xi + om * z
            for (key, f, ref) in ((:sn_lcdf, B.skew_normal_lcdf_stable, sn_lcdf_reference),
                                  (:sn_lccdf, B.skew_normal_lccdf_stable, sn_lccdf_reference))
                r = ref((big(y) - xi) / om, a)
                dr = reference_gradient((y, x, w, al) -> ref((y - x) / w, al), y, xi, om, a)
                value, grad = gradient(f, y, xi, om, a)
                @test value == f(y, xi, om, a)
                @test isfinite(value) && all(isfinite, grad)
                @test value ≈ Float64(r) rtol = cdf_rtol(r)
                # At z = 30 the log CDF is log1m_exp of a log CCDF near -455; its
                # gradient (1.7e-196) is one registered Enzyme loses.
                @test all(isapprox.(grad, dr; rtol = 1e-9, atol = 1e-12 * maximum(abs, dr))) broken =
                    !ENZYME_DOGFOOD && (key, z, a) in ENZYME_TINY_ADJOINT
            end
        end
    end

    # Standard points (u, τ) of U = (Y - μ)/σ = Z + E, E ~ Exp(τ), at μ = -0.2,
    # σ = 1.3 (λ = τ/σ): λσ = 10 (Stan Math collapses to Φ(z)), 37 and 100
    # (it overflows), tiny τ, deep lower and upper tails.
    mu, s = -0.2, 1.3
    for (u, t) in ((0.5, 10.0), (0.5, 37.0), (0.5, 100.0), (-12.0, 1.0), (-8.0, 1.0),
                   (-1e3, 0.5), (0.0, 1e-6), (2.0, 1e-3), (-30.0, 37.0), (5.0, 0.3),
                   (40.0, 2.0), (-2.0, 1e5), (300.0, 1.0), (0.3, 1.0), (-1e6, 3.0))
        @testset "exp_mod_normal u = $u, τ = $t" begin
            y, l = mu + s * u, t / s
            for (key, f, ref) in ((:emg_lcdf, B.exp_mod_normal_lcdf_stable, emg_lcdf_reference),
                                  (:emg_lccdf, B.exp_mod_normal_lccdf_stable, emg_lccdf_reference))
                r = ref((big(y) - mu) / s, big(l) * s)
                dr = reference_gradient((y, m, sg, lm) -> ref((y - m) / sg, lm * sg), y, mu, s, l)
                value, grad = gradient(f, y, mu, s, l)
                @test value == f(y, mu, s, l)
                @test isfinite(value) && all(isfinite, grad)
                @test value ≈ Float64(r) rtol = cdf_rtol(r)
                @test all(isapprox.(grad, dr; rtol = 1e-9, atol = 1e-12 * maximum(abs, dr))) broken =
                    !ENZYME_DOGFOOD && (key, u, t) in ENZYME_TINY_ADJOINT
            end
        end
    end

    # Infinite arguments: ±∞ give -∞ / 0 with a zero, finite gradient.
    for (f, at, v) in ((B.skew_normal_lcdf_stable, -Inf, -Inf), (B.skew_normal_lcdf_stable, Inf, 0.0),
                       (B.skew_normal_lccdf_stable, Inf, -Inf), (B.exp_mod_normal_lcdf_stable, Inf, 0.0),
                       (B.exp_mod_normal_lccdf_stable, -Inf, 0.0))
        @test f(at, 0.1, 1.0, 2.0) == v
        @test all(iszero, gradient(f, at, 0.1, 1.0, 2.0)[2][2:4])
    end
end

"""
`skew_normal_interval_lprob(lo, hi, ξ, ω, α)` and
`exp_mod_normal_interval_lprob(lo, hi, μ, σ, λ)` are log P(lo < Y ≤ hi): on
the side of the median holding less mass, and an 8-point Gauss–Legendre rule
on the stable density where the interval is too narrow for a CDF difference.
The `_standardized` forms take the interval as a center and a log half-width
of the standardized variable, with log σ and log λ for the exponentially
modified normal, so a scale beyond Float64 or a half-width below it is never
formed. Values and Enzyme gradients match the density integrated at 256 bits.
"""
@testitem "slic: skew_normal and exp_mod_normal interval log mass, narrow and in the tails" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EnzymeGradient, SkewEMGCDFReference] begin
    const B = StanBlocks
    xi, om = 0.3, 1.7
    # (z_lo, z_hi, α): straddling, below and above the median, deep tails, and
    # narrow intervals down to 1e-12 of the scale.
    for (a, b, al) in ((-0.5, 0.7, 2.0), (-3.0, -2.5, 5.0), (8.0, 8.5, 2.0), (30.0, 30.5, 2.0),
                       (0.3, 0.3 + 1e-6, 2.0), (0.3, 0.3 + 1e-12, 2.0), (-2.0, -2.0 + 1e-9, 5.0),
                       (1.0, 3.0, -4.0), (-0.01, 0.01, 50.0), (-40.0, -39.0, -0.5))
        @testset "skew_normal ($a, $b], α = $al" begin
            # References at the doubles actually passed: rounding ξ + ωz moves a
            # 1e-12 interval's width by 1e-4 of itself.
            lo, hi = xi + om * a, xi + om * b
            r = Float64(interval_reference(sn_pdf, (big(lo) - xi) / om, (big(hi) - xi) / om, al))
            value = B.skew_normal_interval_lprob(lo, hi, xi, om, al)
            @test value ≈ r rtol = 1e-13 atol = 1e-13
            c, lhw = 0.5 * a + 0.5 * b, log(0.5 * b - 0.5 * a)
            r2 = Float64(interval_reference(sn_pdf, big(c) - exp(big(lhw)), big(c) + exp(big(lhw)), al))
            @test B.skew_normal_interval_lprob_standardized(c, lhw, al) ≈ r2 rtol = 1e-13 atol = 1e-13
            # Gradient in (center, log half-width, α) of the standardized form.
            dr = reference_gradient((c, l, al) -> interval_reference(sn_pdf, c - exp(l), c + exp(l), al), c, lhw, al)
            v2, g2 = gradient(B.skew_normal_interval_lprob_standardized, c, lhw, al)
            @test all(isfinite, g2)
            @test all(isapprox.(g2, dr; rtol = 1e-9, atol = 1e-12 * maximum(abs, dr)))
        end
    end
    mu, s = -0.2, 1.3
    for (a, b, t) in ((-0.5, 0.7, 1.0), (-12.0, -11.5, 1.0), (4.0, 4.5, 1.0), (300.0, 301.0, 1.0),
                      (0.3, 0.3 + 1e-9, 1.0), (0.5, 0.5 + 1e-12, 100.0), (-1e3, -999.0, 0.5),
                      (0.0, 0.25, 1e-6), (-2.0, -1.0, 1e5))
        @testset "exp_mod_normal ($a, $b], τ = $t" begin
            lo, hi, l = mu + s * a, mu + s * b, t / s
            r = Float64(interval_reference(emg_pdf, (big(lo) - mu) / s, (big(hi) - mu) / s, big(l) * s))
            @test B.exp_mod_normal_interval_lprob(lo, hi, mu, s, l) ≈ r rtol = 1e-13 atol = 1e-13
            c, lhw = 0.5 * a + 0.5 * b, log(0.5 * b - 0.5 * a)
            f = (c, lhw, ls, ll) -> B.exp_mod_normal_interval_lprob_standardized(c, lhw, ls, ll)
            r2 = Float64(interval_reference(emg_pdf, big(c) - exp(big(lhw)), big(c) + exp(big(lhw)),
                                            exp(big(log(s)) + big(log(l)))))
            @test f(c, lhw, log(s), log(l)) ≈ r2 rtol = 1e-13 atol = 1e-13
            dr = reference_gradient((c, lw, ls, ll) -> interval_reference(emg_pdf, c - exp(lw), c + exp(lw), exp(ls + ll)),
                                    c, lhw, log(s), log(l))
            v2, g2 = gradient(f, c, lhw, log(s), log(l))
            @test all(isfinite, g2)
            @test all(isapprox.(g2, dr; rtol = 1e-9, atol = 1e-12 * maximum(abs, dr)))
        end
    end
    # A scale and a half-width beyond Float64: σ = e^1000 with log λ = 3.9 is the
    # standard normal to far below double resolution, so the mass of
    # (u - h, u + h], h = e^-1000, is log 2h + log φ(u).
    @test B.exp_mod_normal_interval_lprob_standardized(-2.5, -1000.0, 1000.0, 3.9) ≈
        log(2.0) - 1000.0 - 0.5 * log(2pi) - 2.5^2 / 2 rtol = 1e-15
    @test B.skew_normal_interval_lprob_standardized(0.3, -800.0, 2.0) ≈
        log(2.0) - 800.0 + Float64(log(sn_pdf(big(0.3), big(2.0)))) rtol = 1e-15
    # Infinite and empty intervals.
    @test B.skew_normal_interval_lprob(-Inf, 0.5, 0.0, 1.0, 2.0) == B.skew_normal_lcdf_stable(0.5, 0.0, 1.0, 2.0)
    @test B.exp_mod_normal_interval_lprob(0.5, Inf, 0.0, 1.0, 2.0) == B.exp_mod_normal_lccdf_stable(0.5, 0.0, 1.0, 2.0)
    @test B.skew_normal_interval_lprob(-Inf, Inf, 0.0, 1.0, 2.0) == 0.0
    @test B.exp_mod_normal_interval_lprob(0.5, 0.5, 0.0, 1.0, 2.0) == -Inf
    @test_throws ArgumentError B.skew_normal_interval_lprob(1.0, 0.5, 0.0, 1.0, 2.0)
    @test B.exp_mod_normal_interval_lprob_standardized(0.0, 710.0, 0.0, 0.0) == 0.0

    # stanblocks-use §8.1's rounded-reading mixture, verbatim: the bin
    # (u - h, u + h] of a standardized residual under a mixture of the
    # exponentially modified normal and the unit-variance, centred skew-normal.
    # A `@juliacompat` Julia method resolves calls in its own module.
    using StanBlocks: exp_mod_normal_interval_lprob_standardized, skew_normal_interval_lprob_standardized
    @deffun @juliacompat binmix_lprob(u::real, log_hw::real, log_sigma::real, log_lambda::real,
                                      alpha::real, p::real)::real = begin
        d = alpha / hypot(1., alpha)
        sd = sqrt(1 - 0.6366197723675814 * square(d))   # sd; the mean is d·√(2/π)
        z = u * sd + 0.7978845608028654 * d
        log_sum_exp(log(p) + exp_mod_normal_interval_lprob_standardized(u, log_hw, log_sigma, log_lambda),
                    log1m(p) + skew_normal_interval_lprob_standardized(z, log_hw + log(sd), alpha))
    end
    for (u, lhw, ls, ll, al, p) in ((0.4, log(0.05), 0.0, 0.5, 1.5, 0.1), (-3.0, log(1e-3), 2.0, -1.0, 4.0, 0.3),
                                    (2.5, -900.0, 900.0, -890.0, -2.0, 0.01))
        @testset "binmix $u" begin
            value, grad = gradient(binmix_lprob, u, lhw, ls, ll, al, p)
            @test isfinite(value) && all(isfinite, grad)
            reference = setprecision(BigFloat, 192) do
                d = big(al) / sqrt(1 + big(al)^2)
                sd = sqrt(1 - 2 * d^2 / big(pi)); z = big(u) * sd + d * sqrt(2 / big(pi))
                e = interval_reference_ch(emg_pdf, u, lhw, exp(big(ls) + big(ll)))
                q = interval_reference_ch(sn_pdf, z, big(lhw) + log(sd), al)
                Float64(log(big(p) * exp(e) + (1 - big(p)) * exp(q)))
            end
            @test value ≈ reference rtol = 1e-13 atol = 1e-13
        end
    end
end

"""
The emitted Stan of the stable skew-normal / exponentially modified normal
CDFs and interval masses computes what their Julia methods compute, value and
gradient, through BridgeStan. `censored`, `truncated` and `interval_censored`
of these families select them instead of Stan Math's `skew_normal_lcdf` /
`exp_mod_normal_lcdf`, and `interval_censored` of every family takes its mass
on the side of the median holding less of it, so an interval deep in a normal's
upper tail stays finite. Positive control: Stan Math's own
`exp_mod_normal_lcdf` at λσ = 10 is the normal log CDF Φ(z), 0.054 above the
true value (snag `odl-native-and-s-929bc7d0`).
"""
@testitem "slic: stable skew_normal and exp_mod_normal CDFs and interval masses in Stan" tags=[:slic, :ad, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, EnzymeGradient, SkewEMGCDFReference] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    const B = StanBlocks
    # `k` selects one builtin; the flat parameters (a, b, c, d) shift its
    # arguments, so the gradient at 0 is the builtin's gradient.
    @deffun begin
        @lpxf cdfprobe_lpdf(y::vector[n], k::int, w::real, a::real, b::real, c::real, d::real)::real = begin
            lp = 0.0
            for i in 1:n
                lp = lp + (k == 1 ? skew_normal_lcdf_stable(y[i] + a, b, exp(c), d) :
                          (k == 2 ? skew_normal_lccdf_stable(y[i] + a, b, exp(c), d) :
                          (k == 3 ? exp_mod_normal_lcdf_stable(y[i] + a, b, exp(c), exp(d)) :
                          (k == 4 ? exp_mod_normal_lccdf_stable(y[i] + a, b, exp(c), exp(d)) :
                          (k == 5 ? skew_normal_interval_lprob(y[i] + a, y[i] + w, b, exp(c), d) :
                          (k == 6 ? exp_mod_normal_interval_lprob(y[i] + a, y[i] + w, b, exp(c), exp(d)) :
                          (k == 7 ? exp_mod_normal_interval_lprob_standardized(y[i] + a, w + b, c, d) :
                          (k == 8 ? skew_normal_interval_lprob_standardized(y[i] + a, w + b, d) :
                                    exp_mod_normal_lcdf(y[i] + a, b, exp(c), exp(d))))))))))
            end
            lp
        end
        cdfprobe_lpdfs(y::vector[n], k::int, w::real, a::real, b::real, c::real, d::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = cdfprobe_lpdf(y[i:i], k, w, a, b, c, d)
            end
            out
        end
        cdfprobe_rng(vector[n], k::int, w::real, a::real, b::real, c::real, d::real)::vector[n] = rep_vector(0., n)
    end
    model = @slic (; y = [0.0], k = 1, w = 0.0, b0 = 0.0, c0 = 0.0, d0 = 0.0) begin
        a ~ flat()
        b ~ flat()
        c ~ flat()
        d ~ flat()
        y ~ cdfprobe(k, w, a, b0 + b, c0 + c, d0 + d)
    end
    functions = stan_block(stan_code(model), "functions")
    for name in ("skew_normal_tail_integral(", "exp_mod_normal_std_cdf_gap(", "std_interval_cdf_gap(",
                 "std_normal_inv_mills_cf_excess(", "skew_normal_std_interval_core(")
        @test occursin(name, functions)
    end
    @test stanc_compiles(model)
    julia_law(k, y, w, b, c, d) =
        k == 1 ? B.skew_normal_lcdf_stable(y, b, exp(c), d) :
        k == 2 ? B.skew_normal_lccdf_stable(y, b, exp(c), d) :
        k == 3 ? B.exp_mod_normal_lcdf_stable(y, b, exp(c), exp(d)) :
        k == 4 ? B.exp_mod_normal_lccdf_stable(y, b, exp(c), exp(d)) :
        k == 5 ? B.skew_normal_interval_lprob(y, y + w, b, exp(c), d) :
        k == 6 ? B.exp_mod_normal_interval_lprob(y, y + w, b, exp(c), exp(d)) :
        k == 7 ? B.exp_mod_normal_interval_lprob_standardized(y, w + b, c, d) :
                 B.skew_normal_interval_lprob_standardized(y, w + b, d)
    evaluate(k, y, w, b, c, d) = begin
        problem = instantiate(stan_model(model(; y = [y], k, w, b0 = b, c0 = c, d0 = d)))
        LogDensityProblems.logdensity_and_gradient(problem, zeros(4))
    end
    # (k, y, w, ξ or μ, log ω or log σ, α or log λ)
    for p in ((1, -3.0, 0.0, 0.0, 0.0, 5.0), (1, 0.4, 0.0, 0.1, 0.5, -3.0), (2, 2.0, 0.0, 0.0, 0.0, -5.0),
              (2, 30.0, 0.0, 0.0, 0.0, 2.0), (3, 0.5, 0.0, 0.0, 0.0, log(10.0)), (3, -12.0, 0.0, 0.0, 0.0, 0.0),
              (3, 0.5, 0.0, 0.0, 0.0, log(100.0)), (3, 0.0, 0.0, 0.0, 0.0, log(1e-6)), (4, 40.0, 0.0, 0.0, 0.0, log(2.0)),
              (4, 0.5, 0.0, 0.0, 0.0, log(100.0)), (5, 0.3, 1e-9, 0.0, 0.0, 2.0), (5, 8.0, 0.5, 0.0, 0.0, 2.0),
              (6, 4.0, 0.5, 0.0, 0.0, 0.0), (6, 0.3, 1e-9, 0.0, 0.0, 0.0), (7, 0.3, -20.0, 0.0, 1000.0, 3.9),
              (7, -1.0, -3.0, 0.0, 0.2, 0.1), (8, 0.3, -20.0, 0.0, 0.0, 2.0), (8, -2.0, -0.5, 0.0, 0.0, 5.0))
        @testset "$p" begin
            value, grad = evaluate(p...)
            k, y, w, b, c, d = p
            jv, jg = gradient((a, b, c, d) -> k <= 4 ? julia_law(k, y + a, w, b, c, d) :
                                  (k <= 6 ? julia_law(k, y + a, w - a, b, c, d) : julia_law(k, y + a, w, b, c, d)),
                              0.0, b, c, d)
            @test isfinite(value) && all(isfinite, grad)
            @test value ≈ jv rtol = 1e-14 atol = 1e-14
            @test all(isapprox.(grad, jg; rtol = 1e-10, atol = 1e-12 * maximum(abs, jg)))
        end
    end
    # Stan's reverse mode keeps a gradient registered Enzyme 0.13.216 loses below
    # ~1e-154 (Enzyme snag `reverse-gradient-1717f5f9`): d/dy of the log CCDF at
    # u = -30, τ = 1e-3 is -5e-201, the difference of two terms near 1.5e-196.
    value, grad = evaluate(4, -30.0, 0.0, 0.0, 0.0, log(1e-3))
    dr = reference_gradient((y, m, ls, ll) -> emg_lccdf_reference((y - m) / exp(ls), exp(ls + ll)),
                            -30.0, 0.0, 0.0, log(1e-3))
    @test value ≈ Float64(emg_lccdf_reference(-30.0, 1e-3)) rtol = cdf_rtol(emg_lccdf_reference(-30.0, 1e-3))
    @test all(isapprox.(grad, dr; rtol = 1e-9, atol = 1e-12 * maximum(abs, dr)))

    # Positive control: Stan Math 5.3.0's own exp_mod_normal_lcdf(0.5 | 0, 1, 10)
    # returns log Φ(0.5) = -0.369; the law's value is -0.423.
    native, _ = evaluate(9, 0.5, 0.0, 0.0, 0.0, log(10.0))
    @test native ≈ -0.36894641528865635 rtol = 1e-12
    @test evaluate(3, 0.5, 0.0, 0.0, 0.0, log(10.0))[1] ≈ Float64(emg_lcdf_reference(0.5, 10.0)) rtol = 1e-13

    # The combinators select the stable companions and interval masses.
    combos = @slic (; y = [0.4], lo = [0.2], hi = [0.9], v = [1.0], u = [2.0]) begin
        mu ~ normal(0, 1)
        y ~ censored(skew_normal(mu, 1.0, 3.0); lower = lo, upper = hi)
        v ~ interval_censored(exp_mod_normal(mu, 1.0, 2.0), lo, hi)
        u ~ truncated(exp_mod_normal(mu, 1.0, 2.0); lower = lo)
    end
    code = stan_code(combos)
    @test occursin("skew_normal_lcdf_stable(", code) && occursin("skew_normal_lccdf_stable(", code)
    @test occursin("exp_mod_normal_interval_lprob(", code) && occursin("exp_mod_normal_lccdf_stable(", code)
    @test !occursin(r"\bskew_normal_lc?cdf\(", code) && !occursin(r"\bexp_mod_normal_lc?cdf\(", code)
    @test stanc_compiles(combos)
    problem = instantiate(stan_model(combos))
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, [0.1])
    @test isfinite(value) && all(isfinite, grad)

    # `interval_censored` of any family takes the side with less mass: (40, 41]
    # of N(0, 1), where both log CDFs round to 0, is log(Q(40) - Q(41)).
    deep = @slic (; v = [40.5], lo = [40.0], hi = [41.0]) begin
        mu ~ flat()
        v ~ interval_censored(normal(mu, 1.0), lo, hi)
    end
    @test occursin("interval_logmass_cdf_normal(", stan_code(deep))
    value, grad = LogDensityProblems.logdensity_and_gradient(instantiate(stan_model(deep)), [0.0])
    reference = setprecision(BigFloat, 256) do
        Q(x) = StanBlocks.SpecialFunctions.erfc(big(x) / sqrt(big(2))) / 2
        Float64(log(Q(40) - Q(41)))
    end
    @test value ≈ reference rtol = 1e-13
    @test isfinite(grad[1]) && grad[1] > 0
end
