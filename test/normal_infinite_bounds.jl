"""
`normal_lcdf_stable`/`normal_lccdf_stable` formed `(x - loc)/scale` before
looking at an infinite `x`. That node's scale-partial is infinite there, and
Stan's reverse pass multiplies it by a zero adjoint, 0·∞ = NaN: an interval with
a `-Inf` end, or a truncation with a `±Inf` data bound, had the right value and a
NaN gradient (todo 0x8rwg0; Student-t had the same guard added under snag
native-student-t-b7023f91). Both answer an infinite bound first now. Gradients
are checked against the one-sided closed forms.
"""
@testitem "slic: a normal tail mass at an infinite bound has a finite exact gradient" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    SF = StanBlocks.SpecialFunctions
    # log Φ(z) and d/dz log Φ(z) = φ(z)/Φ(z), in BigFloat.
    logPhi(z) = log(SF.erfc(-big(z) / sqrt(big(2))) / 2)
    mills(z) = exp(-big(z)^2 / 2) / sqrt(2 * big(pi)) / exp(logPhi(z))
    gradient_at(model, theta) = LogDensityProblems.logdensity_and_gradient(
        instantiate(stan_model(model)), theta)
    points = ((0.2, log(0.5)), (-1.3, 0.4), (2.5, -1.0))
    # `w`'s interval with one infinite end is a one-sided normal mass: z is the
    # standardized finite end, and the mass is Φ(z) below it or Φ(-z) above it.
    for (lo, hi, sign) in ((-Inf, 0.9, 1), (-0.4, Inf, -1))
        model = @slic (; w = 0.1, lo, hi) begin
            mu ~ normal(0.0, 1.0)
            ls ~ normal(0.0, 1.0)
            w ~ interval_censored(normal, lo, hi, mu, exp(ls))
        end
        @test stanc_compiles(model)
        finite_end = isinf(lo) ? hi : lo
        for (mu, ls) in points
            value, gradient = gradient_at(model, [mu, ls])
            z = sign * (finite_end - mu) / exp(big(ls))
            # d z/d mu = -sign/σ, d z/d ls = -z.
            expected = [-mu - mills(z) * sign / exp(big(ls)), -ls - mills(z) * z]
            @test isfinite(value) && all(isfinite, gradient)
            @test gradient ≈ expected rtol = 1e-10
        end
    end
    # A truncation whose data bound is infinite has a zero normaliser: its
    # gradient is the plain normal density's.
    for bounds in ((; lower = :lo), (; upper = :hi))
        rhs = Expr(:call, :truncated, :(normal(mu, exp(ls))),
            Expr(:parameters, [Expr(:kw, k, v) for (k, v) in pairs(bounds)]...))
        model = StanBlocks.stan.SlicModel(quote
            mu ~ normal(0.0, 1.0)
            ls ~ normal(0.0, 1.0)
            y ~ $rhs
        end, Dict{Symbol,Any}(:y => 0.3, :lo => -Inf, :hi => Inf), @__MODULE__)
        @test stanc_compiles(model)
        for (mu, ls) in points
            value, gradient = gradient_at(model, [mu, ls])
            r = (0.3 - mu) / exp(ls)
            @test isfinite(value) && all(isfinite, gradient)
            @test gradient ≈ [-mu + r / exp(ls), -ls - 1 + r^2] rtol = 1e-10
        end
    end
end
