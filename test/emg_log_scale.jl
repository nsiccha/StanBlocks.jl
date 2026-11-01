"""
256-bit reference for `exp_mod_normal_lpdf_standardized(u, log_sigma,
log_lambda)`, the exp_mod_normal log density log p(y | μ, σ, λ) at
u = (y - μ)/σ: log λ - log 2 - u²/2 + log erfcx((λσ - u)/√2). `log erfcx(x)`
is `x² + log erfc(x)` through MPFR up to x = 1e4 and its asymptotic series
beyond, where `exp(x²)` would leave MPFR's exponent range (the next term is
below 1e-31 there). Derivatives are central differences at h = 1e-30.
"""
@testsnippet EMGLogScaleReference begin
    emg_logerfcx(x) = x > 1e4 ?
        -log(x) - log(sqrt(big(pi))) + log1p(-1 / (2x^2) + 3 / (4x^4) - 15 / (8x^6)) :
        x^2 + log(StanBlocks.SpecialFunctions.erfc(x))
    emg_reference(u, ls, ll) = setprecision(BigFloat, 256) do
        u, ls, ll = big(u), big(ls), big(ll)
        ll - log(big(2)) - u^2 / 2 + emg_logerfcx((exp(ls + ll) - u) / sqrt(big(2)))
    end
    emg_reference_gradient(u, ls, ll) = setprecision(BigFloat, 256) do
        h = big(1e-30)
        x = big.((u, ls, ll))
        map(1:3) do j
            e = ntuple(i -> i == j ? h : zero(h), 3)
            (emg_reference((x .+ e)...) - emg_reference((x .- e)...)) / 2h
        end
    end
    # Synthetic points (u, log σ, log λ) in every branch: σ and λσ beyond
    # Float64, each side of the switches at log λσ = 20 and u/(λσ) = 1/2, the
    # exponential (z ≥ 0) side, and ordinary scales.
    EMG_POINTS = [
        (-2.5, 1000.0, 3.9),          # σ = e^1000, λσ = e^1003.9: the Mills branch
        (0.8, 1000.0, -6.0),          # σ = e^1000, λσ = e^994
        (-2.5, 1000.0, -999.5),       # σ = e^1000 with λσ = e^0.5: the scaled form
        (-4.0, 750.0, -10.0),         # λσ = e^740
        (-1.0e13, 30.0, 0.0),         # u/(λσ) = -0.94
        (0.7, 10.0, 10.0 - 1e-9),     # either side of log λσ = 20
        (0.7, 10.0, 10.0 + 1e-9),
        (0.5 * exp(21.0) * (1 - 1e-9), 0.0, 21.0),  # either side of u/(λσ) = 1/2
        (0.5 * exp(21.0) * (1 + 1e-9), 0.0, 21.0),
        (2.0e9, 0.0, 21.0),           # u > λσ = 1.3e9: the exponential side
        (0.3, 0.0, 0.0),
        (-1.0, -2.0, 1.0),
        (2.0, 0.5, 0.2),
        (5.0, 0.0, -3.0),             # z = u - λσ > 0
        (-35.0, 0.0, 1.5),            # z < -35: the scaled form's own tail
    ]
end

"""
`exp_mod_normal_lpdf_standardized(u, log_sigma, log_lambda)` evaluates the
exp_mod_normal law from a standardized residual, log scale and log rate, so it
stays finite where σ or λσ exceeds Float64 (snag `evaluate-the-unc-30e61f31`,
from a downstream module whose noise scale grows with an amplitude it carries
as a logarithm). Its `@juliacompat` Julia method runs the branches the emitted
Stan runs: values and Mooncake gradients match the 256-bit reference, central
differences of the gradient match the reference curvature, and at ordinary
scales it equals `exp_mod_normal_lpdf_stable`.
"""
@testitem "slic: exp_mod_normal_lpdf_standardized stays exact past Float64 scales in Julia and under Mooncake" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EMGLogScaleReference] begin
    import Mooncake
    f = StanBlocks.exp_mod_normal_lpdf_standardized
    gradient(g, x...) = begin
        cache = Mooncake.prepare_gradient_cache(g, x...)
        value, grads = Mooncake.value_and_gradient!!(cache, g, x...)
        value, collect(grads[2:end])
    end
    @test Base.return_types(f, (Float64, Float64, Float64)) == [Float64]

    for p in EMG_POINTS
        @testset "$p" begin
            ref = emg_reference(p...)
            dref = emg_reference_gradient(p...)
            value, grad = gradient(f, p...)
            @test isfinite(value) && all(isfinite, grad)
            @test value == f(p...)
            @test value ≈ ref rtol = 1e-13 atol = 1e-12
            @test all(isapprox.(grad, dref; rtol = 1e-10, atol = 1e-12))
        end
    end

    # σ = e^1000 and λσ = e^1003.9: the log density is -log σ - log(2π)/2 - u²/2
    # up to terms of order u/(λσ), far below Float64 resolution.
    @test f(-2.5, 1000.0, 3.9) ≈ -1000.0 - log(2pi) / 2 - 2.5^2 / 2 rtol = 1e-15
    # u > 4e307 with λσ beyond Float64: the log density is below -1e615.
    @test f(1.0e308, 0.0, 709.5) == -Inf

    # Curvature: central differences of the Mooncake gradient against the
    # reference's second derivatives (differences of its central gradient).
    curvature_points = EMG_POINTS[[1, 2, 3, 4, 7, 11, 12, 13, 14, 15]]
    for p in curvature_points
        @testset "curvature $p" begin
            x = collect(p)
            hessian = map(1:3) do j
                h = 1e-5
                e = [i == j ? h : 0.0 for i in 1:3]
                (gradient(f, (x .+ e)...)[2] - gradient(f, (x .- e)...)[2]) / 2h
            end
            reference = setprecision(BigFloat, 256) do
                map(1:3) do j
                    h = big(1e-20)
                    e = ntuple(i -> i == j ? h : zero(h), 3)
                    (emg_reference_gradient((big.(p) .+ e)...) - emg_reference_gradient((big.(p) .- e)...)) / 2h
                end
            end
            scale = maximum(abs, reduce(vcat, reference))
            @test all(isapprox.(reduce(vcat, hessian), Float64.(reduce(vcat, reference)); atol = 1e-6 * scale))
        end
    end

    # At ordinary scales the law equals `exp_mod_normal_lpdf_stable`, value and
    # gradient in (y, μ, σ, λ).
    old(y, mu, s, l) = StanBlocks.exp_mod_normal_lpdf_stable(y, mu, s, l)
    new(y, mu, s, l) = f((y - mu) / s, log(s), log(l))
    for q in ((0.0, 0.4, 0.01, 25.0), (0.0, -0.5, 0.01, 25.0), (1.3, 0.2, 0.7, 2.0),
              (-2.0, 0.5, 1.0, 1e4), (3.0, 0.1, 1.5, 0.3))
        value_old, grad_old = gradient(old, q...)
        value_new, grad_new = gradient(new, q...)
        @test value_new ≈ value_old rtol = 1e-12
        @test grad_new ≈ grad_old rtol = 1e-10
    end
end

"""
The emitted Stan of `exp_mod_normal_lpdf_standardized` matches the same
256-bit reference through BridgeStan, value and gradient in each of its three
arguments, where `exp_mod_normal_lpdf_stable` given the formed σ and λ is not
finite (σ = e^1000 overflows); at ordinary scales the two agree.
"""
@testitem "slic: exp_mod_normal_lpdf_standardized stays exact past Float64 scales in Stan" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, EMGLogScaleReference] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    # `k == 1` evaluates the law from (u, log σ, log λ); `k == 2` forms
    # y = uσ, σ and λ and calls `exp_mod_normal_lpdf_stable(y, 0, σ, λ)`. The
    # flat parameters (a, b, d) shift (u, log σ, log λ), so the gradient at 0
    # is the law's gradient.
    @deffun begin
        @lpxf emglog_lpdf(y::vector[n], k::int, ls::real, ll::real, du::real)::real = begin
            lp = 0.0
            for i in 1:n
                lp = lp + (k == 1 ? exp_mod_normal_lpdf_standardized(y[i] + du, ls, ll) :
                    exp_mod_normal_lpdf_stable((y[i] + du) * exp(ls), 0., exp(ls), exp(ll)))
            end
            lp
        end
        emglog_lpdfs(y::vector[n], k::int, ls::real, ll::real, du::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = emglog_lpdf(y[i:i], k, ls, ll, du)
            end
            out
        end
        emglog_rng(vector[n], k::int, ls::real, ll::real, du::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = exp_mod_normal_rng(0., exp(ls), exp(ll)) / exp(ls) - du
            end
            out
        end
    end
    model = @slic (; y = [0.0], k = 1, ls = 0.0, ll = 0.0) begin
        a ~ flat()
        b ~ flat()
        d ~ flat()
        y ~ emglog(k, ls + b, ll + d, a)
    end
    functions = stan_block(stan_code(model), "functions")
    @test occursin("exp_mod_normal_lpdf_standardized(", functions)
    @test occursin("exp_mod_normal_standardized_core(", functions)
    @test stanc_compiles(model)

    evaluate(p, k) = begin
        u, ls, ll = p
        problem = instantiate(stan_model(model(; y = [u], k, ls, ll)))
        LogDensityProblems.logdensity_and_gradient(problem, [0.0, 0.0, 0.0])
    end
    for p in EMG_POINTS
        @testset "$p" begin
            value, grad = evaluate(p, 1)
            @test isfinite(value) && all(isfinite, grad)
            @test value ≈ emg_reference(p...) rtol = 1e-13 atol = 1e-12
            @test all(isapprox.(grad, emg_reference_gradient(p...); rtol = 1e-10, atol = 1e-12))
        end
    end
    # Positive control: forming σ = e^1000 leaves no finite density.
    @test !isfinite(try evaluate((-2.5, 1000.0, 3.9), 2)[1] catch; NaN end)

    for p in ((0.3, 0.0, 0.0), (-1.0, -2.0, 1.0), (2.0, 0.5, 0.2), (5.0, 0.0, -3.0),
              (-35.0, 0.0, 1.5), (0.7, -4.6, 3.2))
        value_new, grad_new = evaluate(p, 1)
        value_old, grad_old = evaluate(p, 2)
        @test value_new ≈ value_old rtol = 1e-12
        @test grad_new ≈ grad_old rtol = 1e-10
    end
end
