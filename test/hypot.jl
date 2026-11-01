"""
Stan's `hypot(x, y)` is `sqrt(x^2 + y^2)` computed without forming either
square, so it stays finite where a square overflows (`x > 1.34e154`) and
nonzero where one underflows. It was not registered: a body combining a fitted
scale with a per-observation data scale, `s = hypot(sigma, extra)`, failed with
`tracetype not defined for s = hypot(::real, ::vector)::anything!` (snag
`compile-an-odl-b-31b36301`, from a downstream module's observation scale), and
so did a producer that splices Julia's own `Base.hypot` into the body. stanc
2.39 gives `hypot` exactly `fmax`'s signatures, so it takes the same forms: two
ints or reals, a scalar with a vector, row vector, matrix or real array, or two
containers of one kind. `test/elementwise_containers.jl` checks every container
form against stanc.
"""
@testitem "slic: Stan's hypot traces and keeps an observation scale finite past the square's overflow" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # The reported shape: a positive fitted scale combined with a data scale per
    # observation, as the scale of a vectorised normal observation.
    data = (; y = [0.0, 0.5, -0.25], extra = [0.0, 0.3, 1.2])
    model(scale) = StanBlocks.stan.SlicModel(quote
        log_sigma ~ normal(0.0, 1.0)
        sigma = exp(log_sigma)
        s = $scale
        y ~ normal(0.0, s)
    end, data, @__MODULE__)
    by_name = model(:(hypot(sigma, extra)))
    code = stan_code(by_name)
    @test occursin("s = hypot(sigma, extra);", stan_block(code, "transformed parameters"))
    @test stanc_compiles(by_name)
    # BRM's model scope builds `Expr(:call, f, args...)` from the evaluated
    # callable, so it hands StanBlocks `Base.hypot` itself.
    @test stan_code(model(Expr(:call, Base.hypot, :sigma, :extra))) == code

    # Stan's density and gradient in `log_sigma`, against a 256-bit reference:
    # sum over i of normal_lpdf(y[i] | 0, hypot(exp(x), extra[i])), plus the prior.
    reference(x) = setprecision(256) do
        x = BigFloat(x)
        sigma2 = exp(2x)
        lp, grad = -x^2 / 2 - log(2 * BigFloat(pi)) / 2, -x
        for (y, e) in zip(data.y, data.extra)
            s2 = sigma2 + BigFloat(e)^2
            lp += -log(s2) / 2 - log(2 * BigFloat(pi)) / 2 - y^2 / (2s2)
            grad += (-1 + y^2 / s2) * sigma2 / s2
        end
        Float64(lp), Float64(grad)
    end
    stan = instantiate(stan_model(by_name))
    # 1e155 and 1e187 are the scales the downstream module met; their squares
    # overflow Float64.
    for sigma in (0.05, 0.7, 3.0, 1e155, 1e187)
        x = log(sigma)
        lp, grad = BridgeStan.log_density_gradient(stan.model, [x]; propto=false, jacobian=true)
        expected_lp, expected_grad = reference(x)
        @test lp ≈ expected_lp rtol=1e-13
        @test only(grad) ≈ expected_grad rtol=1e-12
    end
    # The control: the same scale spelled with squares is infinite at 1e155, so
    # the observation has zero density there and the log density is -Inf.
    squares = instantiate(stan_model(model(:(sqrt(square(sigma) + square(extra))))))
    @test BridgeStan.log_density(squares.model, [log(1e155)]; propto=false, jacobian=true) == -Inf
    @test isfinite(BridgeStan.log_density(stan.model, [log(1e155)]; propto=false, jacobian=true))

    # The scalar forms: two reals, a real and an int, two ints (data, so
    # transformed data); each a real.
    scalars = (; y = 0.3, n = 3, k = 4)
    scalar_model(rhs) = StanBlocks.stan.SlicModel(quote
        sigma ~ lognormal(0.0, 1.0)
        out = $rhs
        y ~ normal(0.0, out)
    end, scalars, @__MODULE__)
    for (rhs, decl) in (:(hypot(sigma, 2.0)) => "real out = hypot(sigma, 2.0);",
                        :(hypot(n, sigma)) => "real out = hypot(n, sigma);",
                        :(hypot(n, k)) => "real out = hypot(n, k);")
        m = scalar_model(rhs)
        @test occursin(decl, stan_code(m))
        @test stanc_compiles(m)
    end
    # Julia's `hypot` takes any number of arguments; Stan's takes two.
    @test_throws "tracetype" stan_code(scalar_model(:(hypot(sigma, 2.0, 3.0))))
end

@testsnippet HypotShims begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    @deffun @juliacompat begin
        hyp(x::real, y::real)::real = hypot(x, y)
        hyp_vector(x::vector[m], y::vector[m])::vector[m] = hypot(x, y)
        hyp_scale(x::real, y::vector[m])::vector[m] = hypot(x, y)
    end
end

"""
The `@juliacompat` method of `hypot` is Julia's `Base.hypot`, the same IEEE
function as the `std::hypot` Stan Math calls: no intermediate square, `Inf`
when either side is infinite even beside a NaN, NaN otherwise for a NaN side,
and no domain check, so a negative side has its absolute value's length.
Stan's own builtin, evaluated by BridgeStan as a generated quantity, is the
oracle; both agree with a 256-bit reference.
"""
@testitem "slic: @juliacompat hypot matches Stan" tags=[:slic, :bridgestan] setup=[StanBlocksImports, HypotShims] begin
    xs = [3.0, 1e155, 1e187, 1e-200, 1e-320, 0.003, -3.0, 0.0, 5.0, Inf, -Inf, NaN, NaN]
    ys = [4.0, 1e155, 1e186, 1e-200, 1e-320, 0.02, 4.0, 0.0, 0.0, NaN, 1.0, 1.0, Inf]
    reference(x, y) = isinf(x) || isinf(y) ? Inf : isnan(x) || isnan(y) ? NaN :
        setprecision(() -> Float64(sqrt(BigFloat(x)^2 + BigFloat(y)^2)), 256)
    for (x, y) in zip(xs, ys)
        @test isequal(hyp(x, y), reference(x, y)) || isapprox(hyp(x, y), reference(x, y); rtol=eps())
    end
    @test hyp(3, 4) === 5.0
    @test hyp(1e155, 1e155) ≈ sqrt(2) * 1e155 rtol=1e-15
    # Elementwise, with Stan's size check.
    @test isequal(hyp_vector(xs, ys), hyp.(xs, ys))
    @test hyp_scale(1e155, [0.0, 1e154]) == hypot.(1e155, [0.0, 1e154])
    @test_throws DimensionMismatch hyp_vector([1.0], [2.0, 3.0])

    # Stan's values, as generated quantities of a model that reads `mu`.
    model = @slic (; xs, ys, obs = 0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        out = hypot(xs + mu, ys)
    end
    p = instantiate(stan_model(model))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(p.model, [0.0]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1))
    stan = values[[startswith(s, "out.") for s in names]]
    @test length(stan) == length(xs)
    # Two libraries' hypot, each within an ulp of the true length.
    @test all(isequal(a, b) || isapprox(a, b; rtol=2eps()) for (a, b) in zip(stan, hyp_vector(xs, ys)))
end

"""
Mooncake differentiates the Julia `hypot`: its partial derivatives are
`x / hypot(x, y)` and `y / hypot(x, y)`, which Stan Math's reverse mode uses too,
also where the squares would overflow or underflow.
"""
@testitem "slic: @juliacompat hypot is Mooncake differentiable" tags=[:slic, :ad] setup=[StanBlocksImports, HypotShims] begin
    import Mooncake
    gradient(f, x...) = begin
        cache = Mooncake.prepare_gradient_cache(f, x...)
        Mooncake.value_and_gradient!!(cache, f, x...)
    end
    for (x, y) in ((3.0, 4.0), (0.003, 0.02), (-3.0, 4.0), (1e155, 1e154), (1e155, 1.0), (1e-200, 3e-200))
        value, (_, dx, dy) = gradient(hyp, x, y)
        h = hypot(x, y)
        @test value == h
        @test dx ≈ x / h rtol=1e-14
        @test dy ≈ y / h rtol=1e-14 atol=1e-300
    end
    x, y = [3.0, 1e155], [4.0, 1e154]
    _, (_, dx, dy) = gradient((x, y) -> sum(hyp_vector(x, y)), x, y)
    @test dx ≈ x ./ hypot.(x, y) rtol=1e-14
    @test dy ≈ y ./ hypot.(x, y) rtol=1e-14
    @test Base.return_types(hyp, (Float64, Float64)) == [Float64]
    @test Base.return_types(hyp, (Int, Int)) == [Float64]
end
