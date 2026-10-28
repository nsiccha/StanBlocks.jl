"""
Stan's `lchoose(n, k)`, the log of the binomial coefficient, was not
registered, so a `@deffun` body calling it failed to trace with `Could not find
lchoose in model, builtin, … or Main!` (snag `a-deffun-helper-d7d895a9`, from a
downstream module whose count family sums the terms of a binomial). It takes the
forms `lbeta` takes: two ints or reals, a scalar with a vector, row vector,
matrix or real array, or two containers of one kind. `test/elementwise_containers.jl`
checks every container form against stanc.
"""
@testitem "slic: Stan's lchoose traces in a @deffun loop and a custom family" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # The reported shape: a helper whose loop calls `lchoose(int, int)`, behind
    # an `@lhs @lpxf` family. The family is the count of a binomial plus an
    # independent Poisson, summed over the binomial's share `c`.
    @deffun begin
        binpois_logsum(y::int, o::int, p::real, lam::real)::real = begin
            k = min(o, y)
            terms = rep_vector(0.0, k + 1)
            for c in 0:k
                terms[c + 1] = lchoose(o, c) + c * log(p) + (o - c) * log1m(p) + poisson_lpmf(y - c, lam)
            end
            log_sum_exp(terms)
        end
        @lhs @lpxf binpois_lpmf(y::int, o::int, p::real, lam::real)::real = binpois_logsum(y, o, p, lam)
        binpois_lpmfs(y::int, o::int, p::real, lam::real)::real = binpois_logsum(y, o, p, lam)
        binpois_rng(o::int, p::real, lam::real)::int = binomial_rng(o, p) + poisson_rng(lam)
    end
    reported = @slic (; y = 7, o = 5) begin
        p ~ beta(2.0, 2.0)
        lam ~ exponential(1.0)
        y ~ binpois(o, p, lam)
    end
    @test occursin("lchoose(o, c)", stan_block(stan_code(reported), "functions"))
    @test stanc_compiles(reported)

    # Stan's density is the convolution: its differences between two points
    # equal the Julia ones, computed with exact binomial coefficients. The
    # parameters are unconstrained (`logit(p)`, `log(lam)`), Jacobian included.
    convolution(y, o, p, lam) = log(sum(binomial(o, c) * p^c * (1 - p)^(o - c) *
        exp(-lam) * lam^(y - c) / factorial(y - c) for c in 0:min(o, y)))
    julia_lp(u) = begin
        p, lam = 1 / (1 + exp(-u[1])), exp(u[2])
        log(p) + log1p(-p) + -lam + convolution(7, 5, p, lam) + log(p * (1 - p)) + log(lam)
    end
    problem = instantiate(stan_model(reported))
    u0, u1 = [0.0, 0.0], [0.4, 1.1]
    @test LogDensityProblems.logdensity(problem, u1) - LogDensityProblems.logdensity(problem, u0) ≈
        julia_lp(u1) - julia_lp(u0) atol=1e-12

    # The scalar forms: ints, reals, and the two mixed; each a real.
    data = (; n = 6, k = 2, y = 0.3)
    model(rhs) = StanBlocks.stan.SlicModel(quote
        s ~ normal(3.0, 1.0)
        out = $rhs
        y ~ normal(out, 1.0)
    end, data, @__MODULE__)
    for (rhs, decl) in (:(lchoose(n, k)) => "real out = lchoose(n, k);",
                        :(lchoose(s, k)) => "real out = lchoose(s, k);",
                        :(lchoose(n, s)) => "real out = lchoose(n, s);",
                        :(lchoose(s + 1.0, s)) => "real out = lchoose((s + 1.0), s);")
        m = model(rhs)
        @test occursin(decl, stan_code(m))
        @test stanc_compiles(m)
    end
end

@testsnippet LChooseShims begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    @deffun @juliacompat begin
        lch(n::real, k::real)::real = lchoose(n, k)
        lch_vector(n::vector[m], k::vector[m])::vector[m] = lchoose(n, k)
        lch_ints(n::int[m], k::int)::real[m] = lchoose(n, k)
        dg(x::real)::real = digamma(x)
    end
end

"""
The `@juliacompat` method of `lchoose` follows Stan Math's
`binomial_coefficient_log`: exactly 0 at `k == 0`, the symmetric branch for
`k > n / 2`, `lgamma`s below `n + 1 = 10` and `lbeta` above, `-Inf` at
`k == n + 1`, NaN for a NaN argument, and Stan's domain error outside
`n >= -1`, `k >= -1`, `n - k >= -1`. Stan's own builtin, evaluated by BridgeStan
as a generated quantity, is the oracle.
"""
@testitem "slic: @juliacompat lchoose matches Stan" tags=[:slic, :bridgestan] setup=[StanBlocksImports, LChooseShims] begin
    exact(n, k) = Float64(log(binomial(big(n), big(k))))
    for (n, k) in ((5, 2), (5, 3), (10, 3), (9, 4), (50, 25), (200, 7), (10^6, 3))
        @test lch(n, k) ≈ exact(n, k) rtol=1e-14
        @test lch(Float64(n), Float64(k)) == lch(n, k)
    end
    @test lch(5, 0) === 0.0
    @test lch(5, 5) === 0.0
    @test lch(0, 0) === 0.0
    @test lch(3, 4) == -Inf
    @test lch(-1, 0) === 0.0
    # Non-integer arguments: Γ(3.5) / (Γ(2.5) Γ(2)) = 2.5.
    @test lch(2.5, 1.5) ≈ log(2.5) atol=1e-14
    @test isnan(lch(NaN, 1.0)) && isnan(lch(3.0, NaN))
    # Stan's domain: n >= -1, k >= -1 and n - k >= -1 (after the symmetric branch).
    for (n, k) in ((-2.0, 0.0), (3.0, 5.0), (2.0, -1.5), (-1.0, 0.5))
        @test_throws DomainError lch(n, k)
    end
    # Elementwise, with Stan's size check.
    @test lch_vector([5.0, 10.0], [2.0, 3.0]) == [lch(5, 2), lch(10, 3)]
    @test lch_ints([5, 6, 7], 2) == [lch(5, 2), lch(6, 2), lch(7, 2)]
    @test_throws DimensionMismatch lch_vector([5.0], [2.0, 3.0])

    # Stan's values, as generated quantities of a model that reads `mu`.
    ns = [5.0, 5.0, 9.0, 10.0, 50.0, 200.0, 1.0e6, 2.5, 0.3, 7.0, 3.0, -1.0, 12.5]
    ks = [2.0, 3.0, 4.0, 3.0, 25.0, 7.0, 3.0, 1.5, 0.8, 0.0, 4.0, 0.0, 11.0]
    model = @slic (; ns, ks, obs = 0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        out = lchoose(ns + mu, ks)
    end
    p = instantiate(stan_model(model))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(p.model, [0.0]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1))
    stan = values[[startswith(s, "out.") for s in names]]
    julia = lch_vector(ns, ks)
    @test length(stan) == length(ns)
    @test all(isequal(a, b) || isapprox(a, b; rtol=1e-14, atol=1e-15) for (a, b) in zip(stan, julia))
end

"""
Mooncake differentiates the Julia `lchoose`. Its partial derivatives are
`digamma(n + 1) - digamma(n - k + 1)` in `n` and
`digamma(n - k + 1) - digamma(k + 1)` in `k`, on both the `lgamma` and the
`lbeta` branch and after the symmetric branch.
"""
@testitem "slic: @juliacompat lchoose is Mooncake differentiable" tags=[:slic, :ad] setup=[StanBlocksImports, LChooseShims] begin
    import Mooncake
    gradient(f, x...) = begin
        cache = Mooncake.prepare_gradient_cache(f, x...)
        Mooncake.value_and_gradient!!(cache, f, x...)
    end
    for (n, k) in ((5.0, 2.0), (5.0, 3.5), (2.5, 0.7), (40.0, 12.0), (40.0, 31.0), (1.0e4, 3.0))
        value, (_, dn, dk) = gradient(lch, n, k)
        @test value == lch(n, k)
        @test dn ≈ dg(n + 1) - dg(n - k + 1) rtol=1e-10
        @test dk ≈ dg(n - k + 1) - dg(k + 1) rtol=1e-10
    end
    n, k = [5.0, 40.0], [2.0, 12.0]
    _, (_, dn, dk) = gradient((n, k) -> sum(lch_vector(n, k)), n, k)
    @test dn ≈ dg.(n .+ 1) .- dg.(n .- k .+ 1) rtol=1e-10
    @test dk ≈ dg.(n .- k .+ 1) .- dg.(k .+ 1) rtol=1e-10
    @test Base.return_types(lch, (Float64, Float64)) == [Float64]
    @test Base.return_types(lch, (Int, Int)) == [Float64]
end
