"""
A `truncated` draw redraws its base family until it lands in range, so it needs
about 1/mass attempts. A range holding little of the base law's mass, such as
the upper class of a count model whose drawn mean sits far below it, exhausted
the 100000-attempt budget and the draw failed with `reject` (snag
`truncated-rng-re-bd555cae`). After 100 misses the draw now inverts the base
CDF on the range: an integer draw is the exact quantile, a real one is bisected
to float resolution. The draws below come from ranges with masses between 1e-21
and 1e-9, each checked to lie in range and against the exact truncated mean.
"""
@testitem "slic: a truncated draw from a range with little mass inverts the base CDF" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    using StanBlocks.SpecialFunctions: loggamma, erfc

    N = 2000
    draws(model, names...) = begin
        problem = instantiate(model)
        params = BridgeStan.param_names(problem.model; include_tp = true, include_gq = true)
        values = BridgeStan.param_constrain(problem.model, Float64[];
            include_tp = true, include_gq = true, rng = BridgeStan.StanRNG(problem.model, 20261028))
        map(name -> values[findall(p -> startswith(p, "$name."), params)], names)
    end
    matches(d, exact) = abs(mean(d) - exact) < 5 * std(d) / sqrt(length(d))

    # Counts: two-sided, lower- and upper-truncated neg_binomial_2 responses that
    # a prior program omits.
    nb2_lpmf(n, mu, phi) = loggamma(n + phi) - loggamma(n + 1.0) - loggamma(phi) +
        phi * log(phi / (mu + phi)) + n * log(mu / (mu + phi))
    nb2_mean(mu, phi, lo, hi) = begin
        support = lo:min(hi, lo + 20_000)
        lp = nb2_lpmf.(support, mu, phi)
        w = exp.(lp .- maximum(lp))
        sum(w .* support) / sum(w)
    end
    counts = StanBlocks.SlicModel(quote
        two ~ truncated(neg_binomial_2(mu, phi); lower = lo, upper = hi)
        above ~ truncated(neg_binomial_2(mu, phi); lower = lo)
        below ~ truncated(neg_binomial_2(mu, phi); upper = top)
    end, Dict{Symbol,Any}(:mu => fill(70.0, N), :phi => 50.0, :lo => 250, :hi => 10_000_000, :top => 9),
        @__MODULE__, (:two, :above, :below))
    functions = stan_block(stan_code(counts), "functions")
    @test !occursin("exceeded", functions)
    # Stan Math's own `neg_binomial_2_lccdf(249 | 70, 50)` is -Inf; the mass is 5.1e-21.
    @test occursin("neg_binomial_2_lccdf_precise", functions)
    @test occursin("conditioning_bisect", functions)
    @test stanc_compiles(counts)
    for (mu, phi, lo, hi, top) in ((70.0, 50.0, 250, 10_000_000, 9), (70.0, 10.0, 250, 10_000_000, 9),
                                   (1000.0, 10.0, 10, 49, 49), (70.0, 10.0, 60, 80, 80))
        two, above, below = draws(counts(; mu = fill(mu, N), phi, lo, hi, top), :two, :above, :below)
        @test all(x -> lo <= x <= hi, two) && matches(two, nb2_mean(mu, phi, lo, hi))
        @test all(>=(lo), above) && matches(above, nb2_mean(mu, phi, lo, typemax(Int)))
        @test all(x -> 0 <= x <= top, below) && matches(below, nb2_mean(mu, phi, 0, top))
    end
    # A range with no mass representable in floating point still rejects, saying so.
    failure = try
        draws(counts(; mu = fill(70.0, N), phi = 50.0, lo = 1_000_000_000, hi = 1_500_000_000, top = 9), :two)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test occursin("no probability between the bounds is representable", failure)

    # Reals: normal tails on either side and between two bounds, and bounded
    # supports, where a one-sided bracket must stop at the support's edge.
    ϕ(z) = exp(-z^2 / 2) / sqrt(2pi)
    Q(z) = erfc(z / sqrt(2)) / 2
    reals = StanBlocks.SlicModel(quote
        upper_tail ~ truncated(normal(m, 1.0); lower = a)
        lower_tail ~ truncated(normal(m, 1.0); upper = -a)
        between ~ truncated(normal(m, 1.0); lower = b, upper = b + 1.0)
        unit::vector[n] ~ truncated(beta(2.0, 50.0); lower = 0.5)
        small::vector[n] ~ truncated(gamma(5.0, 1.0); upper = 0.01)
    end, Dict{Symbol,Any}(:m => zeros(N), :n => N, :a => 6.0, :b => 8.0), @__MODULE__,
        (:upper_tail, :lower_tail, :between, :unit, :small))
    @test stanc_compiles(reals)
    upper_tail, lower_tail, between, unit, small = draws(reals, :upper_tail, :lower_tail, :between, :unit, :small)
    @test all(>=(6.0), upper_tail) && matches(upper_tail, ϕ(6) / Q(6))
    @test all(<=(-6.0), lower_tail) && matches(lower_tail, -ϕ(6) / Q(6))
    @test all(x -> 8.0 <= x <= 9.0, between) && matches(between, (ϕ(8) - ϕ(9)) / (Q(8) - Q(9)))
    @test all(x -> 0.5 <= x <= 1.0, unit)
    @test all(x -> 0.0 <= x <= 0.01, small)
    far, = draws(reals(; a = 40.0), :upper_tail)
    @test all(>=(40.0), far) && matches(far, 40.0 + 1 / 40.0 - 2 / 40.0^3)

    # A custom family needs only `_lcdf` for a two-sided `truncated`; its upper
    # tail is then the complement of its log CDF.
    dens = :(lcdf_only_lpdf(y::real, mu::real, sigma::real)::real = normal_lpdf(y, mu, sigma))
    ptws = :(lcdf_only_lpdfs(y::real, mu::real, sigma::real)::real = normal_lpdf(y, mu, sigma))
    rngs = :(lcdf_only_rng(mu::real, sigma::real)::real = normal_rng(mu, sigma))
    cdf = :(lcdf_only_lcdf(y::real, mu::real, sigma::real)::real = normal_lcdf(y, mu, sigma))
    family = StanBlocks.ValueFamily(:lcdf_only, :lpdf, dens, ptws, rngs, @__MODULE__; lcdf = cdf)
    rhs = Expr(:call, :truncated, Expr(:parameters, Expr(:kw, :lower, 4.0), Expr(:kw, :upper, 5.0)),
        Expr(:call, family, 0.0, 1.0))
    custom = StanBlocks.SlicModel(Expr(:block, Expr(:call, :~, :(z::vector[n]), rhs)),
        Dict{Symbol,Any}(:n => N), @__MODULE__, (:z,))
    @test occursin("log1m_exp(lcdf_only_lcdf(", stan_block(stan_code(custom), "functions"))
    @test stanc_compiles(custom)
    z, = draws(custom, :z)
    @test all(x -> 4.0 <= x <= 5.0, z) && matches(z, (ϕ(4) - ϕ(5)) / (Q(4) - Q(5)))
end
