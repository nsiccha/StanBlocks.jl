"""
A family companion called on a whole RaggedVector applies the family group by
group (decision `11dxlp0`, groupwise direct-call adapters). `ys ~ foo(args…)`
over a ragged `ys` already lowered per group, but a direct call such as
`normal_lpdf(ys, mu, s)` inside another `@deffun` traced with no Stan definition
behind it and was refused (`normal_RaggedVector_lpdf` is undeclared), and at top
level it typed as `anything`. Now:

- `foo_lpdf(ys, args…)` is the sum over groups of `foo_lpdf(ys[g], args_g…)`;
- `foo_lpdfs(ys, args…)` is one aggregate density per group (`vector[length(ys)]`),
  the grain of a ragged observation's `<obs>_likelihood`;
- `foo_rng(args…)` with a ragged argument draws each group from the family's sized
  companion and returns a RaggedVector with that argument's ends.

Ragged arguments are sliced per group; scalar and dense ones are shared. A family
with its own method for the whole RaggedVector keeps it.
"""
@testitem "slic: a family companion called on a whole RaggedVector applies the family per group" tags=[:slic, :ragged, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    @deffun begin
        # A custom family whose location is per element, so it is sliced per group.
        @lhs @lpxf gfam_lpdf(y::vector[n], mu::vector[n], s::real)::real = normal_lpdf(y, mu, s)
        gfam_lpdfs(y::vector[n], mu::vector[n], s::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = normal_lpdf(y[i], mu[i], s)
            end
            out
        end
        gfam_rng(mu::real, s::real)::real = normal_rng(mu, s)
        # Stan's vectorised `normal_rng` crashes on an empty vector, so a sized
        # companion returns the empty draw itself (stanblocks-use §8).
        gfam_rng(vector[n], mu::vector[n], s::real)::vector[n] = if n == 0
            rv::vector[n]
            rv
        else
            to_vector(normal_rng(mu, s))
        end
        # A family with its own method for a whole RaggedVector keeps it.
        @lpxf whole_lpdf(y::vector[n], mu::real)::real = normal_lpdf(y, mu, 1.0)
        whole_lpdf(y::RaggedVector, mu::real)::real = -1.0
        whole_lpdfs(y::vector[n], mu::real)::vector[n] = rep_vector(0.0, n)
        whole_rng(mu::real)::real = normal_rng(mu, 1.0)

        builtin_lp(ys::RaggedVector, mu::real, s::real)::real = normal_lpdf(ys, mu, s)
        custom_lp(ys::RaggedVector, mus::RaggedVector, s::real)::real = gfam_lpdf(ys, mus, s)
        custom_lps(ys::RaggedVector, mus::RaggedVector, s::real)::vector[length(ys)] =
            gfam_lpdfs(ys, mus, s)
        count_lp(ks::RaggedVector, lam::real)::real = poisson_lpmf(ks, lam)
        weighted_lp(ys::RaggedVector, ws::RaggedVector, mu::real, s::real)::real =
            weighted_lpdf(ys, normal, ws, mu, s)
        whole_lp(ys::RaggedVector, mu::real)::real = whole_lpdf(ys, mu)
        # A draw inside another function: Stan requires its name to end in `_rng`.
        draw_sum_rng(mus::RaggedVector, s::real)::real = sum(gfam_rng(mus, s).mem)
    end

    # The second group is empty: it contributes nothing to a sum and 0 to its own
    # pointwise entry, and draws nothing.
    ys = [[0.6, 0.8, 1.0], Float64[], [0.4, 0.9]]
    mus = [[0.5, 0.7, 1.2], Float64[], [0.3, 1.1]]
    ws = [[1.0, 2.0, 0.5], Float64[], [1.5, 1.0]]
    ks = [[1, 0, 3], Int[], [2, 2]]
    data = Dict{Symbol,Any}(:ys => ys, :mus => mus, :ws => ws, :ks => ks, :y0 => 0.0)
    nlpdf(y, m, s) = -0.5 * log(2pi) - log(s) - 0.5 * abs2((y - m) / s)
    plpmf(k, l) = k * log(l) - l - sum(log, 1:k; init=0.0)

    body = quote
        mu ~ normal(0, 1)
        s ~ exponential(1)
        lam ~ exponential(1)
        a = builtin_lp(ys, mu, s)
        b = custom_lp(ys, mus, s)
        c = custom_lps(ys, mus, s)
        k = count_lp(ks, lam)
        w = weighted_lp(ys, ws, mu, s)
        o = whole_lp(ys, mu)
        y0 ~ normal(a + b + sum(c) + k + w + o, 10.0)
    end
    m = StanBlocks.SlicModel(body, data, @__MODULE__)
    code = stan_code(m)
    @test occursin("ragged_group_density_normal_lpdf_RaggedVector(", code)
    # Each RaggedVector argument adds its tag; the ragged `mus` is sliced per group.
    @test occursin("ragged_group_density_gfam_lpdf_RaggedVector_RaggedVector(", code)
    @test occursin("gfam_lpdf(getindex_RaggedVector(y, g) | getindex_RaggedVector(args1, g), args2)", code)
    @test occursin("ragged_group_pointwise_gfam_lpdf_RaggedVector_RaggedVector(", code)
    @test occursin("ragged_group_density_poisson_lpmf_RaggedVector(", code)
    @test occursin("ragged_group_density_weighted_lpdf_", code)
    # The family's own whole-RaggedVector method is called, not the adapter.
    @test occursin("whole_RaggedVector_lpdf(", code)
    @test !occursin("ragged_group_density_whole_lpdf", code)
    @test stanc_compiles(m)

    p = instantiate(m)
    sm = p.model
    names = BridgeStan.param_names(sm; include_tp=true)
    value(draw, name) = draw[findfirst(==(name), names)]
    mu, s, lam = 0.3, 0.7, 1.4
    draw = BridgeStan.param_constrain(sm, [mu, log(s), log(lam)]; include_tp=true)
    groupwise(f) = [sum((f(g, i) for i in eachindex(ys[g])); init=0.0) for g in eachindex(ys)]
    @test value(draw, "a") ≈ sum(groupwise((g, i) -> nlpdf(ys[g][i], mu, s)))
    @test value(draw, "b") ≈ sum(groupwise((g, i) -> nlpdf(ys[g][i], mus[g][i], s)))
    @test [value(draw, "c.$g") for g in 1:3] ≈ groupwise((g, i) -> nlpdf(ys[g][i], mus[g][i], s))
    @test value(draw, "c.2") == 0.0
    @test value(draw, "k") ≈ sum(sum(plpmf(k, lam) for k in kg; init=0.0) for kg in ks)
    @test value(draw, "w") ≈ sum(groupwise((g, i) -> ws[g][i] * nlpdf(ys[g][i], mu, s)))
    @test value(draw, "o") == -1.0

    # Draws: each group from the family's sized companion, with the ragged
    # argument's ends. A tiny scale makes the custom draw its location.
    lams = [[0.5, 2.0], [3.0], Float64[]]
    rdata = Dict{Symbol,Any}(:mus => mus, :lams => lams)
    rbody = quote
        d = gfam_rng(mus, 1e-9)
        d3 = d[3]
        counts = poisson_rng(lams)
        cm = counts.mem
        nd = normal_rng(mus, 0.5)
        nd1 = nd[1]
        t = draw_sum_rng(mus, 1e-9)
    end
    r = StanBlocks.SlicModel(rbody, rdata, @__MODULE__)
    rcode = stan_code(r)
    @test occursin("ragged_group_vector_gfam_rng_", rcode)
    @test occursin("ragged_group_int_poisson_rng_", rcode)
    @test stanc_compiles(r)
    rp = instantiate(r)
    rnames = BridgeStan.param_names(rp.model; include_tp=true, include_gq=true)
    rdraw = BridgeStan.param_constrain(rp.model, Float64[]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(rp.model, 1100))
    rvalues(prefix) = rdraw[findall(n -> n == prefix || startswith(n, prefix * "."), rnames)]
    @test rvalues("d3") ≈ mus[3] atol=1e-6
    @test only(rvalues("t")) ≈ sum(sum, mus) atol=1e-6
    @test length(rvalues("nd1")) == length(mus[1]) && all(isfinite, rvalues("nd1"))
    # A discrete family draws integers into an `array[] int` memory, one per element.
    @test length(rvalues("cm")) == sum(length, lams)
    @test all(x -> isinteger(x) && x >= 0, rvalues("cm"))
end
