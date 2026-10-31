"""
Stan's Dirichlet-multinomial family was not registered, so
`y ~ dirichlet_multinomial(alpha, N)` failed to trace with `Could not find
dirichlet_multinomial in model, builtin, Main or Main!` (snag
`dirichlet-multin-91925143`, from a downstream module translating a
Dirichlet-multinomial likelihood). Stan's density reads the total as `sum(y)`
while its RNG needs it, so the family takes `multinomial`'s explicit-`N`
contract: one composition `y ~ dirichlet_multinomial(alpha, N)`, one per row
`Y ~ dirichlet_multinomial(alpha, row_N)`, and one per group of a ragged
observation in a plate. Its density and pointwise likelihood are Stan's
normalised `dirichlet_multinomial_lpmf`; `alpha` must be strictly positive.
"""
@testitem "slic: dirichlet_multinomial has an explicit-N density, pointwise likelihood and draws" tags=[:slic, :regression, :ragged, :plate, :descriptor, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    lgamma(x) = StanBlocks.SpecialFunctions.loggamma(x)
    # The normalised Dirichlet-multinomial log mass of counts `y` (total `sum(y)`).
    dm_lpmf(y, alpha) = begin
        a0, n = sum(alpha), sum(y)
        lgamma(a0) + lgamma(n + 1) - lgamma(n + a0) +
            sum(lgamma(yk + ak) - lgamma(ak) - lgamma(yk + 1) for (yk, ak) in zip(y, alpha))
    end
    lognormal_lpdf(x, mu, s) = -log(x) - 0.5 * log(2pi) - log(s) - 0.5 * abs2((log(x) - mu) / s)
    gq(problem, u; seed = 1) = begin
        names = BridgeStan.param_names(problem.model; include_tp = true, include_gq = true)
        values = BridgeStan.param_constrain(problem.model, u; include_tp = true, include_gq = true,
            rng = BridgeStan.StanRNG(problem.model, seed))
        prefix -> values[[startswith(s, prefix) for s in names]]
    end

    # (A) The reported shape: one composition whose concentration is a sampled
    # scale times fixed proportions.
    y, n, p = [2, 3, 1], 6, [0.2, 0.5, 0.3]
    single = @slic (; y, n, p) begin
        phi ~ lognormal(3, 1)
        y ~ dirichlet_multinomial(phi .* p, n)
    end
    code = stan_code(single)
    @test occursin("dirichlet_multinomial: explicit N must equal sum(obs)", stan_block(code, "functions"))
    @test occursin(r"y ~ dirichlet_multinomial\(", stan_block(code, "model"))
    let g = stan_block(code, "generated quantities")
        @test occursin(r"y_likelihood\s*=\s*dirichlet_multinomial_lpmfs\(y,", g)
        @test occursin(r"y_gen\s*=\s*dirichlet_multinomial_int_rng\([^;]*, n\)", g)
    end
    @test stanc_compiles(single)
    problem = instantiate(stan_model(single))
    # The full normalised density at phi = exp(u): the lognormal prior, its
    # log-Jacobian u, and the Dirichlet-multinomial mass. At u = 0 this is the
    # downstream module's reference value, -9.90632568353638.
    full(u) = (phi = exp(u); lognormal_lpdf(phi, 3, 1) + u + dm_lpmf(y, phi .* p))
    for u in (0.0, -0.7, 1.3)
        @test BridgeStan.log_density(problem.model, [u]; propto = false, jacobian = true) ≈ full(u) rtol = 1e-12
        read = gq(problem, [u])
        @test only(read("y_likelihood")) ≈ dm_lpmf(y, exp(u) .* p) rtol = 1e-12
        draw = read("y_gen")
        @test length(draw) == 3 && all(v -> v >= 0 && v == round(v), draw) && sum(draw) == n
    end
    @test full(0.0) ≈ -9.90632568353638 rtol = 1e-12
    # A supplied total that is not the observed one is rejected, not ignored.
    wrong = instantiate(stan_model(@slic (; y, n = 7, p) begin
        phi ~ lognormal(3, 1)
        y ~ dirichlet_multinomial(phi .* p, n)
    end))
    @test_throws "explicit N must equal sum(obs)" BridgeStan.log_density(wrong.model, [0.0])

    # (B) One composition per row of an integer matrix, sharing alpha.
    Y, row_N, alpha = [2 3 1; 0 4 2; 5 0 0], [6, 6, 5], [0.5, 1.0, 1.5]
    rows = @slic (; Y, row_N, alpha) begin
        phi ~ lognormal(0, 1)
        Y ~ dirichlet_multinomial(phi .* alpha, row_N)
    end
    rcode = stan_code(rows)
    @test occursin(r"Y_likelihood\s*=\s*dirichlet_multinomial_lpmfs\(Y,", rcode)
    @test occursin(r"Y_gen\s*=\s*dirichlet_multinomial_int_rng\([^;]*, row_N\)", rcode)
    @test stanc_compiles(rows)
    let read = gq(instantiate(stan_model(rows)), [0.4]; seed = 7)
        @test read("Y_likelihood") ≈ [dm_lpmf(Y[i, :], exp(0.4) .* alpha) for i in 1:3] rtol = 1e-12
        draw = reshape(read("Y_gen"), 3, 3)    # column-major over [row, category]
        @test vec(sum(draw; dims = 2)) == row_N
    end

    # (C) Unequal composition lengths: one ragged group per plate iteration,
    # each with its own proportions and total, and one likelihood per group.
    ys = [[2, 3, 1], [4, 0], [1, 1, 2, 0]]
    ps = [[0.2, 0.5, 0.3], [0.6, 0.4], [0.1, 0.2, 0.3, 0.4]]
    ns = sum.(ys)
    ragged = @slic (; ys, ps, ns, G = 3) begin
        phi ~ lognormal(3, 1)
        @plate for k in 1:G
            ys[k] ~ dirichlet_multinomial(phi .* ps[k], ns[k])
        end
    end
    @test stanc_compiles(ragged)
    d = stan_descriptor(ragged; name = :ragged_dirichlet_multinomial)
    outs = Dict(o.name => o for o in d.outputs)
    @test outs[:ys_gen].generative == :draw && outs[:ys_gen].segments == [3, 5, 9]
    @test outs[:ys_likelihood].generative == :pointwise_loglik
    let read = gq(instantiate(stan_model(ragged)), [0.2]; seed = 3)
        @test read("ys_likelihood") ≈ [dm_lpmf(ys[k], exp(0.2) .* ps[k]) for k in 1:3] rtol = 1e-12
        draw = read("ys_gen")
        @test [sum(draw[s]) for s in (1:3, 4:5, 6:9)] == ns
    end

    # (D) No observed composition: the draw needs only alpha and N.
    prior = @slic (; N = 10, K = 3) begin
        a ~ lognormal(0, 1)
        counts ~ dirichlet_multinomial(rep_vector(a, K), N)
    end
    @test occursin(r"counts\s*=\s*dirichlet_multinomial_int_rng\([^;]*, N\)", stan_code(prior))
    @test stanc_compiles(prior)
end
