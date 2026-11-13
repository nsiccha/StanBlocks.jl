
"""
Stan 2.36 added the constrained types `row_stochastic_matrix[N, M]` (each row a
simplex) and `column_stochastic_matrix[N, M]` (each column a simplex). A typed
declaration of either failed at trace time with the bare `TypeError: in <:,
expected Type, got a value of type Symbol`, because the type name was not
registered (todo `0khvo8v`, filed from snag `dotted-call-over-fc81a4d9` for a
downstream hidden-Markov module). Both now declare natively, keep both sizes,
and sign as a plain `matrix` in a Stan function. `~ dirichlet(alpha)` samples
one dirichlet per row (column) through the internal families `dirichlet_rows` /
`dirichlet_columns`, which also re-draw an unobserved prior in generated
quantities. Plate cells collect as `array[outer...] <kind>[N, M]`. An unknown
declaration type fails naming the declarable types.
"""
@testitem "slic: row- and column-stochastic matrices declare, sample and re-draw" tags=[:slic, :regression, :shapes, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    model(body, data) = StanBlocks.stan.SlicModel(body, data, @__MODULE__)
    message(body, data) = try
        stan_code(model(body, data))
        ""
    catch err
        sprint(showerror, err)
    end

    # The reported shape: an elementwise prior on the constrained values. No
    # family draws a stochastic matrix but the dirichlet, so the prior-only
    # program keeps it a sampled parameter instead of re-drawing it.
    for T in (:row_stochastic_matrix, :column_stochastic_matrix)
        m = model(quote
            G::$T[K, L] ~ normal(0, 1)
            z ~ normal(sum(G), 1)
        end, (; K = 3, L = 4))
        code = stan_code(m)
        @test occursin("$T[K, L] G;", stan_block(code, "parameters"))
        @test occursin("G ~ normal(0, 1);", stan_block(code, "model"))
        @test stanc_compiles(m)
    end

    # A dirichlet prior is one dirichlet per row (column), each its own family.
    data = (; K = 3, L = 4, y = [1, 2, 3], w = [2, 1, 3])
    fitted = model(quote
        G::row_stochastic_matrix[K, L] ~ dirichlet(rep_vector(1.5, L))
        H::column_stochastic_matrix[K, L] ~ dirichlet(rep_vector(2.0, K))
        y ~ categorical(G[1, :]')
        w ~ categorical(H[:, 2])
    end, data)
    code = stan_code(fitted)
    @test occursin("row_stochastic_matrix[K, L] G;", stan_block(code, "parameters"))
    @test occursin("column_stochastic_matrix[K, L] H;", stan_block(code, "parameters"))
    @test occursin("G ~ dirichlet_rows(rep_vector(1.5, L));", stan_block(code, "model"))
    @test occursin("H ~ dirichlet_columns(rep_vector(2.0, K));", stan_block(code, "model"))
    @test occursin("real dirichlet_rows_lpdf(", code)
    @test occursin("real dirichlet_columns_lpdf(", code)
    @test stanc_compiles(fitted)
    outputs = Dict(o.name => o for o in stan_descriptor(fitted).outputs)
    @test outputs[:G].type === :row_stochastic_matrix
    @test outputs[:H].type === :column_stochastic_matrix
    @test outputs[:G].dims == outputs[:H].dims == (3, 4)

    # Prior-only: both are re-drawn in generated quantities as plain matrices,
    # like a simplex re-draws as a vector.
    prior = model(quote
        G::row_stochastic_matrix[K, L] ~ dirichlet(rep_vector(1.5, L))
        H::column_stochastic_matrix[K, L] ~ dirichlet(rep_vector(2.0, K))
    end, (; K = 3, L = 4))
    code = stan_code(prior)
    @test !occursin("stochastic_matrix", stan_block(code, "parameters"))
    gq = stan_block(code, "generated quantities")
    @test occursin(r"matrix\[K, L\] G = dirichlet_rows_\w*rng\(\(K, L\), rep_vector\(1.5, L\)\);", gq)
    @test occursin(r"matrix\[K, L\] H = dirichlet_columns_\w*rng\(\(K, L\), rep_vector\(2.0, K\)\);", gq)
    @test stanc_compiles(prior)

    # Plate cells collect as arrays of the constrained type.
    plated = model(quote
        G ~ plate(; outer = S) do s
            G::row_stochastic_matrix[K, L] ~ dirichlet(rep_vector(1.0, L))
            G
        end
        y ~ categorical(G[2][1, :]')
    end, (; S = 2, K = 3, L = 4, y = [1, 4]))
    code = stan_code(plated)
    @test occursin(r"array\[S\] row_stochastic_matrix\[K, L\] \w+;", stan_block(code, "parameters"))
    @test occursin("~ dirichlet_rows(", stan_block(code, "model"))
    @test stanc_compiles(plated)

    # A Stan function takes either kind as a `matrix`; one definition serves a
    # stochastic and a plain matrix argument.
    @deffun begin
        "Total of a matrix."
        stoch_matrix_total(M::matrix[m, n])::real = sum(M)
    end
    passed = model(quote
        G::row_stochastic_matrix[K, L] ~ dirichlet(rep_vector(1.0, L))
        H::column_stochastic_matrix[K, L] ~ dirichlet(rep_vector(1.0, K))
        z ~ normal(stoch_matrix_total(G) + stoch_matrix_total(H) + stoch_matrix_total(G * H'), 1)
    end, (; K = 2, L = 3, z = 0.5))
    code = stan_code(passed)
    @test count("real stoch_matrix_total(matrix M)", code) == 1
    @test stanc_compiles(passed)

    # A bound every element meets restates the type and is dropped; any other is
    # refused naming the fix.
    restated = model(quote
        G::row_stochastic_matrix[K, L] ~ beta(2, 2)
        y ~ categorical(G[1, :]')
    end, (; K = 2, L = 3, y = [1, 2]))
    @test occursin("row_stochastic_matrix[K, L] G;", stan_block(stan_code(restated), "parameters"))
    @test occursin("Stan declares no bound or affine transform on a `row_stochastic_matrix`", message(quote
        G::row_stochastic_matrix[K, L] ~ normal(0, 1; lower = 0.2)
    end, (; K = 2, L = 3)))

    # refused: an array of stochastic matrices outside a plate, a missing size,
    # a cell-varying plate size and a size per group; each names its form.
    @test occursin("For an array of them, sample each in a plate cell", message(quote
        G::row_stochastic_matrix[S, K, L] ~ dirichlet(rep_vector(1.0, L))
    end, (; S = 2, K = 2, L = 3)))
    @test occursin("takes exactly two sizes, `column_stochastic_matrix[N, M]`", message(quote
        G::column_stochastic_matrix[K] ~ dirichlet(rep_vector(1.0, K))
    end, (; K = 2)))
    @test occursin("varying-size constrained cell `G::row_stochastic_matrix`", message(quote
        G ~ plate(; outer = S) do s
            G::row_stochastic_matrix[Ks[s], L] ~ dirichlet(rep_vector(1.0, L))
            G
        end
    end, (; S = 2, Ks = [2, 3], L = 4)))
    @test occursin("Ragged constrained `G::row_stochastic_matrix[…]` is not supported", message(quote
        G::row_stochastic_matrix[Ks, L] ~ dirichlet(rep_vector(1.0, L))
    end, (; Ks = [2, 3], L = 4)))
    # refused: an unregistered declaration type names the declarable ones
    # instead of the bare `TypeError`.
    unknown = message(quote
        G::sum_to_zero_matrix[K, K] ~ normal(0, 1)
    end, (; K = 3))
    @test occursin("`sum_to_zero_matrix` is not a Stan type SLIC can declare", unknown)
    @test occursin("`row_stochastic_matrix`, `column_stochastic_matrix`", unknown)
end

@testitem "slic: stochastic matrices fit, size and re-draw natively" tags=[:slic, :regression, :descriptor, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using StanBlocks.SpecialFunctions: loggamma
    model(body, data) = StanBlocks.stan.SlicModel(body, data, @__MODULE__)
    log_dirichlet(x, a) = loggamma(sum(a)) - sum(loggamma, a) + sum((a .- 1) .* log.(x))
    K, L = 3, 4
    a_rows, a_cols = [1.5, 2.0, 0.5, 3.0], [2.0, 1.0, 4.0]
    data = (; K, L, a_rows, a_cols, y = [1, 2, 4], w = [3, 1])

    # Densities: the full (non-propto) log density at random points equals
    # the per-row and per-column dirichlets plus the likelihood.
    fitted = model(quote
        G::row_stochastic_matrix[K, L] ~ dirichlet(a_rows)
        H::column_stochastic_matrix[K, L] ~ dirichlet(a_cols)
        y ~ categorical(G[2, :]')
        w ~ categorical(H[:, 3])
    end, data)
    p = instantiate(fitted)
    # Unconstrained sizes: K × (L - 1) for the rows, (K - 1) × L for the columns.
    @test BridgeStan.param_unc_num(p.model) == K * (L - 1) + (K - 1) * L
    # `param_names` lists one entry per element, first index fastest, matching
    # the descriptor's `dims`.
    names = BridgeStan.param_names(p.model)
    outputs = Dict(o.name => o for o in stan_descriptor(fitted).outputs)
    for name in (:G, :H)
        @test outputs[name].dims == (K, L)
        @test count(n -> startswith(n, "$name."), names) == prod(outputs[name].dims)
    end
    @test names[1:2] == ["G.1.1", "G.2.1"]
    rng = Xoshiro(20261113)
    for _ in 1:3
        theta = randn(rng, BridgeStan.param_unc_num(p.model))
        values = BridgeStan.param_constrain(p.model, theta)
        G = reshape(values[1:K*L], K, L)
        H = reshape(values[K*L+1:2K*L], K, L)
        @test all(sum(G; dims = 2) .≈ 1) && all(G .>= 0)
        @test all(sum(H; dims = 1) .≈ 1) && all(H .>= 0)
        reference = sum(log_dirichlet(G[i, :], a_rows) for i in 1:K) +
            sum(log_dirichlet(H[:, j], a_cols) for j in 1:L) +
            sum(log.(G[2, data.y])) + sum(log.(H[data.w, 3]))
        @test BridgeStan.log_density(p.model, theta; propto = false, jacobian = false) ≈ reference atol = 1e-10
    end

    # Prior-only re-draws: every row (column) is a dirichlet draw.
    prior = model(quote
        G::row_stochastic_matrix[K, L] ~ dirichlet(a_rows)
        H::column_stochastic_matrix[K, L] ~ dirichlet(a_cols)
    end, data)
    q = instantiate(prior)
    @test BridgeStan.param_unc_num(q.model) == 0
    gq_names = BridgeStan.param_names(q.model; include_tp = true, include_gq = true)
    qrng = BridgeStan.StanRNG(q.model, 20261113)
    draws = reduce(hcat, [BridgeStan.param_constrain(q.model, Float64[];
        include_tp = true, include_gq = true, rng = qrng) for _ in 1:4000])
    Gs = [reshape(d[findall(n -> startswith(n, "G."), gq_names)], K, L) for d in eachcol(draws)]
    Hs = [reshape(d[findall(n -> startswith(n, "H."), gq_names)], K, L) for d in eachcol(draws)]
    @test all(G -> all(sum(G; dims = 2) .≈ 1) && all(G .>= 0), Gs)
    @test all(H -> all(sum(H; dims = 1) .≈ 1) && all(H .>= 0), Hs)
    for i in 1:K, k in 1:L
        @test abs(mean(G[i, k] for G in Gs) - a_rows[k] / sum(a_rows)) < 0.02
    end
    for k in 1:K, j in 1:L
        @test abs(mean(H[k, j] for H in Hs) - a_cols[k] / sum(a_cols)) < 0.02
    end

    # Plate cells: one constrained matrix per cell.
    plated = model(quote
        G ~ plate(; outer = S) do s
            G::row_stochastic_matrix[K, L] ~ dirichlet(a_rows)
            G
        end
        y ~ categorical(G[2][1, :]')
    end, (; data..., S = 2))
    r = instantiate(plated)
    @test BridgeStan.param_unc_num(r.model) == 2 * K * (L - 1)
    theta = randn(rng, BridgeStan.param_unc_num(r.model))
    values = BridgeStan.param_constrain(r.model, theta)
    # `array[S] row_stochastic_matrix[K, L]`: the array index runs fastest.
    stored = reshape(values[1:2*K*L], 2, K, L)
    cells = [stored[s, :, :] for s in 1:2]
    @test BridgeStan.log_density(r.model, theta; propto = false, jacobian = false) ≈
        sum(log_dirichlet(c[i, :], a_rows) for c in cells for i in 1:K) +
        sum(log.(cells[2][1, data.y])) atol = 1e-10
end
