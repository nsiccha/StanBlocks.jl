"""
A RaggedVector is a vector of groups, as Julia's vector of vectors: `rv[g]` is
group `g` and `length(rv)` counts groups. Selecting several groups with an index
array failed to trace, for ragged data (`MethodError: no method matching
getindex(::NamedTuple, ::Vector{Int64})`) and for a plate result (`tracetype not
defined for ragged_start(::array[] int, ::array[] int)`). `sum(rv)` transpiled to
`sum_RaggedVector(rv)`, a function nothing defines, so stanc rejected it, and
`mean(rv)` / `sd(rv)` did the same (snag `selecting-severa-027a8191`). Now
`rv[idx]` is the groups `idx`, `sum(rv) == rv[1] + rv[2] + …` and
`mean(rv) == sum(rv) / length(rv)`, and any other call with no Stan definition for
a whole RaggedVector is refused when the program is built.
"""
@testitem "slic: a RaggedVector selects groups by index and sums them as Julia does" tags=[:slic, :ragged, :plate, :descriptor, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    model(body, data) = StanBlocks.stan.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    normlp(y, mu) = sum(@. -0.5 * (y - mu)^2 - 0.5 * log(2pi))
    lp(m, θ) = BridgeStan.log_density(instantiate(m).model, θ; propto=false)

    ys = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    data = (; ys, idx=[1, 3], obs=[0.3, -0.2], y1=0.4, grp=[1, 2, 1],
        ks=[[1, 2], [3, 4], [5, 6]],
        X=[[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0; 9.0 1.0], [2.0;; 3.0]])
    plate_rv = :(rv ~ plate(ys; outer = (3,)) do y; m ~ normal(0, 1); y .* m end)
    s, m3 = 0.6, [0.7, -0.4, 1.3]
    cells = [m3[g] * ys[g] for g in 1:3]   # the plate result, as Julia's loop leaves it

    # The reported shapes. (a) ragged data, (b) a plate result, (c) `sum` of a plate result.
    a = model(:(begin s ~ normal(0, 1); z = ys[idx]; obs ~ normal(s * length(z) .+ sum(z), 1.0) end), data)
    b = model(:(begin $plate_rv; z = rv[idx]; obs ~ normal(sum(z), 1.0) end), data)
    c = model(:(begin $plate_rv; z = sum(rv); obs ~ normal(z, 1.0) end), data)
    c_twin = model(:(begin $plate_rv; z = rv[1] + rv[2] + rv[3]; obs ~ normal(z, 1.0) end), data)
    @test stanc_compiles(a) && stanc_compiles(b) && stanc_compiles(c)
    # The selection's offsets are data: computed once, in transformed data.
    let td = stan_block(stan_code(b), "transformed data")
        @test occursin("ragged_select_ends(rv__pl_end, idx)", td)
        @test occursin("ragged_select_index(rv__pl_end, idx)", td)
    end
    @test occursin("vector[ragged_length(rv__pl_end, 1)] z = ragged_sum(rv, rv__pl_end);", stan_code(c))
    @test lp(a, [s]) ≈ normlp([s], [0.0]) + normlp(data.obs, 2s .+ sum(ys[data.idx]))
    @test lp(b, m3) ≈ normlp(m3, zeros(3)) + normlp(data.obs, sum(cells[data.idx]))
    @test lp(c, m3) ≈ normlp(m3, zeros(3)) + normlp(data.obs, sum(cells))
    @test lp(c, m3) ≈ lp(c_twin, m3)

    # mean, a range, a comparison mask, an arithmetic result, and the flat total.
    for (body, μ) in (
        :(obs ~ normal(mean(rv), 1.0)) => sum(cells) / 3,
        :(obs ~ normal(sum(rv[2:3]), 1.0)) => sum(cells[2:3]),
        :(obs ~ normal(sum(rv[grp .== 1]), 1.0)) => sum(cells[[1, 3]]),
        :(obs ~ normal(sum((2.0 * rv)[idx]), 1.0)) => 2 * sum(cells[data.idx]),
        :(y1 ~ normal(sum(rv.mem), 1.0)) => [sum(sum, cells)],
    )
        m = model(:(begin $plate_rv; $body end), data)
        y = length(μ) == 1 ? [data.y1] : data.obs
        @test lp(m, m3) ≈ normlp(m3, zeros(3)) + normlp(y, μ)
        p = instantiate(m)
        @test all(isfinite, last(LogDensityProblems.logdensity_and_gradient(p, m3)))
    end

    # Integer groups sum to integers; a vector of matrices selects matrices.
    ints = model(:(begin s ~ normal(0, 1); obs ~ normal(s * to_vector(sum(ks[idx])), 1.0) end), data)
    @test occursin("ragged_sum(ks.1[", stan_code(ints))
    @test lp(ints, [s]) ≈ normlp([s], [0.0]) + normlp(data.obs, s * [6.0, 8.0])
    mats = model(:(begin s ~ normal(0, 1); Z = X[idx]; y1 ~ normal(s * (sum(Z[1]) + sum(Z[2])) + length(Z), 1.0) end), data)
    @test lp(mats, [s]) ≈ normlp([s], [0.0]) + normlp([data.y1], [s * (sum(data.X[1]) + sum(data.X[3])) + 2])

    # A RaggedVector `@deffun` argument selects and sums the same way.
    @deffun sum_selected(r::RaggedVector, i::int[k])::vector[2] = sum(r[i])
    f = model(:(begin s ~ normal(0, 1); obs ~ normal(s * sum_selected(ys, idx), 1.0) end), data)
    @test lp(f, [s]) ≈ normlp([s], [0.0]) + normlp(data.obs, s * sum(ys[data.idx]))

    # The descriptor evaluates the new sizes in Julia.
    d = stan_descriptor(model(:(begin $plate_rv; z = sum(rv[idx]); w = sum(rv); obs ~ normal(z + w, 1.0) end), data))
    @test Dict(o.name => o.dims for o in d.outputs if o.name in (:z, :w)) == Dict(:z => (2,), :w => (2,))

    # Groups of different lengths have no elementwise sum (Julia throws too); an index
    # outside 1:length(rv) selects nothing. Stan rejects both with their own message.
    uneq = model(:(begin s ~ normal(0, 1); obs ~ normal(s * sum(ys), 1.0) end), (; ys=[[1.0, 2.0], [3.0]], obs=[0.3, -0.2]))
    @test_throws "every group must have the same length" lp(uneq, [s])
    oor = model(:(begin s ~ normal(0, 1); obs ~ normal(s * sum(ys[idx]), 1.0) end), merge(data, (; idx=[1, 4])))
    @test_throws "selects groups 1 to length(rv)" lp(oor, [s])

    # refused: a function with no Stan definition for a whole RaggedVector would call an
    # undeclared `<f>_RaggedVector` that stanc rejects (snag selecting-severa-027a8191).
    let e = errmsg(model(:(begin s ~ normal(0, 1); obs ~ normal(s * sd(ys), 1.0) end), data))
        @test occursin("sd_RaggedVector", e)
        @test occursin("f(x.mem)", e) && occursin("f(x[g])", e)
    end
end
