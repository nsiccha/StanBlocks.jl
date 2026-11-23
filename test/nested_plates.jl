"""
A collecting plate nested in the cell of another one-axis loop — directly, or
inside a called sub-model — stores its cells over the flattened inner index
`k = ragged_start(<ns>__pl_nend, e) + s - 1` (outer cell `e`, inner cell `s`):
one collection of `sum(<ns>__pl_n)` cells, with the per-outer-cell counts and
their cumulative ends in transformed data, declared before the outer loop. Seen
from outer cell `e`, the nested result is its slice of that storage; a ragged
nested result is group `e` of a `NestedRaggedVector`, and an outer cell that
returns it collects a two-level ragged value (snag `emit-stan-code-f-2c7ed92c`,
follow-up F2). Before, a directly nested do-form plate was refused ("collecting
a `plate(...) do` result inside another compiler-owned loop"), a sub-model's
ragged plate called in a cell was refused, and a dense one failed internally.
"""
@testitem "slic: a plate nested in a plate cell stores its cells over the flattened index" tags=[:slic, :plate, :ragged, :submodel, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    import StanBlocks.stan: stan_descriptor

    # Directly nested, a data count that varies by outer cell (one group empty).
    direct = @slic (; ys=[[0.1, 0.2], Float64[], [0.3]]) begin
        mu ~ normal(0, 1)
        out ~ plate(ys) do y
            inner ~ plate(y) do yi
                t ~ normal(mu, 1)
                yi ~ normal(t, 1)
                t
            end
            inner
        end
    end
    code = stan_code(direct)
    td = stan_block(code, "transformed data")
    @test occursin("array[num_elements_RaggedVector(ys)] int out_inner__pl_n;", td)
    @test occursin(r"out_inner__pl_n\[(plate_i__pl_\d+)\] = ragged_length\(ys\.2, \1\);", td)
    @test occursin("array[num_elements_RaggedVector(ys)] int out_inner__pl_nend = cumulative_sum(out_inner__pl_n);", td)
    @test occursin("vector[sum(out_inner__pl_n)] out_inner_t;", stan_block(code, "parameters"))
    @test occursin(r"out_inner_t\[\(\(ragged_start\(out_inner__pl_nend, (plate_i__pl_\d+)\) \+ (plate_i__pl_\d+)\) - 1\)\] ~ normal\(mu, 1\);",
        stan_block(code, "model"))
    # Each observed element's draw lands in a flat twin laid out like the data.
    @test occursin("vector[num_elements(ys.1)] ys_gen;", stan_block(code, "generated quantities"))
    @test stanc_compiles(direct)
    outputs = Dict(o.name => o for o in stan_descriptor(direct).outputs)
    @test outputs[:ys_gen].generative === :draw
    @test outputs[:ys_gen].segments == [2, 2, 3]
    # Storage over the flattened index is grouped by outer cell.
    @test outputs[:out_inner_t].segments == [2, 2, 3]
    @test outputs[:out].segments == [2, 2, 3]

    # A fixed vector cell nested in `@plate for` (the shape v0.2.37 refused).
    annotated = @slic (; J=2, S=3, y=[0.1, 0.2]) begin
        @plate for j in 1:J
            tau_s ~ plate(; outer=S) do s
                tau::vector[2] ~ normal(0.0, 1.0)
                tau
            end
            y[j] ~ normal(sum(tau_s), 1.0)
        end
    end
    code = stan_code(annotated)
    @test occursin(r"matrix\[2, sum\(loop_j_tau_s__pl_n\)\] loop_j_tau_s_tau;", stan_block(code, "parameters"))
    @test stanc_compiles(annotated)

    # Through a called sub-model, dense cells: an index-only plate and a positional one.
    counted = @slic begin
        b ~ normal(0, 1)
        z ~ plate(; outer=n) do s
            w ~ normal(b, 1)
            w
        end
        return z
    end
    @test stanc_compiles(@slic begin
        out ~ plate(; outer=3) do e
            r ~ counted(; n=2)
            r
        end
    end)
    observed = @slic begin
        b ~ normal(0, 1)
        z ~ plate(y) do yi
            w ~ normal(b, 1)
            yi ~ normal(w, 1)
            w
        end
        return z
    end
    @test stanc_compiles(@slic (; ys=[[0.1, 0.2], [0.3]]) begin
        out ~ plate(ys) do y
            r ~ observed(; y=y)
            r
        end
    end)

    # A sub-model's ragged plate called in a cell over three-level data, with
    # keywords named like the caller's data: the nested result is a two-level value.
    component = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    joint = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            fitted ~ component(; y=yg, at=atg, signal=signal)
            fitted
        end
    end
    data = (; grid=[[1.0, 2.0], [3.0, 4.0, 5.0]], y=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        at=[[[1, 2], Int[]], [[2], [1, 3]]])
    model = joint(; data...)
    code = stan_code(model)
    @test occursin("array[sum(locations_fitted_loc__pl_n)] int locations_fitted_loc__pl_end = cumulative_sum(locations_fitted_loc__pl_len);",
        stan_block(code, "transformed data"))
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[sum(locations_fitted_loc__pl_len)] locations;", gq)
    @test occursin(r"locations\[\s*nested_ragged_start\(locations_fitted_loc__pl_end, locations_fitted_loc__pl_nend, plate_i__pl_\d+\)", gq)
    # Each reading vector's predictive draw and log density.
    @test occursin("vector[num_elements(y.1)] y_gen;", gq)
    @test occursin("vector[num_elements(y.2)] y_likelihood;", gq)
    @test stanc_compiles(model)
    outputs = Dict(o.name => o for o in stan_descriptor(model).outputs)
    @test outputs[:y_gen].generative === :draw
    @test outputs[:y_gen].segments == [2, 2, 3, 5]
    @test outputs[:y_likelihood].generative === :pointwise_loglik
    @test outputs[:y_likelihood].dims == (4,)
    # The two-level result reports the ends of every innermost vector.
    @test outputs[:locations].segments == [2, 2, 3, 5]
    @test outputs[:locations].dims == (5,)

    # The two-level result reads like the data it was built from, after the loop.
    downstream = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            fitted ~ component(; y=yg, at=atg, signal=signal)
            fitted
        end
        total ~ normal(sum(locations[2][2]) + length(locations[1]), 1)
    end
    @test stanc_compiles(downstream(; data..., total=0.5))

    # Still refused, by name: a third level.
    three = @slic (; n=2) begin
        a ~ plate(; outer=n) do i
            b ~ plate(; outer=n) do j
                c ~ plate(; outer=n) do k
                    z ~ normal(0, 1)
                    z
                end
                c
            end
            b
        end
    end
    message = try
        stan_code(three)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("nested two levels deep inside other compiler-owned loops is not supported yet", message)
end

@testitem "slic: a nested plate has the density of its flat spelling" tags=[:slic, :plate, :ragged, :submodel, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems, Random
    rng = Xoshiro(20261121)
    nested = @slic begin
        mu ~ normal(0, 1)
        out ~ plate(ys) do y
            inner ~ plate(y) do yi
                t ~ normal(mu, 1)
                yi ~ normal(t, 1)
                t
            end
            inner
        end
    end
    flat = @slic begin
        mu ~ normal(0, 1)
        out ~ plate(yf) do yi
            t ~ normal(mu, 1)
            yi ~ normal(t, 1)
            t
        end
    end
    ys = [[0.1, 0.2], [0.3], Float64[], [0.4, -0.5, 0.6]]
    pn = instantiate(stan_model(nested(; ys)))
    pf = instantiate(stan_model(flat(; yf=reduce(vcat, ys))))
    @test LogDensityProblems.dimension(pn) == LogDensityProblems.dimension(pf) == 7
    for _ in 1:2
        x = randn(rng, 7)
        ln, gn = LogDensityProblems.logdensity_and_gradient(pn, x)
        lf, gf = LogDensityProblems.logdensity_and_gradient(pf, x)
        @test ln == lf
        @test gn == gf
    end

    # A component holding a ragged plate, called in each cell of a plate over
    # three-level data, has the density of one call per group at model scope.
    component = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    in_cells = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            fitted ~ component(; y=yg, at=atg, signal=signal)
            fitted
        end
    end
    per_group = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1 ~ component(; y=y[1], at=at[1], signal=signal)
        f2 ~ component(; y=y[2], at=at[2], signal=signal)
    end
    data = (; grid=[[1.0, 2.0], [3.0, 4.0, 5.0]], y=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        at=[[[1, 2], Int[]], [[2], [1, 3]]])
    pc = instantiate(stan_model(in_cells(; data...)))
    pg = instantiate(stan_model(per_group(; data...)))
    @test LogDensityProblems.dimension(pc) == LogDensityProblems.dimension(pg) == 3
    for _ in 1:2
        x = randn(rng, 3)
        lc, gc = LogDensityProblems.logdensity_and_gradient(pc, x)
        lg, gg = LogDensityProblems.logdensity_and_gradient(pg, x)
        @test lc == lg
        @test gc == gg
    end
end

"""
A two-level ragged quantity reports both of its levels in the descriptor:
`segments` ends each innermost vector of its flat storage and `groups` ends each
outer group, counted in innermost vectors, as a `NestedRaggedVector` input's
`groups` does. Before, a plate of subject plates reported only the subject
ends, so two designs that differ only in how many subjects each outer group
holds were indistinguishable (snag `read-the-endpoin-b2c91a4c`).
"""
@testitem "slic: a two-level ragged output reports its outer groups" tags=[:slic, :plate, :ragged, :submodel, :regression, :descriptor, :interface] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import StanBlocks.stan: stan_descriptor
    outputs(m) = Dict(o.name => o for o in stan_descriptor(m).outputs)

    # Directly nested: five readings in four subjects, grouped differently.
    nested = @slic begin
        slope ~ normal(0, 1)
        fitted ~ plate(xs) do group
            subject_fit ~ plate(group) do x
                slope .* x
            end
            subject_fit
        end
    end
    for (xs, groups) in (
            ([[[1.0, 2.0], Float64[]], [[3.0], [4.0, 5.0]]], [2, 4]),
            ([[[1.0, 2.0], Float64[], [3.0]], [[4.0, 5.0]]], [3, 4]),
            ([[Float64[]], [[1.0, 2.0], [3.0], [4.0, 5.0]]], [1, 4]),
            ([Vector{Float64}[], [[1.0, 2.0], Float64[], [3.0], [4.0, 5.0]]], [0, 4]))
        o = outputs(nested(; xs))
        @test o[:fitted].segments == o[:fitted_subject_fit].segments == cumsum(length.(reduce(vcat, xs)))
        @test o[:fitted].groups == o[:fitted_subject_fit].groups == groups
        @test o[:fitted].dims == (5,)
    end

    # A one-level ragged result reports no outer groups.
    single = @slic (; ys=[[0.1, 0.2], Float64[], [0.3]]) begin
        mu ~ normal(0, 1)
        out ~ plate(ys) do y
            inner ~ plate(y) do yi
                t ~ normal(mu, 1)
                yi ~ normal(t, 1)
                t
            end
            inner
        end
    end
    o = outputs(single)
    @test o[:out].segments == o[:out_inner_t].segments == [2, 2, 3]
    @test o[:out].groups === o[:out_inner_t].groups === o[:ys_gen].groups === nothing

    # Through a called component: the subject plate's stored result, the outer
    # result and the twins of the three-level observation, on the fit and rebound.
    component = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    joint = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            fitted ~ component(; y=yg, at=atg, signal=signal)
            fitted
        end
    end
    a = (; grid=[[1.0, 2.0], [3.0, 4.0, 5.0]], y=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        at=[[[1, 2], Int[]], [[2], [1, 3]]])
    b = (; y=[[[0.1, 0.2], Float64[], [0.3]], [[0.4, 0.5]]], at=[[[1, 2], Int[], [1]], [[1, 3]]])
    fit = outputs(joint(; a...))
    rebound = outputs(stan_model(joint(; a...))(; b...))
    for (o, groups) in ((fit, [2, 4]), (rebound, [3, 4]))
        # `loc` and its cell binding `fitted` are the component's stored values.
        for name in (:locations, :locations_fitted_loc, :locations_fitted_loc_fitted, :y_gen, :y_likelihood)
            @test o[name].segments == [2, 2, 3, 5]
            @test o[name].groups == groups
        end
        @test o[:signal].groups === nothing
    end
    iface = Dict(e["name"] => e for e in stan_interface(joint(; a...))["outputs"])
    @test iface["locations"]["groups"] == [2, 4]
    @test iface["signal"]["groups"] === nothing
end

"""
A sub-model whose plate is called in each cell of a plate may return several
of that plate's values, a cell binding beside the result (`return (; picked =
loc_picked, fitted = loc)`). The calling cell collects each field with an `=`
binding of its own (`picked = c.picked`), named after it (`locations_picked`)
and laid out two-level like the collected result (snag
`return-two-per-r-eae6be9c`). Before, the sub-model's cell binding still named
the nested loop's last cell after the loop, whose size reads the loop's index,
and the plate failed with "inferred cell size references local
`…plate_i__pl_…`, which cannot be expressed outside this cell".
"""
@testitem "slic: a cell collects every field a nested plate's sub-model returns" tags=[:slic, :plate, :ragged, :submodel, :regression, :descriptor, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    import StanBlocks.stan: stan_descriptor
    using LogDensityProblems, Random
    outputs(m) = Dict(o.name => o for o in stan_descriptor(m).outputs)

    two_fields = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            picked = xs[picks]
            fitted = shift .+ picked
            ys ~ normal(fitted, 1)
            fitted
        end
        return (; picked=loc_picked, fitted=loc)
    end
    result_only = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            picked = xs[picks]
            fitted = shift .+ picked
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    both = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            c ~ two_fields(; y=yg, at=atg, signal=signal)
            picked = c.picked
            c.fitted
        end
    end
    control = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            fitted ~ result_only(; y=yg, at=atg, signal=signal)
            fitted
        end
    end
    # A cell ending in a `begin … end` block ends in that block's lines.
    spliced = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            c ~ two_fields(; y=yg, at=atg, signal=signal)
            begin
                picked = c.picked
                c.fitted
            end
        end
    end
    # Five readings in four subjects, one of them empty; then a rebinding.
    a = (; grid=[[1.0, 2.0], [3.0, 4.0, 5.0]], y=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        at=[[[1, 2], Int[]], [[2], [1, 3]]])
    b = (; y=[[[0.1, 0.2], Float64[], [0.3]], [[0.4, 0.5]]], at=[[[1, 2], Int[], [1]], [[1, 3]]])

    code = stan_code(both(; a...))
    @test code == stan_code(spliced(; a...))
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[sum(locations_c_loc_picked__pl_len)] locations_picked;", gq)
    # Each field's memory, read from the cell's tuple of the two returned values.
    @test occursin(r"locations_picked\[\s*nested_ragged_start\(locations_c_loc_picked__pl_end, locations_c_loc__pl_nend, plate_i__pl_\d+\)", gq)
    @test occursin("] = locations_c.1.1;", gq)
    @test occursin("] = locations_c.2.1;", gq)
    @test stanc_compiles(both(; a...))
    fit = outputs(both(; a...))
    rebound = outputs(stan_model(both(; a...))(; b...))
    for (o, groups) in ((fit, [2, 4]), (rebound, [3, 4]))
        for name in (:locations, :locations_picked, :locations_c_loc_picked, :locations_c_loc)
            @test o[name].segments == [2, 2, 3, 5]
            @test o[name].groups == groups
            @test o[name].dims == (5,)
        end
    end

    # Read in the cell, and after the loop as a `NestedRaggedVector`.
    downstream = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        locations ~ plate(y, at) do yg, atg
            c ~ two_fields(; y=yg, at=atg, signal=signal)
            picked = c.picked
            first_picked = sum(picked[1])
            c.fitted
        end
        total ~ normal(sum(locations_picked[2][2]) + sum(locations[1][1]), 1)
    end
    @test stanc_compiles(downstream(; a..., total=0.5))

    # Natively: the density of the result-only program, and each field's values.
    rng = Xoshiro(20261123)
    pb = instantiate(stan_model(both(; a...)))
    pc = instantiate(stan_model(control(; a...)))
    @test LogDensityProblems.dimension(pb) == LogDensityProblems.dimension(pc) == 3
    names = BridgeStan.param_names(pb.model; include_tp=true, include_gq=true)
    for _ in 1:2
        x = randn(rng, 3)
        lb, gb = LogDensityProblems.logdensity_and_gradient(pb, x)
        lc, gc = LogDensityProblems.logdensity_and_gradient(pc, x)
        @test lb == lc
        @test gb == gc
        draw = Dict(zip(names, BridgeStan.param_constrain(pb.model, x; include_tp=true,
            include_gq=true, rng=BridgeStan.StanRNG(pb.model, 1))))
        gain, shift = x[1], x[2:3]
        picked = gain .* [1.0, 2.0, 2.0, 3.0, 5.0]
        @test [draw["locations_picked.$i"] for i in 1:5] ≈ picked
        @test [draw["locations.$i"] for i in 1:5] ≈ picked .+ shift[[1, 1, 2, 2, 2]]
    end
end
