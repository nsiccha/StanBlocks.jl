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
