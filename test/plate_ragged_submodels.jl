"""
A called `@slic` sub-model may hold a plate whose cells vary in length (a
ragged plate): each call declares that plate's flat memory and its
transformed-data length/end tables at model scope under the call's flattened
names (`f1_loc`, `f1_loc__pl_len`, …) and fills them over the call's own loop
index. The program equals the one with each call's plate written out at model
scope, apart from loop-index names (snag `emit-stan-code-f-2c7ed92c`). Before,
every ragged cell inside a sub-model was refused ("ragged vector cells inside
a called @slic submodel are not supported yet").

Such a sub-model called inside a plate cell or an annotated loop is still
refused: its carriers would have to be promoted through the enclosing loop.
"""
@testitem "slic: a ragged plate inside a called sub-model" tags=[:slic, :plate, :ragged, :submodel, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    import StanBlocks.stan: stan_descriptor
    using LogDensityProblems

    readout = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    two_calls = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1 ~ readout(; y=y1, at=at1, signal=signal)
        f2 ~ readout(; y=y2, at=at2, signal=signal)
    end
    written_out = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1_shift ~ normal(0, 1)
        f1_loc ~ plate(y1, at1, signal) do ys, picks, xs
            fitted = f1_shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        f2_shift ~ normal(0, 1)
        f2_loc ~ plate(y2, at2, signal) do ys, picks, xs
            fitted = f2_shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
    end
    # One subject has no readings in the first call.
    data = (; grid=[[1.0, 2.0], [3.0, 4.0, 5.0]],
        y1=[[0.1, 0.2], Float64[]], at1=[[1, 2], Int[]],
        y2=[[0.3], [0.4, 0.5]], at2=[[2], [1, 3]])
    called = two_calls(; data...)
    reference = written_out(; data...)

    code = stan_code(called)
    @test stanc_compiles(called)
    td = stan_block(code, "transformed data")
    @test occursin("array[num_elements_RaggedVector(y1)] int f1_loc__pl_len;", td)
    @test occursin("array[num_elements_RaggedVector(y2)] int f2_loc_fitted__pl_end = cumulative_sum(f2_loc_fitted__pl_len);", td)
    @test occursin("vector[sum(f1_loc_fitted__pl_len)] f1_loc_fitted;", stan_block(code, "transformed parameters"))
    # The same statements as the plates written out, up to loop-index names
    # (the two plans' transformed-data tables may come in either order).
    lines(s) = sort!(split(replace(s, r"\b(f[12]_)?plate_i__pl_\d+\b" => "i"), '\n'))
    @test lines(code) == lines(stan_code(reference))

    outputs(m) = [(o.name, o.kind, o.segments) for o in stan_descriptor(m).outputs]
    @test outputs(called) == outputs(reference)
    @test (:f1_loc, :generated_quantity, [2, 2]) in outputs(called)
    @test (:f2_loc_fitted, :transformed_parameter, [1, 3]) in outputs(called)

    p_called = instantiate(stan_model(called))
    p_reference = instantiate(stan_model(reference))
    @test LogDensityProblems.dimension(p_called) == LogDensityProblems.dimension(p_reference) == 3
    for x in ([0.3, -0.2, 0.5], [1.1, 0.4, -0.7])
        lc, gc = LogDensityProblems.logdensity_and_gradient(p_called, x)
        lr, gr = LogDensityProblems.logdensity_and_gradient(p_reference, x)
        @test lc == lr
        @test gc == gr
    end

    # Called inside a plate cell, the carriers would need promotion through
    # the enclosing loop: refused, naming the remedy.
    in_cell = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        both ~ plate(ys, ats) do y, at
            f ~ readout(; y=y, at=at, signal=signal)
            f
        end
    end
    message = try
        stan_code(in_cell(; grid=data.grid, ys=[data.y1, data.y2], ats=[data.at1, data.at2]))
        ""
    catch err
        sprint(showerror, err)
    end
    @test_broken isempty(message)
end
