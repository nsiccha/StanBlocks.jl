"""
Verify `slic: a plate cell collects a called submodel's tuple return`.

Snag plate-cell-submo-4867bc5b (from a downstream regression package): a called
`@slic` submodel that returns a named tuple could be read by field at model
scope (`k ~ sub(…); k.lo`), but the same call inside a plate cell was collected
as a `vector[N]` of reals, so `cell.lo` failed with "Trying to access property
`lo` of object of type without named properties (real)". A tuple cell is now
collected as `array[outer…] tuple(…)` and read back as a whole element, so its
fields lower to `cell[i].1`.
"""
@testitem "slic: a plate cell collects a called submodel's tuple return" tags=[:slic, :plate, :regression, :stanc, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    msg(e) = sprint(showerror, e)

    named = @slic begin
        a ~ normal(0, s)
        b ~ normal(1, 0.2)
        return (; lo = a + b * x, hi = a - b * x)
    end
    positional = @slic begin
        a ~ normal(0, s)
        b ~ normal(1, 0.2)
        return (a + b * x, a - b * x)
    end
    data = (; x = [-0.4, 0.2, 0.7], y1 = [0.1, 0.4, 0.8], y2 = [0.3, -0.1, 0.2])

    # The reported shape: `@plate for` with the submodel bound per cell.
    m = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(x)
            cell ~ named(; x = x[i], s)
            y1[i] ~ normal(cell.lo, 0.5)
            y2[i] ~ normal(cell.lo + cell.hi, 0.5)
        end
    end
    code = stan_code(m)
    @test occursin("array[num_elements(x)] tuple(real, real) loop_i_cell;", stan_block(code, "transformed parameters"))
    @test occursin("loop_i_cell[i] = ((loop_i_cell_a[i] + (loop_i_cell_b[i] * x[i])), (loop_i_cell_a[i] - (loop_i_cell_b[i] * x[i])));", code)
    @test occursin("y1[i] ~ normal(loop_i_cell[i].1, 0.5);", stan_block(code, "model"))
    @test occursin("y2[i] ~ normal((loop_i_cell[i].1 + loop_i_cell[i].2), 0.5);", stan_block(code, "model"))
    @test stanc_compiles(m)

    # The descriptor reports the collection with its per-cell components.
    out = only(o for o in stan_descriptor(m).outputs if o.name === :loop_i_cell)
    @test out.type == :ntup && out.dims == (3,)
    @test out.fields == (
        lo = (type = :real, size = (), dims = (), fields = (;)),
        hi = (type = :real, size = (), dims = (), fields = (;)),
    )

    # A positional tuple return reads by position, as at model scope.
    p = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(x)
            cell ~ positional(; x = x[i], s)
            y1[i] ~ normal(cell[1], 0.5)
            y2[i] ~ normal(cell[1] + cell[2], 0.5)
        end
    end
    @test occursin("y2[i] ~ normal((loop_i_cell[i].1 + loop_i_cell[i].2), 0.5);", stan_code(p))
    @test stanc_compiles(p)

    # A do-form plate whose cell RESULT is the tuple collects it the same way;
    # later code reads one element's field.
    r = @slic data begin
        s ~ exponential(1)
        ks ~ plate(x) do xi
            c ~ named(; x = xi, s)
            c
        end
        @plate for i in eachindex(x)
            y1[i] ~ normal(ks[i].lo, 0.5)
        end
    end
    @test occursin("array[num_elements(x)] tuple(real, real) ks;", stan_code(r))
    @test occursin("y1[i] ~ normal(ks[i].1, 0.5);", stan_code(r))
    @test stanc_compiles(r)

    # A record with an integer field cannot be a transformed parameter; the
    # collection follows the mixed-record placement (model local, recomputed in
    # generated quantities).
    mixed = @slic begin
        a ~ normal(0, s)
        return (; lo = a + x, n = 2)
    end
    mr = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(x)
            cell ~ mixed(; x = x[i], s)
            y1[i] ~ normal(cell.lo * cell.n, 0.5)
        end
    end
    mcode = stan_code(mr)
    @test !occursin("loop_i_cell;", stan_block(mcode, "transformed parameters"))
    for b in ("model", "generated quantities")
        @test occursin("array[num_elements(x)] tuple(real, int) loop_i_cell;", stan_block(mcode, b))
        @test occursin("loop_i_cell[i] = ((loop_i_cell_a[i] + x[i]), 2);", stan_block(mcode, b))
    end
    @test stanc_compiles(mr)
    # Stan has no field projection over an array of tuples.
    err = try
        stan_code(@slic data begin
            s ~ exponential(1)
            ks ~ plate(x) do xi
                c ~ named(; x = xi, s)
                c
            end
            y1 ~ normal(ks.lo, 0.5)
        end)
        nothing
    catch e
        e
    end
    @test err !== nothing && occursin("array of tuples", msg(err)) && occursin("ks[i].lo", msg(err))

    # Two plate axes lead the tuple array; fixed-size vector components keep
    # their size.
    grid = @slic (; X = [0.1 0.2; 0.3 0.4; 0.5 0.6], Y = [1.0 2.0; 3.0 4.0; 5.0 6.0]) begin
        s ~ exponential(1)
        @plate for i in 1:3, j in 1:2
            cell ~ named(; x = X[i, j], s)
            Y[i, j] ~ normal(cell.lo + cell.hi, 0.5)
        end
    end
    @test occursin("array[3, 2] tuple(real, real) loop_i_j_cell;", stan_code(grid))
    @test stanc_compiles(grid)
    rows = @slic (; X = [0.1 0.2; 0.3 0.4; 0.5 0.6], y = [1.0, 2.0, 3.0]) begin
        s ~ exponential(1)
        @plate for i in 1:3
            cell ~ named(; x = X[i, :], s)
            y[i] ~ normal(sum(cell.lo) + sum(cell.hi), 0.5)
        end
    end
    @test stanc_compiles(rows)
end

"""
Verify `slic: a plate cell reads a tuple whose component varies by cell`.

Snag a-plate-cell-bin-0b519903 (from a downstream regression package): a called
`@slic` submodel whose tuple return holds a vector whose length depends on the
cell (`return (; curve = gain * x, scale)` over a per-cell `x`) was refused in
both plate forms, because a Stan `array[…] tuple(…)` has one size per
component. Such a tuple is now a Stan local inside each block's loop that reads
it (decision `1u3kvwv`): nothing new is stored, its fields read `cell.1`, and a
do-form side output built from a field is still stored. A plate RESULT such a
tuple has no collected form and stays refused.
"""
@testitem "slic: a plate cell reads a tuple whose component varies by cell" tags=[:slic, :plate, :regression, :stanc, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    msg(e) = sprint(showerror, e)

    named = @slic begin
        a ~ normal(0, s)
        b ~ normal(1, 0.2)
        return (; lo = a + b * x, hi = a - b * x)
    end
    member = @slic begin
        gain ~ normal(0, 1)
        scale ~ lognormal(-0.7, 1)
        return (; curve = gain * x, scale = scale)
    end
    data = (; xs = [[0.1, 0.2], [0.3, -0.4, 0.5]], y = [1.0, 2.0])

    # `@plate for`: both components vary; each block's loop declares the tuple.
    m = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(y)
            cell ~ named(; x = xs[i], s)
            y[i] ~ normal(sum(cell.lo) - sum(cell.hi), 0.5)
        end
    end
    code = stan_code(m)
    @test !occursin("loop_i_cell;", code)
    for (b, read) in (("model", "y[i] ~ normal((sum(loop_i_cell.1) - sum(loop_i_cell.2)), 0.5);"),
                      ("generated quantities", "y_gen[i] = normal_rng((sum(loop_i_cell.1) - sum(loop_i_cell.2)), 0.5);"))
        @test occursin(") loop_i_cell = (", stan_block(code, b))
        @test occursin(read, stan_block(code, b))
    end
    @test stanc_compiles(m)
    @test !any(o -> o.name === :loop_i_cell, stan_descriptor(m).outputs)

    # The do-form shape a downstream emitter writes: a field becomes a side
    # output (stored as flat memory), another the plate result.
    d = @slic data begin
        scales ~ plate(xs) do x
            cell ~ member(; x)
            curve = cell.curve
            cell.scale
        end
        y ~ normal(scales, 1)
    end
    dcode = stan_code(d)
    @test occursin("scales[plate_i__pl_1] = scales_cell.2;", stan_block(dcode, "transformed parameters"))
    @test occursin("vector[sum(scales_curve__pl_len)] scales_curve;", dcode)
    @test occursin("] = scales_cell.1;", dcode)
    @test stanc_compiles(d)
    names = Set(o.name for o in stan_descriptor(d).outputs)
    @test :scales_curve in names && !(:scales_cell in names)

    # Equal lengths read from per-cell data still depend on the cell.
    @test stanc_compiles(@slic (; xs = [[0.1, 0.2], [0.3, -0.4]], y = [1.0, 2.0]) begin
        @plate for i in eachindex(y)
            cell ~ member(; x = xs[i])
            y[i] ~ normal(sum(cell.curve), cell.scale)
        end
    end)

    # A plate whose RESULT is such a tuple has no collected form.
    err = try
        stan_code(@slic data begin
            ks ~ plate(xs) do x
                c ~ member(; x)
                c
            end
            y ~ normal(1, 1)
        end)
        nothing
    catch e
        e
    end
    @test err !== nothing && occursin("collects a tuple across cells", msg(err)) &&
        occursin("curve = c.curve", msg(err))
end

"""
Verify `slic: a plate tuple cell has the density of the inlined submodel`.

The collected tuple is a deterministic transformed parameter: the program with
the submodel called per cell and read by field has the log density of the same
cell with the submodel body written inline, at every point (snag
plate-cell-submo-4867bc5b).
"""
@testitem "slic: a plate tuple cell has the density of the inlined submodel" tags=[:slic, :plate, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    named = @slic begin
        a ~ normal(0, s)
        b ~ normal(1, 0.2)
        return (; lo = a + b * x, hi = a - b * x)
    end
    data = (; x = [-0.4, 0.2, 0.7], y1 = [0.1, 0.4, 0.8], y2 = [0.3, -0.1, 0.2])
    called = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(x)
            cell ~ named(; x = x[i], s)
            y1[i] ~ normal(cell.lo, 0.5)
            y2[i] ~ normal(cell.lo + cell.hi, 0.5)
        end
    end
    inlined = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(x)
            cell_a ~ normal(0, s)
            cell_b ~ normal(1, 0.2)
            y1[i] ~ normal(cell_a + cell_b * x[i], 0.5)
            y2[i] ~ normal((cell_a + cell_b * x[i]) + (cell_a - cell_b * x[i]), 0.5)
        end
    end
    bs(m) = BridgeStan.StanModel(stan_library(stan_code(m)),
        StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(m)))
    mc, mi = bs(called), bs(inlined)
    @test BridgeStan.param_unc_num(mc) == BridgeStan.param_unc_num(mi) == 7
    for theta in ([0.1, 0.2, -0.3, 0.4, 0.5, 0.9, 1.1], [-1.0, 0.0, 0.3, -0.2, 1.4, 0.7, 0.2])
        @test BridgeStan.log_density(mc, theta) ≈ BridgeStan.log_density(mi, theta)
        # The transformed parameters hold each cell's (lo, hi).
        c = BridgeStan.param_constrain(mc, theta; include_tp=true)
        a, b = theta[2:4], theta[5:7]
        @test c[8:end] ≈ vec(permutedims(hcat(a .+ b .* data.x, a .- b .* data.x)))
    end
end

"""
Verify `slic: a cell-varying plate tuple has the density of the inlined submodel`.

A tuple whose component varies by cell is a loop local, recomputed in each
block that reads it: the called program has the log density and gradient of
the same cell with the submodel body written inline, and a do-form side output
built from a field holds that field's values (snag a-plate-cell-bin-0b519903).
"""
@testitem "slic: a cell-varying plate tuple has the density of the inlined submodel" tags=[:slic, :plate, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    named = @slic begin
        a ~ normal(0, s)
        b ~ normal(1, 0.2)
        return (; lo = a + b * x, hi = a - b * x)
    end
    data = (; xs = [[0.1, 0.2], [0.3, -0.4, 0.5]], y = [1.0, 2.0])
    called = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(y)
            cell ~ named(; x = xs[i], s)
            y[i] ~ normal(sum(cell.lo) - 2 * sum(cell.hi), 0.5)
        end
    end
    inlined = @slic data begin
        s ~ exponential(1)
        @plate for i in eachindex(y)
            cell_a ~ normal(0, s)
            cell_b ~ normal(1, 0.2)
            y[i] ~ normal(sum(cell_a .+ cell_b .* xs[i]) - 2 * sum(cell_a .- cell_b .* xs[i]), 0.5)
        end
    end
    bs(m) = BridgeStan.StanModel(stan_library(stan_code(m)),
        StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(m)))
    mc, mi = bs(called), bs(inlined)
    @test BridgeStan.param_unc_num(mc) == BridgeStan.param_unc_num(mi) == 5
    for theta in ([0.1, 0.2, -0.3, 0.4, 0.5], [-1.0, 0.7, 0.3, -0.2, 1.4])
        lc, gc = BridgeStan.log_density_gradient(mc, theta)
        li, gi = BridgeStan.log_density_gradient(mi, theta)
        @test lc ≈ li
        @test gc ≈ gi
    end

    # The do-form side output `curve` holds each cell's `gain * x`.
    member = @slic begin
        gain ~ normal(0, 1)
        scale ~ lognormal(-0.7, 1)
        return (; curve = gain * x, scale = scale)
    end
    d = bs(@slic data begin
        scales ~ plate(xs) do x
            cell ~ member(; x)
            curve = cell.curve
            cell.scale
        end
        y ~ normal(scales, 1)
    end)
    theta = [0.3, -1.2, 0.1, 0.4]
    names = BridgeStan.param_names(d; include_tp=true, include_gq=true)
    vals = BridgeStan.param_constrain(d, theta; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(d, 1))
    curve = vals[startswith.(names, "scales_curve.")]
    @test curve ≈ vcat(theta[1] .* data.xs[1], theta[2] .* data.xs[2])
    @test vals[startswith.(names, "scales.")] ≈ exp.(theta[3:4])
end
