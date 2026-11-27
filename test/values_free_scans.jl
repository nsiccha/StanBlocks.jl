"""
A values-free `maybecv(:y)` forecast of a series observed by a `@scan` (snag
values-free-fore-ae8fef99): the forecast draws the whole trajectory in generated
quantities. The first step is drawn from the stationary start, the lags read
the draws, and the fit's parameters stay parameters. The draw is named after
the fit program's twin (`y_gen`, or a nested-vector `ys_gen` with the data's
segments). The scan may observe the series at model scope, in a called
submodel through a keyword (`node ~ f(; obs = y)`), inline in a plate cell
over its do-block parameter, or in a plate cell passing that parameter to a
called submodel, which is the shape a downstream module emits for a grouped
AR(1) response.
"""
@testitem "slic: a values-free forecast of an observed @scan draws whole trajectories" tags=[:slic, :scan, :plate, :ragged, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    maybecv = StanBlocks.stan.maybecv
    ops(d) = Tuple(o.name for o in d.operations)
    output(d, name) = only(o for o in d.outputs if o.name == name)
    msg(f) = try; f(); nothing; catch e; StanBlocks.diagnostic(e).message; end

    stream = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                obs[t] ~ normal(mu[t] + phi * (obs[t-1] - mu[t-1]), sigma)
            end
        end
    end
    mu = randn(5)
    scope = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
            end
        end
    end, Dict{Symbol,Any}(:y => randn(5), :mu => mu, :T => 5), @__MODULE__)
    called = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        node ~ stream(; obs = y, mu, n = T, phi, sigma)
    end, Dict{Symbol,Any}(:y => randn(5), :mu => mu, :T => 5), @__MODULE__)
    for (fit, draw) in ((scope, "y"), (called, "node_obs"))
        forecast = fit(; y = maybecv(:y))
        code = stan_code(forecast)
        @test occursin("real<lower=-1, upper=1> phi;", stan_block(code, "parameters"))
        @test occursin("real<lower=0.0> sigma;", stan_block(code, "parameters"))
        @test !occursin("y", stan_block(code, "data"))
        @test !occursin("normal", stan_block(code, "model"))
        gq = stan_block(code, "generated quantities")
        @test occursin("$draw[1] = normal_rng(mu[1], (sigma / sqrt((1 - (phi ^ 2)))));", gq)
        # Each lag reads the trajectory drawn so far.
        @test occursin(Regex("\\(phi \\* \\(" * draw * "\\[\\(\\w+ - 1\\)\\] - mu\\[\\(\\w+ - 1\\)\\]\\)\\)"), gq)
        @test occursin("vector[T] y_gen = $draw;", gq)
        @test !occursin("y_likelihood", code)
        d = stan_descriptor(forecast)
        @test ops(d) == (:transpile, :instantiate, :predict)
        @test !(:y in StanBlocks.required_inputs(d))
        @test (output(d, :y_gen).generative, output(d, :y_gen).source) == (:draw, :y)
        @test stanc_compiles(forecast)
    end

    # In a plate cell, the cell's do-block parameter bound to the mark: an inline
    # scan over it, or a called submodel's keyword. The count and the lengths
    # come from the other positional.
    ys = [randn(3), randn(4), randn(2)]; mus = [randn(3), randn(4), randn(2)]
    inline = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        cells ~ plate(ys, mus) do y, m
            @scan begin
                y[1] ~ normal(m[1], sigma / sqrt(1 - phi^2))
                for t in 2:length(m)
                    y[t] ~ normal(m[t] + phi * (y[t-1] - m[t-1]), sigma)
                end
            end
        end
    end, Dict{Symbol,Any}(:ys => ys, :mus => mus), @__MODULE__)
    cell_called = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        cells ~ plate(ys, mus) do y, m
            node ~ stream(; obs = y, mu = m, n = length(m), phi, sigma)
        end
    end, Dict{Symbol,Any}(:ys => ys, :mus => mus), @__MODULE__)
    for (fit, draw) in ((inline, "y"), (cell_called, "cells_node_obs"))
        @test output(stan_descriptor(fit), :ys_gen).segments == [3, 7, 9]
        forecast = fit(; ys = maybecv(:ys))
        code = stan_code(forecast)
        @test !occursin("ys", stan_block(code, "data"))
        @test !occursin("for(", stan_block(code, "model"))
        gq = stan_block(code, "generated quantities")
        @test occursin("$draw[ragged_start($(draw)__pl_end, plate_i__pl_1)] = normal_rng(mus.1[ragged_start(mus.2, plate_i__pl_1)]", gq)
        # Each lag reads the cell's trajectory drawn so far.
        @test occursin("$draw[((ragged_start($(draw)__pl_end, plate_i__pl_1) + (", gq)
        @test occursin("vector[sum($(draw)__pl_len)] ys_gen = $draw;", gq)
        @test !occursin("ys_likelihood", code)
        d = stan_descriptor(forecast)
        @test ops(d) == (:transpile, :instantiate, :predict)
        @test StanBlocks.required_inputs(d) == (:mus,)
        @test (output(d, :ys_gen).generative, output(d, :ys_gen).source, output(d, :ys_gen).segments) ==
            (:draw, :ys, [3, 7, 9])
        @test output(d, :phi).kind == :parameter
        @test stanc_compiles(forecast)
    end
    # Bound with its values, the cell's series is observed as before.
    @test occursin("ys.1[ragged_start(ys.2, plate_i__pl_1)] ~ normal(", stan_block(stan_code(cell_called), "model"))

    # Refused: a length read off the values-free series, in the cell or in the
    # submodel, has no values to read; a cell that does not observe its
    # parameter, or observes a slice of it outside a scan, has nothing to draw.
    reader = @slic begin
        n = length(obs)
        @scan begin
            obs[1] ~ normal(mu[1], sigma)
            for t in 2:n
                obs[t] ~ normal(mu[t] + 0.5 * obs[t - 1], sigma)
            end
        end
    end
    in_cell = msg(() -> stan_code(StanBlocks.SlicModel(quote
        sigma ~ exponential(1)
        cells ~ plate(ys, mus) do y, m
            node ~ stream(; obs = y, mu = m, n = length(y), phi = 0.5, sigma)
        end
    end, Dict{Symbol,Any}(:ys => maybecv(:ys), :mus => mus), @__MODULE__)))
    @test occursin("plate `cells`: do-block parameter `y` is the cell's slice of `ys`, bound to " *
        "`maybecv(:ys)`, which has no values to read", in_cell)
    in_submodel = msg(() -> stan_code(StanBlocks.SlicModel(quote
        sigma ~ exponential(1)
        cells ~ plate(ys, mus) do y, m
            node ~ reader(; obs = y, mu = m, sigma)
        end
    end, Dict{Symbol,Any}(:ys => maybecv(:ys), :mus => mus), @__MODULE__)))
    @test occursin("`cells_node ~ …`: keyword `obs` is bound to the values-free `maybecv(:ys)`, " *
        "which has no values to read", in_submodel)
    for cell in (:(t), :(y[1] ~ normal(0.0, 1.0)))
        unobserved = msg(() -> stan_code(StanBlocks.SlicModel(quote
            mu ~ plate(y, t) do y, t
                $cell
                t
            end
        end, Dict{Symbol,Any}(:t => [[0.1, 0.2], [0.3]], :y => maybecv(:y)), @__MODULE__)))
        @test occursin("so the cell must observe its do-block parameter `y`: whole (`y ~ …`), element " *
            "by element in an observed `@scan`", unobserved)
    end
end

"""
The values-free forecast's trajectory is the fit program's whole-trajectory
twin: given the same parameter values and seed, its `ys_gen` equals the fit
program's `ys_gen` (stanblocks-use §35), here through a timed AR(1) submodel
with integer time gaps, a regression mean and a per-group persistence.
"""
@testitem "slic: a values-free forecast of an observed @scan equals the fit's trajectory twin" tags=[:slic, :scan, :plate, :ragged, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    maybecv = StanBlocks.stan.maybecv
    timed = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                k = time[t] - time[t - 1]
                obs[t] ~ normal(mu[t] + phi ^ k * (obs[t - 1] - mu[t - 1]),
                    sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
            end
        end
    end
    xs = [[0.3, -1.2, 0.8], [1.1, 0.2, -0.4, 0.9], [-0.7, 0.5]]
    ts = [[1, 2, 4], [1, 3, 4, 7], [2, 5]]
    ys = [[0.1, -0.5, 0.4], [1.0, 0.3, -0.2, 0.6], [-0.9, 0.2]]
    fit = StanBlocks.SlicModel(quote
        b ~ normal(0, 1); sigma ~ exponential(1)
        z ~ normal(0, 0.5; n = 2)
        cells ~ plate(ys, xs, ts, g) do y, x, t, gi
            m = b .* x
            node ~ timed(; obs = y, mu = m, sigma, phi = tanh(z[gi]), n = length(t), time = t)
        end
    end, Dict{Symbol,Any}(:ys => ys, :xs => xs, :ts => ts, :g => [1, 2, 1]), @__MODULE__)
    forecast = fit(; ys = maybecv(:ys))
    draws(bm, th, seed) = begin
        values = BridgeStan.param_constrain(bm, th; include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(bm, seed))
        names = BridgeStan.param_names(bm; include_tp=true, include_gq=true)
        values[findall(n -> startswith(n, "ys_gen."), names)]
    end
    mktempdir() do dir
        cd(dir) do
            a = stan_instantiate(fit).model
            f = stan_instantiate(forecast).model
            @test BridgeStan.param_unc_num(f) == BridgeStan.param_unc_num(a) == 4
            for (th, seed) in (([0.4, log(0.3), 0.8, -1.1], 1), ([-1.3, log(1.7), -0.2, 0.6], 7))
                y = draws(f, th, seed)
                @test length(y) == 9
                @test y == draws(a, th, seed)
            end
            # Given the parameters every entry is drawn around its mean, the
            # fitted regression `b .* x`: a tiny sigma pins it there.
            th = [0.4, log(1e-6), 0.8, -1.1]
            m = 0.4 .* reduce(vcat, xs)
            @test all(abs.(draws(f, th, 3) .- m) .< 1e-3)
        end
    end
end
