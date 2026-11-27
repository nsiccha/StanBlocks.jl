"""
A series an observed `@scan` reads back may hold `missing` entries (snag
`bind-an-observed-62bb34a3`; user picks on decision `0iazaa6`: a gap before
the series' last observed entry is imputed, and the program uses §14's names).
Observed entries keep their conditional densities, each missing entry after the
last observed one is drawn in generated quantities from its own step law (the
conditional forecast), a gap before it is a parameter whose prior is its step
law, and the completed series is an output named after the data, laid out like
it. `y_gen` stays the whole trajectory from the setup law. The same holds at
model scope, over nested vectors in a plate (the container clause, a
`plate(ys) do y` cell parameter) and through a called submodel's `obs`
keyword, the shape a downstream module emits for a grouped AR(1) response
replayed onto its last observed rows and the rows to forecast.
"""
@testitem "slic: a partly missing series an observed @scan reads back" tags=[:slic, :scan, :plate, :ragged, :missing, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    msg(f) = try; f(); nothing; catch e; StanBlocks.diagnostic(e).message; end
    output(d, name) = only(o for o in d.outputs if o.name == name)
    ops(d) = Tuple(o.name for o in d.operations)
    M(body, data) = StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    ar = quote
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
            end
        end
    end
    head = :(phi ~ uniform(-1, 1); sigma ~ exponential(1))
    scope(y) = M(quote $head; $ar end, (; y, mu = zeros(length(y)), T = length(y)))

    # Model scope, a missing tail: the observed values are data, the steps
    # observe them through their position codes, and the tail is drawn.
    tail = scope([0.3, 0.1, missing, missing])
    code = stan_code(tail)
    data = stan_block(code, "data")
    @test occursin("vector[y_obs_n] y_obs;", data) && occursin("array[y_ii_n] int y_ii;", data)
    @test !occursin("vector[y_n] y;", data)
    @test occursin("vector[y_ii_n] y__h = merge_scan_missing(y_obs, y_ii);", stan_block(code, "transformed data"))
    @test !occursin("y_mis", stan_block(code, "parameters"))
    model = stan_block(code, "model")
    @test occursin("if((y_ii[1] > 0)) {", model)
    @test occursin("y_obs[y_ii[t]] ~ normal((mu[t] + (phi * (y__h[(t - 1)] - mu[(t - 1)]))), sigma);", model)
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[y_ii_n] y;", gq)
    @test occursin("y[t] = y__h[t];", gq)
    @test occursin("y[t] = normal_rng((mu[t] + (phi * (y[(t - 1)] - mu[(t - 1)]))), sigma);", gq)
    @test occursin("y_gen[t] = normal_rng((mu[t] + (phi * (y_gen[(t - 1)] - mu[(t - 1)]))), sigma);", gq)
    @test occursin("y_obs_likelihood[y_ii[t]] = normal_lpdfs(y_obs[y_ii[t]],", gq)
    @test !occursin("y_likelihood", code)
    d = stan_descriptor(tail)
    @test ops(d) == (:transpile, :instantiate, :fit, :predict, :pointwise_loglik)
    @test (output(d, :y).generative, output(d, :y).source) == (:completed, :y)
    @test (output(d, :y_gen).generative, output(d, :y_gen).source) == (:draw, :y)
    @test (output(d, :y_obs_likelihood).generative, output(d, :y_obs_likelihood).source) == (:pointwise_loglik, :y_obs)
    @test stanc_compiles(tail)

    # A gap before the last observed entry is a parameter; its step law is its prior.
    gap = scope([0.3, missing, 0.1, missing])
    code = stan_code(gap)
    @test occursin("vector[y_mis_n] y_mis;", stan_block(code, "parameters"))
    model = stan_block(code, "model")
    @test occursin("vector[y_ii_n] y__h = merge_scan_missing(y_obs, y_mis, y_ii);", model)
    @test occursin("if((y_ii[t] < 0)) {", model)
    @test occursin("y__h[t] ~ normal((mu[t] + (phi * (y__h[(t - 1)] - mu[(t - 1)]))), sigma);", model)
    @test output(stan_descriptor(gap), :y_mis).kind == :parameter
    @test stanc_compiles(gap)

    # A series with no observed entry is drawn whole, from the setup law.
    @test stanc_compiles(scope([missing, missing, missing]))
    # Data that admit `missing` but hold none are ordinary data.
    complete = stan_code(scope(Union{Missing,Float64}[0.3, 0.1, 0.2]))
    @test occursin("vector[y_n] y;", stan_block(complete, "data"))
    @test complete == stan_code(scope([0.3, 0.1, 0.2]))

    # After the statement holding the scan, the body's `y` is the completed series.
    after = M(quote $head; $ar; total = sum(y) end, (; y = [0.3, missing], mu = zeros(2), T = 2))
    @test occursin("real total = sum(y);", stan_block(stan_code(after), "generated quantities"))
    # A per-step local the observation reads is recomputed for the forecast.
    lagged = M(quote
        nu ~ normal(0, 1); phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(nu, sigma)
            for t in 2:T
                e = y[t-1] - nu
                y[t] ~ normal(nu + phi * e, sigma)
            end
        end
    end, (; y = [0.3, 0.1, missing], T = 3))
    gq = stan_block(stan_code(lagged), "generated quantities")
    @test occursin("real loop_t_e__cmp = (y[(t - 1)] - nu);", gq)
    @test stanc_compiles(lagged)

    # Nested vectors: through a called submodel's keyword, the container clause,
    # and a `plate(ys) do y` cell parameter; the twins carry the data's groups.
    stream = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                obs[t] ~ normal(mu[t] + phi * (obs[t-1] - mu[t-1]), sigma)
            end
        end
    end
    ys = Vector{Union{Missing,Float64}}[[0.3, 0.1, missing, missing], [0.2, missing, missing], [missing, missing]]
    mus = [zeros(4), zeros(3), zeros(2)]
    called = M(quote
        $head
        @plate for p in eachindex(ys)
            node ~ stream(; obs = ys[p], mu = mus[p], n = length(ys[p]), phi, sigma)
        end
    end, (; ys, mus))
    container = M(quote
        $head
        @plate for y in ys
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(y)
                    y[t] ~ normal(phi * y[t-1], sigma)
                end
            end
        end
    end, (; ys))
    cell = M(quote
        $head
        cells ~ plate(ys, mus) do y, m
            node ~ stream(; obs = y, mu = m, n = length(y), phi, sigma)
        end
    end, (; ys, mus))
    for m in (called, container, cell)
        code = stan_code(m)
        @test occursin("tuple(vector[ys_obs_mem_n], array[ys_obs_ends_n] int) ys_obs;", stan_block(code, "data"))
        @test occursin("tuple(array[ys_ii_mem_n] int, array[ys_ii_ends_n] int) ys_ii;", stan_block(code, "data"))
        @test occursin("vector[num_elements(ys_ii.1)] ys;", stan_block(code, "generated quantities"))
        d = stan_descriptor(m)
        @test ops(d) == (:transpile, :instantiate, :fit, :predict, :pointwise_loglik)
        @test (output(d, :ys).generative, output(d, :ys).source, output(d, :ys).segments) == (:completed, :ys, [4, 7, 9])
        @test only(o for o in d.operations if o.name === :predict).outputs == (:ys_gen, :ys)
        @test (output(d, :ys_gen).generative, output(d, :ys_gen).segments) == (:draw, [4, 7, 9])
        @test output(d, :ys_obs_likelihood).segments == [2, 3, 3]
        @test stanc_compiles(m)
    end
    @test stanc_compiles(M(quote
        $head
        @plate for p in eachindex(ys)
            node ~ stream(; obs = ys[p], mu = mus[p], n = length(ys[p]), phi, sigma)
        end
    end, (; ys = Vector{Union{Missing,Float64}}[[0.3, missing, 0.1, missing], [0.2, missing]], mus = [zeros(4), zeros(2)])))

    # Re-binding a traced model re-splits the series; one with a new gap needs a new trace.
    sm = stan_model(tail)
    rebound = StanBlocks.block(sm(; y = [0.5, missing, missing, missing]), :data).content
    @test StanBlocks.getvalue(rebound[:y_obs]) == [0.5]
    @test StanBlocks.getvalue(rebound[:y_ii]) == [1, 0, 0, 0]
    @test_throws ErrorException sm(; y = [0.5, missing, 0.2, missing])

    # Refused, each naming the series: a value read of it outside its scan, a
    # series no scan or `~` observes, integer values, a gap under a law that
    # does not cover the real line, a carried array filled from it, and a
    # likelihood term reading the completed series' draws.
    @test occursin("so only the observed `@scan` that reads it back may read its values, but `mean(y_ii)` reads them",
        msg(() -> stan_code(M(quote $head; ybar = mean(y); z ~ normal(ybar, 1); $ar end,
            (; y = [0.3, missing], mu = zeros(2), T = 2, z = 0.1)))))
    @test occursin("no `~` statement observes it whole and no observed `@scan` reads it back",
        msg(() -> stan_code(M(quote b ~ normal(0, 1); z ~ normal(b * sum(y), 1) end, (; y = [0.3, missing], z = 0.1)))))
    @test occursin("supports for continuous (floating-point) values",
        msg(() -> stan_code(scope([1, 2, missing]))))
    @test occursin("needs a law with support on the whole real line; `lognormal(0, s)`",
        msg(() -> stan_code(M(quote
            s ~ exponential(1)
            @scan begin
                y[1] ~ lognormal(0, s)
                for t in 2:T; y[t] ~ lognormal(log(y[t-1]), s); end
            end
        end, (; y = [1.0, missing, 2.0], T = 3)))))
    @test occursin("`err[1] = y[1] - nu` reads `y`, which holds `missing` entries",
        msg(() -> stan_code(M(quote
            nu ~ normal(0, 1); phi ~ uniform(-1, 1); sigma ~ exponential(1)
            @scan begin
                err[1] = y[1] - nu; y[1] ~ normal(nu, sigma)
                for t in 2:T; err[t] = y[t] - nu; y[t] ~ normal(nu + phi * err[t-1], sigma); end
            end
        end, (; y = [0.3, 0.1, missing], T = 3)))))
    @test occursin("is a likelihood term, but it reads `y`: a random draw",
        msg(() -> stan_code(M(quote $head; $ar; z ~ normal(sum(y), 1) end,
            (; y = [0.3, missing], mu = zeros(2), T = 2, z = 0.1)))))
end

"""
The density of a partly missing series is that of its observed prefix (a
missing tail adds no term), or with a gap imputed, the complete series'
density at the imputed value; the completed series repeats the observed values
and draws each missing entry from its conditional law given the last observed
one. Checked through a timed AR(1) submodel over nested vectors with integer
time gaps, a regression mean and one series with no observed entry.
"""
@testitem "slic: a partly missing series matches its observed density and forecasts conditionally" tags=[:slic, :scan, :plate, :ragged, :missing, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    using Statistics: mean, std
    BS = BridgeStan
    M(body, data) = StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    scope(y, mu) = M(quote
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
            end
        end
    end, (; y, mu, T = length(y)))
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
    panel(ys, xs, ts) = M(quote
        b ~ normal(0, 1); sigma ~ exponential(1); phi ~ uniform(-1, 1)
        cells ~ plate(ys, xs, ts) do y, x, t
            m = b .* x
            node ~ timed(; obs = y, mu = m, sigma, phi, n = length(t), time = t)
        end
    end, (; ys, xs, ts))
    gq(bm, th, seed, name) = begin
        v = BS.param_constrain(bm, th; include_tp=true, include_gq=true, rng=BS.StanRNG(bm, seed))
        n = BS.param_names(bm; include_tp=true, include_gq=true)
        v[findall(x -> x == name || startswith(x, name * "."), n)]
    end
    mu = [0.2, -0.1, 0.4, 0.0]
    mktempdir() do dir
        cd(dir) do
            # Model scope, a missing tail.
            miss = stan_instantiate(scope([0.3, 0.1, missing, missing], mu)).model
            prefix = stan_instantiate(scope([0.3, 0.1], mu[1:2])).model
            @test BS.param_unc_num(miss) == 2
            for th in ([0.4, log(0.5)], [-1.2, log(1.7)])
                @test BS.log_density(miss, th) ≈ BS.log_density(prefix, th) atol=1e-10
                @test BS.log_density_gradient(miss, th)[2] ≈ BS.log_density_gradient(prefix, th)[2] atol=1e-10
            end
            phi, sigma = 0.7, 0.5
            th = [2atanh(phi), log(sigma)]
            draws = reduce(hcat, [gq(miss, th, s, "y") for s in 1:4000])
            @test all(draws[1, :] .== 0.3) && all(draws[2, :] .== 0.1)
            m3 = mu[3] + phi * (0.1 - mu[2])
            @test mean(draws[3, :]) ≈ m3 atol=0.04
            @test std(draws[3, :]) ≈ sigma atol=0.03
            @test mean(draws[4, :]) ≈ mu[4] + phi * (m3 - mu[3]) atol=0.04
            @test std(draws[4, :]) ≈ sigma * sqrt(1 + phi^2) atol=0.03
            sd1 = sigma / sqrt(1 - phi^2)
            lpdf(x, m, s) = -0.5 * ((x - m) / s)^2 - log(s) - 0.5 * log(2pi)
            @test gq(miss, th, 1, "y_obs_likelihood") ≈
                [lpdf(0.3, mu[1], sd1), lpdf(0.1, mu[2] + phi * (0.3 - mu[1]), sigma)] atol=1e-10

            # Model scope, a gap: the imputed value is the first unconstrained coordinate.
            gap = stan_instantiate(scope([0.3, missing, 0.1, missing], mu)).model
            @test BS.param_unc_num(gap) == 3
            for (th, v) in (([0.4, log(0.5)], 0.25), ([-1.2, log(1.7)], -0.8))
                full = stan_instantiate(scope([0.3, v, 0.1], mu[1:3])).model
                @test BS.log_density(gap, [v; th]) ≈ BS.log_density(full, th) atol=1e-10
                @test gq(gap, [v; th], 5, "y")[1:3] == [0.3, v, 0.1]
            end

            # Nested vectors through a called submodel, time gaps, a series with
            # no observed entry.
            xs = [[0.3, -1.2, 0.8, 0.1], [1.1, 0.2, -0.4], [-0.7, 0.5]]
            ts = [[1, 2, 4, 5], [1, 3, 4], [2, 5]]
            ys = Vector{Union{Missing,Float64}}[[0.1, -0.5, missing, missing], [1.0, missing, missing], [missing, missing]]
            fc = stan_instantiate(panel(ys, xs, ts)).model
            ref = stan_instantiate(panel([[0.1, -0.5], [1.0]], [xs[1][1:2], xs[2][1:1]], [ts[1][1:2], ts[2][1:1]])).model
            @test BS.param_unc_num(fc) == 3
            for th in ([0.4, log(0.3), 0.8], [-1.3, log(1.7), -0.2])
                @test BS.log_density(fc, th) ≈ BS.log_density(ref, th) atol=1e-10
            end
            b, sigma, phi = 0.4, 0.3, tanh(0.4)
            th = [b, log(sigma), 0.8]
            draws = reduce(hcat, [gq(fc, th, s, "ys") for s in 1:4000])
            @test size(draws, 1) == 9
            @test all(draws[1, :] .== 0.1) && all(draws[2, :] .== -0.5) && all(draws[5, :] .== 1.0)
            m = b .* xs[1]
            @test mean(draws[3, :]) ≈ m[3] + phi^2 * (-0.5 - m[2]) atol=0.03
            @test std(draws[3, :]) ≈ sigma * sqrt((1 - phi^4) / (1 - phi^2)) atol=0.03
            @test mean(draws[8, :]) ≈ b * xs[3][1] atol=0.05
            @test std(draws[8, :]) ≈ sigma / sqrt(1 - phi^2) atol=0.05
            @test length(gq(fc, th, 1, "ys_obs_likelihood")) == 3
        end
    end
end
