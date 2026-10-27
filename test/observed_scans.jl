"""
A `@scan` may observe DATA and read it back at a lag (snag
`observed-scan-sc-6bfe7467`). The observations are the conditional likelihood:
each reads the observed past. The predictive twin `y_gen` is the whole
trajectory — the scan the same body runs with `y` omitted — so its lags read
`y_gen`, never the observed `y` (a one-step-ahead draw is not a draw of `y`).
Each observation contributes its pointwise conditional log density to a
per-element `y_likelihood`. Before this, a setup statement on data refused
(`@scan: setup y[1]: y is data`).
"""
@testitem "slic: an observed @scan conditions on its past and simulates whole trajectories" tags=[:slic, :scan, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    import StanBlocks.stan: maybecv
    ar1 = @slic (; y = randn(5), mu = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
            end
        end
    end
    code = stan_code(ar1)
    @test strip(stan_block(code, "transformed parameters")) == "{\n}"
    mb = stan_block(code, "model")
    @test occursin("y[1] ~ normal(mu[1], (sigma / sqrt((1 - (phi ^ 2)))));", mb)
    @test occursin("y[t] ~ normal((mu[t] + (phi * (y[(t - 1)] - mu[(t - 1)]))), sigma);", mb)
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[y_n] y_gen;", gq)
    @test occursin("vector[y_n] y_likelihood;", gq)
    @test occursin("y_gen[1] = normal_rng(mu[1], (sigma / sqrt((1 - (phi ^ 2)))));", gq)
    @test occursin("y_gen[t] = normal_rng((mu[t] + (phi * (y_gen[(t - 1)] - mu[(t - 1)]))), sigma);", gq)
    @test occursin("y_likelihood[t] = normal_lpdfs(y[t], (mu[t] + (phi * (y[(t - 1)] - mu[(t - 1)]))), sigma);", gq)
    @test !occursin("normal_rng((mu[t] + (phi * (y[(t - 1)]", gq)     # no one-step-ahead draw
    @test stanc_compiles(ar1)
    d = stan_descriptor(ar1)
    @test Set(o.name for o in d.operations) == Set([:transpile, :instantiate, :fit, :predict, :pointwise_loglik])
    roles = Dict(o.name => (o.generative, o.source, o.dims) for o in d.outputs)
    @test roles[:y_gen] == (:draw, :y, (5,))
    @test roles[:y_likelihood] == (:pointwise_loglik, :y, (5,))

    # The same body with `y` omitted is the prior-predictive trajectory, unchanged.
    prior = @slic (; mu = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
            end
        end
    end
    pgq = stan_block(stan_code(prior), "generated quantities")
    @test occursin("y[t] = normal_rng((mu[t] + (phi * (y[(t - 1)] - mu[(t - 1)]))), sigma);", pgq)

    # Held out: the likelihood leaves the model; both twins stay.
    held = @slic (; y = maybecv(:y, randn(5)), T = 5) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, sigma)
            for t in 2:T
                y[t] ~ normal(phi * y[t-1], sigma)
            end
        end
    end
    hcode = stan_code(held)
    @test !occursin("y[t] ~", stan_block(hcode, "model"))
    @test occursin("y_gen[t] = normal_rng((phi * y_gen[(t - 1)]), sigma);", stan_block(hcode, "generated quantities"))
    @test Set(o.name for o in stan_descriptor(held).operations) == Set([:transpile, :instantiate, :predict, :pointwise_loglik])
    @test stanc_compiles(held)

    # An array observed per step and never read back keeps its per-cell twin.
    ss = @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            h[1] ~ normal(0, 1)
            y[1] ~ normal(h[1], 1)
            for t in 2:T
                h[t] ~ normal(phi * h[t-1], s)
                y[t] ~ normal(h[t], 1)
            end
        end
    end
    sgq = stan_block(stan_code(ss), "generated quantities")
    @test occursin("y_gen[t] = normal_rng(h[t], 1);", sgq)
    @test !occursin("y_likelihood", sgq)
    @test stanc_compiles(ss)
end

"""
Higher-order and moving-average recurrences: AR(2) setups read the observed
array too, and a deterministic statement that reads the observations and feeds
one of them (a carried residual) is replayed along the simulated trajectory
(`err_gen`). Latent arrays the observations read are not re-drawn: like every
predictive twin, the trajectory is drawn given the posterior latents.
"""
@testitem "slic: an observed @scan replays the residuals its observations read" tags=[:slic, :scan, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    ar2 = @slic (; y = randn(6), T = 6) begin
        a1 ~ normal(0, 0.5); a2 ~ normal(0, 0.5); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            y[2] ~ normal(a1 * y[1], sigma)
            for t in 3:T
                y[t] ~ normal(a1 * y[t-1] + a2 * y[t-2], sigma)
            end
        end
    end
    gq = stan_block(stan_code(ar2), "generated quantities")
    @test occursin("y_gen[2] = normal_rng((a1 * y_gen[1]), sigma);", gq)
    @test occursin("y_gen[t] = normal_rng(((a1 * y_gen[(t - 1)]) + (a2 * y_gen[(t - 2)])), sigma);", gq)
    @test stanc_compiles(ar2)

    arma = @slic (; y = randn(6), T = 6) begin
        phi ~ uniform(-1, 1); theta ~ normal(0, 0.5); sigma ~ exponential(1); mu ~ normal(0, 1)
        @scan begin
            err[1] = y[1] - mu
            y[1] ~ normal(mu, sigma)
            for t in 2:T
                nu = mu + phi * y[t-1] + theta * err[t-1]
                err[t] = y[t] - nu
                y[t] ~ normal(nu, sigma)
            end
        end
    end
    code = stan_code(arma)
    tp = stan_block(code, "transformed parameters")
    # The step local `nu` is a local of each loop that reads it (`1rgglep`).
    @test occursin("real loop_t_nu = (mu + (phi * y[(t - 1)]) + (theta * err[(t - 1)]));\n        err[t] = (y[t] - loop_t_nu);", tp)
    gq = stan_block(code, "generated quantities")
    # The draw at step t precedes the residual that reads it.
    @test occursin(join([
        "real loop_t_nu_gen = (mu + (phi * y_gen[(t - 1)]) + (theta * err_gen[(t - 1)]));",
        "y_gen[t] = normal_rng(loop_t_nu_gen, sigma);",
        "err_gen[t] = (y_gen[t] - loop_t_nu_gen);"], "\n        "), gq)
    @test occursin("y_gen[1] = normal_rng(mu, sigma);\n    err_gen[1] = (y_gen[1] - mu);", gq)
    @test stanc_compiles(arma)

    # Each step draws as its observation's own twin would: truncated, clamped,
    # unweighted; an integer series draws integers.
    hof(obs) = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ $(obs(:(0.0), :(1.0), 1))
            for t in 2:T
                y[t] ~ $(obs(:(phi * y[t-1]), :s, :t))
            end
        end
    end, Dict{Symbol,Any}(:y => abs.(randn(5)), :w => ones(5), :T => 5), @__MODULE__)
    truncated_gq = stan_block(stan_code(hof((m, s, i) -> :(normal($m, $s; lower = 0)))), "generated quantities")
    @test occursin("y_gen[t] = lower_conditioning_normal_rng(0, (phi * y_gen[(t - 1)]), s);", truncated_gq)
    censored_m = hof((m, s, i) -> :(censored(normal($m, $s); lower = 0.1)))
    @test occursin("y_gen[t] = lower_clamping_normal_rng(0.1, (phi * y_gen[(t - 1)]), s);",
        stan_block(stan_code(censored_m), "generated quantities"))
    @test stanc_compiles(censored_m)
    weighted_m = hof((m, s, i) -> :(weighted(normal($m, $s), w[$i])))
    @test occursin("y_gen[t] = weighted_normal_rng(w[t], (phi * y_gen[(t - 1)]), s);",
        stan_block(stan_code(weighted_m), "generated quantities"))
    @test stanc_compiles(weighted_m)
    counts = @slic (; y = [1, 3, 2, 0, 4], T = 5) begin
        a ~ normal(0, 1); b ~ normal(0, 0.3)
        @scan begin
            y[1] ~ poisson(exp(a))
            for t in 2:T
                y[t] ~ poisson(exp(a + b * y[t-1]))
            end
        end
    end
    gq = stan_block(stan_code(counts), "generated quantities")
    @test occursin("array[y_n] int y_gen;", gq)
    @test occursin("y_gen[t] = poisson_rng(exp((a + (b * y_gen[(t - 1)]))));", gq)
    @test stanc_compiles(counts)

    mixed = @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1); r ~ normal(0, 1)
        @scan begin
            h[1] ~ normal(0, 1)
            y[1] ~ normal(h[1], 1)
            for t in 2:T
                h[t] ~ normal(phi * h[t-1], s)
                y[t] ~ normal(h[t] + r * y[t-1], 1)
            end
        end
    end
    code = stan_code(mixed)
    @test occursin("vector[T] h;", stan_block(code, "parameters"))
    @test occursin("y_gen[t] = normal_rng((h[t] + (r * y_gen[(t - 1)])), 1);", stan_block(code, "generated quantities"))
    @test stanc_compiles(mixed)
end

"""
An observed scan nested in a plate: dense data observed per cell (`y[t, j]`),
or a group of nested-vector data — written through the container clause
(`for y in ys`), as `ys[p][t]`, or in a `plate(…) do` cell. The trajectory twin
is laid out like the data (a flat `ys_gen` for nested vectors, with the data's
`segments`), and so is the per-element likelihood.
"""
@testitem "slic: an observed @scan nests in plate cells, dense and ragged" tags=[:slic, :scan, :plate, :ragged, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    dense = @slic (; y = randn(5, 3), J = 3, T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                y[1, j] ~ normal(0, s / sqrt(1 - phi^2))
                for t in 2:T
                    y[t, j] ~ normal(phi * y[t-1, j], s)
                end
            end
        end
    end
    code = stan_code(dense)
    @test occursin("y[t, j] ~ normal((phi * y[(t - 1), j]), s);", stan_block(code, "model"))
    gq = stan_block(code, "generated quantities")
    @test occursin("matrix[y_m, y_n] y_gen;", gq)
    @test occursin("matrix[y_m, y_n] y_likelihood;", gq)
    @test occursin("y_gen[t, j] = normal_rng((phi * y_gen[(t - 1), j]), s);", gq)
    @test count("y_gen[t, j] =", gq) == 1
    @test stanc_compiles(dense)
    roles = Dict(o.name => (o.generative, o.dims) for o in stan_descriptor(dense).outputs)
    @test roles[:y_gen] == (:draw, (5, 3))
    @test roles[:y_likelihood] == (:pointwise_loglik, (5, 3))

    ys = [randn(3), randn(4), randn(2)]
    container = @slic (; ys) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for y in ys
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(y)
                    y[t] ~ normal(phi * y[t-1], sigma)
                end
            end
        end
    end
    indexed = @slic (; ys, P = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:P
            @scan begin
                ys[p][1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(ys[p])
                    ys[p][t] ~ normal(phi * ys[p][t-1], sigma)
                end
            end
        end
    end
    doform = @slic (; ys) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        r ~ plate(ys; outer = length(ys)) do y
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(y)
                    y[t] ~ normal(phi * y[t-1], sigma)
                end
            end
        end
    end
    for m in (container, indexed, doform)
        code = stan_code(m)
        mb = stan_block(code, "model")
        @test occursin(r"ys\.1\[ragged_start\(ys\.2, \w+\)\] ~ normal\(0, ", mb)
        gq = stan_block(code, "generated quantities")
        @test occursin("vector[num_elements(ys.1)] ys_gen;", gq)
        @test occursin("vector[num_elements(ys.1)] ys_likelihood;", gq)
        @test occursin(r"ys_gen\[\(\(ragged_start\(ys\.2, \w+\) \+ t\) - 1\)\] = normal_rng\(\(phi \* ys_gen\[", gq)
        @test stanc_compiles(m)
        d = stan_descriptor(m)
        @test :pointwise_loglik in Set(o.name for o in d.operations)
        out = Dict(o.name => o for o in d.outputs)
        @test (out[:ys_gen].generative, out[:ys_gen].segments, out[:ys_gen].dims) == (:draw, [3, 7, 9], (9,))
        @test (out[:ys_likelihood].generative, out[:ys_likelihood].segments) == (:pointwise_loglik, [3, 7, 9])
    end
end

"""
A do-block parameter may reuse the name of the data it slices
(`plate(y) do y`), as in Julia, and the cell lowers exactly like the
distinct-name spelling (snag `observed-scan-in-63843c19`). The compiler writes
outer references into the cell (the per-cell accessor, a nested scan's
`y[i][t]`), so a parameter that shadows an outer name is renamed before the cell
is traced. Before this, an observed scan over the shadowed group refused
(`@scan: unsupported statement LHS (y[plate_i__pl_1])[t]`).
"""
@testitem "slic: a plate do-parameter may shadow the data it slices" tags=[:slic, :scan, :plate, :ragged, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    ys = [[0.12, 0.31, 0.2], [-0.05, 0.1, 0.42, 0.3], [0.3, -0.2]]
    observed_shadow = @slic (; y = ys) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        level ~ normal(0, 1; n = 3)
        y_cell ~ plate(level, y; outer = (3,)) do level, y
            @scan begin
                y[1] ~ normal(level, sigma / sqrt(1 - phi^2))
                for t in 2:length(y)
                    y[t] ~ normal(level + phi * (y[t-1] - level), sigma)
                end
            end
        end
    end
    observed_distinct = @slic (; y = ys) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        level ~ normal(0, 1; n = 3)
        y_cell ~ plate(level, y; outer = (3,)) do l, yy
            @scan begin
                yy[1] ~ normal(l, sigma / sqrt(1 - phi^2))
                for t in 2:length(yy)
                    yy[t] ~ normal(l + phi * (yy[t-1] - l), sigma)
                end
            end
        end
    end
    code = stan_code(observed_shadow)
    @test code == stan_code(observed_distinct)
    @test occursin(r"y\.1\[ragged_start\(y\.2, \w+\)\] ~ normal\(level\[\w+\], ", stan_block(code, "model"))
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[num_elements(y.1)] y_gen;", gq)
    @test occursin(r"\(y_gen\[\(\(loop_t__pl_inv\d+\[\w+\] \+ \(t - 1\)\) - 1\)\] - level\[\w+\]\)", gq)
    @test stanc_compiles(observed_shadow)
    out = Dict(o.name => o for o in stan_descriptor(observed_shadow).outputs)
    @test (out[:y_gen].generative, out[:y_gen].segments) == (:draw, [3, 7, 9])
    @test (out[:y_likelihood].generative, out[:y_likelihood].segments) == (:pointwise_loglik, [3, 7, 9])

    # A latent scan and a plain in-cell observation over the shadowed group.
    latent(shadow) = shadow ? @slic((; y = ys), begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        res ~ plate(y) do y
            @scan begin
                h[1] ~ normal(0, 1)
                for t in 2:length(y)
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
            y ~ normal(h, 1.0)
        end
    end) : @slic((; y = ys), begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        res ~ plate(y) do yy
            @scan begin
                h[1] ~ normal(0, 1)
                for t in 2:length(yy)
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
            yy ~ normal(h, 1.0)
        end
    end)
    @test stan_code(latent(true)) == stan_code(latent(false))
    @test stanc_compiles(latent(true))
    # A parameter may shadow the data ANOTHER positional slices: `z` reads the
    # outer `y`, not the cell's `y` (a slice of `x`).
    cross(shadow) = shadow ? @slic((; x = randn(3), y = randn(3)), begin
        res ~ plate(x, y; outer = (3,)) do y, z
            z ~ normal(y, 1.0)
        end
    end) : @slic((; x = randn(3), y = randn(3)), begin
        res ~ plate(x, y; outer = (3,)) do a, b
            b ~ normal(a, 1.0)
        end
    end)
    @test stan_code(cross(true)) == stan_code(cross(false))
    @test occursin(r"y\[(\w+)\] ~ normal\(x\[\1\], 1\.0\);", stan_block(stan_code(cross(true)), "model"))
    # Messages keep the author's parameter name.
    err = try
        stan_code(@slic (; y = randn(3)) begin
            res ~ plate(y; outer = (4,)) do y
                y ~ normal(0, 1)
            end
        end)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("positional `y` (param `y`) has length 3", err)
end

"""
A scan nested in a plate may run to a per-cell length (`for t in 2:T[j]`, or
`length(y)` of a ragged cell). Its carried arrays and per-step locals are then
stored as the plate's flat memory, and an element of cell `j` is
`mem[ragged_start(ends, j) + t - 1]`. Before this, the inner sampling refused
(`Plate-generated indexed sampling of h[…] requires the plate emitter to register
its outer declaration first`).
"""
@testitem "slic: a @scan nested in a plate runs to a per-cell length" tags=[:slic, :scan, :plate, :ragged, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    latent(data) = @slic data begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                h[1] ~ normal(0, 1)
                for t in 2:T[j]
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
        end
    end
    prior = latent((; T = [3, 4, 5], J = 3))
    code = stan_code(prior)
    @test strip(stan_block(code, "parameters")) == "{\n}"
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[sum(h__pl_len)] h;", gq)
    @test occursin("h[ragged_start(h__pl_end, j)] = normal_rng(0, 1);", gq)
    @test occursin("h[((ragged_start(h__pl_end, j) + t) - 1)] = normal_rng((phi * h[((ragged_start(h__pl_end, j) + (t - 1)) - 1)]), s);", gq)
    @test stanc_compiles(prior)

    obs = [randn(3), randn(4), randn(5)]
    fitted = @slic (; T = [3, 4, 5], J = 3, obs) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                h[1] ~ normal(0, 1)
                for t in 2:T[j]
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
            obs[j] ~ normal(h, 1.0)
        end
    end
    code = stan_code(fitted)
    @test occursin("vector[sum(h__pl_len)] h;", stan_block(code, "parameters"))
    @test stanc_compiles(fitted)

    noncentered = @slic (; T = [3, 4, 5], J = 3, obs) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                e1 ~ std_normal()
                x[1] = s * e1
                for t in 2:T[j]
                    eps ~ std_normal()
                    x[t] = phi * x[t-1] + s * eps
                end
            end
        end
        obs ~ normal(x, 1.0)
    end
    code = stan_code(noncentered)
    td = stan_block(code, "transformed data")
    @test occursin("loop_t_eps__pl_len[j] = (T[j] - 1);", td)
    @test occursin("vector[sum(loop_t_eps__pl_len)] loop_t_eps;", stan_block(code, "parameters"))
    @test stanc_compiles(noncentered)

    container = @slic (; ys = obs) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for y in ys
            @scan begin
                h[1] ~ normal(0, 1)
                for t in 2:length(y)
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
            y ~ normal(h, 1.0)
        end
    end
    @test stanc_compiles(container)
    @test Dict(o.name => o.segments for o in stan_descriptor(container).outputs)[:h] == [3, 7, 12]

    # Support reaches the flat memory: one shared bound is scalar; a setup and a
    # recurrence with different supports take a per-element bound vector.
    bounded(step) = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                h[1] ~ exponential(1)
                for t in 2:T[j]
                    h[t] ~ $step
                end
            end
            obs[j] ~ normal(h, 1.0)
        end
    end, Dict{Symbol,Any}(:T => [3, 4, 5], :J => 3, :obs => obs), @__MODULE__)
    shared = stan_code(bounded(:(exponential(1 / (1 + h[t-1])))))
    @test occursin("vector<lower=0.0>[sum(h__pl_len)] h;", stan_block(shared, "parameters"))
    mixed = bounded(:(normal(phi * h[t-1], s)))
    code = stan_code(mixed)
    @test occursin("vector<lower=h__lower>[sum(h__pl_len)] h;", stan_block(code, "parameters"))
    td = stan_block(code, "transformed data")
    @test occursin("h__lower[ragged_start(h__pl_end, j)] = 0.0;", td)
    @test occursin("h__lower[((ragged_start(h__pl_end, j) + t) - 1)] = negative_infinity();", td)
    @test stanc_compiles(mixed)

    # A plain varying-length plate cell keeps its family's support too.
    cell = @slic (; K = [2, 3], G = 2, obs = [randn(2), randn(3)]) begin
        z ~ plate(; outer = G) do g
            zz::vector[K[g]] ~ exponential(1)
            zz
        end
        obs ~ normal(z, 1.0)
    end
    @test occursin("vector<lower=0.0>[sum(z_zz__pl_len)] z_zz;", stan_block(stan_code(cell), "parameters"))
end

"""
An integer that a nested `@scan` computes per step — a step local such as the
time gap `k = d[t] - d[t - 1]`, or a carried int array — promotes to the
enclosing plate's storage like a real-valued one: `array[S, T - 1] int` for a
fixed length, flat int memory for a per-cell length. It is data, so it is
computed once in `transformed data`. The plate used to drop it from promotion as
if it were a per-cell `findall` index; its definition sits inside the scan, so
nothing inlined it, and stanc rejected the model's read of an in-cell
declaration (`Identifier "loop_t_k" not in scope`; snag
`deterministic-st-a7afc47c`).
"""
@testitem "slic: a nested @scan promotes its integer step locals and carried arrays" tags=[:slic, :scan, :plate, :ragged, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    ys = [[0.1, 0.2, 0.4], [0.3, 0.5], [0.2, 0.1, 0.0, 0.3]]
    ts = [[1, 2, 4], [1, 3], [2, 3, 4, 7]]
    gaps = @slic (; ys, ts, n = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        res ~ plate(ys, ts; outer = (n,)) do y, d
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(d)
                    k = d[t] - d[t - 1]
                    y[t] ~ normal(phi^k * y[t - 1], sigma * sqrt((1 - phi^(2 * k)) / (1 - phi^2)))
                end
            end
            nothing
        end
    end
    code = stan_code(gaps)
    td = stan_block(code, "transformed data")
    @test occursin("array[sum(loop_t_k__pl_len)] int loop_t_k;", td)
    @test occursin("loop_t_k[((ragged_start(loop_t_k__pl_end, plate_i__pl_1) + (t - 1)) - 1)] = (", td)
    @test occursin("phi ^ loop_t_k[((ragged_start(loop_t_k__pl_end, plate_i__pl_1) + (t - 1)) - 1)]", stan_block(code, "model"))
    @test !occursin(r"(?m)^\s*int loop_t_k;", code)
    @test stanc_compiles(gaps)

    # Omitting the response simulates the trajectory, reading the same gaps.
    prior = @slic (; ts, n = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        res ~ plate(ts; outer = (n,)) do d
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(d)
                    k = d[t] - d[t - 1]
                    y[t] ~ normal(phi^k * y[t - 1], sigma * sqrt((1 - phi^(2 * k)) / (1 - phi^2)))
                end
            end
            nothing
        end
    end
    code = stan_code(prior)
    @test occursin("array[sum(loop_t_k__pl_len)] int loop_t_k;", stan_block(code, "transformed data"))
    @test occursin("phi ^ loop_t_k[", stan_block(code, "generated quantities"))
    @test stanc_compiles(prior)

    # The `@plate for` spelling of the same model.
    sugar = @slic (; ys, ts, n = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for i in 1:n
            @scan begin
                ys[i][1] ~ normal(0, sigma)
                for t in 2:length(ts[i])
                    k = ts[i][t] - ts[i][t - 1]
                    ys[i][t] ~ normal(phi^k * ys[i][t - 1], sigma)
                end
            end
        end
    end
    @test occursin("array[sum(loop_t_k__pl_len)] int loop_t_k;", stan_block(stan_code(sugar), "transformed data"))
    @test stanc_compiles(sugar)

    # Fixed length: a step local and a carried int array, one row per cell.
    D = [1 2 4; 1 3 5; 2 3 7]
    Y = [0.1 0.2 0.3; 0.2 0.1 0.0; 0.4 0.5 0.3]
    dense = @slic (; Y, D, T = 3, S = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for j in 1:S
            @scan begin
                c[1] = D[1, j]
                Y[1, j] ~ normal(0, sigma)
                for t in 2:T
                    k = D[t, j] - D[t - 1, j]
                    c[t] = c[t - 1] + k
                    Y[t, j] ~ normal(phi^k * Y[t - 1, j] + 0.01 * c[t], sigma)
                end
            end
        end
    end
    code = stan_code(dense)
    td = stan_block(code, "transformed data")
    @test occursin("array[S, (T - 1)] int loop_t_k;", td)   # locals take the first filled array's prefix
    @test occursin("array[S, T] int c;", td)
    @test occursin("(phi ^ loop_t_k[j][(t - 1)])", stan_block(code, "model"))
    @test stanc_compiles(dense)
end

"""
A per-cell integer binding in a plate stays an integer (`array[N] int`), so it
can index. It used to collect as a real `vector[N]`, and indexing with it failed
to trace — which a nested scan hits whenever it lifts a step-invariant index
such as `ragged_start(ys.ends, p)` out of its recurrence.
"""
@testitem "slic: a plate collects integer cell bindings as integers" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    sugar = @slic (; xs = randn(9), P = 3) begin
        phi ~ uniform(-1, 1)
        @plate for p in 1:P
            k = 2 * p + 1
            h[p] ~ normal(phi * xs[k], 1)
        end
    end
    td = stan_block(stan_code(sugar), "transformed data")
    @test occursin("array[P] int loop_p_k;", td)
    @test stanc_compiles(sugar)
    doform = @slic (; xs = randn(9), P = 3) begin
        phi ~ uniform(-1, 1)
        r ~ plate(; outer = P) do p
            k = 2 * p + 1
            hh ~ normal(phi * xs[k], 1)
            hh
        end
    end
    @test occursin("array[P] int r_k;", stan_block(stan_code(doform), "transformed data"))
    @test stanc_compiles(doform)
end

"""
An observed scan observes every element exactly once, in step order, given
earlier elements only; each refusal names the observed array.
"""
@testitem "slic: an observed @scan refuses what is not a sequential observation" tags=[:slic, :scan, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    refusal(f) = try
        stan_code(f()); ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("the setup observes no element of `y`", refusal(() -> @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            z ~ normal(0, 1)
            for t in 2:T
                y[t] ~ normal(phi * y[t-1], s)
            end
        end
    end))
    @test occursin("the scan runs to `T` = 4 but `y` has 5 elements", refusal(() -> @slic (; y = randn(5), T = 4) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            for t in 2:T
                y[t] ~ normal(phi * y[t-1], s)
            end
        end
    end))
    @test occursin("`y[t - 2]` reads before the setup", refusal(() -> @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            for t in 2:T
                y[t] ~ normal(phi * y[t-2], s)
            end
        end
    end))
    @test occursin("reads its own current value", refusal(() -> @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            for t in 2:T
                e = y[t] - phi * y[t-1]
                y[t] ~ normal(e, s)
            end
        end
    end))
    @test occursin("the model already binds `y_gen`", refusal(() -> @slic (; y = randn(5), T = 5, y_gen = 1.0) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            for t in 2:T
                y[t] ~ normal(phi * y[t-1] + y_gen, s)
            end
        end
    end))
    @test occursin("a scan reads back only its own cell", refusal(() -> @slic (; y = randn(5, 3), J = 3, T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 2:J
            @scan begin
                y[1, j] ~ normal(0, 1)
                for t in 2:T
                    y[t, j] ~ normal(phi * y[t-1, j-1], s)
                end
            end
        end
    end) * refusal(() -> @slic (; y = randn(5, 3), J = 3, T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                y[1, j] ~ normal(0, 1)
                for t in 2:T
                    y[t, j] ~ normal(phi * y[t-1, 1], s)
                end
            end
        end
    end))
    @test occursin("is observed only in the setup but read back in the loop", refusal(() -> @slic (; y = randn(5), T = 5) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @scan begin
            y[1] ~ normal(0, 1)
            for t in 2:T
                h[t] ~ normal(phi * y[t-1], s)
            end
        end
    end))
end

"""
The emitted programs evaluate the densities they claim (BridgeStan against
independent Julia references), and `y_gen` is a trajectory: at `phi = 0.9` its
last element has the stationary variance `1/(1 - 0.81)`, where a one-step-ahead
draw would have variance 1.
"""
@testitem "slic: observed @scan densities and trajectories match references" tags=[:slic, :scan, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems, Statistics, Random
    import BridgeStan
    Random.seed!(7)
    normlpdf(x, m, s) = -0.5 * log(2pi) - log(s) - 0.5 * ((x - m) / s)^2
    ar1(y, mu, phi, s) = normlpdf(y[1], mu[1], s / sqrt(1 - phi^2)) +
        sum(normlpdf(y[t], mu[t] + phi * (y[t-1] - mu[t-1]), s) for t in 2:length(y); init=0.0)
    gq(bm, th, seed) = BridgeStan.param_constrain(bm, th; include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(bm, seed))
    idx(bm, prefix) = findall(n -> startswith(n, prefix * "."), BridgeStan.param_names(bm; include_tp=true, include_gq=true))
    mktempdir() do dir
        cd(dir) do
            T = 8; y = cumsum(randn(T)) .* 0.3; mu = 0.1 .* (1:T)
            top = @slic (; y, mu, T) begin
                phi ~ uniform(-1, 1); sigma ~ exponential(1)
                @scan begin
                    y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
                    for t in 2:T
                        y[t] ~ normal(mu[t] + phi * (y[t-1] - mu[t-1]), sigma)
                    end
                end
            end
            bm = stan_instantiate(top).model
            for (phi, s) in ((0.3, 0.7), (0.85, 1.3), (-0.5, 0.4))
                th = BridgeStan.param_unconstrain(bm, [phi, s])
                @test BridgeStan.log_density(bm, th; propto=false, jacobian=false) ≈ ar1(y, mu, phi, s) + log(0.5) - s
                @test sum(gq(bm, th, 1)[idx(bm, "y_likelihood")]) ≈ ar1(y, mu, phi, s)
            end
            th = BridgeStan.param_unconstrain(bm, [0.9, 1.0])
            draws = reduce(hcat, [gq(bm, th, k)[idx(bm, "y_gen")] for k in 1:4000])
            @test 4.0 < var(draws[T, :]) < 6.5
            @test cor(draws[T, :], draws[T-1, :]) > 0.8

            ys = [randn(3), randn(5), randn(2)]
            ragged = @slic (; ys) begin
                phi ~ uniform(-1, 1); sigma ~ exponential(1)
                @plate for y in ys
                    @scan begin
                        y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                        for t in 2:length(y)
                            y[t] ~ normal(phi * y[t-1], sigma)
                        end
                    end
                end
            end
            bm = stan_instantiate(ragged).model
            for (phi, s) in ((0.4, 0.8), (-0.7, 1.5))
                th = BridgeStan.param_unconstrain(bm, [phi, s])
                lik = sum(ar1(g, zeros(length(g)), phi, s) for g in ys)
                @test BridgeStan.log_density(bm, th; propto=false, jacobian=false) ≈ lik + log(0.5) - s
                @test sum(gq(bm, th, 1)[idx(bm, "ys_likelihood")]) ≈ lik
            end
            th = BridgeStan.param_unconstrain(bm, [0.9, 1.0])
            draws = reduce(hcat, [gq(bm, th, k)[idx(bm, "ys_gen")] for k in 1:4000])
            @test 4.0 < var(draws[8, :]) < 6.5            # last element of group 2
            @test abs(cor(draws[3, :], draws[4, :])) < 0.1  # groups are independent

            Ts = [3, 4, 5]; obs = [randn(n) for n in Ts]
            percell = @slic (; T = Ts, J = 3, obs) begin
                phi ~ uniform(-1, 1); s ~ exponential(1)
                @plate for j in 1:J
                    @scan begin
                        h[1] ~ normal(0, 1)
                        for t in 2:T[j]
                            h[t] ~ normal(phi * h[t-1], s)
                        end
                    end
                    obs[j] ~ normal(h, 0.5)
                end
            end
            prob = stan_instantiate(percell)
            @test LogDensityProblems.dimension(prob) == 2 + sum(Ts)
            x = 0.1 .* randn(2 + sum(Ts))
            c = BridgeStan.param_constrain(prob.model, x)
            phi, s, h = c[1], c[2], c[3:end]
            ends = cumsum(Ts); starts = ends .- Ts .+ 1
            ref = log(0.5) - s
            for j in 1:3
                hj = h[starts[j]:ends[j]]
                ref += normlpdf(hj[1], 0, 1) + sum(normlpdf(hj[t], phi * hj[t-1], s) for t in 2:Ts[j])
                ref += sum(normlpdf(obs[j][t], hj[t], 0.5) for t in 1:Ts[j])
            end
            @test BridgeStan.log_density(prob.model, x; propto=false, jacobian=false) ≈ ref
        end
    end
end

"""
A nested scan's integer step local evaluates to the density of the same model
written with the local inlined: an AR(1) over irregular time gaps `k`.
"""
@testitem "slic: a nested @scan's integer step local matches a reference density" tags=[:slic, :scan, :plate, :ragged, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    normlpdf(x, m, s) = -0.5 * log(2pi) - log(s) - 0.5 * ((x - m) / s)^2
    ys = [[0.1, 0.2, 0.4], [0.3, 0.5], [0.2, 0.1, 0.0, 0.3]]
    ts = [[1, 2, 4], [1, 3], [2, 3, 4, 7]]
    gaps = @slic (; ys, ts, n = 3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        res ~ plate(ys, ts; outer = (n,)) do y, d
            @scan begin
                y[1] ~ normal(0, sigma / sqrt(1 - phi^2))
                for t in 2:length(d)
                    k = d[t] - d[t - 1]
                    y[t] ~ normal(phi^k * y[t - 1], sigma * sqrt((1 - phi^(2 * k)) / (1 - phi^2)))
                end
            end
            nothing
        end
    end
    ref(phi, s) = log(0.5) - s + sum(zip(ys, ts)) do (y, d)
        normlpdf(y[1], 0, s / sqrt(1 - phi^2)) + sum(2:length(d)) do t
            k = d[t] - d[t - 1]
            normlpdf(y[t], phi^k * y[t - 1], s * sqrt((1 - phi^(2k)) / (1 - phi^2)))
        end
    end
    mktempdir() do dir
        cd(dir) do
            bm = stan_instantiate(gaps).model
            for (phi, s) in ((0.3, 0.7), (0.85, 1.3), (-0.5, 0.4))
                th = BridgeStan.param_unconstrain(bm, [phi, s])
                @test BridgeStan.log_density(bm, th; propto=false, jacobian=false) ≈ ref(phi, s)
            end
        end
    end
end
