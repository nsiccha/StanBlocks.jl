"""
An annotated `@scan` inside a called `@slic` submodel (snag
`scan-inside-call-2ea17cef`). Its carried arrays are declared at the root under
the submodel's namespace (`hh_h`), and the submodel's own spelling (`h`) names
them in the setup, the lag reads and after the loop (`return h`). The three
modes behave as at model scope: fitted, prior-only (the whole block is a
generated-quantities simulation) and cross-validation. Called per `@plate`
cell, the enclosing plate collects each cell's trajectory. Before this, the
setup refused (`Indexed sampling h[…] ~ … needs h to be declared first`).
"""
@testitem "slic: a latent @scan runs inside a called submodel" tags=[:slic, :scan, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    import StanBlocks.stan: maybecv
    latent = @slic begin
        @scan begin
            h[1] ~ normal(0, sigma)
            for t in 2:n
                h[t] ~ normal(phi * h[t-1], sigma)
            end
        end
        return h
    end
    called(data; observe=true) = observe ? @slic(data, begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        hh ~ latent(; n = 5, phi, sigma)
        y ~ normal(hh, 1.0)
    end) : @slic(data, begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        hh ~ latent(; n = 5, phi, sigma)
    end)
    y = randn(5)
    code = stan_code(called((; y)))
    @test occursin("vector[5] hh_h;", stan_block(code, "parameters"))
    @test !occursin("loopprobe", code)
    @test occursin("vector[5] hh = hh_h;", stan_block(code, "transformed parameters"))
    mb = stan_block(code, "model")
    @test occursin("hh_h[1] ~ normal(0, sigma);", mb)
    @test occursin("hh_h[hh_t] ~ normal((phi * hh_h[(hh_t - 1)]), sigma);", mb)
    @test occursin("y ~ normal(hh, 1.0);", mb)
    @test stanc_compiles(called((; y)))

    prior = called((;); observe=false)
    pcode = stan_code(prior)
    @test strip(stan_block(pcode, "parameters")) == "{\n}"
    @test occursin("hh_h[hh_t] = normal_rng((phi * hh_h[(hh_t - 1)]), sigma);", stan_block(pcode, "generated quantities"))
    @test stanc_compiles(prior)

    held = called((; y = maybecv(:y, y)))
    hcode = stan_code(held)
    @test !occursin("y ~", stan_block(hcode, "model"))
    @test occursin("hh_h[hh_t] ~ normal(", stan_block(hcode, "model"))
    @test stanc_compiles(held)

    # Called per plate cell: the plate collects each cell's trajectory.
    percell(data; observe=true) = observe ? @slic(data, begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            hp ~ latent(; n = 5, phi, sigma)
            Y[:, p] ~ normal(hp, 1.0)
        end
    end) : @slic(data, begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            hp ~ latent(; n = 5, phi, sigma)
        end
    end)
    fitted = percell((; Y = randn(5, 3)))
    code = stan_code(fitted)
    @test occursin("matrix[5, 3] loop_p_hp_h;", stan_block(code, "parameters"))
    @test occursin("loop_p_hp_h[loop_p_hp_t, p] ~ normal((phi * loop_p_hp_h[(loop_p_hp_t - 1), p]), sigma);",
        stan_block(code, "model"))
    @test stanc_compiles(fitted)
    pprior = percell((;); observe=false)
    @test strip(stan_block(stan_code(pprior), "parameters")) == "{\n}"
    @test stanc_compiles(pprior)
    # A per-cell length: the trajectories are the plate's flat memory.
    ragged = @slic (; T = [3, 4, 2], obs = [randn(3), randn(4), randn(2)]) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            hp ~ latent(; n = T[p], phi, sigma)
            obs[p] ~ normal(hp, 0.5)
        end
    end
    code = stan_code(ragged)
    @test occursin("vector[sum(loop_p_hp_h__pl_len_1)] loop_p_hp_h__pl_mem_1;", stan_block(code, "parameters"))
    @test occursin("loop_p_hp_h__pl_mem_1[ragged_start(loop_p_hp_h__pl_end_1, p)] ~ normal(0, sigma);",
        stan_block(code, "model"))
    @test stanc_compiles(ragged)

    # Non-centered, with per-step locals: at model scope and per cell.
    noncentered = @slic begin
        @scan begin
            e1 ~ std_normal()
            h[1] = e1
            for t in 2:n
                eps ~ std_normal()
                h[t] = phi * h[t-1] + s * eps
            end
        end
        return h
    end
    top = @slic (; y = randn(5)) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        hh ~ noncentered(; n = 5, phi, s)
        y ~ normal(hh, 1.0)
    end
    code = stan_code(top)
    @test occursin("vector[(5 - 1)] hh_h_eps;", stan_block(code, "parameters"))
    @test occursin("hh_h[hh_t] = ((phi * hh_h[(hh_t - 1)]) + (s * hh_h_eps[(hh_t - 1)]));",
        stan_block(code, "transformed parameters"))
    @test stanc_compiles(top)
    cells = @slic (; Y = randn(5, 3)) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for p in 1:3
            hp ~ noncentered(; n = 5, phi, s)
            Y[:, p] ~ normal(hp, 1.0)
        end
    end
    code = stan_code(cells)
    @test occursin("matrix[(5 - 1), 3] loop_p_hp_h_eps;", stan_block(code, "parameters"))
    @test occursin("loop_p_hp_h_eps[(loop_p_hp_t - 1), p] ~ std_normal();", stan_block(code, "model"))
    @test stanc_compiles(cells)
    lengths = @slic (; T = [3, 4, 2], obs = [randn(3), randn(4), randn(2)]) begin
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for p in 1:3
            hp ~ noncentered(; n = T[p], phi, s)
            obs[p] ~ normal(hp, 0.5)
        end
    end
    @test stanc_compiles(lengths)
end

"""
An observed `@scan` inside a called submodel observes the CALLER's data: a
keyword bound to a data variable (`obs = y`), a column (`obs = Y[:, p]`) or a
nested-vector group (`obs = ys[p]`) is observed exactly as the same scan written
in the caller (`y[t]`, `Y[t, p]`, `ys[p][t]`): the trajectory is `y_gen` /
`Y_gen` / `ys_gen` and the likelihood `y_likelihood` / … , both twins of the
caller's data. Before this, the observation's metadata went to a new data
variable `<lhs>_obs` instead of `y` (an indexed observation of a keyword then
observed `node_obs[2]`), and the scan failed to trace. Unbound, the keyword is
an ordinary latent array: the prior-predictive trajectory.
"""
@testitem "slic: an observed @scan in a called submodel observes the caller's data" tags=[:slic, :scan, :plate, :ragged, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    import StanBlocks.stan: maybecv
    stream = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                obs[t] ~ normal(mu[t] + phi * (obs[t-1] - mu[t-1]), sigma)
            end
        end
    end
    called(data) = @slic data begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        node ~ stream(; obs = y, mu, n = 5, phi, sigma)
    end
    inline(data) = @slic data begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @scan begin
            y[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for node_t in 2:5
                y[node_t] ~ normal(mu[node_t] + phi * (y[node_t-1] - mu[node_t-1]), sigma)
            end
        end
    end
    y = randn(5); mu = randn(5)
    for data in ((; y, mu), (; y = maybecv(:y, y), mu))
        @test stan_code(called(data)) == stan_code(inline(data))
        @test stanc_compiles(called(data))
    end
    d = stan_descriptor(called((; y, mu)))
    @test Set(o.name for o in d.operations) == Set([:transpile, :instantiate, :fit, :predict, :pointwise_loglik])
    roles = Dict(o.name => (o.generative, o.source, o.dims) for o in d.outputs)
    @test roles[:y_gen] == (:draw, :y, (5,))
    @test roles[:y_likelihood] == (:pointwise_loglik, :y, (5,))
    @test Set(o.name for o in stan_descriptor(called((; y = maybecv(:y, y), mu))).operations) ==
        Set([:transpile, :instantiate, :predict, :pointwise_loglik])

    # Unbound, the keyword is a latent array: the prior-predictive trajectory.
    unbound = @slic (; mu) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        node ~ stream(; mu, n = 5, phi, sigma)
    end
    ucode = stan_code(unbound)
    @test strip(stan_block(ucode, "parameters")) == "{\n}"
    @test occursin("node_obs[node_t] = normal_rng(", stan_block(ucode, "generated quantities"))
    @test stanc_compiles(unbound)

    # A plain indexed observation of a keyword observes the caller's `y`.
    pair = @slic begin
        m ~ normal(0, 1)
        obs[1] ~ normal(m, 1)
        obs[2] ~ normal(m, 2)
    end
    pcode = stan_code(@slic (; y = randn(2)) begin
        node ~ pair(; obs = y)
    end)
    @test !occursin("node_obs", pcode)
    @test occursin("y[2] ~ normal(node_m, 2);", stan_block(pcode, "model"))
    @test occursin("y_gen[2] = normal_rng(node_m, 2);", stan_block(pcode, "generated quantities"))

    # Called per plate cell: a column of a matrix, and a nested-vector group.
    dense = @slic (; Y = randn(5, 3), M = randn(5, 3)) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            node ~ stream(; obs = Y[:, p], mu = M[:, p], n = 5, phi, sigma)
        end
    end
    code = stan_code(dense)
    @test occursin("Y[loop_p_node_t, p] ~ normal(", stan_block(code, "model"))
    gq = stan_block(code, "generated quantities")
    @test occursin("matrix[Y_m, Y_n] Y_gen;", gq)
    @test occursin("matrix[Y_m, Y_n] Y_likelihood;", gq)
    @test occursin("Y_gen[(loop_p_node_t - 1), p]", gq)
    @test stanc_compiles(dense)
    roles = Dict(o.name => (o.generative, o.source, o.dims) for o in stan_descriptor(dense).outputs)
    @test roles[:Y_gen] == (:draw, :Y, (5, 3))
    @test roles[:Y_likelihood] == (:pointwise_loglik, :Y, (5, 3))

    ys = [randn(3), randn(4), randn(2)]; mus = [randn(3), randn(4), randn(2)]
    grouped(data) = @slic data begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            node ~ stream(; obs = ys[p], mu = mus[p], n = length(ys[p]), phi, sigma)
        end
    end
    g = grouped((; ys, mus))
    code = stan_code(g)
    @test occursin(r"ys\.1\[ragged_start\(ys\.2, p\)\] ~ normal\(", stan_block(code, "model"))
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[num_elements(ys.1)] ys_gen;", gq)
    @test occursin("vector[num_elements(ys.1)] ys_likelihood;", gq)
    @test stanc_compiles(g)
    d = stan_descriptor(g)
    @test Set(o.name for o in d.operations) == Set([:transpile, :instantiate, :fit, :predict, :pointwise_loglik])
    out = Dict(o.name => o for o in d.outputs)
    @test (out[:ys_gen].generative, out[:ys_gen].source, out[:ys_gen].segments) == (:draw, :ys, [3, 7, 9])
    @test (out[:ys_likelihood].generative, out[:ys_likelihood].segments) == (:pointwise_loglik, [3, 7, 9])
    held = grouped((; ys = maybecv(:ys, ys), mus))
    @test !occursin("~ normal((", stan_block(stan_code(held), "model"))
    @test Set(o.name for o in stan_descriptor(held).operations) == Set([:transpile, :instantiate, :predict, :pointwise_loglik])
    @test stanc_compiles(held)

    # Irregular time gaps: an integer step local over a per-subject group is
    # data, computed once in transformed data, as in the caller's spelling.
    gapped = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                k = time[t] - time[t-1]
                obs[t] ~ normal(mu[t] + phi^k * (obs[t-1] - mu[t-1]), sigma * sqrt((1 - phi^(2k)) / (1 - phi^2)))
            end
        end
    end
    days = [[1, 2, 4], [1, 3, 4, 7], [2, 5]]
    g = @slic (; ys, mus, days) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in eachindex(ys)
            node ~ gapped(; obs = ys[p], mu = mus[p], time = days[p], n = length(ys[p]), phi, sigma)
        end
    end
    code = stan_code(g)
    @test occursin("array[sum(loop_p_node_loop_t_k__pl_len_1)] int loop_p_node_loop_t_k__pl_mem_1;",
        stan_block(code, "transformed data"))
    @test stanc_compiles(g)
    @test Set(o.name for o in stan_descriptor(g).operations) == Set([:transpile, :instantiate, :fit, :predict, :pointwise_loglik])

    # Anything else has no trajectory to simulate: say what to bind instead.
    err = try
        stan_code(@slic (; Y = randn(3, 5)) begin
            phi ~ uniform(-1, 1); sigma ~ exponential(1)
            node ~ stream(; obs = Y[1, :], mu = rep_vector(0.0, 5), n = 5, phi, sigma)
        end)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("`obs` is observed and read back, so it must name data", err)
end

"""
The called submodel's observed scan evaluates the same density as the scan
written in the caller, at model scope and per plate cell over nested-vector
groups, and its `ys_likelihood` sums to it.
"""
@testitem "slic: an observed @scan in a called submodel matches the inline density" tags=[:slic, :scan, :plate, :ragged, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    import BridgeStan
    using Random
    Random.seed!(11)
    stream = @slic begin
        @scan begin
            obs[1] ~ normal(mu[1], sigma / sqrt(1 - phi^2))
            for t in 2:n
                obs[t] ~ normal(mu[t] + phi * (obs[t-1] - mu[t-1]), sigma)
            end
        end
    end
    ys = [randn(3), randn(5), randn(2)]; mus = [randn(3), randn(5), randn(2)]
    called = @slic (; ys, mus) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            node ~ stream(; obs = ys[p], mu = mus[p], n = length(ys[p]), phi, sigma)
        end
    end
    inline = @slic (; ys, mus) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1)
        @plate for p in 1:3
            @scan begin
                ys[p][1] ~ normal(mus[p][1], sigma / sqrt(1 - phi^2))
                for t in 2:length(ys[p])
                    ys[p][t] ~ normal(mus[p][t] + phi * (ys[p][t-1] - mus[p][t-1]), sigma)
                end
            end
        end
    end
    gq(bm, th, seed) = BridgeStan.param_constrain(bm, th; include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(bm, seed))
    idx(bm, prefix) = findall(n -> startswith(n, prefix * "."), BridgeStan.param_names(bm; include_tp=true, include_gq=true))
    mktempdir() do dir
        cd(dir) do
            a = stan_instantiate(called).model
            b = stan_instantiate(inline).model
            for (phi, s) in ((0.3, 0.7), (-0.6, 1.4))
                th = BridgeStan.param_unconstrain(a, [phi, s])
                lpa = BridgeStan.log_density(a, th; propto=false, jacobian=false)
                @test lpa ≈ BridgeStan.log_density(b, th; propto=false, jacobian=false)
                @test sum(gq(a, th, 1)[idx(a, "ys_likelihood")]) ≈ lpa - (log(0.5) - s)
                @test length(idx(a, "ys_gen")) == 10
            end
        end
    end
end
