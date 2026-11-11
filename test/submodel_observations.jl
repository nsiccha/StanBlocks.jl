@testitem "slic: sub-models observe their own scoped data" tags=[:slic, :submodel, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    import StanBlocks.stan: stan_data
    model(body, data=(; y=0.1)) = StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)

    sub = @slic (; obs=[1.0, 2.0]) begin
        m ~ normal(0.0, 1.0)
        obs ~ normal(m, 1.0)
        return m
    end
    fitted = model(:(begin s ~ $sub(); y ~ normal(s, 1.0) end))
    code = stan_code(fitted)
    @test stan_data(stan_model(fitted))[:s_obs] == [1.0, 2.0]
    @test occursin("s_obs ~ normal(s_m, 1.0)", stan_block(code, "model"))
    @test occursin("s_obs_gen", stan_block(code, "generated quantities"))
    @test occursin("s_obs_likelihood", stan_block(code, "generated quantities"))
    @test stanc_compiles(fitted)
    let m=0.3, p=instantiate(fitted)
        normal_lp(y, mu) = sum(@. -0.5 * (y - mu)^2 -0.5 * log(2pi))
        expected = normal_lp([m], 0.0) + normal_lp([1.0, 2.0], m) + normal_lp([0.1], m)
        @test BridgeStan.log_density(p.model, [m]; propto=false, jacobian=false) ≈ expected
    end

    missing_sub = @slic (; obs=[1.0, missing, 2.0]) begin
        m ~ normal(0.0, 1.0)
        obs ~ normal(m, 1.0)
        return m
    end
    imputed = model(:(begin s ~ $missing_sub(); y ~ normal(s, 1.0) end))
    code = stan_code(imputed)
    @test stan_data(stan_model(imputed))[:s_obs_obs] == [1.0, 2.0]
    @test occursin("s_obs_obs_gen", stan_block(code, "generated quantities"))
    @test occursin("s_obs_obs_likelihood", stan_block(code, "generated quantities"))
    @test stanc_compiles(imputed)

    loop_sub = @slic (; obs=[0.1, 0.2, 0.3, 0.4]) begin
        m ~ normal(0.0, 1.0)
        @plate for i in 1:4
            obs[i] ~ normal(m, 1.0)
        end
        return m
    end
    looped = model(:(begin s ~ $loop_sub(); y ~ normal(s, 1.0) end))
    code = stan_code(looped)
    @test occursin("s_obs_gen", code)
    @test stanc_compiles(looped)

    # Repeated whole-vector observations need a separate per-cell layout for
    # their predictive twins. Refuse them rather than overwrite one shared twin.
    cells = model(:(begin
        means ~ plate(; outer=2) do i
            s ~ $sub()
            s
        end
        y ~ normal(sum(means), 1.0)
    end))
    message = try
        stan_code(cells)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("cannot repeatedly observe the whole data binding `means_s_obs`", message)
    @test occursin("index the observed data by the loop's cell", message)

    # The plate traces a cell twice. A shared bound input read by each cell
    # remains data of the original shape instead of becoming per-cell storage.
    read_sub = @slic (; obs=[1.0, 2.0]) begin
        m ~ normal(0.0, 1.0)
        return m + sum(obs)
    end
    cells = model(:(begin
        means ~ plate(; outer=2) do i
            s ~ $read_sub()
            s
        end
        y ~ normal(sum(means), 1.0)
    end))
    code = stan_code(cells)
    @test stan_data(stan_model(cells))[:means_s_obs] == [1.0, 2.0]
    @test occursin("vector[means_s_obs_n] means_s_obs;", stan_block(code, "data"))
    @test stanc_compiles(cells)
end
