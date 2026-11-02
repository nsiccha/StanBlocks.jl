"""
A plate cell or an annotated-loop body may bind a name that is also bound
outside it only when the binding is declared `local` (snag
`reflect-a-fixed-d777e837`, user decision `02bhgp2`): `local x …` declares a
cell binding that shadows the outer `x` in the whole body, as `local` does in a
Julia closure, and it emits exactly the program the same cell emits when no
outer `x` exists. Unmarked, such a binding is refused with the remedies, since
it is far more often a forgotten index (`y ~ …` for `y[i] ~ …`) than a new
variable. Before this, every do-form spelling crashed at root scope with an
internal emitter error (a per-cell LKJ prior next to a same-named data input
was the report), and inside a called sub-model a caller's keyword silently
replaced the cell's binding.
"""
@testitem "slic: a local cell binding shadows an outer name" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    errmsg(f) = try (stan_code(f()); "") catch e; sprint(showerror, e) end
    corr = [1.0 0.4; 0.4 1.0]
    cells = [zeros(2), zeros(2)]

    # The report: a per-cell correlation prior whose name a data field also uses.
    control = @slic (; K = 2, y = cells) begin
        out ~ plate(y) do yi
            Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
            yi ~ multi_normal_cholesky(rep_vector(0.0, K), Omega)
            nothing
        end
    end
    shadowing = @slic (; K = 2, y = cells, Omega = corr) begin
        out ~ plate(y) do yi
            local Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
            yi ~ multi_normal_cholesky(rep_vector(0.0, K), Omega)
            nothing
        end
    end
    unshadowed = @slic (; K = 2, y = cells) begin
        out ~ plate(y) do yi
            local Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
            yi ~ multi_normal_cholesky(rep_vector(0.0, K), Omega)
            nothing
        end
    end
    code = stan_code(control)
    @test occursin("cholesky_factor_corr[K] out_Omega;", stan_block(code, "parameters"))
    @test stan_code(shadowing) == code       # the data field changes nothing
    @test stan_code(unshadowed) == code      # `local` on an unbound name is harmless
    @test stanc_compiles(shadowing)

    # Unmarked, the same cell is refused with the three remedies.
    msg = errmsg(() -> @slic (; K = 2, y = cells, Omega = corr) begin
        out ~ plate(y) do yi
            Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
            yi ~ multi_normal_cholesky(rep_vector(0.0, K), Omega)
            nothing
        end
    end)
    @test occursin("plate `out`: the cell binds `Omega`, which is already bound outside it (a data input)", msg)
    @test occursin("local Omega::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)", msg)
    @test occursin("index it", msg)

    # A forgotten index is caught, not turned into a latent per-cell parameter.
    msg = errmsg(() -> @slic (; N = 2, y = [0.1, 0.2]) begin
        mu ~ normal(0.0, 1.0)
        out ~ plate(; outer = N) do i
            y ~ normal(mu, 1.0)
            nothing
        end
    end)
    @test occursin("the cell binds `y`, which is already bound outside it (a data input)", msg)

    # A shadowed model-scope parameter stays its own variable outside the cell.
    two = @slic (; y = [0.0, 1.0], z = 0.5) begin
        sigma ~ exponential(1.0)
        out ~ plate(y) do yi
            local sigma ~ exponential(1.0)
            yi ~ normal(0.0, sigma)
            nothing
        end
        z ~ normal(0.0, sigma)
    end
    code = stan_code(two)
    pars = stan_block(code, "parameters")
    @test occursin("real<lower=0.0> sigma;", pars) && occursin("out_sigma;", pars)
    model = stan_block(code, "model")
    @test occursin("normal(0.0, out_sigma[", model) && occursin("z ~ normal(0.0, sigma);", model)
    @test stanc_compiles(two)
    @test occursin("a model-scope parameter", errmsg(() -> @slic (; y = [0.0, 1.0]) begin
        sigma ~ exponential(1.0)
        out ~ plate(y) do yi
            sigma ~ exponential(1.0)
            yi ~ normal(0.0, sigma)
            nothing
        end
    end))

    # `local x = …`, collected as the cell result; the unused data input is not declared.
    assigned = @slic (; y = [0.0, 1.0], mu = 3.0) begin
        out ~ plate(y) do yi
            local mu = 2.0
            yi ~ normal(mu, 1.0)
            mu
        end
    end
    code = stan_code(assigned)
    @test occursin("real out_mu = 2.0;", stan_block(code, "transformed data"))
    @test !occursin(r"real mu;", stan_block(code, "data"))
    @test stanc_compiles(assigned)
end

@testitem "slic: local is refused where it cannot shadow" tags=[:slic, :plate, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    errmsg(f) = try (stan_code(f()); "") catch e; sprint(showerror, e) end
    # The binding shadows the outer name in the whole cell, so an earlier read
    # (or its own RHS) cannot see the outer value.
    @test occursin("`mu` is read before (or by) the `local mu` statement", errmsg(() -> @slic (; y = [0.1, 0.2], mu = 3.0) begin
        out ~ plate(y) do yi
            a = mu * 2.0
            local mu ~ normal(a, 1.0)
            yi ~ normal(mu, 1.0)
            nothing
        end
    end))
    @test occursin("`mu` is read before (or by) the `local mu` statement", errmsg(() -> @slic (; y = [0.1, 0.2], mu = 3.0) begin
        out ~ plate(y) do yi
            local mu ~ normal(mu, 1.0)
            yi ~ normal(mu, 1.0)
            nothing
        end
    end))
    # Model scope has no enclosing scope to shadow.
    @test occursin("not valid at model scope", errmsg(() -> @slic (; y = [0.1, 0.2]) begin
        local mu ~ normal(0.0, 1.0)
        y ~ normal(mu, 1.0)
    end))
    # A do-block parameter is already the cell's own name.
    @test occursin("`yi` is this cell's do-block parameter", errmsg(() -> @slic (; y = [0.1, 0.2]) begin
        out ~ plate(y) do yi
            local yi ~ normal(0.0, 1.0)
            nothing
        end
    end))
    # `local` takes one binding.
    @test occursin("`local` declares one binding of a name", errmsg(() -> @slic (; y = [0.1, 0.2]) begin
        out ~ plate(y) do yi
            local yi
            nothing
        end
    end))
end

@testitem "slic: local in an annotated loop body" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    errmsg(f) = try (stan_code(f()); "") catch e; sprint(showerror, e) end
    # Unmarked, a bare binding of a data name is still the forgotten-index refusal,
    # which now also names `local`.
    msg = errmsg(() -> @slic (; N = 2, y = [0.1, 0.2]) begin
        mu ~ normal(0.0, 1.0)
        @plate for i in 1:N
            y ~ normal(mu, 1.0)
        end
    end)
    @test occursin("`y` is bound outside the loop", msg) && occursin("index it (`y[i]`)", msg)
    @test occursin("(`local y ~ normal(mu, 1.0)`)", msg)
    # Declared `local`, it is a stored per-iteration local, as for an unbound name.
    shadowing = @slic (; N = 2, y = [0.1, 0.2], obs = [1.0, 2.0]) begin
        mu ~ normal(0.0, 1.0)
        @plate for i in 1:N
            local y ~ normal(mu, 1.0)
            obs[i] ~ normal(y, 1.0)
        end
    end
    plain = @slic (; N = 2, obs = [1.0, 2.0]) begin
        mu ~ normal(0.0, 1.0)
        @plate for i in 1:N
            y ~ normal(mu, 1.0)
            obs[i] ~ normal(y, 1.0)
        end
    end
    @test stan_code(shadowing) == stan_code(plain)
    @test occursin("vector[N] loop_i_y;", stan_block(stan_code(shadowing), "parameters"))
    @test stanc_compiles(shadowing)
end

@testitem "slic: a sub-model caller's keyword does not reach a cell binding" tags=[:slic, :plate, :submodel, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block
    model(body, data) = StanBlocks.stan.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    unmarked = @slic begin
        out ~ plate(y) do yi
            sigma = 1.0
            yi ~ normal(0.0, sigma)
            nothing
        end
    end
    marked = @slic begin
        out ~ plate(y) do yi
            local sigma = 1.0
            yi ~ normal(0.0, sigma)
            nothing
        end
    end
    y = [0.1, 0.2]
    # It used to override the cell's binding in every cell; now it is refused.
    msg = errmsg(model(:(begin h ~ $unmarked(; y = y, sigma = 2.0) end), (; y)))
    @test occursin("the cell binds `sigma`, which is already bound outside it (a data input or caller keyword)", msg)
    # Declared `local`, the cell keeps its own value; the caller's keyword is unused.
    code = stan_code(model(:(begin h ~ $marked(; y = y, sigma = 2.0) end), (; y)))
    @test occursin("real h_out_sigma = 1.0;", stan_block(code, "transformed data"))
    @test occursin("normal(0.0, h_out_sigma)", stan_block(code, "model"))
    # Without the keyword, both spellings emit the same program as before.
    plain = stan_code(model(:(begin h ~ $unmarked(; y = y) end), (; y)))
    @test plain == stan_code(model(:(begin h ~ $marked(; y = y) end), (; y)))
    @test occursin("real h_out_sigma = 1.0;", stan_block(plain, "transformed data"))
end
