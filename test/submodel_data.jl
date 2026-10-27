"""
A sub-model's own data belongs to its call, as its parameters do (snag
`submodel-instanc-09f67aba`). Data a sub-model binds itself — fixed with
`Base.merge(sub, (; k = v))`, bound with `sub(; k = v)` before the call, or given
to `@slic (; k = v)` — is declared under the receiving LHS (`hx_k`). It used to
be declared as plain `k`: two calls fixing `rho_iso` to 0.3 and 0.9 emitted one
`real rho_iso;` that both read, and the 0.9 was dropped. Data the caller passes
by keyword keeps the caller's name, and two bindings that still declare one
data name with different values are refused.
"""
@testitem "slic: a sub-model's own data is scoped to its call" tags=[:slic, :submodel, :descriptor, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    import StanBlocks.stan: stan_data
    model(body, data) = StanBlocks.stan.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__)
    datablock(m) = stan_block(stan_code(m), "data")
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end

    # The reported shape: one sub-model, two calls, each with its own fixed value.
    sub = @slic begin
        sigma ~ exponential(1.0)
        rho_iso ~ lognormal(0.0, 1.0)
        return sigma * rep_vector(rho_iso, n)
    end
    a = Base.merge(sub, (; rho_iso = 0.3))
    b = Base.merge(sub, (; rho_iso = 0.9))
    y = [0.1, 0.2]
    two = model(:(begin hx ~ $a(; n = 2); hu ~ $b(; n = 2); y ~ normal(hx + hu, 1.0) end), (; y))
    code = stan_code(two)
    db = stan_block(code, "data")
    @test occursin("real hx_rho_iso;", db) && occursin("real hu_rho_iso;", db)
    @test !occursin(r"real rho_iso;", db)
    tp = stan_block(code, "transformed parameters")
    @test occursin("rep_vector(hx_rho_iso, 2)", tp) && occursin("rep_vector(hu_rho_iso, 2)", tp)
    d = stan_data(stan_model(two))
    @test d[:hx_rho_iso] == 0.3 && d[:hu_rho_iso] == 0.9 && !haskey(d, :rho_iso)
    @test stanc_compiles(two)
    # Each call reads its own value: the density uses 0.3 for `hx` and 0.9 for `hu`.
    let σ = [0.7, 1.4], p = instantiate(two)
        normlp(y, mu) = sum(@. -0.5 * (y - mu)^2 - 0.5 * log(2pi))
        lp = BridgeStan.log_density(p.model, log.(σ); propto=false, jacobian=false)
        @test lp ≈ -sum(σ) + normlp(y, fill(0.3σ[1] + 0.9σ[2], 2))
    end
    # The descriptor lists both as required inputs, and a traced model re-binds one by its name.
    desc = stan_descriptor(two)
    @test StanBlocks.required_inputs(desc) == (:hx_rho_iso, :hu_rho_iso, :y)
    @test stan_data(stan_model(two)(; hu_rho_iso = 1.5))[:hu_rho_iso] == 1.5
    # The same fixed model called twice declares the value once per call.
    same = model(:(begin hx ~ $a(; n = 2); hu ~ $a(; n = 2); y ~ normal(hx + hu, 1.0) end), (; y))
    let d = stan_data(stan_model(same))
        @test d[:hx_rho_iso] == 0.3 && d[:hu_rho_iso] == 0.3
    end

    # Bound with the call operator, or given to `@slic (; …)`; a vector's size is scoped too.
    subk = @slic begin
        z ~ normal(0.0, 1.0)
        return z * k
    end
    k2, k3 = subk(; k = 2.0), subk(; k = 3.0)
    called = model(:(begin hx ~ $k2(); hu ~ $k3(); y ~ normal(hx + hu, 1.0) end), (; y = 0.1))
    let d = stan_data(stan_model(called))
        @test d[:hx_k] == 2.0 && d[:hu_k] == 3.0
    end
    @test occursin("real hx = (hx_z * hx_k);", stan_code(called))
    @test stanc_compiles(called)
    subd = @slic (; k = 2.0, v = [1.0, 2.0, 3.0]) begin
        z ~ normal(0.0, 1.0)
        return z * k + sum(v)
    end
    defined = model(:(begin hx ~ subd(); y ~ normal(hx, 1.0) end), (; y = 0.1))
    @test occursin("int hx_v_n;", datablock(defined)) && occursin("vector[hx_v_n] hx_v;", datablock(defined))
    @test stan_data(stan_model(defined))[:hx_v] == [1.0, 2.0, 3.0]
    @test stanc_compiles(defined)

    # A caller's keyword keeps the caller's name; a literal keyword is inlined.
    caller = model(:(begin hx ~ subk(; k = kk); hu ~ subk(; k = 4.0); y ~ normal(hx + hu, 1.0) end), (; y = 0.1, kk = 5.0))
    @test occursin("real kk;", datablock(caller)) && !occursin("hx_k", stan_code(caller))
    @test occursin("real hu = (hu_z * 4.0);", stan_code(caller))

    # Nested calls compose the prefixes; a call in a plate cell takes the cell's namespace.
    outer = model(:(begin w ~ $k2(); return w end), (;))
    nested = model(:(begin o ~ $outer(); y ~ normal(o, 1.0) end), (; y = 0.1))
    @test stan_data(stan_model(nested))[:o_w_k] == 2.0
    @test stanc_compiles(nested)
    cells = model(:(begin rv ~ plate(; outer = 3) do i; z ~ $k2(); z end; y ~ normal(rv, 1.0) end), (; y = [0.1, 0.2, 0.3]))
    @test occursin("real rv_z_k;", datablock(cells))
    @test stanc_compiles(cells)

    # A named sub-model's positional value bound in Julia is its call's data; one
    # bound in the body is the caller's expression.
    @slic offset_by(slope) = begin
        intercept ~ normal(0, 1)
        return intercept + slope
    end
    named = model(:(begin mu ~ $(offset_by(0.5)); nu ~ offset_by(x); y ~ normal(mu + nu, 1.0) end), (; y = 0.1, x = 0.7))
    @test occursin("real mu = (mu_intercept + mu_slope);", stan_code(named))
    @test occursin("real nu = (nu_intercept + x);", stan_code(named))

    # One data name, one value: a model name equal to a call's data name must agree.
    clash = model(:(begin hx ~ $k2(); y ~ normal(hx + hx_k, 1.0) end), (; y = 0.1, hx_k = 5.0))
    @test occursin("Two bindings declare the data variable `hx_k` with different values: 2.0 and 5.0", errmsg(clash))
    agree = model(:(begin hx ~ $k2(); y ~ normal(hx + hx_k, 1.0) end), (; y = 0.1, hx_k = 2.0))
    @test stanc_compiles(agree)
end
