"""
The producer can omit a response while retaining its distribution and ragged
predictors. The compiler owns shape, one predictive draw, and its public layout;
censoring still cannot introduce a sampled latent parameter.
"""
@testitem "slic: declared unbound ragged observations preserve shape and activity" tags=[:slic, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block
    declare(body, data; observations=(:y,)) =
        StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__, observations)
    dense = declare(quote
        mu ~ normal(0.0, 1.0)
        y ~ censored(normal, mu, 0.2; lower=lo)
    end, (; lo=[0.1, 0.1]))
    ragged = declare(quote
        sigma ~ exponential(1.0)
        y ~ normal(mu, sigma)
    end, (; mu=[[1.0, 2.0], [3.0]]))
    combined = declare(quote
        sigma ~ exponential(1.0)
        y ~ censored(normal(mu, sigma); lower=lo)
    end, (; mu=[[1.0, 2.0], Float64[], [3.0]], lo=[[0.1, 0.2], Float64[], [0.3]]))
    produced = declare(quote
        sigma ~ exponential(1.0)
        loc ~ plate(ts; outer=length(ts)) do t
            2.0 .* t
        end
        y ~ censored(normal(loc, sigma); lower=lo)
    end, (; ts=[[1.0, 2.0], [3.0]], lo=0.1))
    for (m, segments) in ((dense, nothing), (ragged, [2, 3]),
                          (combined, [2, 2, 3]), (produced, [2, 3]))
        traced = stan_model(m)
        code = stan_code(traced)
        @test strip(stan_block(code, "parameters")) == "{\n}"
        @test strip(stan_block(code, "model")) == "{\n}"
        @test !occursin("y_likelihood", code)
        d = stan_descriptor(traced)
        @test stan_operation(d, :predict).outputs == (:y_gen,)
        output = only(o for o in d.outputs if o.name == :y_gen)
        @test output.generative == :draw
        @test output.source == :y
        @test output.segments == segments
        @test !(:fit in getproperty.(d.operations, :name))
        @test !(:pointwise_loglik in getproperty.(d.operations, :name))
    end
    # Rebinding must recompute boundaries from current data, without retracing.
    rebound = stan_model(ragged)(; mu=[[1.0], [2.0, 3.0, 4.0]])
    @test only(o for o in stan_descriptor(rebound).outputs if o.name == :y_gen).segments == [1, 4]
    rebound_plate = stan_model(produced)(; ts=[[1.0], [2.0, 3.0, 4.0]])
    @test only(o for o in stan_descriptor(rebound_plate).outputs if o.name == :y_gen).segments == [1, 4]

    # A normal missing response can still be a latent parameter if observed
    # data depend on it. A censoring law has atoms and must still reject there.
    for response in (:(y ~ censored(normal, 0.0, 1.0; lower=lo)),
                     :(y ~ interval_censored(normal, lo, 10.0, 0.0, 1.0)))
        latent = declare(quote
            $response
            z ~ normal(y, 1.0)
        end, (; lo=0.1, z=1.0))
        @test_throws "priors are not supported" stan_code(latent)
    end
    undeclared = declare(quote
        y ~ censored(normal, 0.0, 1.0; lower=lo)
    end, (; lo=0.1); observations=())
    @test_throws "priors are not supported" stan_code(undeclared)

    # A scalar-event family takes one draw per group, rather than replicating
    # draws to the number of categories in each probability vector.
    categorical = declare(:(y ~ categorical(p)), (; p=[[0.3, 0.7], [1.0]]))
    cd = stan_descriptor(categorical)
    co = only(o for o in cd.outputs if o.name == :y_gen)
    @test co.segments === nothing
    @test co.type == :int

    # Preserve ordinary inferred support if a missing continuous response
    # actually becomes latent because observed data depend on it.
    latent_positive = declare(quote
        y ~ exponential(rate)
        z ~ normal(y, 1.0)
    end, (; rate=[[1.0, 2.0], [3.0]], z=[[1.1, 2.1], [3.1]]))
    lc = stan_code(latent_positive)
    @test occursin("vector<lower=0.0>", stan_block(lc, "parameters"))
    @test !occursin("y_gen", lc)
end

@testitem "slic: declared unbound ragged censoring stanc acceptance" tags=[:slic, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    for (family, data) in (
        (:(censored(normal, 0.0, 1.0; lower=lo)), (; lo=[0.1, 0.2])),
        (:(normal(mu, 1.0)), (; mu=[[1.0, 2.0], [3.0]])),
        (:(censored(normal(mu, 1.0); lower=lo)), (; mu=[[1.0, 2.0], Float64[], [3.0]], lo=[[0.1, 0.2], Float64[], [0.3]])),
        (:(censored(normal, 0.0, 1.0; lower=lo)), (; lo=[[0.1, 0.2], [0.3]])),
        (:(poisson(mu)), (; mu=[[1.0, 2.0], [3.0]])),
        (:(categorical(mu)), (; mu=[[0.3, 0.7], [1.0]])),
    )
        m = StanBlocks.SlicModel(:(y ~ $family), Dict{Symbol,Any}(pairs(data)), @__MODULE__, (:y,))
        @test occursin("y_gen", stan_code(m))
        @test stanc_compiles(m)
    end
end

@testitem "slic: declared unbound ragged censoring prediction execution" tags=[:slic, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    m = StanBlocks.SlicModel(quote
        sigma ~ exponential(1.0)
        loc ~ plate(ts; outer=length(ts)) do t
            2.0 .* t
        end
        y ~ censored(normal(loc, sigma); lower=lo, upper=hi)
    end, Dict{Symbol,Any}(
        :ts=>[[0.5, 1.0], Float64[], [1.5]],
        :lo=>[[0.1, 0.2], Float64[], [0.3]],
        :hi=>[[1.1, 2.1], Float64[], [3.1]],
    ), @__MODULE__, (:y,))
    d = stan_descriptor(m)
    p = stan_execute(d, :instantiate)
    @test LogDensityProblems.dimension(p) == 0
    pred = stan_execute(d, :predict; problem=p, draws=zeros(0, 6), seed=17)
    @test keys(pred) == (:y_gen,)
    @test size(pred.y_gen) == (3, 6)
    @test all([0.1, 0.2, 0.3] .<= pred.y_gen .<= [1.1, 2.1, 3.1])
    output = only(o for o in d.outputs if o.name == :y_gen)
    @test output.segments == [2, 2, 3]
    # The twin is an alias of the completed memory, so no second RNG is used.
    all_names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    raw = BridgeStan.param_constrain(p.model, Float64[]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 17))
    mem_indices = findall(n -> startswith(n, "y__obs_mem_"), all_names)
    twin_indices = findall(n -> startswith(n, "y_gen"), all_names)
    @test length(mem_indices) == 3
    @test raw[mem_indices] == raw[twin_indices]
end

"""
Per-cell unbound INTEGER responses (snag omitted-integer-f59f449a, Bruno TGI
prior): a `~`-defined `int[K]` plate cell is a sampled response, not an
ephemeral index array — it promotes to an outer collection and
forward-simulates in generated quantities, exactly like a continuous cell.
Deterministic int cells (findall/boolean-mask) keep inlining (plate-cell-int).
"""
@testitem "slic: per-cell unbound integer responses forward-simulate" tags=[:slic, :plate, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block

    @deffun begin
        "Category family mimicking Bruno `tgi_category`."
        @lhs @lpxf tcat_lpmf(y::int[n], r::vector[n])::real = sum(r)
        tcat_rng(r::vector[n])::int[n] = begin
            rv::int[n]
            for i in 1:n
                rv[i] = 1
            end
            rv
        end
        tcat_rng(int[n], r::vector[n])::int[n] = tcat_rng(r)
        "Binary family mimicking Bruno `tgi_response`."
        @lhs @lpxf tresp_lpmf(y::int[n], r::vector[n])::real = sum(r)
        tresp_rng(r::vector[n])::int[n] = begin
            rv::int[n]
            for i in 1:n
                rv[i] = 0
            end
            rv
        end
        tresp_rng(int[n], r::vector[n])::int[n] = tresp_rng(r)
    end

    # Group 2 is EMPTY — the empty-group acceptance. The cell mirrors the BRM
    # kernel count-form emission: bound ragged inputs slice per cell while the
    # omitted integer responses sample per cell beside a continuous response,
    # a per-cell parameter, computed intermediates, and a findall index.
    tcol = [[0.1, 0.2, 0.3], Float64[], [0.4, 0.5]]
    cmt = [[1, 2, 1], Int[], [2, 2]]
    m = @slic (; tcol, cmt, nsub = 3) begin
        pred ~ plate(tcol, cmt; outer = (nsub,)) do t, c
            gg ~ normal(0.0, 1.0)
            r = gg * t
            yy ~ normal(r, 1.0)
            cc ~ tcat(r)
            bb ~ tresp(r)
            ipk = findall(c .== 1)
            mu_pk = sum(r[ipk])
            r
        end
    end
    code = stan_code(m)
    @test strip(stan_block(code, "parameters")) == "{\n}"
    @test strip(stan_block(code, "model")) == "{\n}"
    gq = stan_block(code, "generated quantities")
    @test occursin(r"array\[sum\(pred_cc__pl_len_\d+\)\] int pred_cc__pl_mem_\d+;", gq)
    @test occursin(r"array\[sum\(pred_bb__pl_len_\d+\)\] int pred_bb__pl_mem_\d+;", gq)
    @test occursin("tcat_int_rng(", gq)
    @test occursin("tresp_int_rng(", gq)
    # The findall index still inlines at its use (plate-cell-int preserved).
    @test occursin("findall(", gq)
    @test !occursin("boolmask_idx", stan_block(code, "transformed data"))
    d = stan_descriptor(m)
    for prefix in ("pred_cc__pl_mem_", "pred_bb__pl_mem_")
        output = only(o for o in d.outputs if startswith(string(o.name), prefix))
        @test output.kind == :generated_quantity
        @test output.type == :int
        @test output.segments == [3, 3, 5]
    end
    # Rebinding recomputes the integer boundaries from current data.
    rebound = stan_model(m)(;
        tcol = [[0.1], [0.2, 0.3, 0.4], [0.5]],
        cmt = [[1], [2, 2, 1], [2]],
    )
    rd = stan_descriptor(rebound)
    for prefix in ("pred_cc__pl_mem_", "pred_bb__pl_mem_")
        output = only(o for o in rd.outputs if startswith(string(o.name), prefix))
        @test output.segments == [1, 4, 5]
    end

    # A fixed-width int cell collects to a dense `array[N, K] int` instead.
    dense = @slic (; nsub = 2, w = [0.5, -0.2, 0.1]) begin
        pred ~ plate(; outer = (nsub,)) do g
            cc ~ tcat(w)
            sum(w)
        end
    end
    dcode = stan_code(dense)
    @test occursin(
        r"array\[nsub, \w+\] int pred_cc;",
        stan_block(dcode, "generated quantities"),
    )
    dd = stan_descriptor(dense)
    dout = only(o for o in dd.outputs if o.name == :pred_cc)
    @test dout.kind == :generated_quantity
    @test dout.type == :int
    @test dout.segments === nothing

    # An int-array plate RESULT stays unsupported (MVP) — sampled int SIDE
    # cells promote, but the loop's own return has no int outer form yet.
    bad = @slic (; nsub = 2, w = [0.5, -0.2, 0.1]) begin
        pred ~ plate(; outer = (nsub,)) do g
            cc ~ tcat(w)
            cc
        end
    end
    @test_throws "cannot collect an int-array result" stan_code(bad)

    # A ragged int cell inside a called submodel names its own center in the
    # loud refusal (the vector wording stays pinned by the submodel item).
    @slic int_ragged_in_sub(dv) = begin
        b ~ plate(dv; outer = 3) do t
            z ~ tcat(t)
            sum(t)
        end
        return b
    end
    @test_throws "ragged int cells inside a called" stan_code(@slic (;
        dv = [[0.1, 0.2], [0.3], [0.4]], y = 0.3,
    ) begin
        b ~ int_ragged_in_sub(dv)
        y ~ normal(sum(b), 1.0)
    end)
end

@testitem "slic: per-cell unbound integer responses execute with zero dims" tags=[:slic, :plate, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles

    @deffun begin
        "Category family with real draws, mimicking Bruno `tgi_category`."
        @lhs @lpxf tcat_lpmf(y::int[n], r::vector[n])::real = sum(r)
        tcat_rng(r::vector[n])::int[n] = begin
            rv::int[n]
            for i in 1:n
                rho = normal_rng(r[i], 1.0)
                rv[i] = 4
                if rho < 0.0
                    rv[i] = 1
                end
            end
            rv
        end
        tcat_rng(int[n], r::vector[n])::int[n] = tcat_rng(r)
        "Binary family with real draws, mimicking Bruno `tgi_response`."
        @lhs @lpxf tresp_lpmf(y::int[n], r::vector[n])::real = sum(r)
        tresp_rng(r::vector[n])::int[n] = begin
            rv::int[n]
            for i in 1:n
                rv[i] = bernoulli_rng(0.5)
            end
            rv
        end
        tresp_rng(int[n], r::vector[n])::int[n] = tresp_rng(r)
    end

    m = @slic (; tcol = [[0.1, 0.2, 0.3], [0.4, 0.5]], nsub = 2) begin
        pred ~ plate(tcol; outer = (nsub,)) do t
            gg ~ normal(0.0, 1.0)
            r = gg * t
            cc ~ tcat(r)
            bb ~ tresp(r)
            r
        end
    end
    @test stanc_compiles(m)
    p = instantiate(stan_model(m))
    @test LogDensityProblems.dimension(p) == 0
    names = BridgeStan.param_names(p.model; include_tp = true, include_gq = true)
    cc_idx = findall(n -> startswith(n, "pred_cc__pl_mem_"), names)
    bb_idx = findall(n -> startswith(n, "pred_bb__pl_mem_"), names)
    r_idx = findall(n -> startswith(n, "pred_r__pl_mem_"), names)
    @test length(cc_idx) == 5
    @test length(bb_idx) == 5
    draw_at(seed) = BridgeStan.param_constrain(
        p.model, Float64[]; include_tp = true, include_gq = true,
        rng = BridgeStan.StanRNG(p.model, seed),
    )
    draw1, draw2 = draw_at(17), draw_at(17)
    @test draw1 == draw2
    @test all(x -> x == 1 || x == 4, draw1[cc_idx])
    @test all(x -> x == 0 || x == 1, draw1[bb_idx])
    # Distinct seeds draw distinctly (pinned on the continuous carrier, where
    # an accidental collision has probability zero).
    @test draw_at(18)[r_idx] != draw1[r_idx]
end
