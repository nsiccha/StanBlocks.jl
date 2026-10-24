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

@testitem "slic: declared missing plate cells keep public predictive names" tags=[:slic, :plate, :descriptor, :stanc, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    data = Dict{Symbol,Any}(
        :ts => [[0.2, 0.8], [0.3]],
        :lo => [[0.05, 0.05], [0.05]],
        :obs => [[0.4, 0.7], [0.2]],
        :nsub => 2,
    )
    for family in (:(censored(normal, tt, scale; lower=ll)), :(normal(tt, scale)))
        m = StanBlocks.SlicModel(quote
            scale ~ exponential(1.0)
            pred ~ plate(ts, lo; outer=(nsub,)) do tt, ll
                z ~ $family
                tt
            end
            fitted ~ plate(obs, ts; outer=(nsub,)) do yy, tt
                yy ~ normal(tt, scale)
                tt
            end
        end, data, @__MODULE__, (:pred_z,))
        code = stan_code(m)
        gq = stan_block(code, "generated quantities")
        @test occursin(r"pred_z_gen\s*=\s*pred_z__pl_mem_\d+;", gq)
        @test !occursin("pred_z_likelihood", code)
        @test stanc_compiles(m)

        d = stan_descriptor(m)
        outputs = Dict(o.name => o for o in d.outputs)
        @test outputs[:pred_z_gen].generative == :draw
        @test outputs[:pred_z_gen].source == :pred_z
        @test outputs[:pred_z_gen].segments == [2, 3]
        @test outputs[:obs_gen].segments == [2, 3]
        @test outputs[:obs_likelihood].segments == [2, 3]
        @test :pred_z_gen in stan_operation(d, :predict).outputs
        @test :obs_likelihood in stan_operation(d, :pointwise_loglik).outputs
        @test :fit in getproperty.(d.operations, :name)

        rebound = stan_model(m)(; ts=[[0.2], [0.3, 0.4]], lo=[[0.05], [0.05, 0.05]],
            obs=[[0.4], [0.2, 0.3]])
        rd = stan_descriptor(rebound)
        @test only(o for o in rd.outputs if o.name == :pred_z_gen).segments == [1, 3]
    end

    censored_cell = quote
        pred ~ plate(ts, lo; outer=(nsub,)) do tt, ll
            z ~ censored(normal, tt, 1.0; lower=ll)
            tt
        end
    end
    # Refused: without a declaration the missing cell sample is a latent prior,
    # while censored threshold atoms are observation-only (stanblocks-use §8).
    undeclared = StanBlocks.SlicModel(censored_cell, data, @__MODULE__)
    @test_throws "censored priors are not supported" stan_code(undeclared)

    latent = StanBlocks.SlicModel(quote
        pred ~ plate(ts, lo, obs; outer=(nsub,)) do tt, ll, yy
            z ~ censored(normal, tt, 1.0; lower=ll)
            yy ~ normal(z, 1.0)
            tt
        end
    end, data, @__MODULE__, (:pred_z,))
    # Refused: declaring a stem cannot turn an actually fitted latent into
    # an observation; its censored law remains invalid as a parameter prior.
    @test_throws "censored priors are not supported" stan_code(latent)
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
Per-cell unbound INTEGER responses (snag omitted-integer-f59f449a, a prior-only
downstream model): a `~`-defined `int[K]` plate cell is a sampled response, not an
ephemeral index array — it promotes to an outer collection and
forward-simulates in generated quantities, exactly like a continuous cell.
Deterministic int cells (findall/boolean-mask) keep inlining (plate-cell-int).
"""
@testitem "slic: per-cell unbound integer responses forward-simulate" tags=[:slic, :plate, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block

    @deffun begin
        "Synthetic ordinal category family."
        @lhs @lpxf tcat_lpmf(y::int[n], r::vector[n])::real = sum(r)
        tcat_rng(r::vector[n])::int[n] = begin
            rv::int[n]
            for i in 1:n
                rv[i] = 1
            end
            rv
        end
        tcat_rng(int[n], r::vector[n])::int[n] = tcat_rng(r)
        "Synthetic binary response family."
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
        "Synthetic ordinal category family with real draws."
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
        "Synthetic binary response family with real draws."
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

"""
A values-free `maybecv(:y)` (snag held-out-observa-58a2eb13, design decision
1kj6hva) binds an observation with no values as a quantity to predict given an
earlier fit: a forecast. What the observation reads stays a parameter, `y_gen`
takes the shape of its other arguments, and there is neither a `y` input nor a
`y_likelihood`. It is reached from the fit program's own body by rebinding data,
including through a `plate` positional.
"""
@testitem "slic: a values-free maybecv observation is predicted given a fit" tags=[:slic, :plate, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block
    maybecv = StanBlocks.stan.maybecv
    ops(d) = getproperty.(d.operations, :name)
    output(d, name) = only(o for o in d.outputs if o.name == name)

    body = quote
        b ~ normal(0.0, 1.0)
        s ~ exponential(1.0)
        y ~ normal(b .+ x, s)
    end
    fit = StanBlocks.SlicModel(body, Dict{Symbol,Any}(:x => [0.4, 0.5], :y => [0.1, 0.2]), @__MODULE__)
    forecast = fit(; y = maybecv(:y))
    code = stan_code(forecast)
    @test occursin("real b;", stan_block(code, "parameters"))
    @test occursin("real<lower=0.0> s;", stan_block(code, "parameters"))
    @test !occursin("y", stan_block(code, "model"))
    @test !occursin("y", stan_block(code, "data"))
    @test occursin("vector[x_n] y = normal_vector_rng(x_n, (b + x), s);", code)
    @test occursin("vector[x_n] y_gen = y;", code)
    @test !occursin("y_likelihood", code)
    d = stan_descriptor(forecast)
    @test ops(d) == (:transpile, :instantiate, :predict)
    @test StanBlocks.required_inputs(d) == (:x,)
    @test output(d, :y_gen).generative == :draw
    @test output(d, :y_gen).source == :y
    @test output(d, :b).kind == :parameter
    # Omitting `y` is still the prior program, and the fit program is unchanged.
    prior = StanBlocks.SlicModel(body, Dict{Symbol,Any}(:x => [0.4, 0.5]), @__MODULE__, (:y,))
    @test strip(stan_block(stan_code(prior), "parameters")) == "{\n}"
    @test occursin("y ~ normal((b + x), s);", stan_block(stan_code(fit), "model"))

    # A second, bound observation keeps its likelihood, so the program is fitted.
    mixed = StanBlocks.SlicModel(quote
        mu ~ normal(0.0, 1.0); sy ~ exponential(1.0); sz ~ exponential(1.0)
        y ~ normal(mu .+ x, sy)
        z ~ normal(mu, sz)
    end, Dict{Symbol,Any}(:x => [0.1, 0.2], :z => [0.3, 0.4, 0.5], :y => maybecv(:y)), @__MODULE__)
    md = stan_descriptor(mixed)
    @test :fit in ops(md)
    @test output(md, :y_gen).generative == :draw
    @test output(md, :sy).kind == :parameter
    @test !occursin("y_likelihood", stan_code(mixed))

    # Ragged and censored: the arguments give the groups and the bounds.
    ragged = StanBlocks.SlicModel(quote
        s ~ exponential(1.0)
        y ~ censored(normal(mu, s); lower = lo)
    end, Dict{Symbol,Any}(:mu => [[1.0, 2.0], [3.0]], :lo => [[0.1, 0.2], [0.3]], :y => maybecv(:y)), @__MODULE__)
    rd = stan_descriptor(ragged)
    @test ops(rd) == (:transpile, :instantiate, :predict)
    @test output(rd, :y_gen).segments == [2, 3]
    @test output(rd, :s).kind == :parameter

    # A plate positional bound to the mark: the cell's observation is the draw,
    # named after the data input as in the fit program, and the cell's values
    # (and what they read) stay parameters and transformed parameters.
    cell_body = quote
        b ~ normal(0.0, 1.0)
        logsigma ~ normal(-1.0, 1.0)
        sigma = exp(logsigma)
        mu ~ plate(y, t; outer = (nsub,)) do y, t
            m = b .+ 2.0 .* t
            y ~ normal(m, sigma)
            m
        end
    end
    cell_data = Dict{Symbol,Any}(:t => [[0.1, 0.2, 0.3], [0.4]], :nsub => 2)
    cell_fit = StanBlocks.SlicModel(cell_body, merge(cell_data, Dict{Symbol,Any}(:y => [[1.0, 2.0, 3.0], [4.0]])), @__MODULE__)
    cell_forecast = cell_fit(; y = maybecv(:y))
    cc = stan_code(cell_forecast)
    @test occursin("real b;", stan_block(cc, "parameters"))
    @test occursin("real logsigma;", stan_block(cc, "parameters"))
    @test !occursin("for(", stan_block(cc, "model"))
    @test occursin("for(", stan_block(stan_code(cell_fit), "model"))
    @test occursin(r"y_gen\s*=\s*mu_y__pl_mem_\d+;", stan_block(cc, "generated quantities"))
    cd = stan_descriptor(cell_forecast)
    @test ops(cd) == (:transpile, :instantiate, :predict)
    @test !(:y in StanBlocks.required_inputs(cd))
    @test output(cd, :y_gen).source == :y
    @test output(cd, :y_gen).segments == [3, 4]
    fit_d = stan_descriptor(cell_fit)
    @test output(fit_d, :y_gen).segments == [3, 4]
    rebound = stan_model(cell_forecast)(; t = [[0.1], [0.2, 0.3]])
    @test output(stan_descriptor(rebound), :y_gen).segments == [1, 3]
    # The count may come from another positional instead of `outer`.
    counted = StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0)
        mu ~ plate(y, t) do y, t
            m = b .+ t
            y ~ normal(m, 1.0)
            m
        end
    end, Dict{Symbol,Any}(:t => [[0.1, 0.2], [0.3]], :y => maybecv(:y)), @__MODULE__)
    @test output(stan_descriptor(counted), :y_gen).segments == [2, 3]
    # A dense per-cell observation.
    dense_cell = StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0); s ~ exponential(1.0)
        theta ~ plate(y, x; outer = n) do y, x
            y ~ normal(b + x, s)
        end
    end, Dict{Symbol,Any}(:x => [0.1, 0.2, 0.3], :n => 3, :y => maybecv(:y)), @__MODULE__)
    dd = stan_descriptor(dense_cell)
    @test ops(dd) == (:transpile, :instantiate, :predict)
    @test output(dd, :y_gen).dims == (3,)
    @test output(dd, :s).kind == :parameter
    # The annotated loop's model-scope array is the observation under its own name.
    looped = StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0); s ~ exponential(1.0)
        @plate for i in 1:n
            y[i] ~ normal(b + x[i], s)
        end
    end, Dict{Symbol,Any}(:x => [0.1, 0.2, 0.3], :n => 3, :y => maybecv(:y)), @__MODULE__)
    ld = stan_descriptor(looped)
    @test ops(ld) == (:transpile, :instantiate, :predict)
    @test output(ld, :y_gen).source == :y
    @test output(ld, :b).kind == :parameter
    @test !occursin("for(", stan_block(stan_code(looped), "model"))

    # Refused: a mark no `~` observes would silently change nothing (dev §1).
    @test_throws "no `z ~ …` statement observes it" stan_code(fit(; z = maybecv(:z)))
    # Refused: a values-free observation has no values to read, only to draw.
    @test_throws "has no values to read" stan_code(StanBlocks.SlicModel(quote
        s ~ exponential(1.0); y ~ normal(rep_vector(0.0, length(y)), s)
    end, Dict{Symbol,Any}(:y => maybecv(:y)), @__MODULE__))
    # Refused: the mark names the observation it is bound to.
    @test_throws "write `y = maybecv(:y)`" stan_code(fit(; y = maybecv(:w)))
    # Refused: a traced model re-binds data only, and the mark changes the program.
    @test_throws "cannot be data" stan_model(fit)(; y = maybecv(:y))
    # Refused: a marked positional has no slices, so only a whole `y ~ …` observes it.
    @test_throws "must be observed whole" stan_code(StanBlocks.SlicModel(quote
        mu ~ plate(y, t) do y, t
            y[1] ~ normal(0.0, 1.0)
            t
        end
    end, Dict{Symbol,Any}(:t => [[0.1, 0.2], [0.3]], :y => maybecv(:y)), @__MODULE__))
    # Refused: a plate with no other positional and no `outer` has no cell count.
    @test_throws "pass `outer=N`" stan_code(StanBlocks.SlicModel(quote
        s ~ exponential(1.0)
        mu ~ plate(y) do y
            y ~ normal(0.0, s)
        end
    end, Dict{Symbol,Any}(:y => maybecv(:y)), @__MODULE__))
end

@testitem "slic: a values-free maybecv forecast stanc acceptance" tags=[:slic, :plate, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    maybecv = StanBlocks.stan.maybecv
    @test stanc_compiles(StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0); s ~ exponential(1.0); y ~ normal(b .+ x, s)
    end, Dict{Symbol,Any}(:x => [0.4, 0.5], :y => maybecv(:y)), @__MODULE__))
    @test stanc_compiles(StanBlocks.SlicModel(quote
        s ~ exponential(1.0); y ~ censored(normal(mu, s); lower = lo)
    end, Dict{Symbol,Any}(:mu => [[1.0, 2.0], [3.0]], :lo => [[0.1, 0.2], [0.3]], :y => maybecv(:y)), @__MODULE__))
    @test stanc_compiles(StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0)
        logsigma ~ normal(-1.0, 1.0)
        mu ~ plate(y, t; outer = (nsub,)) do y, t
            m = b .+ 2.0 .* t
            y ~ normal(m, exp(logsigma))
            m
        end
    end, Dict{Symbol,Any}(:t => [[0.1, 0.2, 0.3], [0.4]], :nsub => 2, :y => maybecv(:y)), @__MODULE__))
end

@testitem "slic: a values-free maybecv forecast executes from fit draws" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    maybecv = StanBlocks.stan.maybecv
    # Fit-shaped unconstrained draws feed the forecast: with b = 5 and a tiny
    # scale every predicted entry sits at its mean.
    forecast = StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0); s ~ exponential(1.0); y ~ normal(b .+ x, s)
    end, Dict{Symbol,Any}(:x => [0.4, 0.5], :y => maybecv(:y)), @__MODULE__)
    d = stan_descriptor(forecast)
    p = stan_execute(d, :instantiate)
    @test LogDensityProblems.dimension(p) == 2
    draws = repeat([5.0, log(1e-4)], 1, 3)
    pred = stan_execute(d, :predict; problem = p, draws, seed = 3)
    @test keys(pred) == (:y_gen,)
    @test size(pred.y_gen) == (2, 3)
    @test all(abs.(pred.y_gen .- [5.4, 5.5]) .< 0.01)

    cell = StanBlocks.SlicModel(quote
        b ~ normal(0.0, 1.0)
        logsigma ~ normal(-1.0, 1.0)
        mu ~ plate(y, t; outer = (nsub,)) do y, t
            m = b .+ 2.0 .* t
            y ~ normal(m, exp(logsigma))
            m
        end
    end, Dict{Symbol,Any}(:t => [[0.1, 0.2, 0.3], [0.4]], :nsub => 2, :y => maybecv(:y)), @__MODULE__)
    cd = stan_descriptor(cell)
    cp = stan_execute(cd, :instantiate)
    @test LogDensityProblems.dimension(cp) == 2
    cpred = stan_execute(cd, :predict; problem = cp, draws = [5.0, log(1e-4)], seed = 3)
    @test all(abs.(cpred.y_gen .- (5.0 .+ 2.0 .* [0.1, 0.2, 0.3, 0.4])) .< 0.01)
end
