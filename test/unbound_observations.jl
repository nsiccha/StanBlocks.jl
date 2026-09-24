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
