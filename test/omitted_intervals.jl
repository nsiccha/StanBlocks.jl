"""
Removing only a declared response keeps its upper-only interval statement.
Retained controls carry the draw's shape; interval evidence needs an observation.
"""
@testitem "slic: omitted upper-only intervals retain predictive shapes" tags=[:slic, :descriptor, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    declare(body, data; observations=(:y,)) =
        StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__, observations)
    body = quote
        mu ~ normal(0.0, 1.0)
        y ~ interval_censored(normal(mu, scale); upper=hi)
    end
    for (data, segments) in (
        ((; hi=0.5, scale=0.8), nothing),
        ((; hi=[0.5, 0.6], scale=0.8), nothing),
        ((; hi=0.5, scale=[0.8, 0.9]), nothing),
        ((; hi=Float64[], scale=0.8), nothing),
        ((; hi=[[0.5, 0.6], Float64[], [0.7]], scale=0.8), [2, 2, 3]),
        ((; hi=0.5, scale=[[0.8, 0.9], Float64[], [1.0]]), [2, 2, 3]),
    )
        m = declare(body, data)
        traced = stan_model(m)
        code = stan_code(traced)
        @test strip(stan_block(code, "parameters")) == "{\n}"
        @test strip(stan_block(code, "model")) == "{\n}"
        @test !occursin("y_likelihood", code)
        @test stanc_compiles(m)
        d = stan_descriptor(traced)
        @test stan_operation(d, :predict).outputs == (:y_gen,)
        output = only(o for o in d.outputs if o.name == :y_gen)
        @test output.generative == :draw
        @test output.source == :y
        @test output.segments == segments
    end
    # The same missing endpoint composes through a weighted interval and a
    # discrete base law. Weights carry no extra predictive dimension.
    for family in (
        :(weighted(interval_censored(normal(0.0, 1.0); upper=hi), w)),
        :(interval_censored(poisson(2.0); upper=hi)),
    )
        m = declare(:(y ~ $family), (; hi=[2.0, 3.0], w=[0.5, 1.0]))
        @test stanc_compiles(m)
        @test stan_operation(stan_descriptor(m), :predict).outputs == (:y_gen,)
        @test !occursin("y_likelihood", stan_code(m))
    end
    # A declaration does not make an interval likelihood into a latent prior.
    latent = declare(quote
        y ~ interval_censored(normal(0.0, 1.0); upper=hi)
        z ~ normal(y, 1.0)
    end, (; hi=0.5, z=0.1))
    @test_throws "priors are not supported" stan_code(latent)
    undeclared = declare(:(y ~ interval_censored(normal(0.0, 1.0); upper=hi)),
        (; hi=0.5); observations=())
    @test_throws "has no data" stan_code(undeclared)

    # Rebinding the retained upper bound updates the ragged output layout.
    traced = stan_model(declare(body, (; hi=[[0.5, 0.6], [0.7]], scale=0.8)))
    rebound = traced(; hi=[[0.5], [0.6, 0.7, 0.8]])
    @test only(o for o in stan_descriptor(rebound).outputs if o.name == :y_gen).segments == [1, 4]

    cell = declare(quote
        pred ~ plate(hi; outer=length(hi)) do hh
            y ~ interval_censored(normal(0.0, 1.0); upper=hh)
            hh
        end
    end, (; hi=[[0.5, 0.6], Float64[], [0.7]]); observations=(:pred_y,))
    @test stanc_compiles(cell)
    cd = stan_descriptor(cell)
    @test stan_operation(cd, :predict).outputs == (:pred_y_gen,)
    @test only(o for o in cd.outputs if o.name == :pred_y_gen).segments == [2, 2, 3]
end

@testitem "slic: omitted interval predictions draw the base law with one twin" tags=[:slic, :bridgestan, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    declare(body, data) =
        StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__, (:y,))
    body = :(y ~ interval_censored(normal(0.0, 1.0); upper=hi))
    for hi in ([0.0, 0.0, 0.0], [[0.0, 0.0], Float64[], [0.0]])
        m = declare(body, (; hi))
        d = stan_descriptor(m)
        p = stan_execute(d, :instantiate)
        @test LogDensityProblems.dimension(p) == 0
        pred = stan_execute(d, :predict; problem=p, draws=zeros(0, 128), seed=17)
        @test size(pred.y_gen) == (3, 128)
        @test any(pred.y_gen .> 0.0) # upper is evidence, not a predictive cutoff
        @test any(pred.y_gen .< 0.0)
        @test abs(sum(pred.y_gen) / length(pred.y_gen)) < 0.2
        control = declare(:(y ~ interval_censored(normal(0.0, 1.0); lower=lo, upper=hi)),
            (; lo=-10.0, hi))
        dc = stan_descriptor(control)
        pc = stan_execute(dc, :instantiate)
        reference = stan_execute(dc, :predict; problem=pc, draws=zeros(0, 128), seed=17)
        @test pred.y_gen == reference.y_gen
        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        raw = BridgeStan.param_constrain(p.model, Float64[]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(p.model, 17))
        original = findall(n -> startswith(n, "y.") || startswith(n, "y__obs_mem_"), names)
        twin = findall(n -> startswith(n, "y_gen"), names)
        @test length(original) == length(twin) == 3
        @test raw[original] == raw[twin]
    end
    # With y bound, the unchanged statement still measures (y, hi]. Its
    # explicit-endpoint equivalent must have exactly the same program.
    observed = declare(body, (; y=[-1.0, -0.5], hi=[0.5, 1.0]))
    explicit = declare(:(y ~ interval_censored(normal(0.0, 1.0), y, hi)),
        (; y=[-1.0, -0.5], hi=[0.5, 1.0]))
    @test stan_code(observed) == stan_code(explicit)
    p = instantiate(observed)
    using StanBlocks.SpecialFunctions: erf
    cdf(x) = (1 + erf(x / sqrt(2))) / 2
    expected = log(cdf(0.5) - cdf(-1.0)) + log(cdf(1.0) - cdf(-0.5))
    @test BridgeStan.log_density(p.model, Float64[]; propto=false) ≈ expected atol=1e-12
end
