"""
A self-bound endpoint belongs to the observed censoring event. Removing only
that response removes the endpoint, while independent thresholds still clamp.
"""
@testitem "slic: omitted self-bound censoring preserves the source and predictive shape" tags=[:slic, :descriptor, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    declare(body, data; observations=(:y,)) =
        StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__, observations)
    for rhs in (
        :(censored(normal(mu, sigma); lower=y)),
        :(censored(normal, mu, sigma; upper=y)),
        :(censored(normal(mu, sigma); lower=y, upper=y)),
        :(weighted(censored(normal(mu, sigma); lower=y), w)),
        :(weighted(censored, w, normal, mu, sigma; upper=y)),
    ), (mu, segments) in (
        (0.0, nothing), ([0.0, 0.5], nothing), (Float64[], nothing),
        ([[0.0, 0.5], Float64[], [-0.5]], [2, 2, 3]),
    )
        m = declare(:(y ~ $rhs), (; mu, sigma=0.8, w=0.5))
        code = stan_code(m)
        @test strip(stan_block(code, "parameters")) == "{\n}"
        @test strip(stan_block(code, "model")) == "{\n}"
        @test !occursin("y_likelihood", code)
        @test stanc_compiles(m)
        d = stan_descriptor(m)
        @test stan_operation(d, :predict).outputs == (:y_gen,)
        output = only(o for o in d.outputs if o.name == :y_gen)
        @test output.source == :y
        @test output.generative == :draw
        @test output.segments == segments
    end
    # The other endpoint retains its own shape, even with scalar family args.
    for rhs in (:(censored(normal(0.0, 1.0); lower=y, upper=hi)),
                :(censored(normal(0.0, 1.0); lower=lo, upper=y)))
        m = declare(:(y ~ $rhs), (; lo=[-0.5, -0.2], hi=[0.5, 0.2]))
        @test stanc_compiles(m)
        @test only(o for o in stan_descriptor(m).outputs if o.name == :y_gen).size in ((:hi_n,), (:lo_n,))
    end
    # Plate discovery and promoted flat storage keep the logical observation.
    cell = declare(quote
        pred ~ plate(mu; outer=length(mu)) do mm
            y ~ censored(normal(mm, 0.8); lower=y)
            mm
        end
    end, (; mu=[[0.0, 0.5], Float64[], [-0.5]]); observations=(:pred_y,))
    @test stanc_compiles(cell)
    cd = stan_descriptor(cell)
    @test stan_operation(cd, :predict).outputs == (:pred_y_gen,)
    @test only(o for o in cd.outputs if o.name == :pred_y_gen).segments == [2, 2, 3]

    for endpoint in (:lower, :upper)
        rhs = Expr(:call, :censored, Expr(:parameters, Expr(:kw, endpoint, :y)),
            :(normal(mu, 0.8)))
        body = quote
            mu ~ normal(0.0, 1.0)
            y ~ $rhs
        end
        forecast = declare(body, (; y=StanBlocks.stan.maybecv(:y)))
        @test stanc_compiles(forecast)
        @test occursin("real mu;", stan_block(stan_code(forecast), "parameters"))
        @test !occursin("y_likelihood", stan_code(forecast))
    end
    discrete = declare(:(y ~ censored(poisson(rate); lower=y)), (; rate=[1.0, 2.0]))
    @test stanc_compiles(discrete)
    @test occursin("poisson", stan_code(discrete))
    traced = stan_model(declare(:(y ~ censored(normal(mu, 0.8); upper=y)),
        (; mu=[[0.0, 0.5], [0.2]])))
    rebound = traced(; mu=[[0.0], [0.2, 0.3, 0.5]])
    @test only(o for o in stan_descriptor(rebound).outputs if o.name == :y_gen).segments == [1, 4]

    # Observation declarations do not make a censored latent prior legal.
    latent = declare(quote
        y ~ censored(normal(0.0, 1.0); lower=y)
        z ~ normal(y, 1.0)
    end, (; z=0.1))
    @test_throws "priors are not supported" stan_code(latent)
    undeclared = declare(:(y ~ censored(normal(0.0, 1.0); lower=y)), (;); observations=())
    @test_throws "Could not find y" stan_code(undeclared)
    # Missing independent bounds and response-dependent family arguments still
    # need their values; only an endpoint identical to the observation erases.
    @test_throws "Could not find lo" stan_code(declare(:(y ~ censored(normal(0.0, 1.0); lower=lo)), (;)))
    @test_throws "Could not find y" stan_code(declare(:(y ~ censored(normal(y, 1.0); lower=y)), (;)))
    @test_throws "Could not find y" stan_code(declare(:(y ~ censored(normal(0.0, 1.0); lower=y-1)), (;)))
end

@testitem "slic: self-bound censoring draws once and preserves observed evidence" tags=[:slic, :bridgestan, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    declare(body, data) = StanBlocks.SlicModel(body, Dict{Symbol,Any}(pairs(data)), @__MODULE__, (:y,))
    for rhs in (:(censored(normal(mu, 0.8); lower=y)),
                :(censored(normal(mu, 0.8); upper=y))),
        mu in ([-0.5, 0.0, 0.5], [[-0.5, 0.0], Float64[], [0.5]])
        m = declare(:(y ~ $rhs), (; mu))
        base = declare(:(y ~ normal(mu, 0.8)), (; mu))
        @test stan_code(m) == stan_code(base)
        d = stan_descriptor(m)
        p = stan_execute(d, :instantiate)
        @test LogDensityProblems.dimension(p) == 0
        pred = stan_execute(d, :predict; problem=p, draws=zeros(0, 128), seed=17)
        @test size(pred.y_gen) == (3, 128)
        @test any(pred.y_gen .> 0.0)
        @test any(pred.y_gen .< 0.0)
        @test abs(sum(pred.y_gen) / length(pred.y_gen)) < 0.2
        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        raw = BridgeStan.param_constrain(p.model, Float64[]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(p.model, 17))
        original = findall(n -> startswith(n, "y.") || startswith(n, "y__obs_mem_"), names)
        twin = findall(n -> startswith(n, "y_gen"), names)
        @test length(original) == length(twin) == 3
        @test raw[original] == raw[twin]
    end
    # Erasing one endpoint leaves the independent threshold's clamp intact.
    mixed = StanBlocks.SlicModel(quote
        yl ~ censored(normal(mu, 0.8); lower=yl, upper=hi)
        yu ~ censored(normal(mu, 0.8); lower=lo, upper=yu)
    end, Dict{Symbol,Any}(:mu=>[0.0, 0.0], :lo=>[-0.5, -0.2], :hi=>[0.5, 0.2]),
        @__MODULE__, (:yl, :yu))
    d = stan_descriptor(mixed)
    p = stan_execute(d, :instantiate)
    @test LogDensityProblems.dimension(p) == 0
    pred = stan_execute(d, :predict; problem=p, draws=zeros(0, 128), seed=19)
    @test all(pred.yl_gen .<= [0.5, 0.2])
    @test all(pred.yu_gen .>= [-0.5, -0.2])
    @test any(pred.yl_gen .== [0.5, 0.2])
    @test any(pred.yu_gen .== [-0.5, -0.2])

    using StanBlocks.SpecialFunctions: erf
    cdf(x) = (1 + erf(x / sqrt(2))) / 2
    for (rhs, tail) in ((:(censored(normal(mu, 0.8); lower=y)), cdf),
                        (:(censored(normal(mu, 0.8); upper=y)), x -> cdf(-x)))
        body = quote
            mu ~ normal(0.0, 1.0)
            y ~ $rhs
        end
        y = [-0.5, 0.2, 1.1]
        m = declare(body, (; y))
        observed_problem = instantiate(m)
        location = 0.3
        value, gradient = BridgeStan.log_density_gradient(observed_problem.model, [location]; propto=false)
        oracle(x) = -0.5*x^2 - 0.5*log(2pi) + sum(log(tail((v-x)/0.8)) for v in y)
        @test value ≈ oracle(location) atol=1e-12
        @test only(gradient) ≈ (oracle(location+1e-5)-oracle(location-1e-5))/2e-5 atol=1e-8
    end
end
