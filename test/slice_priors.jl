"""
One vector parameter can take different prior families on different coordinates:
slice statements (`tau[1:2] ~ …`) and `tau ~ vcat(…)` fold every coordinate's
support into the ONE Stan declaration, and re-draws stay inside it. Before this an
indexed prior never reached the declaration, so `tau[1] ~ exponential(1.5)`
emitted an unbounded `vector[2] tau` (decision 2026-10-16T14-13-42-706-0c3ako4,
snag heterogeneous-pe-7668c283).
"""
@testitem "slic: slice statements fold per-coordinate support into one declaration" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # Every coordinate bounded the same way: one scalar bound.
    uniform = @slic (; y=0.3) begin
        tau::vector[2]
        tau[1] ~ exponential(1.5)
        tau[2] ~ normal(0.0, 1.0; lower=0.)
        y ~ normal(sum(tau), 1.)
    end
    code = stan_code(uniform)
    @test occursin("vector<lower=0.0>[2] tau;", stan_block(code, "parameters"))
    @test occursin("tau[1] ~ exponential(1.5);", stan_block(code, "model"))
    @test occursin("tau[2] ~ normal(0.0, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(uniform)

    # Mixed support: a literal per-coordinate bound vector; an unbounded
    # coordinate is Stan's `negative_infinity()` (identity transform).
    mixed = @slic (; y=0.3, rate=[1.5, 3.0]) begin
        tau::vector[3]
        tau[1:2] ~ exponential(rate)
        tau[3] ~ normal(0.0, 1.0)
        y ~ normal(sum(tau), 1.)
    end
    code = stan_code(mixed)
    @test occursin("vector<lower=[0.0, 0.0, negative_infinity()]'>[3] tau;", code)
    @test occursin("tau[1:2] ~ exponential(rate);", code)
    @test stanc_compiles(mixed)

    # A `flat` declaration states the support (Stan's own style); a slice whose
    # family has a wider support is narrowed to it.
    flat_body = quote
        tau::vector[3] ~ flat(; lower=0.)
        tau[1:2] ~ exponential(rate)
        tau[3] ~ normal(0.0, 1.0)
    end
    fitted = StanBlocks.SlicModel(Base.remove_linenums!(Expr(:block, flat_body.args..., :(y ~ normal(sum(tau), 1.)))),
        Dict{Symbol,Any}(:y => 0.3, :rate => [1.5, 3.0]), @__MODULE__)
    code = stan_code(fitted)
    @test occursin("vector<lower=0.0>[3] tau;", stan_block(code, "parameters"))
    @test occursin("tau[3] ~ normal(0.0, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(fitted)
    # Prior-only: `flat` has nothing to draw; each slice re-draws its own block
    # inside the declared support (a truncated draw for the half-normal).
    prior = StanBlocks.SlicModel(flat_body, Dict{Symbol,Any}(:rate => [1.5, 3.0]), @__MODULE__)
    code = stan_code(prior)
    gq = stan_block(code, "generated quantities")
    @test strip(stan_block(code, "parameters")) == "{\n}"
    @test occursin("vector<lower=0.0>[3] tau;", gq)
    @test occursin("tau[3] = lower_conditioning_normal_rng(0.0, 0.0, 1.0);", gq)
    @test stanc_compiles(prior)

    # The intersection of a declaration's and a slice's support, per bound.
    capped = @slic (; y=0.3) begin
        tau::vector[3] ~ flat(; upper=5.)
        tau[1:2] ~ exponential(2.)
        tau[3] ~ normal(0., 1.)
        y ~ normal(sum(tau), 1.)
    end
    @test occursin("vector<lower=[0.0, 0.0, negative_infinity()]', upper=5.0>[3] tau;", stan_code(capped))
    @test stanc_compiles(capped)

    # Data indices: the per-coordinate bound is built in transformed data, so it
    # stays correct when the data are re-bound.
    indexed = @slic (; y=0.3, pos=[1, 3], free=[2]) begin
        tau::vector[3]
        tau[pos] ~ exponential(1.)
        tau[free] ~ normal(0., 1.)
        y ~ normal(sum(tau), 1.)
    end
    code = stan_code(indexed)
    @test occursin("vector<lower=tau__lower>[3] tau;", stan_block(code, "parameters"))
    @test occursin("tau__lower[pos[tau__lower_i]] = 0.0;", stan_block(code, "transformed data"))
    @test occursin("tau__lower[free[tau__lower_i]] = negative_infinity();", stan_block(code, "transformed data"))
    @test stanc_compiles(indexed)

    # Prior-only bare declaration: the re-draws honour each block's support.
    bare_prior = @slic begin
        tau::vector[2]
        tau[1] ~ exponential(1.5)
        tau[2] ~ normal(0.0, 1.0; lower=0.)
    end
    gq = stan_block(stan_code(bare_prior), "generated quantities")
    @test occursin("tau[1] = exponential_rng(1.5);", gq)
    @test occursin("tau[2] = lower_conditioning_normal_rng(0.0, 0.0, 1.0);", gq)

    # refused: each coordinate takes exactly one prior (decision 0c3ako4)
    err = try
        stan_code(@slic (; y=0.3) begin
            tau::vector[3]
            tau[1] ~ exponential(1.)
            tau[1:2] ~ normal(0., 1.)
            y ~ normal(sum(tau), 1.)
        end)
        nothing
    catch e
        e
    end
    @test err isa StanBlocks.StanBlocksError
    @test occursin("`tau[1]` is sampled twice", sprint(showerror, err))

    # refused: a whole-vector prior already gives every coordinate its prior
    # (decision 0c3ako4); the message names the slice and `vcat` spellings.
    err = try
        stan_code(@slic (; y=0.3) begin
            tau::vector[3] ~ normal(0., 1.)
            tau[1] ~ exponential(1.)
            y ~ normal(sum(tau), 1.)
        end)
        nothing
    catch e
        e
    end
    @test err isa StanBlocks.StanBlocksError
    @test occursin("already has a whole-vector prior", sprint(showerror, err))
    @test occursin("vcat", sprint(showerror, err))

    # A slice of a matrix that keeps the declaration's support is unchanged...
    rows = @slic (; y=0.3) begin
        tau::matrix[2, 2]
        tau[1, :] ~ normal(0., 1.)
        tau[2, :] ~ normal(1., 1.)
        y ~ normal(sum(tau), 1.)
    end
    @test occursin("matrix[2, 2] tau;", stan_code(rows))
    @test stanc_compiles(rows)
    # ...but one bringing its own support is refused: Stan declares a
    # per-coordinate bound on a vector, and dropping the bound (the old
    # behaviour) samples outside the family's support.
    err = try
        stan_code(@slic (; y=0.3) begin
            tau::matrix[2, 2]
            tau[1, :] ~ exponential(1.)
            tau[2, :] ~ normal(0., 1.)
            y ~ normal(sum(tau), 1.)
        end)
        nothing
    catch e
        e
    end
    @test err isa StanBlocks.StanBlocksError
    @test occursin("per coordinate on a vector", sprint(showerror, err))

    # refused: Stan declares an affine transform for the whole variable, never
    # for a slice of it.
    err = try
        stan_code(@slic (; y=0.3) begin
            tau::vector[2]
            tau[1] ~ normal(0., 1.; multiplier=2.)
            tau[2] ~ normal(0., 1.)
            y ~ normal(sum(tau), 1.)
        end)
        nothing
    catch e
        e
    end
    @test err isa StanBlocks.StanBlocksError
    @test occursin("affine transform belongs to the whole declaration", sprint(showerror, err))
end

@testitem "slic: vcat(...) is one statement per vector and lowers to slice statements" tags=[:slic, :regression, :stanc, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    symbolic = @slic (; y=0.3, rate=[1.5, 3.0]) begin
        tau ~ vcat(exponential(rate), normal(0., 1.; lower=0.))
        y ~ normal(sum(tau), 1.)
    end
    code = stan_code(symbolic)
    @test occursin("vector<lower=0.0>[(rate_n + 1)] tau;", stan_block(code, "parameters"))
    @test occursin("tau[1:rate_n] ~ exponential(rate);", stan_block(code, "model"))
    @test occursin("tau[(rate_n + 1)] ~ normal(0.0, 1.0);", stan_block(code, "model"))
    @test stanc_compiles(symbolic)

    # The blocks tile the vector, so the carrier needs no default fill.
    mixed = @slic (; y=0.3, rate=[1.5, 3.0]) begin
        tau ~ vcat(exponential(rate), normal(0., 1.))
        y ~ normal(sum(tau), 1.)
    end
    td = stan_block(stan_code(mixed), "transformed data")
    @test occursin("tau__lower[tau__lower_i] = 0.0;", td)
    @test occursin("tau__lower[(rate_n + 1)] = negative_infinity();", td)
    @test count("tau__lower[", td) == 2
    @test stanc_compiles(mixed)

    # The generated-prior shape a downstream regression package emits: literal
    # hyperparameter vectors, one statement for the whole coefficient vector.
    literal = @slic (; y=0.3) begin
        tau::vector[6] ~ vcat(exponential([1.5, 1.5, 3.0, 1.5]), normal([0., 0.], [1., 0.5]; lower=0.))
        y ~ normal(sum(tau), 1.)
    end
    @test occursin("vector<lower=0.0>[6] tau;", stan_code(literal))
    @test stanc_compiles(literal)

    # Same program as the hand-written slices.
    lowered = @slic (; y=0.3) begin
        tau ~ vcat(exponential([1.5, 3.0]), normal(0., 1.))
        y ~ normal(sum(tau), 1.)
    end
    sliced = @slic (; y=0.3) begin
        tau::vector[3]
        tau[1:2] ~ exponential([1.5, 3.0])
        tau[3] ~ normal(0., 1.)
        y ~ normal(sum(tau), 1.)
    end
    @test stan_code(sliced) == stan_code(lowered)

    # One whole-name statement: `Base.merge` replaces a bare declaration's prior
    # in one step and keeps its declared length.
    base = @slic begin
        tau::vector[n]
        y ~ normal(sum(tau), 1.)
    end
    merged = Base.merge(base, :(tau ~ vcat(exponential([1.5, 3.0]), normal(0., 1.; lower=0.))))(; n=3, y=0.5)
    @test occursin("vector<lower=0.0>[n] tau;", stan_code(merged))
    @test stanc_compiles(merged)
    err = try
        stan_code(Base.merge(base, :(tau ~ vcat(exponential([1.5, 3.0]), normal(0., 1.))))(; n=4, y=0.5))
        nothing
    catch e
        e
    end
    @test err isa StanBlocks.StanBlocksError
    @test occursin("the blocks cover 3 coordinates, not 4", sprint(showerror, err))

    # An observed vcat: one slice observation per block, with `y_gen` and a
    # per-coordinate `y_likelihood`.
    observed = @slic (; y=[0.3, 0.4, 0.5]) begin
        s ~ exponential(1.)
        y ~ vcat(exponential(s; n=2), normal(0.0, s))
    end
    code = stan_code(observed)
    gq = stan_block(code, "generated quantities")
    @test occursin("y[1:2] ~ exponential(s);", stan_block(code, "model"))
    @test occursin("vector[y_n] y_likelihood;", gq)
    @test occursin("y_likelihood[3] = normal_lpdfs(y[3], 0.0, s);", gq)
    @test occursin("y_gen[3] = normal_rng(0.0, s);", gq)
    @test stanc_compiles(observed)
    d = stan_descriptor(observed)
    @test :pointwise_loglik in [o.name for o in d.operations]
    lik = only(o for o in d.outputs if o.name == :y_likelihood)
    @test lik.generative == :pointwise_loglik && lik.source == :y
end

@testitem "slic: @scan setup and recurrence keep their own support" tags=[:slic, :scan, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # The setup's support used to bound the WHOLE carried array, silently
    # truncating the unbounded recurrence to `vector<lower=0.0>[3] x`.
    literal = @slic (; y=randn(3)) begin
        @scan begin
            x[1] ~ exponential(1.)
            for t in 2:3
                x[t] ~ normal(x[t-1], 1.)
            end
        end
        y ~ normal(x, 1.)
    end
    code = stan_code(literal)
    @test occursin("vector<lower=[0.0, negative_infinity(), negative_infinity()]'>[3] x;", code)
    @test stanc_compiles(literal)

    symbolic = @slic (; y=randn(4), T=4) begin
        @scan begin
            x[1] ~ exponential(1.)
            for t in 2:T
                x[t] ~ normal(x[t-1], 1.)
            end
        end
        y ~ normal(x, 1.)
    end
    code = stan_code(symbolic)
    @test occursin("vector<lower=x__lower>[T] x;", stan_block(code, "parameters"))
    @test occursin("x__lower[1] = 0.0;", stan_block(code, "transformed data"))
    @test stanc_compiles(symbolic)

    # Setup and recurrence that agree keep one scalar bound.
    shared = @slic (; y=randn(3)) begin
        @scan begin
            x[1] ~ exponential(1.)
            for t in 2:3
                x[t] ~ lognormal(log(x[t-1]), 1.)
            end
        end
        y ~ normal(x, 1.)
    end
    @test occursin("vector<lower=0.0>[3] x;", stan_code(shared))

    # A @scan nested in a @plate for keeps a support its statements share.
    nested = @slic (; y=randn(6), S=2) begin
        @plate for j in 1:S
            @scan begin
                h[1] ~ normal(0., 1.)
                for t in 2:3
                    h[t] ~ normal(h[t-1], 1.)
                end
            end
        end
        y ~ normal(to_vector(h), 1.)
    end
    @test stanc_compiles(nested)
    # Capability gap, not a refusal of the shape: a per-coordinate bound across
    # a nested scan's promoted axes is not built yet (todo StanBlocks 2026-10-16T17-32-53-378-1eyqyrh).
    @test_broken try
        stan_code(@slic (; y=randn(6), S=2) begin
            @plate for j in 1:S
                @scan begin
                    h[1] ~ exponential(1.)
                    for t in 2:3
                        h[t] ~ normal(h[t-1], 1.)
                    end
                end
            end
            y ~ normal(to_vector(h), 1.)
        end)
        true
    catch
        false
    end
end

@testitem "slic: per-coordinate support constrains and draws natively" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    import BridgeStan
    rate = [1.5, 3.0]
    y = 0.3
    mixed = @slic (; y, rate) begin
        tau ~ vcat(exponential(rate), normal(0., 1.))
        y ~ normal(sum(tau), 1.)
    end
    mktempdir() do dir
        cd(dir) do
            prob = stan_instantiate(mixed)
            @test LogDensityProblems.dimension(prob) == 3
            # Coordinates 1:2 are log-transformed, coordinate 3 is the identity.
            for x in ([0.1, -0.3, -1.2], [-0.7, 0.4, 2.0])
                tau = BridgeStan.param_constrain(prob.model, x)
                @test tau[1:2] ≈ exp.(x[1:2])
                @test tau[3] ≈ x[3]
            end
            # Log density (with Jacobian) up to a constant, against a hand computation.
            hand(x) = begin
                tau = [exp(x[1]), exp(x[2]), x[3]]
                -sum(rate .* tau[1:2]) - tau[3]^2 / 2 - (y - sum(tau))^2 / 2 + x[1] + x[2]
            end
            a, b = [0.1, -0.3, -1.2], [-0.7, 0.4, 2.0]
            @test LogDensityProblems.logdensity(prob, a) - LogDensityProblems.logdensity(prob, b) ≈ hand(a) - hand(b)

            # Prior-only `flat`-declared vector: every draw stays inside lower=0.
            prior = @slic (; rate) begin
                tau::vector[3] ~ flat(; lower=0.)
                tau[1:2] ~ exponential(rate)
                tau[3] ~ normal(0., 1.)
            end
            pprob = stan_instantiate(prior)
            @test LogDensityProblems.dimension(pprob) == 0
            rng = BridgeStan.StanRNG(pprob.model, 1234)
            draws = reduce(hcat, [BridgeStan.param_constrain(pprob.model, Float64[];
                include_tp=true, include_gq=true, rng) for _ in 1:400])
            @test all(>=(0), draws)
            @test 0.6 < sum(draws[3, :]) / 400 < 0.95    # half-normal mean ≈ 0.80
        end
    end
end
