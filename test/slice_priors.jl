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

    # Blocks of literal length fix the coordinates even when the declaration
    # spells its length as data, so literal per-block bounds stay a literal
    # bound vector on the declaration, not a `transformed data` carrier filled
    # element by element (snag vcat-per-coordin-201e2b14).
    sized = @slic (; y=0.3) begin
        n_coef = 2
        beta::vector[n_coef] ~ vcat(uniform(0.2, 0.8; lower=0.2, upper=0.8), uniform(0.3, 0.9; lower=0.3, upper=0.9))
        y ~ normal(sum(beta), 1.)
    end
    code = stan_code(sized)
    @test occursin("vector<lower=[0.2, 0.3]', upper=[0.8, 0.9]'>[n_coef] beta;", stan_block(code, "parameters"))
    @test !occursin("beta__lower", code) && !occursin("beta__upper", code)
    @test occursin("beta[2] ~ uniform(0.3, 0.9);", stan_block(code, "model"))
    @test stanc_compiles(sized)
    open_coord = @slic (; y=0.3, n=2) begin
        beta::vector[n] ~ vcat(normal(0., 1.), uniform(0., 2.; lower=0., upper=2.))
        y ~ normal(sum(beta), 1.)
    end
    code = stan_code(open_coord)
    @test occursin("vector<lower=[negative_infinity(), 0.0]', upper=[positive_infinity(), 2.0]'>[n] beta;", code)
    @test !occursin("beta__lower", code)
    @test stanc_compiles(open_coord)

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
    mixed_merged = Base.merge(base, :(tau ~ vcat(exponential([1.5, 3.0]), normal(0., 1.))))(; n=3, y=0.5)
    @test occursin("vector<lower=[0.0, 0.0, negative_infinity()]'>[n] tau;", stan_code(mixed_merged))
    @test stanc_compiles(mixed_merged)
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

# Before snag a-vcat-prior-fam-4a4fae26 a cell refused `tau ~ vcat(…)` ("not
# supported yet; declare `tau` at model scope"), and its other spellings — slice
# statements on a bare cell declaration, or the bare declaration alone —
# re-declared the cell over its promoted storage and failed with `tracetype not
# defined for ::vector[Colon(), ::int]`.
@testitem "slic: a vcat(...) prior in a plate or loop cell" tags=[:slic, :plate, :regression, :stanc, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    # The reporter's shape: per-stratum scale priors, every coordinate `lower=0`.
    shared = @slic (; n_terms=2, n_strata=3, y=[0.4, -0.2, 1.1]) begin
        tau_s ~ plate(; outer=(n_strata,)) do s
            tau::vector[n_terms] ~ vcat(exponential(0.5; lower=0.0), lognormal(0.0, 0.5; lower=0.0))
            tau
        end
        y ~ normal(to_vector(tau_s[1, :]), to_vector(tau_s[2, :]))
    end
    code = stan_code(shared)
    @test occursin("matrix<lower=0.0>[n_terms, n_strata] tau_s_tau;", stan_block(code, "parameters"))
    model = stan_block(code, "model")
    @test occursin("tau_s_tau[1, plate_i__pl_1] ~ exponential(0.5);", model)
    @test occursin("tau_s_tau[2, plate_i__pl_1] ~ lognormal(0.0, 0.5);", model)
    @test stanc_compiles(shared)

    # Mixed support: a transformed-data bound shaped like the cell storage.
    mixed = @slic (; n_strata=3, y=[0.4, -0.2, 1.1]) begin
        tau_s ~ plate(; outer=n_strata) do s
            tau ~ vcat(normal(0.0, 1.0), exponential(1.0))
            tau
        end
        y ~ normal(to_vector(tau_s[1, :]), to_vector(tau_s[2, :]))
    end
    code = stan_code(mixed)
    @test occursin("matrix<lower=tau_s_tau__lower>[2, n_strata] tau_s_tau;", stan_block(code, "parameters"))
    td = stan_block(code, "transformed data")
    @test occursin("tau_s_tau__lower[1, plate_i__pl_1] = negative_infinity();", td)
    @test occursin("tau_s_tau__lower[2, plate_i__pl_1] = 0.0;", td)
    @test stanc_compiles(mixed)
    out = only(o for o in stan_descriptor(mixed).outputs if o.name == :tau_s_tau)
    @test out.dims == (2, 3)
    @test isequal(out.constraint_values.lower, [-Inf -Inf -Inf; 0.0 0.0 0.0])

    # Slice statements on a bare cell declaration are the same program.
    slices = @slic (; n_terms=2, n_strata=3, y=[0.4, -0.2, 1.1]) begin
        tau_s ~ plate(; outer=(n_strata,)) do s
            tau::vector[n_terms]
            tau[1] ~ exponential(0.5)
            tau[2] ~ lognormal(0.0, 0.5)
            tau
        end
        y ~ normal(to_vector(tau_s[1, :]), to_vector(tau_s[2, :]))
    end
    @test stan_code(slices) == stan_code(shared)
    # A bare cell declaration alone is an improper-flat cell parameter.
    bare = @slic (; n_strata=3, y=[0.4, -0.2, 1.1]) begin
        tau_s ~ plate(; outer=n_strata) do s
            tau::vector[2]
            tau
        end
        y ~ normal(to_vector(tau_s[1, :]), 1.0)
    end
    code = stan_code(bare)
    @test occursin("matrix[2, n_strata] tau_s_tau;", stan_block(code, "parameters"))
    @test !occursin("tau_s_tau[", stan_block(code, "model"))
    @test stanc_compiles(bare)

    # Prior only: the cells re-draw in generated quantities, each block inside
    # its own support.
    prior = @slic (; n_strata=3) begin
        tau_s ~ plate(; outer=n_strata) do s
            tau ~ vcat(normal(0.0, 1.0; lower=0.0), exponential(1.0))
            tau
        end
    end
    gq = stan_block(stan_code(prior), "generated quantities")
    @test occursin("tau_s_tau[1, plate_i__pl_1] = lower_conditioning_normal_rng(0.0, 0.0, 1.0);", gq)
    @test occursin("tau_s_tau[2, plate_i__pl_1] = exponential_rng(1.0);", gq)
    @test stanc_compiles(prior)

    # Annotated loops: a `@plate for` local, its indexed-array spelling, and a
    # `@scan` step local.
    local_cell = @slic (; S=3, y=[0.4, -0.2, 1.1]) begin
        @plate for s in 1:S
            tau ~ vcat(normal(0.0, 1.0), exponential(1.0))
            y[s] ~ normal(tau[1], tau[2])
        end
    end
    code = stan_code(local_cell)
    @test occursin("matrix<lower=loop_s_tau__lower>[2, S] loop_s_tau;", stan_block(code, "parameters"))
    @test occursin("loop_s_tau[2, s] ~ exponential(1.0);", stan_block(code, "model"))
    @test stanc_compiles(local_cell)
    array_cell = @slic (; S=3, y=[0.4, -0.2, 1.1]) begin
        @plate for s in 1:S
            tau[:, s] ~ vcat(normal(0.0, 1.0), exponential(1.0))
            y[s] ~ normal(sum(tau[:, s]), 1.0)
        end
    end
    @test occursin("matrix<lower=tau__lower>[2, S] tau;", stan_block(stan_code(array_cell), "parameters"))
    @test stanc_compiles(array_cell)
    step = @slic (; T=4, y=[0.4, -0.2, 1.1, 0.3]) begin
        @scan begin
            x[1] ~ normal(0.0, 1.0)
            for t in 2:T
                tau ~ vcat(normal(0.0, 1.0), exponential(1.0))
                x[t] ~ normal(x[t-1] + tau[1], tau[2])
            end
        end
        y ~ normal(x, 1.0)
    end
    @test occursin("matrix<lower=loop_t_tau__lower>[2, (T - 1)] loop_t_tau;", stan_block(stan_code(step), "parameters"))
    @test stanc_compiles(step)

    # A plate in a called submodel, with support every coordinate shares.
    scales = @slic begin
        tau_s ~ plate(; outer=n) do s
            tau ~ vcat(exponential(0.5), lognormal(0.0, 0.5))
            tau
        end
        return tau_s
    end
    called = @slic (; n=3, y=[0.4, -0.2, 1.1]) begin
        t ~ scales(; n)
        y ~ normal(to_vector(t[1, :]), to_vector(t[2, :]))
    end
    code = stan_code(called)
    @test occursin("matrix<lower=0.0>[2, n] t_tau_s_tau;", stan_block(code, "parameters"))
    @test occursin("t_tau_s_tau[2, t_plate_i__pl_1] ~ lognormal(0.0, 0.5);", stan_block(code, "model"))
    @test stanc_compiles(called)

    # Not built yet. Per-coordinate support in a submodel's plate (todo
    # `StanBlocks/todos/2026-10-25T18-37-35-165-1i9cmkt`):
    mixed_scales = @slic begin
        tau_s ~ plate(; outer=n) do s
            tau ~ vcat(normal(0.0, 1.0), exponential(1.0))
            tau
        end
        return tau_s
    end
    @test_broken transpiles(@slic((; n=3, y=[0.4, -0.2, 1.1]), begin
        t ~ mixed_scales(; n)
        y ~ normal(to_vector(t[1, :]), to_vector(t[2, :]))
    end); re=false)
    # A `vcat` local in a nested annotated loop, like a typed local there (todo
    # `StanBlocks/todos/2026-10-25T18-37-35-254-1ich1nr`):
    @test_broken transpiles(@slic((; J=2, S=3, y=randn(2, 3)), begin
        @plate for j in 1:J
            @plate for s in 1:S
                tau ~ vcat(normal(0.0, 1.0), exponential(1.0))
                y[j, s] ~ normal(tau[1], tau[2])
            end
        end
    end); re=false)
    # Observing a plate positional's cell with `vcat` (todo
    # `StanBlocks/todos/2026-10-25T18-37-35-329-1i5w32n`):
    @test_broken transpiles(@slic((; Y=randn(2, 3)), begin
        mu ~ normal(0.0, 1.0)
        r ~ plate(EachCol(Y); outer=3) do yi
            yi ~ vcat(normal(mu, 1.0), normal(mu, 2.0))
        end
    end); re=false)
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
    @test occursin("matrix[3, S] h;", stan_block(stan_code(nested), "parameters"))

    # A recurrence bound that reads the loop index varies along it, even with
    # no setup statement to differ from. It used to fold as one scalar bound,
    # `vector<lower=L[t]>[3] x`, naming `t` outside its loop.
    indexed = @slic (; y=randn(3), L=[0., 1., 2.]) begin
        @scan begin
            for t in 1:3
                x[t] ~ normal(0., 1.; lower=L[t])
            end
        end
        y ~ normal(x, 1.)
    end
    code = stan_code(indexed)
    @test occursin("vector<lower=x__lower>[3] x;", stan_block(code, "parameters"))
    @test occursin("x__lower[t] = L[t];", stan_block(code, "transformed data"))
    @test stanc_compiles(indexed)

    # Nested in a @plate for, each cell's setup and recurrence keep their own
    # support: a bound matrix over the promoted axes (todo
    # StanBlocks 2026-10-16T17-32-53-378-1eyqyrh). It used to be refused, and
    # before that the setup's support bounded every coordinate.
    per_cell = @slic (; y=randn(6), S=2) begin
        @plate for j in 1:S
            @scan begin
                h[1] ~ exponential(1.)
                for t in 2:3
                    h[t] ~ normal(h[t-1], 1.)
                end
            end
        end
        y ~ normal(to_vector(h), 1.)
    end
    code = stan_code(per_cell)
    @test occursin("matrix<lower=h__lower>[3, S] h;", stan_block(code, "parameters"))
    td = stan_block(code, "transformed data")
    @test occursin("matrix[3, S] h__lower;", td)
    @test occursin("h__lower[1, j] = 0.0;", td)
    @test occursin("h__lower[t, j] = negative_infinity();", td)
    @test stanc_compiles(per_cell)

    # A bound may read data through the loop indices, and every enclosing
    # axis gets its own coordinates (`array[S] matrix[3, R]`, indexed `[j, t, i]`).
    axes = @slic (; y=randn(3), S=2, R=2, L=[0., 1.], U=[2., 3., 4.]) begin
        @plate for i in 1:R, j in 1:S
            @scan begin
                h[1] ~ normal(0., 1.; lower=L[j])
                for t in 2:3
                    h[t] ~ normal(h[t-1], 1.; upper=U[t])
                end
            end
        end
        y ~ normal(h[2][:, 1], 1.)
    end
    code = stan_code(axes)
    @test occursin("array[S] matrix<lower=h__lower, upper=h__upper>[3, R] h;", stan_block(code, "parameters"))
    td = stan_block(code, "transformed data")
    @test occursin("h__lower[j, 1, i] = L[j];", td)
    @test occursin("h__upper[j, t, i] = U[t];", td)
    @test stanc_compiles(axes)
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
