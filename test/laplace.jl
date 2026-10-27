"""
`z ~ laplace_marginal(prior) do z … end` integrates a Gaussian latent out with
Stan's embedded Laplace approximation (snag `laplace-marginal-b2fb4189`, design
decision `1wf36ma`). The model block gets one compiler-written
`target += laplace_marginal(...)`; the latent is drawn in generated quantities by
`laplace_latent_rng`; the do block replays there, so its observation keeps its
`y_gen` / `y_likelihood` twins and the descriptor still sees it as observed.
"""
@testitem "slic: laplace_marginal integrates a Gaussian latent out" tags=[:slic, :laplace, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    rng = Xoshiro(1)
    X = randn(rng, 12, 4); y = randn(rng, 12)
    m = @slic (; y, X, k=4) begin
        s ~ exponential(1.0)
        tau ~ exponential(1.0)
        z::vector[k] ~ laplace_marginal(normal(0.0, tau)) do z
            y ~ normal(X * z, s)
        end
    end
    code = stan_code(m)
    params = stan_block(code, "parameters")
    model = stan_block(code, "model")
    gq = stan_block(code, "generated quantities")
    @test !occursin(r"\bz;", params)
    @test occursin(r"(?s)target \+= laplace_marginal\(closure_\d+, \(y, X, s\), k, closure_\d+, \(tau, k\)\);", model)
    @test !occursin("y ~", model)
    @test occursin(r"(?s)vector\[k\] z = laplace_latent_rng\(closure_\d+, \(y, X, s\), k, closure_\d+, \(tau, k\)\);", gq)
    @test occursin("y_likelihood = normal_lpdfs(y, (X * z), s)", gq)
    @test occursin("y_gen = normal_vector_rng(y_n, (X * z), s)", gq)
    # The likelihood function: the do block with each observation's density.
    @test occursin(r"(?s)real closure_\d+\(\s*vector laplace_theta,\s*vector y,\s*matrix X,\s*real s\s*\)", code)
    @test occursin("real laplace_lp1 = normal_lpdf(y | (X * z), s);", code)
    @test stanc_compiles(m)

    d = stan_descriptor(m)
    @test [op.name for op in d.operations] == [:transpile, :instantiate, :fit, :predict, :pointwise_loglik]
    yi = only(i for i in d.inputs if i.name === :y)
    @test yi.observed && !yi.held_out
    zo = only(o for o in d.outputs if o.name === :z)
    @test zo.kind === :generated_quantity
    @test only(o for o in d.outputs if o.name === :y_gen).generative === :draw

    # Integrating the latent out keeps every output name of the sampled spelling.
    sampled = @slic (; y, X, k=4) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        z::vector[k] ~ normal(0.0, tau)
        eta = X * z
        y ~ normal(eta, s)
    end
    integrated = @slic (; y, X, k=4) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        z::vector[k] ~ laplace_marginal(normal(0.0, tau)) do z
            eta = X * z
            y ~ normal(eta, s)
        end
    end
    output_names(m) = sort([o.name for o in stan_descriptor(m).outputs])
    @test output_names(integrated) == output_names(sampled) == [:eta, :s, :tau, :y_gen, :y_likelihood, :z]
end

"""
For a Gaussian likelihood the Laplace approximation is exact, so the program's
log density and gradient equal the analytic marginal's, for each prior family
and with a non-zero prior mean (Stan's prior has mean zero; StanBlocks shifts
the latent by the mean). Constants are compared too (`propto=false`).
"""
@testitem "slic: laplace_marginal of a Gaussian likelihood is its exact marginal" tags=[:slic, :laplace, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    rng = Xoshiro(2)
    X = randn(rng, 12, 4); y = randn(rng, 12); mu = randn(rng, 4)
    exact_zero = @slic (; y, X) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        y ~ multi_normal(rep_vector(0.0, 12), square(tau) * tcrossprod(X) + diag_matrix(rep_vector(square(s), 12)))
    end
    exact_mean = @slic (; y, X, mu) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        y ~ multi_normal(X * mu, square(tau) * tcrossprod(X) + diag_matrix(rep_vector(square(s), 12)))
    end
    families = [
        exact_zero => @slic((; y, X, k=4), begin
            s ~ exponential(1.0); tau ~ exponential(1.0)
            z::vector[k] ~ laplace_marginal(normal(0.0, tau)) do z
                y ~ normal(X * z, s)
            end
        end),
        exact_mean => @slic((; y, X, mu), begin
            s ~ exponential(1.0); tau ~ exponential(1.0)
            z ~ laplace_marginal(normal(mu, tau)) do z
                y ~ normal(X * z, s)
            end
        end),
        exact_mean => @slic((; y, X, mu), begin
            s ~ exponential(1.0); tau ~ exponential(1.0)
            z ~ laplace_marginal(multi_normal(mu, square(tau) * identity_matrix(4))) do z
                y ~ normal(X * z, s)
            end
        end),
        exact_mean => @slic((; y, X, mu), begin
            s ~ exponential(1.0); tau ~ exponential(1.0)
            z ~ laplace_marginal(multi_normal_cholesky(mu, tau * identity_matrix(4))) do z
                y ~ normal(X * z, s)
            end
        end),
        exact_mean => @slic((; y, X, mu), begin
            s ~ exponential(1.0); tau ~ exponential(1.0)
            z ~ laplace_marginal(multi_normal_prec(mu, identity_matrix(4) / square(tau))) do z
                y ~ normal(X * z, s)
            end
        end),
    ]
    for (exact, lap) in families
        pe, pl = instantiate(stan_model(exact)), instantiate(stan_model(lap))
        for _ in 1:5
            u = 0.5 .* randn(rng, 2)
            le, ge = BridgeStan.log_density_gradient(pe.model, u; propto=false)
            ll, gl = BridgeStan.log_density_gradient(pl.model, u; propto=false)
            @test ll ≈ le atol=1e-9
            @test gl ≈ ge atol=1e-9
        end
    end
    # A standard-normal prior needs the latent's declared length.
    stdn = @slic (; y, X) begin
        s ~ exponential(1.0)
        z::vector[4] ~ laplace_marginal(std_normal()) do z
            y ~ normal(X * z, s)
        end
    end
    exact_stdn = @slic (; y, X) begin
        s ~ exponential(1.0)
        y ~ multi_normal(rep_vector(0.0, 12), tcrossprod(X) + diag_matrix(rep_vector(square(s), 12)))
    end
    ps, pse = instantiate(stan_model(stdn)), instantiate(stan_model(exact_stdn))
    for v in (-0.4, 0.3)
        @test BridgeStan.log_density(ps.model, [v]; propto=false) ≈ BridgeStan.log_density(pse.model, [v]; propto=false) atol=1e-9
    end
end

"""
`hessian_block_size` defaults to the latent's length (a dense Hessian, always
correct); a smaller one declares a block-diagonal likelihood Hessian. Any `_tol`
control selects `laplace_marginal_tol` / `laplace_latent_tol_rng`, with Stan Math
5.3.0's defaults for the controls left out and `theta_init` (the latent's
Newton start) shifted by the prior mean.
"""
@testitem "slic: laplace_marginal block size and solver controls" tags=[:slic, :laplace, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    rng = Xoshiro(3)
    c = rand(rng, 0:5, 10)
    diag1 = @slic (; c, n=10) begin
        mu ~ normal(0.0, 1.0); tau ~ exponential(1.0)
        u::vector[n] ~ laplace_marginal(normal(mu, tau); hessian_block_size=1) do u
            c ~ poisson_log(u)
        end
    end
    dense = @slic (; c, n=10) begin
        mu ~ normal(0.0, 1.0); tau ~ exponential(1.0)
        u::vector[n] ~ laplace_marginal(normal(mu, tau)) do u
            c ~ poisson_log(u)
        end
    end
    tolm = @slic (; c, n=10) begin
        mu ~ normal(0.0, 1.0); tau ~ exponential(1.0)
        u::vector[n] ~ laplace_marginal(normal(mu, tau); hessian_block_size=1, tol=1e-12,
                theta_init=rep_vector(1.0, n)) do u
            c ~ poisson_log(u)
        end
    end
    @test occursin(r"(?s)laplace_marginal\(closure_\d+, \(c, mu\), 1, closure_\d+, \(tau, n\)\)", stan_block(stan_code(diag1), "model"))
    @test occursin(r"(?s)laplace_marginal\(closure_\d+, \(c, mu\), n, closure_\d+, \(tau, n\)\)", stan_block(stan_code(dense), "model"))
    tcode = stan_code(tolm)
    @test occursin(r"(?s)target \+= laplace_marginal_tol\(.*\(\(rep_vector\(1\.0, n\) - mu\), 1\.0e-12, 500, 1, 1000, 1\)\s*\);", stan_block(tcode, "model"))
    @test occursin("laplace_latent_tol_rng(", stan_block(tcode, "generated quantities"))
    @test all(stanc_compiles, (diag1, dense, tolm))

    p1, p2, p3 = (instantiate(stan_model(m)) for m in (diag1, dense, tolm))
    for _ in 1:5
        v = 0.5 .* randn(rng, 2)
        a, ga = BridgeStan.log_density_gradient(p1.model, v; propto=false)
        b, gb = BridgeStan.log_density_gradient(p2.model, v; propto=false)
        t = BridgeStan.log_density(p3.model, v; propto=false)
        @test isfinite(a) && all(isfinite, ga)
        # The two Hessian structures run separate Newton solves, each stopping
        # at Stan's default `tol = 1.49012e-8`, so they agree to the solver's
        # tolerance rather than to rounding (CI measured 3e-9 in the density
        # and 4e-8 in the gradient).
        @test a ≈ b atol=1e-6
        @test ga ≈ gb atol=1e-6
        @test t ≈ a atol=1e-4
    end
    # The latent draw is named after the latent.
    pnames = BridgeStan.param_names(p1.model; include_gq=true)
    @test pnames[3:4] == ["u.1", "u.2"]
    draw = BridgeStan.param_constrain(p1.model, [0.1, 0.2]; include_gq=true, rng=BridgeStan.StanRNG(p1.model, 1))
    @test all(isfinite, draw)
end

"""
An observation in the do block may use any family its `~` accepts: a custom
`@lpxf` family, a distribution combinator (its bound validated by the replayed
`~` against the data itself), a discrete family. A one-capture tuple renders as
Stan's `(x,)`.
"""
@testitem "slic: laplace_marginal observes through custom families and combinators" tags=[:slic, :laplace, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    @deffun begin
        @lhs @lpxf lapshift_lpdf(y::vector[n], loc::vector[n], s::real)::real = normal_lpdf(y, loc .+ 0.5, s)
        lapshift_lpdfs(y::vector[n], loc::vector[n], s::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n; out[i] = normal_lpdf(y[i], loc[i] + 0.5, s); end
            out
        end
        lapshift_rng(vector[n], loc::vector[n], s::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n; out[i] = normal_rng(loc[i] + 0.5, s); end
            out
        end
    end
    rng = Xoshiro(5)
    y = max.(randn(rng, 6), -0.5); b = rand(rng, 0:1, 6); lo = -0.5
    custom = @slic (; y) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        z::vector[6] ~ laplace_marginal(normal(0.0, tau)) do z
            y ~ lapshift(z, s)
        end
    end
    exact = @slic (; y) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        y ~ normal(0.5, sqrt(square(tau) + square(s)))
    end
    cens = @slic (; y, lo) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        z::vector[6] ~ laplace_marginal(normal(0.0, tau)) do z
            y ~ censored(normal(z, s); lower=lo)
        end
    end
    bern = @slic (; b) begin
        tau ~ exponential(1.0)
        z::vector[6] ~ laplace_marginal(normal(0.0, tau); hessian_block_size=1) do z
            b ~ bernoulli_logit(z)
        end
    end
    @test occursin(r"(?s)laplace_marginal\(closure_\d+, \(b,\), 1, closure_\d+, \(tau, 6\)\)", stan_block(stan_code(bern), "model"))
    @test all(stanc_compiles, (custom, cens, bern))
    pc, pe = instantiate(stan_model(custom)), instantiate(stan_model(exact))
    for _ in 1:3
        u = 0.4 .* randn(rng, 2)
        @test BridgeStan.log_density(pc.model, u; propto=false) ≈ BridgeStan.log_density(pe.model, u; propto=false) atol=1e-9
    end
    for m in (cens, bern)
        p = instantiate(stan_model(m))
        lp, g = BridgeStan.log_density_gradient(p.model, 0.3 .* randn(rng, LogDensityProblems.dimension(p)); propto=false)
        @test isfinite(lp) && all(isfinite, g)
    end
end

"""
In a plate cell the latent is the cell's integrated latent: the model loop holds
one `target += laplace_marginal(...)` per cell, the latent fills its cell storage
in generated quantities, and a ragged observation keeps its flat `<obs>_gen` and
per-group `<obs>_likelihood` twins with the data's `segments`. `@plate for` over a
ragged observation, the do-form and a dense column all work; per-cell Gaussian
cells match their exact marginals.
"""
@testitem "slic: laplace_marginal in plate cells" tags=[:slic, :laplace, :plate, :descriptor, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    counts = [[1, 0, 3], [2, 2], [0, 1, 4, 1]]
    mus = [[0.1, 0.2, 0.0], [0.3, -0.1], [0.0, 0.5, 0.2, 0.1]]
    platefor = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        @plate for g in eachindex(counts)
            u ~ laplace_marginal(normal(mus[g], tau)) do u
                counts[g] ~ poisson_log(u)
            end
        end
    end
    doform = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        cells ~ plate(counts, mus) do c, m
            u ~ laplace_marginal(normal(m, tau)) do u
                c ~ poisson_log(u)
            end
            nothing
        end
    end
    dense = @slic (; C = [1 0; 2 1; 0 3], M = [0.1 0.2; 0.0 0.3; -0.1 0.4]) begin
        tau ~ exponential(1.0)
        @plate for j in 1:2
            u ~ laplace_marginal(normal(M[:, j], tau)) do u
                C[:, j] ~ poisson_log(u)
            end
        end
    end
    for m in (platefor, doform, dense)
        code = stan_code(m)
        @test occursin(r"(?s)for\(\w+ in 1:\S+\) \{\s*target \+= laplace_marginal\(", stan_block(code, "model"))
        @test occursin("laplace_latent_rng(", stan_block(code, "generated quantities"))
        @test stanc_compiles(m)
    end
    @test occursin("counts_likelihood[g] = poisson_log_lpmf(", stan_code(platefor))
    @test occursin("C_gen[:, j] = poisson_log_int_rng(", stan_code(dense))
    d = stan_descriptor(platefor)
    @test [op.name for op in d.operations] == [:transpile, :instantiate, :fit, :predict, :pointwise_loglik]
    @test only(o for o in d.outputs if o.name === :counts_gen).segments == [3, 5, 9]

    # A do-block binding is a cell local, as in the sampled spelling, so the
    # outputs are the same: in a `@plate for` it is a local of each loop that
    # reads it (`1rgglep`), in the do-form it is stored under the plate's
    # namespace.
    output_names(m) = sort([o.name for o in stan_descriptor(m).outputs])
    sampled_for = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        @plate for g in eachindex(counts)
            u ~ normal(mus[g], tau)
            eta = u .+ 0.0
            counts[g] ~ poisson_log(eta)
        end
    end
    integrated_for = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        @plate for g in eachindex(counts)
            u ~ laplace_marginal(normal(mus[g], tau)) do u
                eta = u .+ 0.0
                counts[g] ~ poisson_log(eta)
            end
        end
    end
    sampled_do = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        cells ~ plate(counts, mus) do c, m
            u ~ normal(m, tau)
            eta = u .+ 0.0
            c ~ poisson_log(eta)
            nothing
        end
    end
    integrated_do = @slic (; counts, mus) begin
        tau ~ exponential(1.0)
        cells ~ plate(counts, mus) do c, m
            u ~ laplace_marginal(normal(m, tau)) do u
                eta = u .+ 0.0
                c ~ poisson_log(eta)
            end
            nothing
        end
    end
    @test output_names(integrated_for) == output_names(sampled_for) == [:counts_gen, :counts_likelihood, :loop_g_u, :tau]
    @test output_names(integrated_do) == output_names(sampled_do) == [:cells_eta, :cells_u, :counts_gen, :counts_likelihood, :tau]
    @test all(stanc_compiles, (integrated_for, integrated_do))

    rng = Xoshiro(4)
    ys = [randn(rng, 3), randn(rng, 2), randn(rng, 4)]
    ms = [0.1 .* randn(rng, 3), 0.1 .* randn(rng, 2), 0.1 .* randn(rng, 4)]
    lap = @slic (; ys, ms) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        @plate for g in eachindex(ys)
            z ~ laplace_marginal(normal(ms[g], tau)) do z
                ys[g] ~ normal(z, s)
            end
        end
    end
    exact = @slic (; ys, ms) begin
        s ~ exponential(1.0); tau ~ exponential(1.0)
        @plate for g in eachindex(ys)
            ys[g] ~ normal(ms[g], sqrt(square(tau) + square(s)))
        end
    end
    pl, pe = instantiate(stan_model(lap)), instantiate(stan_model(exact))
    for _ in 1:5
        u = 0.5 .* randn(rng, 2)
        a, ga = BridgeStan.log_density_gradient(pl.model, u; propto=false)
        b, gb = BridgeStan.log_density_gradient(pe.model, u; propto=false)
        @test a ≈ b atol=1e-9
        @test ga ≈ gb atol=1e-9
    end
end

"""
With no bound observation in its do block nothing informs the latent: it is
drawn from its prior, and the block replays as ordinary statements. With no
data at all that is the prior simulation (no sampled parameter); with every
observation `maybecv`-held-out the observations keep their twins; a bound and an
unbound observation together put only the bound one in the likelihood.
"""
@testitem "slic: laplace_marginal with nothing or part of its block bound" tags=[:slic, :laplace, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    c = [1, 0, 3, 2]; w = [0.5, 1.5, -0.2, 0.1]
    base = @slic begin
        mu ~ normal(0.0, 1.0); tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(mu, tau)) do u
            c ~ poisson_log(u)
            w ~ normal(u, 1.0)
        end
    end
    prior = base()
    pcode = stan_code(prior)
    @test !occursin("laplace", pcode)
    @test occursin("vector[4] u = normal_vector_rng(4, mu, tau);", stan_block(pcode, "generated quantities"))
    @test !occursin(r"\breal\b", stan_block(pcode, "parameters"))
    @test [op.name for op in stan_descriptor(prior).operations] == [:transpile, :instantiate]

    held = base(; c=StanBlocks.stan.maybecv(:c, c), w=StanBlocks.stan.maybecv(:w, w))
    hcode = stan_code(held)
    @test !occursin("laplace", hcode)
    @test occursin("c_likelihood = poisson_log_lpmfs(c, u);", hcode)
    dh = stan_descriptor(held)
    @test [op.name for op in dh.operations] == [:transpile, :instantiate, :predict, :pointwise_loglik]
    @test all(i -> i.held_out, (i for i in dh.inputs if i.name in (:c, :w)))

    mixed = base(; c)
    mcode = stan_code(mixed)
    @test occursin(r"(?s)target \+= laplace_marginal\(closure_\d+, \(c, mu\), 4,", stan_block(mcode, "model"))
    @test occursin("vector[4] w = normal_vector_rng(4, u, 1.0);", stan_block(mcode, "generated quantities"))
    @test all(stanc_compiles, (prior, held, mixed, base(; c, w)))
end

"""
Shapes that cannot be an embedded Laplace approximation are refused at trace
time, each naming what to write instead.
"""
@testitem "slic: laplace_marginal refusals name the fix" tags=[:slic, :laplace] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    c = [1, 0, 3, 2]; w = [0.5, 1.5, -0.2, 0.1]
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("needs a `do` block", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u ~ laplace_marginal(normal(0.0, tau; n=4))
    end))))
    @test occursin("must be one of `std_normal(…)`", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(student_t(3, 0.0, tau)) do u
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("unknown option `tolerance`", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau); tolerance=1e-3) do u
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("does not follow from its prior `normal(…)`", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u ~ laplace_marginal(normal(0.0, tau)) do u
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("samples `s`, which is not data", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0); s ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            s ~ normal(0, 1)
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("`v` is sampled inside the do block of `u` without data", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            v ~ normal(u, 1.0)
            c ~ poisson_log(v)
        end
    end))))
    @test occursin("samples the latent `u` itself", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            u ~ normal(0, 1)
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("written as `=` bindings and observations", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            for i in 1:4
                c[i] ~ poisson_log(u[i])
            end
        end
    end))))
    @test occursin("must be a data `int`", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau); hessian_block_size=tau) do u
            c ~ poisson_log(u)
        end
    end))))
    @test occursin("observes the whole ragged `counts`", msg(() -> stan_code(@slic((; counts=[[1, 0], [2]], n=3), begin
        tau ~ exponential(1.0)
        u::vector[n] ~ laplace_marginal(normal(0.0, tau)) do u
            counts ~ poisson_log(u)
        end
    end))))
    stream = @slic begin
        obs ~ poisson_log(eta)
    end
    @test occursin("embeds the sub-model `stream`", msg(() -> stan_code(@slic((; c), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            node ~ stream(; obs=c, eta=u)
        end
    end))))
    later = msg(() -> stan_code(@slic((; c, w), begin
        tau ~ exponential(1.0)
        u::vector[4] ~ laplace_marginal(normal(0.0, tau)) do u
            c ~ poisson_log(u)
        end
        w ~ normal(u, 1.0)
    end)))
    @test occursin("which `laplace_marginal` integrates out of the model block", later)
    @test occursin("Observe `w` inside the do block of `u`", later)
end
