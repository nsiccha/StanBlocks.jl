"""
A simplex parameter whose Dirichlet concentration depends on parameters is
sampled in the log-space stick-breaking frame (snag `hierarchical-spa-1a7808bd`,
decision `18ro7g0`). Stan's own `simplex` holds its weights in linear space: at
a small concentration they underflow to 0, where `dirichlet` answers +Inf, and a
sampled concentration makes a funnel. The model is unchanged: the simplex keeps
its name, values and output column; only its emission differs (a
`<name>_free` parameter, a `<name>_log` transformed parameter, and the density
of `<name>_free` in the model block). `dirichlet(a; frame = :native | :stick)`
overrides the choice.
"""
@testitem "slic: simplex ~ dirichlet with a sampled concentration lowers to the stick frame" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    data = (; K = 4, y = [1, 2, 2, 3])
    hier = @slic data begin
        alpha ~ lognormal(0, 1)
        beta::simplex[K] ~ dirichlet(rep_vector(1.0, K))
        theta::simplex[K] ~ dirichlet(alpha * K * beta)
        y ~ categorical(theta)
    end
    code = stan_code(hier)
    params, tp, model = (stan_block(code, b) for b in ("parameters", "transformed parameters", "model"))
    # A data concentration keeps Stan's simplex; a sampled one takes the stick frame.
    @test occursin("simplex[K] beta;", params)
    @test occursin("beta ~ dirichlet(", model)
    @test occursin("vector[(K - 1)] theta_free;", params)
    @test !occursin("simplex[K] theta;", params)
    @test occursin("vector[K] theta_log = dirichlet_stick_log_simplex(theta_free, (alpha * K * beta));", tp)
    @test occursin("simplex[K] theta = exp(theta_log);", tp)
    @test occursin("target += dirichlet_stick_lpdf(theta_free | (alpha * K * beta));", model)
    @test !occursin("theta ~ dirichlet", model)
    @test stanc_compiles(hier)
    # The descriptor still reports the simplex as the model's output.
    outs = Dict(o.name => o for o in stan_descriptor(hier).outputs)
    @test outs[:theta].type === :simplex && outs[:theta].dims == (4,)
    @test outs[:theta].generative === :posterior
    @test outs[:theta_free].kind === :parameter && outs[:theta_free].dims == (3,)

    # The bare spelling lowers the same way.
    bare = @slic data begin
        alpha ~ lognormal(0, 1)
        theta ~ dirichlet(rep_vector(alpha, K))
        y ~ categorical(theta)
    end
    @test occursin("theta_free;", stan_block(stan_code(bare), "parameters"))
    @test stanc_compiles(bare)

    # `frame = :native` keeps Stan's simplex for a sampled concentration,
    # `frame = :stick` takes the stick frame for a data one.
    native = @slic data begin
        alpha ~ lognormal(0, 1)
        theta::simplex[K] ~ dirichlet(rep_vector(alpha, K); frame = :native)
        y ~ categorical(theta)
    end
    @test occursin("simplex[K] theta;", stan_block(stan_code(native), "parameters"))
    @test occursin("theta ~ dirichlet(", stan_block(stan_code(native), "model"))
    @test !occursin("frame", stan_code(native))
    forced = @slic (; data..., a = [0.5, 0.5, 0.5, 0.5]) begin
        theta ~ dirichlet(a; frame = :stick)
        y ~ categorical(theta)
    end
    @test occursin("theta_free;", stan_block(stan_code(forced), "parameters"))
    @test stanc_compiles(forced)

    # An array of simplexes lowers element by element.
    arr = @slic (; N = 3, K = 4, a = [0.3, 2.0, 0.05, 5.0], y1 = [1], y2 = [4]) begin
        c ~ lognormal(0, 1)
        theta::simplex[N, K] ~ dirichlet(c * a)
        y1 ~ categorical(theta[1])
        y2 ~ categorical(theta[2])
    end
    acode = stan_code(arr)
    @test occursin("array[N] vector[(K - 1)] theta_free;", stan_block(acode, "parameters"))
    @test occursin("array[N] vector[K] theta_log = dirichlet_stick_log_simplex(theta_free, (c * a));", acode)
    @test occursin("array[N] simplex[K] theta = exp(theta_log);", acode)
    @test stanc_compiles(arr)

    # A simplex plate cell lowers cell by cell: the map fills transformed
    # parameters in the loop, the density joins the model loop.
    pl = @slic (; N = 3, K = 4, a = [0.3, 2.0, 0.05, 5.0], y = [1, 2, 4]) begin
        c::vector[N] ~ lognormal(0, 1)
        theta ~ plate(; outer = N) do i
            t::simplex[K] ~ dirichlet(c[i] * a)
            y[i] ~ categorical(t)
            t
        end
    end
    pcode = stan_code(pl)
    ptp = stan_block(pcode, "transformed parameters")
    @test occursin("array[N] vector[(K - 1)] theta_t_free;", stan_block(pcode, "parameters"))
    @test occursin("array[N] simplex[K] theta_t;", ptp)
    @test occursin(r"theta_t_log\[(\w+)\] = dirichlet_stick_log_simplex\(theta_t_free\[\1\], \(c\[\1\] \* a\)\);", ptp)
    @test occursin(r"target \+= dirichlet_stick_lpdf\(theta_t_free\[(\w+)\] \| \(c\[\1\] \* a\)\);", stan_block(pcode, "model"))
    @test stanc_compiles(pl)
    plnative = @slic (; N = 3, K = 4, a = [0.3, 2.0, 0.05, 5.0], y = [1, 2, 4]) begin
        c::vector[N] ~ lognormal(0, 1)
        theta ~ plate(; outer = N) do i
            t::simplex[K] ~ dirichlet(c[i] * a; frame = :native)
            y[i] ~ categorical(t)
            t
        end
    end
    @test occursin("array[N] simplex[K] theta_t;", stan_block(stan_code(plnative), "parameters"))

    # Refusals name the fix.
    @test occursin("takes `:stick`", errmsg(@slic data begin
        alpha ~ lognormal(0, 1); theta ~ dirichlet(rep_vector(alpha, K); frame = :ilr); y ~ categorical(theta)
    end))
    @test occursin("not a fresh `simplex` parameter", errmsg(@slic (; z = [0.2, 0.8]) begin
        alpha ~ lognormal(0, 1); z ~ dirichlet(rep_vector(alpha, 2); frame = :stick)
    end))
    @test occursin("Rename the model's `theta_free`", errmsg(@slic data begin
        alpha ~ lognormal(0, 1); theta_free ~ normal(0, 1)
        theta::simplex[K] ~ dirichlet(rep_vector(alpha, K)); y ~ categorical(theta)
    end))
end


@testitem "slic: the stick-frame simplex density is exact down to tiny concentrations" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports] begin
    using BridgeStan, Random, LinearAlgebra
    const SF = StanBlocks.SpecialFunctions
    # log|d outs / d ins| at unconstrained point q, by central differences.
    fd_logdet(m, q, ins, outs; h = 1e-6) = begin
        J = zeros(length(outs), length(ins))
        for (j, i) in enumerate(ins)
            e = zeros(length(q)); e[i] = h
            J[:, j] = (BridgeStan.param_constrain(m, q .+ e; include_tp = true)[outs] .-
                       BridgeStan.param_constrain(m, q .- e; include_tp = true)[outs]) ./ 2h
        end
        logabsdet(J)[1]
    end
    # The constrained entries `X.i…k` matching each free coordinate `X_free.i…k`.
    free_and_values(m, X) = begin
        unc = BridgeStan.param_unc_names(m)
        names = BridgeStan.param_names(m; include_tp = true)
        ins = findall(n -> startswith(n, "$(X)_free."), unc)
        outs = [findfirst(==(replace(unc[i], "$(X)_free." => "$(X).")), names) for i in ins]
        ins, outs
    end
    pick(names, prefix) = findall(n -> n == prefix || startswith(n, prefix * "."), names)

    # The stick density is the density of the free coordinates: Stan's simplex
    # density (without its own Jacobian) plus log|d theta / d theta_free|.
    body(frame) = quote
        alpha ~ lognormal(0, 1)
        beta::simplex[K] ~ dirichlet(rep_vector(1.0, K))
        theta::simplex[K] ~ dirichlet(alpha * K * beta; frame = $(QuoteNode(frame)))
        y ~ categorical(theta)
    end
    data = (; K = 4, y = [1, 2, 2, 3])
    ms = stan_instantiate(StanBlocks.SlicModel(body(:stick), data, @__MODULE__)).model
    mn = stan_instantiate(StanBlocks.SlicModel(body(:native), data, @__MODULE__)).model
    names = BridgeStan.param_names(ms; include_tp = true)
    ins, outs = free_and_values(ms, "theta")
    outs = outs[1:3]   # theta[1:K-1]
    Random.seed!(1)
    compared = Ref(0)
    for _ in 1:60
        q = randn(BridgeStan.param_unc_num(ms))
        v = BridgeStan.param_constrain(ms, q; include_tp = true)
        minimum(v[pick(names, "theta")]) > 1e-6 || continue
        qn = BridgeStan.param_unconstrain(mn, vcat(v[pick(names, "alpha")], v[pick(names, "beta")], v[pick(names, "theta")]))
        @test BridgeStan.log_density(ms, q; propto = false, jacobian = false) ≈
              BridgeStan.log_density(mn, qn; propto = false, jacobian = false) + fd_logdet(ms, q, ins, outs) atol = 1e-6
        compared[] += 1
    end
    @test compared[] > 30

    # Per stick, Beta(V | a, b) |dV/ds| in the frame centred by concentration,
    # in 512-bit arithmetic: the emitted density matches it to 1e-9 however
    # small the concentration (a split into Dirichlet(x | a) + log|dx/ds| lost
    # it entirely below 1e-15).
    setprecision(BigFloat, 512) do
        lσ(s) = s > 0 ? -log1p(exp(-s)) : s - log1p(exp(s))
        σ(s) = 1 / (1 + exp(-s))
        scale(a) = exp(log(a) / (a^2 + 1))
        exact(s, a) = sum(1:length(s)) do k
            b = sum(a[k+1:end]); A, B = scale(a[k]), scale(b)
            ls, lm = lσ(s[k]), lσ(-s[k]); p, q = ls / A, lm / B
            l = max(p, q) + log1p(exp(-abs(p - q)))
            a[k] / A * ls + b / B * lm - (a[k] + b) * l - SF.logbeta(a[k], b) + log(σ(-s[k]) / A + σ(s[k]) / B)
        end
        iso = @slic begin
            theta::simplex[K] ~ dirichlet(a; frame = :stick)
            y ~ normal(theta[2], 1.0)
        end
        s = [-3.0, 1.0, -2.0]
        for e in (0, 3, 9, 15, 17, 30)
            a = [10.0^-e, 0.5, 10.0^-e, 2.0]
            m = stan_instantiate(iso(; K = 4, a, y = 0.3)).model
            lp, g = BridgeStan.log_density_gradient(m, s; propto = false)
            th2 = BridgeStan.param_constrain(m, s; include_tp = true)[findfirst(==("theta.2"), BridgeStan.param_names(m; include_tp = true))]
            likelihood = -0.5 * log(2pi) - 0.5 * (0.3 - th2)^2
            @test lp - likelihood ≈ Float64(exact(big.(s), big.(a))) atol = 1e-9
            @test all(isfinite, g)
            h = 1e-6
            fd = [(BridgeStan.log_density(m, s .+ h .* (1:3 .== j); propto = false) -
                   BridgeStan.log_density(m, s .- h .* (1:3 .== j); propto = false)) / 2h for j in 1:3]
            @test g ≈ fd rtol = 1e-5 atol = 1e-7
        end
        # The weights underflow, their logs do not.
        m = stan_instantiate(iso(; K = 4, a = [1e-3, 0.5, 1e-3, 2.0], y = 0.3)).model
        v = BridgeStan.param_constrain(m, [-12.0, 3.0, -15.0]; include_tp = true)
        tn = BridgeStan.param_names(m; include_tp = true)
        logs = v[pick(tn, "theta_log")]
        @test all(isfinite, logs) && minimum(logs) < -10000
        @test sum(v[pick(tn, "theta")]) ≈ 1
    end

    # A hand-written program includes the same definitions and evaluates the
    # same density as the program SLIC emits.
    iso = @slic begin
        theta::simplex[K] ~ dirichlet(a; frame = :stick)
        y ~ categorical(theta)
    end
    mi = stan_instantiate(iso(; K = 4, a = [0.3, 2.0, 0.05, 5.0], y = [2])).model
    inc = StanBlocks.stick_simplex_include(mktempdir())
    handwritten = """
    functions {
    #include stick_simplex.stan
    }
    data { int<lower=2> K; vector<lower=0>[K] a; int<lower=1, upper=K> y; }
    parameters { vector[K - 1] theta_free; }
    transformed parameters {
      vector[K] theta_log = dirichlet_stick_log_simplex(theta_free, a);
      vector[K] theta = exp(theta_log);
    }
    model {
      target += dirichlet_stick_lpdf(theta_free | a);
      y ~ categorical(theta);
    }
    """
    lib = StanBlocks.stan_library(handwritten;
        stanc_args=["--include-paths=" * dirname(inc)], dependencies=[inc])
    mh = BridgeStan.StanModel(lib, """{"K": 4, "a": [0.3, 2.0, 0.05, 5.0], "y": 2}"""; warn = false)
    for _ in 1:10
        s = 1.5 .* randn(3)
        @test BridgeStan.log_density(mh, s; propto = false) ≈ BridgeStan.log_density(mi, s; propto = false) atol = 1e-10
    end

    # The plate form: Stan's simplex density plus each cell's log-Jacobian.
    pbody(frame) = quote
        c::vector[N] ~ lognormal(0, 1)
        theta ~ plate(; outer = N) do i
            t::simplex[K] ~ dirichlet(c[i] * a; frame = $(QuoteNode(frame)))
            y[i] ~ categorical(t)
            t
        end
    end
    pdata = (; N = 3, K = 4, a = [0.3, 2.0, 0.05, 5.0], y = [1, 2, 4])
    mps = stan_instantiate(StanBlocks.SlicModel(pbody(:stick), pdata, @__MODULE__)).model
    mpn = stan_instantiate(StanBlocks.SlicModel(pbody(:native), pdata, @__MODULE__)).model
    pnames = BridgeStan.param_names(mps; include_tp = true)
    pins, pouts = free_and_values(mps, "theta_t")
    pcompared = Ref(0)
    for _ in 1:60
        q = randn(BridgeStan.param_unc_num(mps))
        v = BridgeStan.param_constrain(mps, q; include_tp = true)
        minimum(v[pick(pnames, "theta_t")]) > 1e-6 || continue
        qn = BridgeStan.param_unconstrain(mpn, vcat(v[pick(pnames, "c")], v[pick(pnames, "theta_t")]))
        @test BridgeStan.log_density(mps, q; propto = false, jacobian = false) ≈
              BridgeStan.log_density(mpn, qn; propto = false, jacobian = false) + fd_logdet(mps, q, pins, pouts) atol = 1e-6
        pcompared[] += 1
    end
    @test pcompared[] > 5
end
