"""
A bare `tmp = …` in a `@plate for` / `@scan` body is per-iteration scratch
(user decision `1rgglep`): a Stan local inside the loop body of each block that
reads it, recomputed there, never stored per cell and never written to the
draws (snag `plate-cell-deter-3703bd7f`). A data-only one keeps its
transformed-data storage, `~` locals and indexed arrays stay stored, and the
`plate(...) do` form keeps storing every cell binding.
"""
@testitem "slic: an annotated loop's `=` local is a loop local, not an output" tags=[:slic, :plate, :scan, :ragged, :descriptor, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    output_names(m) = Set(o.name for o in stan_descriptor(m).outputs)

    # The snag's shape: a ragged per-cell vector computed from a parameter.
    x = [[0.1, 0.4, -0.3, 0.2], [0.5, -0.1, 0.0], [0.3, 0.2, -0.4, 0.1, 0.6]]
    y = [[1.1, 0.4, 0.9, 1.2], [0.5, 1.1, 1.0], [0.3, 1.2, 0.4, 1.1, 0.6]]
    ragged = @slic (; x, y) begin
        b ~ normal(0, 1)
        sigma ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(y)
            m = exp.(b .* x[i])
            y[i] ~ normal(m, sigma)
        end
    end
    code = stan_code(ragged)
    decl = "vector[(1 + (ragged_end_RaggedVector(x, i) - ragged_start_RaggedVector(x, i)))] loop_i_m = jbroadcasted_exp((b .* getindex_RaggedVector(x, i)));"
    @test occursin(decl, stan_block(code, "model"))
    @test occursin(decl, stan_block(code, "generated quantities"))
    @test !occursin("loop_i_m__pl", code)
    @test !occursin("loop_i_m", stan_block(code, "transformed parameters"))
    @test output_names(ragged) == Set([:b, :sigma, :y_gen, :y_likelihood])
    @test stanc_compiles(ragged)

    # Each block's loop declares the locals its statements read, in source
    # order, including those read only through another local.
    xs, ys = randn(5), randn(5)
    chain = @slic (; xs, ys, zobs = 0.3) begin
        b ~ normal(0, 1); sigma ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(ys)
            a = b * xs[i]
            m = exp(a)
            q = m * 3
            mu[i] = a + 1
            out[i] = q - 1
            ys[i] ~ normal(m, sigma)
        end
        zobs ~ normal(sum(mu), 1)
    end
    code = stan_code(chain)
    tp = stan_block(code, "transformed parameters")
    @test occursin("real loop_i_a = (b * xs[i]);\n        mu[i] = (loop_i_a + 1);", tp)
    @test !occursin("loop_i_m", tp) && !occursin("loop_i_q", tp)
    mb = stan_block(code, "model")
    @test occursin("real loop_i_a = (b * xs[i]);\n        real loop_i_m = exp(loop_i_a);\n        ys[i] ~ normal(loop_i_m, sigma);", mb)
    @test !occursin("loop_i_q", mb)
    gq = stan_block(code, "generated quantities")
    @test occursin("real loop_i_q = (loop_i_m * 3);\n        out[i] = (loop_i_q - 1);", gq)
    @test output_names(chain) == Set([:b, :sigma, :mu, :out, :ys_gen, :zobs_likelihood, :zobs_gen])
    @test stanc_compiles(chain)

    # A local carries no constraint: a constrained center declares its plain
    # container, a bounded value a plain real.
    constrained = @slic (; ys) begin
        b ~ normal(0, 1)
        Omega::cov_matrix[2] ~ wishart(4.0, diag_matrix(rep_vector(1.0, 2)))
        @plate for i in eachindex(ys)
            t ~ exponential(1)
            u = t
            C = Omega
            ys[i] ~ normal(b + C[1, 2], u)
        end
    end
    mb = stan_block(stan_code(constrained), "model")
    @test occursin("real loop_i_u = loop_i_t[i];", mb)
    @test occursin("matrix[2, 2] loop_i_C = Omega;", mb)
    @test :loop_i_t in output_names(constrained)
    @test stanc_compiles(constrained)

    # `@scan` step locals; a nested scan reads its enclosing plate's local.
    scan = @slic (; y = randn(6), T = 6) begin
        phi ~ normal(0, 0.5); s ~ exponential(1)
        @scan begin
            h[1] ~ normal(0, 1)
            for t in 2:T
                m = phi * h[t-1]
                h[t] ~ normal(m, s)
            end
        end
        y ~ normal(h, 1)
    end
    @test occursin("real loop_t_m = (phi * h[(t - 1)]);\n        h[t] ~ normal(loop_t_m, s);", stan_block(stan_code(scan), "model"))
    @test !(:loop_t_m in output_names(scan))
    @test stanc_compiles(scan)
    nested = @slic (; y = randn(4, 3), S = 3, T = 4) begin
        phi ~ normal(0.0, 0.5); ls ~ normal(0, 1)
        @plate for j in 1:S
            sj = exp(ls) * j
            @scan begin
                e1 ~ std_normal()
                x[1] = sj * e1
                for t in 2:T
                    eps ~ std_normal()
                    m = phi * x[t-1]
                    x[t] = m + sj * eps
                end
            end
            y[:, j] ~ normal(x, 1.0)
        end
    end
    tp = stan_block(stan_code(nested), "transformed parameters")
    @test occursin("real loop_j_sj = (loop_j__pl_inv1 * j);\n        x[1, j] = (loop_j_sj * e1[j]);", tp)
    @test occursin("real loop_t_m = (phi * x[(t - 1), j]);\n            x[t, j] = (loop_t_m + (loop_j_sj * loop_t_eps[(t - 1), j]));", tp)
    @test isempty(intersect(output_names(nested), Set([:loop_j_sj, :loop_t_m])))
    @test stanc_compiles(nested)
    # A typed local is a local too, also in a nested loop (where the enclosing
    # loop used to promote it and refuse its type).
    typed = @slic (; y = randn(4, 3), S = 3, T = 4) begin
        phi ~ normal(0.0, 0.5); s ~ exponential(1.0)
        @plate for j in 1:S
            @scan begin
                x[1] ~ normal(0, 1)
                for t in 2:T
                    v::vector[2] = [phi, 1.0] .* x[t-1]
                    x[t] ~ normal(sum(v), s)
                end
            end
            y[:, j] ~ normal(x, 1.0)
        end
    end
    @test occursin("vector[2] loop_t_v = ([phi, 1.0]' .* x[(t - 1), j]);\n            x[t, j] ~ normal(sum(loop_t_v), s);",
        stan_block(stan_code(typed), "model"))
    @test stanc_compiles(typed)

    # The do-form stores its cell bindings, as before.
    stored = @slic (; xs, ys) begin
        b ~ normal(0, 1)
        res ~ plate(xs, ys) do xi, yi
            m = exp(b * xi)
            yi ~ normal(m, 1)
            m + 1
        end
    end
    @test occursin("vector[num_elements(xs)] res_m;", stan_block(stan_code(stored), "transformed parameters"))
    @test :res_m in output_names(stored)
end

@testitem "slic: a loop local has the log density of its stored spelling" tags=[:slic, :plate, :ragged, :regression, :bridgestan] setup=[StanBlocksImports] begin
    using LogDensityProblems
    x = [[0.1, 0.4, -0.3, 0.2], [0.5, -0.1, 0.0], [0.3, 0.2, -0.4, 0.1, 0.6]]
    y = [[1.1, 0.4, 0.9, 1.2], [0.5, 1.1, 1.0], [0.3, 1.2, 0.4, 1.1, 0.6]]
    # `m` as a loop local, and the same value stored as the array `m[i] = …`.
    scratch = @slic (; x, y) begin
        b ~ normal(0, 1)
        sigma ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(y)
            t ~ normal(0, 1)
            m = exp.(b .* x[i]) .+ t
            s2 = sigma ^ 2 .+ 0.1 .* m .^ 2
            y[i] ~ normal(m, sqrt.(s2))
        end
    end
    stored = @slic (; x, y) begin
        b ~ normal(0, 1)
        sigma ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(y)
            t ~ normal(0, 1)
            m[i] = exp.(b .* x[i]) .+ t
            s2[i] = sigma ^ 2 .+ 0.1 .* m[i] .^ 2
            y[i] ~ normal(m[i], sqrt.(s2[i]))
        end
    end
    mktempdir() do dir
        cd(dir) do
            ps, pt = stan_instantiate(scratch), stan_instantiate(stored)
            @test LogDensityProblems.dimension(ps) == LogDensityProblems.dimension(pt) == 5
            rng = Random.Xoshiro(3)
            for _ in 1:5
                q = 0.5 .* randn(rng, 5)
                ls, gs = LogDensityProblems.logdensity_and_gradient(ps, q)
                lt, gt = LogDensityProblems.logdensity_and_gradient(pt, q)
                @test ls ≈ lt
                @test gs ≈ gt
            end
            # The draws drop exactly the two stored arrays, 24 values each draw.
            names_s = BridgeStan.param_names(ps.model; include_tp=true, include_gq=true)
            names_t = BridgeStan.param_names(pt.model; include_tp=true, include_gq=true)
            @test length(names_t) - length(names_s) == 24
            @test setdiff(names_t, names_s) == filter(n -> startswith(n, "m.") || startswith(n, "s2."), names_t)
            # The predictive twins read the same values.
            q = 0.5 .* randn(rng, 5)
            draw(p) = BridgeStan.param_constrain(p.model, q; include_tp=true, include_gq=true,
                rng=BridgeStan.StanRNG(p.model, 11))
            ds, dt = draw(ps), draw(pt)
            twins(names) = findall(n -> startswith(n, "y_likelihood") || startswith(n, "y_gen"), names)
            @test ds[twins(names_s)] ≈ dt[twins(names_t)]
        end
    end
end
