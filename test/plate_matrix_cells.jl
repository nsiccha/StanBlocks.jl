"""
A plate cell may bind a matrix to a local, as it binds a vector (snag
`binding-a-per-ce-e16efa8d`): a cell names its covariance before observing with
it, or keeps a solver's matrix result to read several rows of it. A matrix cell
binding is collected like a vector one. With fixed sizes it is
`array[outer...] matrix[R, C]`, read and written as the whole element `f[i]`.
When its row or column count depends on the cell, it is flat memory, as a
varying-length vector cell is: each cell's matrix is stored column-major in its
span, read back with `to_matrix(span, f__pl_rows[i], f__pl_cols[i])`, and after
the loop `f` is a `RaggedMatrix`. Both used to be refused with "unsupported
per-cell type `matrix`".
"""
@testitem "slic: a plate cell binds a fixed-size matrix" tags=[:slic, :plate, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    # A matrix local, a matrix result read after the loop, and two outer axes.
    local_and_result = @slic (; Y = [0.1 0.2; 0.3 0.1; 0.2 0.5], obs = [0.1, 0.2]) begin
        g ~ normal(0, 1)
        S ~ plate(; outer = 3) do i
            v = exp(g) .* (1.0 .+ Y[i, :]' .^ 2)
            C = diag_matrix(v) .+ 0.1
            Y[i, :]' ~ multi_normal(rep_vector(0.0, 2), C)
            C
        end
        obs ~ multi_normal(rep_vector(0.0, 2), S[2])
    end
    code = stan_code(local_and_result)
    tp = stan_block(code, "transformed parameters")
    @test occursin("array[3] matrix[Y_n, Y_n] S_C;", tp)
    @test occursin("array[3] matrix[Y_n, Y_n] S;", tp)
    @test occursin("S_C[plate_i__pl_1] = (diag_matrix(S_v[:, plate_i__pl_1]) + 0.1);", tp)
    @test occursin("multi_normal(rep_vector(0.0, 2), S_C[plate_i__pl_1])", stan_block(code, "model"))
    @test occursin("obs ~ multi_normal(rep_vector(0.0, 2), S[2]);", stan_block(code, "model"))
    @test stanc_compiles(local_and_result)
    dims = Dict(o.name => o.dims for o in stan_descriptor(local_and_result).outputs)
    @test dims[:S_C] == (3, 2, 2)
    @test dims[:S] == (3, 2, 2)

    two_axes = @slic (; Y = [0.3 0.1 0.2; 0.5 0.4 0.1]) begin
        g ~ normal(0, 1)
        th ~ plate(; outer = (2, 3)) do i, j
            C = diag_matrix(rep_vector(exp(g) + Y[i, j]^2, 2))
            t ~ multi_normal(rep_vector(0.0, 2), C)
            sum(t)
        end
    end
    code = stan_code(two_axes)
    @test occursin("array[2, 3] matrix[2, 2] th_C;", code)
    @test occursin(r"th_C\[plate_i1__pl_\d+, plate_i2__pl_\d+\] = diag_matrix", code)
    @test stanc_compiles(two_axes)

    # A sampled fixed-size matrix cell is a parameter `array[N] matrix[R, C]`.
    sampled = @slic (; obs = [0.1, 0.2, 0.3]) begin
        s ~ normal(0, 1; lower = 0)
        th ~ plate(; outer = 3) do i
            Z::matrix[2, 3] ~ normal(0, s)
            sum(Z)
        end
        obs ~ normal(th, 1)
    end
    code = stan_code(sampled)
    @test occursin("array[3] matrix[2, 3] th_Z;", stan_block(code, "parameters"))
    @test occursin("th_Z[plate_i__pl_1] ~ normal(0, s);", stan_block(code, "model"))
    @test stanc_compiles(sampled)

    # A matrix local inside a called sub-model cell is promoted under its namespace.
    @slic matrix_cell(k::int, s) = begin
        C = diag_matrix(rep_vector(s, k))
        t ~ multi_normal(rep_vector(0.0, k), C)
        return sum(t)
    end
    called = @slic (; obs = [0.1, 0.2, 0.3]) begin
        s ~ normal(0, 1; lower = 0)
        th ~ plate(; outer = 3) do i
            c ~ matrix_cell(2, s)
            c
        end
        obs ~ normal(th, 1)
    end
    @test occursin("array[3] matrix[2, 2] th_c_C;", stan_code(called))
    @test stanc_compiles(called)
end

@testitem "slic: a plate cell binds a matrix whose size varies by cell" tags=[:slic, :plate, :ragged, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    # The snag's shape: each cell observes its group jointly, with a covariance
    # whose size is the group's length.
    data = (; y = [[0.1, 0.2, 0.15], [0.2, 0.1], [0.3, 0.05, 0.1, 0.2]])
    joint = @slic data begin
        s0 ~ normal(0, 1; lower = 0)
        g ~ normal(0, 1)
        @plate for s in eachindex(y)
            v = exp(g) .* (1.0 .+ y[s] .^ 2)
            c = sqrt(s0) .* v
            ccov = diag_matrix(v) .+ c * c' ./ (1.0 + sum(v))
            y[s] ~ multi_normal(g .* v, ccov)
        end
    end
    code = stan_code(joint)
    td = stan_block(code, "transformed data")
    @test occursin("loop_s_ccov__pl_rows[s] = (1 + (ragged_end_RaggedVector(y, s) - ragged_start_RaggedVector(y, s)));", td)
    @test occursin("loop_s_ccov__pl_len[s] = (loop_s_ccov__pl_rows[s] * loop_s_ccov__pl_cols[s]);", td)
    tp = stan_block(code, "transformed parameters")
    @test occursin("vector[sum(loop_s_ccov__pl_len)] loop_s_ccov;", tp)
    @test occursin("loop_s_ccov[ragged_start(loop_s_ccov__pl_end, s):ragged_end(loop_s_ccov__pl_end, s)] = to_vector(", tp)
    reader = r"to_matrix\(\s*loop_s_ccov\[ragged_start\(loop_s_ccov__pl_end, s\):ragged_end\(loop_s_ccov__pl_end, s\)\],\s*loop_s_ccov__pl_rows\[s\],\s*loop_s_ccov__pl_cols\[s\]\s*\)"
    @test occursin(reader, stan_block(code, "model"))
    # The observation keeps its ragged twins, which read the stored covariance.
    gq = stan_block(code, "generated quantities")
    @test occursin("vector[num_elements(y.1)] y_gen;", gq)
    @test occursin("vector[num_elements(y.2)] y_likelihood;", gq)
    @test !occursin("diag_matrix", gq)
    @test stanc_compiles(joint)
    out = only(o for o in stan_descriptor(joint).outputs if o.name === :loop_s_ccov)
    @test out.kind === :transformed_parameter
    @test out.segments == [9, 13, 29]           # 3x3, 2x2 and 4x4 cells, column-major
    @test out.dims == (29,)

    # A varying-size matrix RESULT is a `RaggedMatrix` after the loop: `M[2]` is
    # the second cell's matrix.
    result = @slic (; ys = data.y, obs = [0.1, 0.2]) begin
        g ~ normal(0, 1)
        M ~ plate(ys) do yi
            exp(g) .* (diag_matrix(1.0 .+ yi .^ 2) .+ 0.1)
        end
        obs ~ multi_normal(rep_vector(0.0, 2), M[2])
    end
    code = stan_code(result)
    @test occursin("to_matrix(M[ragged_start(M__pl_end, 2):ragged_end(M__pl_end, 2)], M__pl_rows[2], M__pl_cols[2])",
        stan_block(code, "model"))
    @test stanc_compiles(result)
    @test only(o.segments for o in stan_descriptor(result).outputs if o.name === :M) == [9, 13, 29]

    # A matrix read several times in its cell is computed once, then read.
    reused = @slic (; ys = data.y, ts = [[1.0, 2.0, 3.0], [1.0, 2.0], [0.5, 1.0, 2.0, 3.0]]) begin
        k ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(ys)
            X = append_row(exp(-k .* ts[i])', exp(-2k .* ts[i])')
            ys[i] ~ normal(X[1, :]' .+ X[2, :]', 0.5)
        end
    end
    code = stan_code(reused)
    @test count("append_row", code) == 1
    @test stanc_compiles(reused)

    # A matrix built from data alone (each cell's 0/1 matrix summing its zones
    # into its groups, stored column-major in a ragged data value, with per-cell
    # row and column counts) is filled once in transformed data, then read
    # (snag a-brm-plate-cell-a9b491d3).
    grp = [[1, 1, 2], [1, 1], [1, 2, 3, 3]]
    summing_data = (; agg = [vec(Float64[g[z] == b for b in 1:maximum(g), z in eachindex(g)]) for g in grp],
        n_groups = maximum.(grp), n_zones = length.(grp), x = [[1.0, 2.0, 0.5], [0.3, 0.7], [1.0, 0.2, 0.4, 0.9]],
        y = [[0.5, -0.1], [0.2], [0.1, -0.3, 0.4]])
    summed = @slic summing_data begin
        g ~ normal(0, 1)
        @plate for c in eachindex(n_zones)
            A = reshape(agg[c], n_groups[c], n_zones[c])
            m = exp(g) .* x[c]
            y[c] ~ normal(A * m, A * (m .^ 2) .+ 0.1)
        end
    end
    code = stan_code(summed)
    td = stan_block(code, "transformed data")
    @test occursin("loop_c_A__pl_rows[c] = n_groups[c];", td)
    @test occursin("loop_c_A__pl_cols[c] = n_zones[c];", td)
    @test occursin("loop_c_A[ragged_start(loop_c_A__pl_end, c):ragged_end(loop_c_A__pl_end, c)] = to_vector(reshape(getindex_RaggedVector(agg, c), n_groups[c], n_zones[c]));", td)
    @test count("getindex_RaggedVector(agg, c)", code) == 1
    @test !occursin("loop_c_A;", stan_block(code, "transformed parameters"))
    @test stanc_compiles(summed)

    # With no likelihood the cell, its matrix included, is a generated quantity.
    prior = @slic (; ys = data.y) begin
        g ~ normal(0, 1)
        @plate for s in eachindex(ys)
            v = exp(g) .* (1.0 .+ ys[s] .^ 2)
            C = diag_matrix(v)
            z ~ multi_normal(v, C)
        end
    end
    code = stan_code(prior)
    @test occursin("vector[sum(loop_s_C__pl_len)] loop_s_C;", stan_block(code, "generated quantities"))
    @test stanc_compiles(prior)

    # A sampled matrix whose size varies by cell is refused; the reshape it
    # names works.
    @test_throws "has a size that varies by cell, which is not supported yet" stan_code(@slic (; ys = data.y, K = [3, 2, 4]) begin
        s ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(ys)
            Z::matrix[K[i], 2] ~ normal(0, s)
            ys[i] ~ normal(Z[:, 1], 1)
        end
    end)
    reshaped = @slic (; ys = data.y, K = [3, 2, 4]) begin
        s ~ normal(0, 1; lower = 0)
        @plate for i in eachindex(ys)
            z::vector[K[i] * 2] ~ normal(0, s)
            Z = to_matrix(z, K[i], 2)
            ys[i] ~ normal(Z[:, 1], 1)
        end
    end
    @test stanc_compiles(reshaped)
end

@testitem "slic: a matrix plate cell local has the density of its inline spelling" tags=[:slic, :plate, :ragged, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems, Random
    using StanBlocks.LinearAlgebra: Diagonal, dot, logdet
    data = (; y = [[0.1, 0.2, 0.15], [0.2, 0.1], [0.3, 0.05, 0.1, 0.2]])
    bound = @slic data begin
        s0 ~ normal(0, 1; lower = 0)
        g ~ normal(0, 1)
        @plate for s in eachindex(y)
            v = exp(g) .* (1.0 .+ y[s] .^ 2)
            c = sqrt(s0) .* v
            ccov = diag_matrix(v) .+ c * c' ./ (1.0 + sum(v))
            y[s] ~ multi_normal(g .* v, ccov)
        end
    end
    inline = @slic data begin
        s0 ~ normal(0, 1; lower = 0)
        g ~ normal(0, 1)
        @plate for s in eachindex(y)
            v = exp(g) .* (1.0 .+ y[s] .^ 2)
            c = sqrt(s0) .* v
            y[s] ~ multi_normal(g .* v, diag_matrix(v) .+ c * c' ./ (1.0 + sum(v)))
        end
    end
    # Fixed sizes: every group has two members.
    fixed_data = (; Y = [0.1 0.2; 0.3 0.1; 0.2 0.5])
    fixed_bound = @slic fixed_data begin
        g ~ normal(0, 1)
        th ~ plate(; outer = 3) do i
            v = exp(g) .* (1.0 .+ Y[i, :]' .^ 2)
            C = diag_matrix(v) .+ 0.1
            Y[i, :]' ~ multi_normal(rep_vector(0.0, 2), C)
            nothing
        end
    end
    fixed_inline = @slic fixed_data begin
        g ~ normal(0, 1)
        th ~ plate(; outer = 3) do i
            v = exp(g) .* (1.0 .+ Y[i, :]' .^ 2)
            Y[i, :]' ~ multi_normal(rep_vector(0.0, 2), diag_matrix(v) .+ 0.1)
            nothing
        end
    end
    # A data-only matrix of per-cell size, filled once in transformed data.
    grp = [[1, 1, 2], [1, 1], [1, 2, 3, 3]]
    summing_data = (; agg = [vec(Float64[g[z] == b for b in 1:maximum(g), z in eachindex(g)]) for g in grp],
        n_groups = maximum.(grp), n_zones = length.(grp), x = [[1.0, 2.0, 0.5], [0.3, 0.7], [1.0, 0.2, 0.4, 0.9]],
        y = [[0.5, -0.1], [0.2], [0.1, -0.3, 0.4]])
    summed_bound = @slic summing_data begin
        g ~ normal(0, 1)
        @plate for c in eachindex(n_zones)
            A = reshape(agg[c], n_groups[c], n_zones[c])
            m = exp(g) .* x[c]
            y[c] ~ normal(A * m, A * (m .^ 2) .+ 0.1)
        end
    end
    summed_inline = @slic summing_data begin
        g ~ normal(0, 1)
        @plate for c in eachindex(n_zones)
            m = exp(g) .* x[c]
            y[c] ~ normal(reshape(agg[c], n_groups[c], n_zones[c]) * m,
                reshape(agg[c], n_groups[c], n_zones[c]) * (m .^ 2) .+ 0.1)
        end
    end
    # An independent plain-Julia joint density of the varying-size model, on
    # the unconstrained scale (`s0 = exp(theta[1])`), up to a constant.
    reference(theta) = begin
        s0, g = exp(theta[1]), theta[2]
        lp = -s0^2 / 2 + theta[1] - g^2 / 2
        for yi in data.y
            v = exp(g) .* (1 .+ yi .^ 2)
            c = sqrt(s0) .* v
            C = Diagonal(v) .+ c * c' ./ (1 + sum(v))
            d = yi .- g .* v
            lp -= (dot(d, C \ d) + logdet(C)) / 2
        end
        lp
    end
    summed_reference(theta) = begin
        g = only(theta)
        lp = -g^2 / 2
        for (a, nr, nc, xi, yi) in zip(summing_data.agg, summing_data.n_groups, summing_data.n_zones,
                                       summing_data.x, summing_data.y)
            A, m = reshape(a, nr, nc), exp(g) .* xi
            sd = A * (m .^ 2) .+ 0.1
            lp -= sum(log.(sd) .+ (yi .- A * m) .^ 2 ./ (2 .* sd .^ 2))
        end
        lp
    end
    rng = Xoshiro(20261025)
    for (a, b, ref) in ((bound, inline, reference), (fixed_bound, fixed_inline, nothing),
                        (summed_bound, summed_inline, summed_reference))
        pa, pb = instantiate(a), instantiate(b)
        n = LogDensityProblems.dimension(pa)
        @test n == LogDensityProblems.dimension(pb)
        points = [0.5 .* randn(rng, n) for _ in 1:3]
        lps = map(points) do theta
            lpa, ga = LogDensityProblems.logdensity_and_gradient(pa, theta)
            lpb, gb = LogDensityProblems.logdensity_and_gradient(pb, theta)
            @test lpa == lpb
            @test ga == gb
            lpa
        end
        ref === nothing && continue
        @test lps[2] - lps[1] ≈ ref(points[2]) - ref(points[1]) atol = 1e-10
        @test lps[3] - lps[1] ≈ ref(points[3]) - ref(points[1]) atol = 1e-10
    end
end

@testitem "slic: a plate cell binds a per-subject pmx solve result" tags=[:slic, :plate, :ragged, :regression, :stanc, :bridgestan, :torsten] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    @deffun pmx_cell_theta(cl::real, v::real, ka::real)::real[3] = begin
        th::real[3]
        th[1] = cl
        th[2] = v
        th[3] = ka
        th
    end
    # Two subjects with event tables of different lengths: each cell's solve is
    # a `matrix[2, n_events]` whose column count varies by subject.
    data = (; ts = [[0.0, 0.5, 1.0, 2.0, 4.0, 8.0], [0.0, 1.0, 3.0, 6.0]],
        amts = [[100.0; zeros(5)], [80.0; zeros(3)]], rates = [zeros(6), zeros(4)],
        iis = [zeros(6), zeros(4)], evids = [[1; zeros(Int, 5)], [1; zeros(Int, 3)]],
        cmts = [[1; fill(2, 5)], [1; fill(2, 3)]], addls = [zeros(Int, 6), zeros(Int, 4)],
        sss = [zeros(Int, 6), zeros(Int, 4)],
        ys = [[0.01, 1.2, 1.6, 1.5, 1.1, 0.5], [0.02, 1.0, 1.1, 0.6]])
    # Both compartments are read from ONE solve bound to `x`; the inline spelling
    # has to solve twice.
    bound = @slic data begin
        cl ~ lognormal(log(5.0), 0.5)
        v ~ lognormal(log(50.0), 0.5)
        ka ~ lognormal(log(1.2), 0.5)
        sigma ~ exponential(10.0)
        pred ~ plate(ts, amts, rates, iis, evids, cmts, addls, sss, ys; outer = 2) do t, a, r, i, e, c, ad, s, yi
            x = pmx_solve_onecpt(to_array_1d(t), to_array_1d(a), to_array_1d(r), to_array_1d(i), e, c, ad, s,
                pmx_cell_theta(cl, v, ka))
            mu = to_vector(x[2, :]) / v .+ 0.001 .* to_vector(x[1, :])
            yi ~ normal(mu, sigma)
            mu
        end
    end
    inline = @slic data begin
        cl ~ lognormal(log(5.0), 0.5)
        v ~ lognormal(log(50.0), 0.5)
        ka ~ lognormal(log(1.2), 0.5)
        sigma ~ exponential(10.0)
        pred ~ plate(ts, amts, rates, iis, evids, cmts, addls, sss, ys; outer = 2) do t, a, r, i, e, c, ad, s, yi
            mu = to_vector(pmx_solve_onecpt(to_array_1d(t), to_array_1d(a), to_array_1d(r), to_array_1d(i), e, c, ad, s,
                    pmx_cell_theta(cl, v, ka))[2, :]) / v .+
                0.001 .* to_vector(pmx_solve_onecpt(to_array_1d(t), to_array_1d(a), to_array_1d(r), to_array_1d(i), e, c, ad, s,
                    pmx_cell_theta(cl, v, ka))[1, :])
            yi ~ normal(mu, sigma)
            mu
        end
    end
    code = stan_code(bound)
    @test count("pmx_solve_onecpt(", code) == 1
    @test count("pmx_solve_onecpt(", stan_code(inline)) == 2
    @test occursin("vector[sum(pred_x__pl_len)] pred_x;", code)
    @test only(o.segments for o in stan_descriptor(bound).outputs if o.name === :pred_x) == [12, 20]

    torsten = joinpath(homedir(), ".local/opt/torsten")
    if !isdir(torsten)
        @test_skip "Torsten build not provisioned at $torsten"
    else
        tstanc = joinpath(torsten, "cmdstan/bin/stanc")
        for m in (bound, inline)
            result = withenv("STANC_PATH" => tstanc) do
                stanc_check(stan_code(m); warn_pedantic=false)
            end
            result.ok || @info "Torsten stanc rejected" result.output
            @test result.ok
        end
        make_args = ["STAN_THREADS=true", "STAN=$torsten/cmdstan/stan/",
            "MATH=$torsten/cmdstan/stan/lib/stan_math/", "STANC=$torsten/cmdstan/bin/stanc"]
        pb, pi = instantiate(bound; make_args), instantiate(inline; make_args)
        for q in ([0.1, -0.2, 0.05, -0.3], [-0.3, 0.2, 0.4, 0.1])
            lb, gb = LogDensityProblems.logdensity_and_gradient(pb, q)
            li, gi = LogDensityProblems.logdensity_and_gradient(pi, q)
            @test lb == li
            @test gb ≈ gi atol = 1e-10
        end
    end
end
