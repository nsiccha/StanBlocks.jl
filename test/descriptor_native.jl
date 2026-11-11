@testitem "slic: Julia metadata and ragged layouts need no native build" tags=[:slic, :descriptor, :regression] setup=[StanBlocksImports] begin
    @deffun @juliacompat julia_bound(k::real)::real = sqrt(k)
    model = @slic (; xs=[[1.0, 2.0], Float64[], [3.0]], k=4.0, y=0.0) begin
        a ~ normal(0.0, 1.0; lower=julia_bound(k))
        loc ~ plate(xs) do x
            a .* x
        end
        scale ~ plate(loc) do l
            exp.(l)
        end
        y ~ normal(a, 1.0)
    end
    mktemp() do path, io
        # A regular file cannot hold compiled artifacts, even if this machine
        # has a Stan toolchain. Both descriptors must stay on the Julia path.
        withenv("STANBLOCKS_BUILD_DIR" => path) do
            outputs(m) = Dict(o.name => o for o in stan_descriptor(m).outputs)
            d = outputs(model)
            @test d[:a].constraint_values.lower == 2.0
            @test d[:loc].segments == d[:scale].segments == [2, 2, 3]
            rebound = StanBlocks.stan_model(model)(; xs=[[1.0], [2.0, 3.0]], k=9.0)
            d2 = outputs(rebound)
            @test d2[:a].constraint_values.lower == 3.0
            @test d2[:loc].segments == d2[:scale].segments == [1, 3]
        end
    end
end

@testitem "slic: algebraic data bounds work in executable bundles" tags=[:slic, :descriptor, :regression, :bridgestan] setup=[StanBlocksImports] begin
    definition = :(data_root(k::real)::real = begin
        root = solve_newton_tol(x -> rep_vector(x[1]^2 - k, 1),
            rep_vector(1.0, 1), 1e-12, 1e-13, 100)
        root[1]
    end)
    body = quote
        theta ~ restricted(uniform(0.0, 2.0); lower=data_root(k))
        y ~ normal(theta, 1.0)
    end
    for marker in (:stanonly, :juliacompat)
        bundle = compile_slic_bundle((; k=0.25, y=0.0), Pair{String,Any}[], "body" => body;
            udf_definitions=["data_root" => (; definition, markers=marker)],
            name=:data_root_bound)
        output(d) = only(o for o in d.outputs if o.name === :theta)
        @test output(bundle.descriptor).constraint_values.lower ≈ 0.5
        rebound = bundle.model(; k=1.0)
        @test output(stan_descriptor(rebound)).constraint_values.lower ≈ 1.0
        @test output(stan_descriptor(bundle.model)).constraint_values.lower ≈ 0.5
        native = instantiate(bundle.model)
        @test only(BridgeStan.param_constrain(native.model, [0.0])) ≈ 1.25
    end
end

@testitem "slic: native metadata resolves ragged loops and matrix bounds" tags=[:slic, :descriptor, :regression, :bridgestan] setup=[StanBlocksImports] begin
    @deffun begin
        cell_size(k::int)::int = max(k, 0)
        matrix_bound(x::matrix[m, n])::matrix[m, n] = 2.0 .* x
    end
    model = @slic (; K=[2, 0, 1], L=[0.1 0.2; 0.3 0.4], y=0.0) begin
        b ~ plate(; outer=length(K)) do g
            z::vector[cell_size(K[g])] ~ std_normal()
            z
        end
        a::matrix[2, 2] ~ normal(0.0, 1.0; lower=matrix_bound(L))
        y ~ normal(sum(a), 1.0)
    end
    outputs(m) = Dict(o.name => o for o in stan_descriptor(m).outputs)
    d = outputs(model)
    @test d[:b].dims == (3,)
    @test d[:b].segments == [2, 2, 3]
    @test d[:b_z].segments == [2, 2, 3]
    @test d[:a].constraint_values.lower ≈ [0.2 0.4; 0.6 0.8]
    rebound = StanBlocks.stan_model(model)(; K=[0, 0, 0], L=[0.4 0.3; 0.2 0.1])
    d2 = outputs(rebound)
    @test d2[:b].dims == (0,)
    @test d2[:b].segments == [0, 0, 0]
    @test d2[:a].constraint_values.lower ≈ [0.8 0.6; 0.4 0.2]
end

@testitem "slic: native metadata batches solvers and slices transformed data" tags=[:slic, :descriptor, :regression, :bridgestan] setup=[StanBlocksImports] begin
    @deffun begin
        root_bounds(k::vector[n])::vector[n] =
            solve_powell_tol(x -> x .* x .- k, rep_vector(0.7, n), 1e-10, 1e-8, 1000)
        root_newton(k::real)::real = solve_newton(x -> x .* x .- k, rep_vector(1.0, 1))[1]
        root_powell(k::real)::real = solve_powell(x -> x .* x .- k, rep_vector(1.0, 1))[1]
        positive_count(x::vector[n])::int = begin
            count = 0
            for i in 1:n
                if x[i] > 0
                    count += 1
                end
            end
            count
        end
        must_not_run(x::real)::real = begin
            reject("unneeded transformed data ran")
            x
        end
    end
    model = @slic (; k=[0.25, 1.0], x=[1.0, -1.0, 2.0], bad=1.0, captured_only=4.0, y=0.0) begin
        unused = must_not_run(bad)
        lo = root_bounds(k)
        n = positive_count(x)
        a::vector[2] ~ normal(0.0, 1.0; lower=lo)
        b::vector[n] ~ std_normal()
        c ~ normal(0.0, 1.0; lower=root_newton(k[1]), upper=root_powell(k[2]))
        direct ~ normal(0.0, 1.0; lower=solve_newton_tol(
            z -> z .* z .- captured_only, rep_vector(1.0, 1), 1e-12, 1e-12, 100)[1])
        y ~ normal(sum(a) + sum(b) + c + direct + unused, 1.0)
    end
    outputs(m) = Dict(o.name => o for o in stan_descriptor(m).outputs)
    original_code = stan_code(model)
    d = outputs(model)
    @test d[:a].constraint_values.lower ≈ [0.5, 1.0]
    @test d[:b].dims == (2,)
    @test d[:c].constraint_values.lower ≈ 0.5 atol=2e-6
    @test d[:c].constraint_values.upper ≈ 1.0 atol=2e-6
    @test d[:direct].constraint_values.lower ≈ 2.0
    rebound = StanBlocks.stan_model(model)(; k=[2.25, 6.25], x=[-1.0], captured_only=9.0)
    d2 = outputs(rebound)
    @test d2[:a].constraint_values.lower ≈ [1.5, 2.5]
    @test d2[:b].dims == (0,)
    @test d2[:c].constraint_values.lower ≈ 1.5 atol=2e-6
    @test d2[:c].constraint_values.upper ≈ 2.5 atol=2e-6
    @test d2[:direct].constraint_values.lower ≈ 3.0
    @test stan_code(model) == original_code
    @test outputs(model)[:a].constraint_values.lower ≈ [0.5, 1.0]
end

@testitem "slic: metadata preserves Julia and native numerical failures" tags=[:slic, :descriptor, :regression, :bridgestan] setup=[StanBlocksImports] begin
    @deffun @juliacompat julia_sqrt(k::real)::real = sqrt(k)
    julia_model = @slic (; k=-1.0, y=0.0) begin
        a ~ normal(0.0, 1.0; lower=julia_sqrt(k))
        y ~ normal(a, 1.0)
    end
    error_text(f) = try
        f()
        "NO ERROR"
    catch err
        sprint(showerror, err)
    end
    @test occursin("DomainError", error_text(() -> stan_descriptor(julia_model)))
    @deffun no_root(k::real)::real = solve_newton_tol(
        x -> x .* x .+ k, rep_vector(1.0, 1), 1e-12, 1e-13, 2)[1]
    native_model = @slic (; k=1.0, y=0.0) begin
        a ~ normal(0.0, 1.0; lower=no_root(k))
        y ~ normal(a, 1.0)
    end
    message = error_text(() -> stan_descriptor(native_model))
    @test occursin("native evaluation of data-only metadata failed", message)
    @test occursin("no_root", message)
    @test occursin("param_constrain() failed", message)
    @test !occursin("@juliacompat", message)
end
