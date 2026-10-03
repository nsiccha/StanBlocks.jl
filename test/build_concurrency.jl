using TestItemRunner

@testitem "thread safety: immutable artifact identities and atomic source aliases" tags=[:slic, :regression] begin
    using StanBlocks
    code = "parameters { real x; } model { x ~ normal(0, 1); }"
    one = StanBlocks._default_build_path(code, (; make_args=["STAN_THREADS=true"]))
    two = StanBlocks._default_build_path(code, (; make_args=["STAN_THREADS=false"]))
    @test one != two
    @test one == StanBlocks._default_build_path(code, (; make_args=["STAN_THREADS=true"]))
    @test one != StanBlocks._default_build_path(code * "\n", (; make_args=["STAN_THREADS=true"]))
    @test one != StanBlocks._default_build_path(code, (; make_args=["STAN_THREADS=true"]); filename="other.stan")
    mktempdir() do dir
        relocated = withenv("STANBLOCKS_BUILD_DIR" => dir) do
            StanBlocks._default_build_path(code, (; make_args=["STAN_THREADS=true"]))
        end
        @test basename(relocated) != basename(one)
        plain_context = withenv("BRIDGESTAN_AD_HESSIAN" => nothing) do
            StanBlocks._native_build_context(dir, String[])
        end
        hessian_context = withenv("BRIDGESTAN_AD_HESSIAN" => "true") do
            StanBlocks._native_build_context(dir, String[])
        end
        @test StanBlocks._default_build_path(code, plain_context) !=
            StanBlocks._default_build_path(code, hessian_context)
        path = joinpath(dir, "alias.stan")
        a, b = repeat("a", 100_000), repeat("b", 100_000)
        StanBlocks._atomic_build_write(path, a)
        readers = [Threads.@spawn begin
            for _ in 1:100
                bytes = read(path, String)
                bytes == a || bytes == b || error("partial source publication")
                yield()
            end
        end for _ in 1:2]
        for i in 1:20
            StanBlocks._atomic_build_write(path, isodd(i) ? b : a)
        end
        foreach(fetch, readers)
        @test read(path, String) == a
        # The artifact path is immutable, even if a caller damages the cache.
        @test_throws ErrorException StanBlocks._write_immutable_stan_source(path, b)
        @test read(path, String) == a

        # Refused: replacing a directory must preserve it and surface the
        # native failure, never fall back to copying (atomic publication).
        directory = joinpath(dir, "keep-directory")
        mkpath(directory)
        marker = joinpath(directory, "marker")
        write(marker, a)
        before = sort(readdir(dir))
        @test_throws Base.IOError StanBlocks._atomic_build_write(directory, b)
        @test read(marker, String) == a
        @test sort(readdir(dir)) == before

        if Sys.iswindows()
            # A real Windows handle blocks rename until its reader closes.
            ready = Channel{Nothing}(1)
            reader = Threads.@spawn open(path, "r") do io
                put!(ready, nothing)
                sleep(0.05)
                @test read(io, String) == a
            end
            take!(ready)
            try
                StanBlocks._atomic_build_write(path, b)
            finally
                fetch(reader)
            end
            @test read(path, String) == b

            # Persistent OS denial must surface rather than report success
            # or delete the old source (dev §1; atomic publication contract).
            before = sort(readdir(dir))
            open(path, "r") do io
                @test_throws Base.IOError StanBlocks._atomic_build_write(path, a)
                @test read(io, String) == b
            end
            @test read(path, String) == b
            @test sort(readdir(dir)) == before
        end
    end
end

@testitem "thread safety: shared native artifact across processes" tags=[:slic, :regression, :bridgestan] begin
    using StanBlocks
    worker = joinpath(@__DIR__, "fixtures", "native_build_worker.jl")
    implementation = joinpath(dirname(pathof(StanBlocks)), "slic_stan", "build.jl")
    project = dirname(Base.active_project())
    mktempdir() do dir
        logs = [joinpath(dir, "worker-$i.log") for i in 1:2]
        streams = open.(logs, "w")
        processes = Base.Process[]
        try
            for i in 1:2
                cmd = `$(Base.julia_cmd()) --startup-file=no --compile=yes --threads=1 --project=$project $worker $implementation $dir $i`
                push!(processes, run(pipeline(cmd; stdout=streams[i], stderr=streams[i]); wait=false))
            end
            ready = timedwait(() -> all(i -> isfile(joinpath(dir, "ready-$i")), 1:2) ||
                any(process_exited, processes), 120)
            if ready !== :ok || !all(i -> isfile(joinpath(dir, "ready-$i")), 1:2)
                error("native workers did not reach the bounded start gate:\n" *
                    join(read.(logs, String), "\n"))
            end
            write(joinpath(dir, "start"), "start")
            foreach(wait, processes)
            @test all(success, processes)
        finally
            for process in processes
                process_running(process) && kill(process)
                wait(process)
            end
            close.(streams)
        end
        results = map(logs) do log
            output = read(log, String)
            result = match(r"(?m)^RESULT (.+)$", output)
            result === nothing && error("native worker returned no result:\n" * output)
            parse(Float64, result.captures[1])
        end
        @test results == [-0.5, -2.0]
        libraries = String[]
        for (root, _, files) in walkdir(dir), file in files
            endswith(file, "_model.so") && push!(libraries, joinpath(root, file))
        end
        @test length(libraries) == 1
        @test count(endswith(".stan"), readdir(dirname(only(libraries)))) == 1
    end
end

@testitem "thread safety: native cold warm and explicit-path construction" tags=[:slic, :regression, :bridgestan] begin
    using StanBlocks
    using LogDensityProblems: logdensity
    import BridgeStan
    y = [0.25, -0.5]
    a = stan_model(@slic (; y) begin
        alpha ~ normal(0.0, 1.0)
        y ~ normal(alpha, 1.0)
    end)
    b = stan_model(@slic (; y) begin
        beta ~ normal(10.0, 1.0)
        y ~ normal(beta, 1.0)
    end)
    before = copy(y)
    mktempdir() do dir
        withenv("STANBLOCKS_BUILD_DIR" => dir) do
            problems = fetch.([Threads.@spawn(stan_instantiate(m)) for m in (a, b, a, b)])
            @test BridgeStan.param_names.(getproperty.(problems, :model)) ==
                [["alpha"], ["beta"], ["alpha"], ["beta"]]
            densities = [logdensity(p, [0.0]) for p in problems]
            @test densities[1] == densities[3]
            @test densities[2] == densities[4]
            @test densities[2] - densities[1] ≈ -50.0
            @test problems[1] !== problems[3]
            @test BridgeStan.name(problems[1].model) != BridgeStan.name(problems[2].model)
            warm = fetch.([Threads.@spawn(stan_instantiate(a)) for _ in 1:2])
            @test all(p -> logdensity(p, [0.0]) == densities[1], warm)

            alias = joinpath(dir, "model.stan")
            p1 = stan_instantiate(a; path=alias)
            p2 = @test_logs (:warn, r"overwriting stale Stan source") stan_instantiate(b; path=alias)
            p3 = stan_instantiate(b; path=alias)
            @test BridgeStan.param_names(p1.model) == ["alpha"]
            @test BridgeStan.param_names(p2.model) == ["beta"]
            @test BridgeStan.param_names(p3.model) == ["beta"]
            @test logdensity(p2, [0.0]) == logdensity(p3, [0.0])
            @test read(alias, String) == stan_code(b)
            raced = fetch.([Threads.@spawn(stan_instantiate(m; path=alias)) for m in (a, b)])
            @test BridgeStan.param_names.(getproperty.(raced, :model)) == [["alpha"], ["beta"]]
            @test logdensity.(raced, Ref([0.0])) == densities[1:2]
            @test read(alias, String) in (stan_code(a), stan_code(b))

            # A data-constructor failure must release all locks; it does not
            # corrupt an already published library or cache a mutable problem.
            invalid = StanBlocks.JSON.json(merge(StanBlocks.stan_data(a), Dict(:y => "bad")))
            @test_throws Exception StanBlocks._instantiate_artifact(stan_code(a), invalid;
                path=nothing, make_args=["STAN_THREADS=true"], nan_on_error=true, warn=false)
            @test logdensity(stan_instantiate(a), [0.0]) == densities[1]

            # Failed compilation must release locks and discard private staging.
            # Repeating the failure exercises the same artifact lock twice.
            invalid_source = "parameters { this is not Stan; }"
            for _ in 1:2
                @test_throws Exception StanBlocks._instantiate_artifact(invalid_source, "{}";
                    path=nothing, make_args=["STAN_THREADS=true"], nan_on_error=true, warn=false)
            end
            @test !any(startswith("build-"), [name for (_, dirs, _) in walkdir(dir) for name in dirs])
            @test logdensity(stan_instantiate(a), [0.0]) == densities[1]

            # A different optimization option gets its own immutable artifact.
            # Keep the threading ABI consistent across all loaded libraries.
            optimized = stan_instantiate(a; make_args=["STAN_THREADS=true", "O=0"])
            @test logdensity(optimized, [0.0]) == densities[1]
            @test BridgeStan.name(optimized.model) != BridgeStan.name(problems[1].model)
        end
    end
    @test y == before
end
