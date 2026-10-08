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
        # Windows rename briefly holds delete access to the new file, which
        # can deny an ordinary reader's open. Retry only that sharing error;
        # every successful read must still contain one complete source.
        read_alias = Base.retry(read;
            delays=Base.ExponentialBackOff(n=10, first_delay=0.01,
                max_delay=0.25, factor=2.0, jitter=0.0),
            check=(_, err) -> Sys.iswindows() && err isa SystemError &&
                err.errnum == Base.Libc.EACCES)
        readers = [Threads.@spawn begin
            for _ in 1:100
                bytes = read_alias(path, String)
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

@testitem "build lock: a dead holder's lock is reclaimed, a live holder's never" tags=[:slic, :regression] begin
    using StanBlocks
    using Test: TestLogger
    using Logging: with_logger
    # A cooperating holder in another process, on Julia's stdlib Pidfile lock
    # as an external replay builder or an older StanBlocks takes it: it never
    # refreshes the file, so only its liveness protects it.
    # Pkg.test hides @stdlib from children; resolve it through the test manifest.
    project = dirname(Base.active_project())
    holder_code = "const FileWatching = Base.require(Base.PkgId(Base.UUID(" *
        "\"7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee\"), \"FileWatching\")); " *
        "lock = FileWatching.Pidfile.mkpidlock(ARGS[1]); println(\"HELD\"); flush(stdout); sleep(3600)"
    acquire(path) = Threads.@spawn StanBlocks._with_build_lock(path; dead_age=1, poll=0.25) do
        read(path, String)
    end
    own = "$(getpid()) $(gethostname())"
    mktempdir() do dir
        path = joinpath(dir, "toolchain", ".stanblocks-build.pid")
        mkpath(dirname(path))
        holder = open(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $holder_code $path`, "r")
        logger = TestLogger()
        local waiter
        local held
        try
            readline(holder) == "HELD" || error("lock holder did not start")
            # Pidfile writes "<pid> <host>" before mkpidlock returns.
            held = read(path, String)
            @test held == "$(getpid(holder)) $(gethostname())"
            waiter = with_logger(() -> acquire(path), logger)
            # Refused: expiring a live holder's lock on a timeout would let a
            # competitor into a slow build (build.jl lock contract).
            @test timedwait(() -> istaskdone(waiter), 5) === :timed_out
            kill(holder, Base.SIGKILL)  # killed mid-build: its file stays
        finally
            process_running(holder) && kill(holder, Base.SIGKILL)
            wait(holder)
        end
        # The waiter may reclaim before `wait(holder)` returns: Windows reports
        # the pid gone before the exit reaches this task. So check what it
        # reclaimed, the killed holder's own file, not whether that file still
        # exists at this instant.
        @test timedwait(() -> istaskdone(waiter), 60) === :ok
        @test fetch(waiter) == own
        reclaimed = [log.kwargs[:record] for log in logger.logs
                     if occursin("reclaimed a build lock", log.message)]
        @test reclaimed == [held]
        @test !isfile(path) && !isfile(path * ".reclaim")

        # A holder on another host cannot be checked, so it is left alone.
        write(path, "1 elsewhere.invalid")
        sleep(1.5)
        waiter = acquire(path)
        @test timedwait(() -> istaskdone(waiter), 3) === :timed_out
        rm(path)
        @test fetch(waiter) == own

        # A holder killed between creating and writing its file leaves it empty.
        write(path, "")
        sleep(1.5)
        @test fetch(acquire(path)) == own

        # External builders share the toolchain lock through the public helper.
        lock_path = joinpath(realpath(dir), ".stanblocks-build.pid")
        @test StanBlocks.with_toolchain_lock(() -> read(lock_path, String), dir) == own
        @test !isfile(lock_path)
    end
end

@testitem "build lock: waiters on a dead holder's lock enter one at a time" tags=[:slic, :regression] begin
    using StanBlocks
    # Julia's Pidfile `stale_age` removal is check-then-remove: waiters that
    # see the same stale file delete each other's fresh locks and all enter.
    worker = joinpath(@__DIR__, "fixtures", "build_lock_waiter.jl")
    implementation = joinpath(dirname(pathof(StanBlocks)), "slic_stan", "build.jl")
    # Pkg.test hides @stdlib from children; resolve it through the test manifest.
    project = dirname(Base.active_project())
    holder_code = "const FileWatching = Base.require(Base.PkgId(Base.UUID(" *
        "\"7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee\"), \"FileWatching\")); " *
        "lock = FileWatching.Pidfile.mkpidlock(ARGS[1]); println(\"HELD\"); flush(stdout); sleep(3600)"
    mktempdir() do dir
        path = joinpath(dir, ".stanblocks-build.pid")
        holder = open(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $holder_code $path`, "r")
        try
            readline(holder) == "HELD" || error("lock holder did not start")
        finally
            kill(holder, Base.SIGKILL)
            wait(holder)
        end
        @test isfile(path)
        n = 4
        workers = [open(`$(Base.julia_cmd()) --startup-file=no --project=$project $worker $implementation $path $dir $i`, "r")
                   for i in 1:n]
        outputs = String[]
        try
            ready = timedwait(() -> all(i -> isfile(joinpath(dir, "ready-$i")), 1:n) ||
                any(process_exited, workers), 120)
            ready === :ok && all(i -> isfile(joinpath(dir, "ready-$i")), 1:n) ||
                error("build-lock workers did not reach the start gate")
            write(joinpath(dir, "start"), "start")
            finished = timedwait(() -> all(process_exited, workers), 120)
            finished === :ok || error("build-lock workers did not finish")
        finally
            foreach(w -> process_running(w) && kill(w, Base.SIGKILL), workers)
            foreach(wait, workers)
            append!(outputs, read.(workers, String))
        end
        @test all(success, workers)
        intervals = sort!([parse.(Float64, m.captures) for output in outputs
                           for m in eachmatch(r"(?m)^INTERVAL (\S+) (\S+)$", output)])
        @test length(intervals) == n
        @test all(k -> intervals[k+1][1] >= intervals[k][2], 1:n-1)
    end
end

@testitem "build lock: a reader of a dead lock defers its reclaim, never fails it" tags=[:slic, :regression] begin
    using StanBlocks
    using Test: TestLogger
    using Logging: with_logger
    acquire(path, logger) = with_logger(logger) do
        Threads.@spawn StanBlocks._with_build_lock(path; dead_age=1, poll=0.25) do
            read(path, String)
        end
    end
    reclaimed(logger) = [log.kwargs[:record] for log in logger.logs
                         if occursin("reclaimed a build lock", log.message)]
    own = "$(getpid()) $(gethostname())"
    mktempdir() do dir
        path = joinpath(dir, ".stanblocks-build.pid")
        # A holder killed between creating and writing its file leaves it
        # empty, dead once older than `dead_age`.
        write(path, "")
        sleep(1.5)
        # Waiters read the record with delete sharing, so a waiter reading it
        # never blocks another's reclaim, on Windows included.
        logger = TestLogger()
        file = Base.Filesystem.open(path, Base.Filesystem.JL_O_RDONLY)
        try
            @test fetch(acquire(path, logger)) == own
        finally
            close(file)
        end
        @test reclaimed(logger) == [""]

        # A reader without delete sharing (an IOStream on Windows; an older
        # StanBlocks reads one) makes the reclaim's aside-rename fail with
        # EBUSY. The waiter keeps waiting until it can reclaim; it never fails.
        write(path, "")
        sleep(1.5)
        logger = TestLogger()
        local waiter
        open(path, "r") do io
            waiter = acquire(path, logger)
            if Sys.iswindows()
                @test timedwait(() -> istaskdone(waiter), 3) === :timed_out
            else
                @test timedwait(() -> istaskdone(waiter), 30) === :ok
            end
        end
        @test fetch(waiter) == own
        @test reclaimed(logger) == [""]
        @test !isfile(path)
    end
end

@testitem "build lock: a holder is alive while its lock file is open, whatever pid it recorded" tags=[:slic, :regression] begin
    using StanBlocks
    using Test: TestLogger
    using Logging: with_logger
    # A holder in another pid namespace (a sandboxed agent's build) records its
    # namespace-local pid. Seen from here that pid names some unrelated live
    # process, and a host holder's pid is invisible from inside a sandbox. Only
    # Linux can ask the kernel whether the file is still open for writing.
    acquire(path, logger) = with_logger(logger) do
        Threads.@spawn StanBlocks._with_build_lock(path; dead_age=1, poll=0.25) do
            read(path, String)
        end
    end
    reclaimed(logger) = [log.kwargs[:record] for log in logger.logs
                         if occursin("reclaimed a build lock", log.message)]
    own = "$(getpid()) $(gethostname())"
    mktempdir() do dir
        path = joinpath(dir, ".stanblocks-build.pid")
        # A dead holder whose recorded pid names a live process: this one.
        write(path, own)
        sleep(1.5)
        logger = TestLogger()
        waiter = acquire(path, logger)
        if Sys.islinux()
            @test timedwait(() -> istaskdone(waiter), 30) === :ok
            @test fetch(waiter) == own
            @test reclaimed(logger) == [own]
        else
            # Elsewhere a live pid keeps the lock, the documented limit.
            @test timedwait(() -> istaskdone(waiter), 3) === :timed_out
            rm(path)
            @test fetch(waiter) == own
        end
        @test !isfile(path)

        Sys.islinux() || return
        # A live holder whose recorded pid this waiter cannot see: the record
        # names an exited process, but the holder still has the file open.
        gone = run(`true`; wait=false)
        record = "$(getpid(gone)) $(gethostname())"
        wait(gone)
        write(path, record)
        holder_code = "file = open(ARGS[1], \"a\"); println(\"HELD\"); flush(stdout); sleep(3600)"
        holder = open(`$(Base.julia_cmd()) --startup-file=no -e $holder_code $path`, "r")
        logger = TestLogger()
        local waiter
        try
            readline(holder) == "HELD" || error("lock holder did not start")
            sleep(1.5)
            waiter = acquire(path, logger)
            # Refused: the holder never refreshes, so its open file alone
            # protects its build from a competitor (build.jl lock contract).
            @test timedwait(() -> istaskdone(waiter), 5) === :timed_out
        finally
            kill(holder, Base.SIGKILL)
            wait(holder)
        end
        @test timedwait(() -> istaskdone(waiter), 30) === :ok
        @test fetch(waiter) == own
        @test reclaimed(logger) == [record]
        @test !isfile(path) && !isfile(path * ".reclaim")
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
