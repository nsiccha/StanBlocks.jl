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

@testitem "artifact identity: make arguments naming toolchain paths key on what they name" tags=[:slic, :regression] begin
    # Snag instantiate-cach-ad8bb476: a toolchain behind a moved link, or
    # rebuilt in place, kept the make_args text and reused a library built
    # against the old headers.
    using StanBlocks
    code = "parameters { real x; } model { x ~ normal(0, 1); }"
    key(home, args) = StanBlocks._default_build_path(code, StanBlocks._native_build_context(home, args))
    # No build environment variable names a path here, so only make_args do.
    quiet = [name => nothing for name in keys(ENV) if name in ("MATH", "USER_HEADER", "CXX",
        "CC", "AR", "LD", "RANLIB", "MAKE") || startswith(name, "TBB") ||
        startswith(name, "BRIDGESTAN") || (startswith(name, "STAN") && !startswith(name, "STANBLOCKS_"))]
    mktempdir() do dir
        home = mkpath(joinpath(dir, "bridgestan"))
        tree(name, header) = begin
            root = joinpath(dir, name)
            write(joinpath(mkpath(joinpath(root, "stan", "src")), "solver.hpp"), header)
            mkpath(joinpath(root, "stan", "lib", "math"))
            write(joinpath(mkpath(joinpath(root, "bin")), "stanc"), "stanc")
            root
        end
        old, new = tree("toolchain-1", "old"), tree("toolchain-2", "fixed")
        withenv(quiet...) do
            # Values that name no path leave the context, and so every
            # existing key, exactly as before.
            plain = StanBlocks._native_build_context(home, ["STAN_THREADS=true", "O=0", "STAN_THREADS="])
            @test !haskey(plain, :toolchain)

            args(root) = ["STAN_THREADS=true", "STAN=$root/stan/",
                "MATH=$root/stan/lib/math/", "STANC=$root/bin/stanc"]
            context = StanBlocks._native_build_context(home, args(old))
            @test context.toolchain.paths == ["STAN" => realpath(joinpath(old, "stan")),
                "MATH" => realpath(joinpath(old, "stan", "lib", "math")),
                "STANC" => realpath(joinpath(old, "bin", "stanc"))]
            # MATH lies inside STAN, so its tree is stamped once.
            @test first.(context.toolchain.stamps) == [realpath(joinpath(old, "bin", "stanc")),
                realpath(joinpath(old, "stan"))]
            @test key(home, args(old)) == key(home, args(old))
            @test key(home, args(old)) != key(home, args(new))

            # A relative value names a path in the BridgeStan home, where make runs.
            mkpath(joinpath(home, "stan-fork"))
            relative = StanBlocks._native_build_context(home, ["STAN=stan-fork/"])
            @test relative.toolchain.paths == ["STAN" => realpath(joinpath(home, "stan-fork"))]

            # A link moved to another tree selects another library, and moving
            # it back selects the first one again. The link resolves to the
            # tree it names; its text stays in the key as make receives it.
            if !Sys.iswindows()
                toolchain(root) = StanBlocks._native_build_context(home, args(root)).toolchain
                link = joinpath(dir, "toolchain")
                symlink(old, link)
                via_old = key(home, args(link))
                @test toolchain(link) == toolchain(old)
                rm(link); symlink(new, link)
                @test toolchain(link) == toolchain(new)
                @test key(home, args(link)) != via_old
                rm(link); symlink(old, link)
                @test key(home, args(link)) == via_old
            end

            # Rebuilding a header or the stanc binary in place under the same
            # path selects another library.
            before = key(home, args(old))
            write(joinpath(old, "stan", "src", "solver.hpp"), "rebuilt")
            header_rebuilt = key(home, args(old))
            @test header_rebuilt != before
            write(joinpath(old, "bin", "stanc"), "stanc 2")
            @test key(home, args(old)) != header_rebuilt
            # A file added anywhere in the tree counts too.
            stanc_rebuilt = key(home, args(old))
            write(joinpath(old, "stan", "lib", "math", "extra.hpp"), "extra")
            @test key(home, args(old)) != stanc_rebuilt
        end
        # Build environment variables name toolchain paths as make arguments
        # do. The PATH-style search lists and BRIDGESTAN (the home itself,
        # where make writes its objects) are not toolchain paths.
        math = joinpath(new, "stan", "lib", "math")
        @test StanBlocks._toolchain_paths(home, String[], ["MATH" => math]).paths ==
            ["MATH" => realpath(math)]
        @test isempty(StanBlocks._toolchain_paths(home, String[],
            ["PATH" => new, "LD_LIBRARY_PATH" => new, "BRIDGESTAN" => home, "O" => "3"]).paths)
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
            istaskdone(waiter) || rm(path)  # fail, never hang, on a regression
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

@testitem "artifact library: a published library loads while another build holds the toolchain lock" tags=[:slic, :regression] begin
    # Snag toolchain-build-ec701757: every call read its build context under the
    # host-wide toolchain lock, so loading an already-built library queued
    # behind every other process's compile on that toolchain.
    using StanBlocks
    code = "parameters { real x; } model { x ~ normal(0, 1); }\n"
    args = ["STAN_THREADS=true"]
    mktempdir() do dir
        # A toolchain home without a Makefile: nothing here can compile.
        home = realpath(mkpath(joinpath(dir, "bridgestan")))
        withenv("BRIDGESTAN" => home, "STANBLOCKS_BUILD_DIR" => joinpath(dir, "build")) do
            # An artifact as a finished build publishes it: source, then library.
            source = StanBlocks._default_build_path(code, StanBlocks._native_build_context(home, args))
            library = joinpath(realpath(mkpath(dirname(source))), splitext(basename(source))[1] * "_model.so")
            write(source, code)
            write(library, "")
            held, release = Channel{Nothing}(1), Channel{Nothing}(1)
            holder = Threads.@spawn StanBlocks.with_toolchain_lock(home) do
                put!(held, nothing)
                take!(release)
            end
            take!(held)
            local probe, hit, miss
            try
                # Control: the lock a build takes stays held throughout.
                probe = Threads.@spawn StanBlocks.with_toolchain_lock(() -> nothing, home)
                hit = Threads.@spawn StanBlocks._artifact_library(code; make_args=args)
                @test timedwait(() -> istaskdone(hit), 30) === :ok
                # Fail, never hang, on a regression: the holder waits for us.
                @test istaskdone(hit) && fetch(hit) == library
                # A library not yet built still waits for the toolchain.
                miss = Threads.@spawn StanBlocks._artifact_library("// other\n" * code; make_args=args)
                @test timedwait(() -> istaskdone(miss), 3) === :timed_out
                @test !istaskdone(probe)
            finally
                put!(release, nothing)
                fetch(holder)
            end
            fetch(probe)
            # Once the toolchain is free the miss reaches the compiler.
            @test timedwait(() -> istaskdone(miss), 60) === :ok
            @test istaskfailed(miss)
            @test occursin("Makefile does not exist", sprint(showerror, miss.result))
            # The lock-free path still refuses a damaged artifact source.
            write(source, "// damaged\n" * code)
            @test_throws ErrorException StanBlocks._artifact_library(code; make_args=args)
            @test read(source, String) == "// damaged\n" * code
        end
    end
end

@testitem "artifact library: distinct models compile at once; only shared toolchain files take its lock" tags=[:slic, :regression] begin
    # Snag host-wide-toolch-da758255: every model compile held the host-wide
    # toolchain lock, so a host compiled one model at a time. Only the files
    # all builds share (stanc, the bridge object, Stan Math's libraries) need
    # it; once they are current a model build writes only into its staging.
    using StanBlocks
    if !Sys.iswindows()  # the stand-in Makefile needs a POSIX make and shell
        code = "parameters { real x; } model { x ~ normal(0, 1); }\n"
        args = ["STAN_THREADS=true"]
        fixture = joinpath(@__DIR__, "fixtures", "toolchain.mk")
        mktempdir() do dir
            function toolchain(name, makefile)
                home = realpath(mkpath(joinpath(dir, name)))
                write(joinpath(home, "Makefile"), makefile)
                home, mkpath(joinpath(dir, name * "-rendezvous"))
            end
            buildlog(home) = readlines(joinpath(home, "build.log"))
            home, rendezvous = toolchain("bridgestan", read(fixture, String))
            withenv("BRIDGESTAN" => home, "STANBLOCKS_BUILD_DIR" => joinpath(dir, "build"),
                    "STANBLOCKS_TEST_RENDEZVOUS" => rendezvous) do
                sources = ["// model $i\n" * code for i in 1:2]
                builds = [Threads.@spawn StanBlocks._artifact_library(s; make_args=args) for s in sources]
                # Each model recipe waits for the other's: a serialized compile
                # fails after its 30 s wait instead of hanging here.
                @test timedwait(() -> all(istaskdone, builds), 120) === :ok
                @test all(istaskdone, builds) && !any(istaskfailed, builds)
                libraries = fetch.(builds)
                @test allunique(libraries) && read.(libraries, String) == sources
                # The shared files were built once, under the toolchain lock;
                # both models compiled at the same time without it.
                shared = ["bin/stanc locked", "src/bridge.o locked",
                          "lib/sundials.a locked", "lib/tbb.so locked"]
                @test buildlog(home) == [shared; "model unlocked"; "model unlocked"]
                # Another build variant's own shared file is built under the lock too.
                variant = StanBlocks._artifact_library(code; make_args=[args; "VARIANT=hess"])
                @test read(variant, String) == code
                @test buildlog(home)[end-1:end] == ["src/bridge_hess.o locked", "model unlocked"]
                # A published library is returned again without another build.
                @test StanBlocks._artifact_library(sources[1]; make_args=args) == libraries[1]
                @test length(buildlog(home)) == 8
            end
            # A Makefile that names no shared targets keeps the old, conservative
            # shape: the whole compile holds the toolchain lock.
            makefile = join(filter(!startswith("print-%"), readlines(fixture)), '\n') * '\n'
            home, rendezvous = toolchain("unnamed", makefile)
            write(joinpath(rendezvous, "other"), "")
            withenv("BRIDGESTAN" => home, "STANBLOCKS_BUILD_DIR" => joinpath(dir, "build"),
                    "STANBLOCKS_TEST_RENDEZVOUS" => rendezvous) do
                @test read(StanBlocks._artifact_library(code; make_args=args), String) == code
                @test buildlog(home) == ["bin/stanc locked", "src/bridge.o locked",
                    "lib/sundials.a locked", "lib/tbb.so locked", "model locked"]
            end
        end
    end
end

@testitem "artifact library: a toolchain it may not write compiles once its shared files are current" tags=[:slic, :regression] begin
    # Snag stanblocks-toolc-6d8cf00a: every model build created the toolchain
    # lock inside the BridgeStan tree, so a tree this process may not write (a
    # read-only container mount, a store path, another user's install) failed
    # with EROFS or EACCES although its shared files were current. Only building
    # those files needs the lock, and nothing here can build them there.
    using StanBlocks
    # The stand-in Makefile needs a POSIX make and shell. Root writes into a
    # directory whatever its mode, so a chmod cannot take the tree from it.
    if !Sys.iswindows() && ccall(:geteuid, Cuint, ()) != 0
        code = "parameters { real x; } model { x ~ normal(0, 1); }\n"
        args = ["STAN_THREADS=true"]
        fixture = joinpath(@__DIR__, "fixtures", "toolchain.mk")
        mktempdir() do dir
            function toolchain(name, makefile)
                local tree = realpath(mkpath(joinpath(dir, name)))
                write(joinpath(tree, "Makefile"), makefile)
                tree
            end
            home = toolchain("bridgestan", read(fixture, String))
            unnamed = toolchain("unnamed", join(filter(!startswith("print-%"), readlines(fixture)), '\n') * '\n')
            lockfile = joinpath(home, ".stanblocks-build.pid")
            rendezvous = mkpath(joinpath(dir, "rendezvous"))
            write(joinpath(rendezvous, "other"), "")  # each model builds alone here
            log = joinpath(dir, "build.log")
            # A HOME the installer lock cannot be created in either.
            user = mkpath(joinpath(dir, "user"))
            build(source; make_args=args) = StanBlocks._artifact_library(source; make_args)
            withenv("BRIDGESTAN" => home, "STANBLOCKS_BUILD_DIR" => joinpath(dir, "build"),
                    "STANBLOCKS_TEST_RENDEZVOUS" => rendezvous, "STANBLOCKS_TEST_BUILD_LOG" => log) do
                # Writable: the shared files are built once, under the lock.
                @test read(build("// 1\n" * code), String) == "// 1\n" * code
                shared = ["bin/stanc locked", "src/bridge.o locked",
                          "lib/sundials.a locked", "lib/tbb.so locked"]
                @test readlines(log) == [shared; "model unlocked"]
                foreach(path -> chmod(path, 0o555), (home, unnamed, user))
                try
                    withenv("HOME" => user) do
                        # Its shared files current, the tree compiles a distinct model
                        # without the lock, and nothing is written into it.
                        @test read(build("// 2\n" * code), String) == "// 2\n" * code
                        @test readlines(log) == [shared; "model unlocked"; "model unlocked"]
                        @test !ispath(lockfile)
                        # A flag variant whose bridge object is missing is not built:
                        # the call names the file and the tree instead.
                        err = try
                            build(code; make_args=[args; "VARIANT=hess"])
                        catch e
                            e
                        end
                        @test err isa ErrorException
                        @test occursin("./src/bridge_hess.o", err.msg) && occursin(home, err.msg)
                        @test length(readlines(log)) == 6
                        # refused: a tree that cannot hold the toolchain lock and whose
                        # Makefile names no shared targets gives no way to tell that a
                        # model build writes only next to its source (`0cb7c06` keeps
                        # such builds under the lock for that reason).
                        withenv("BRIDGESTAN" => unnamed) do
                            err = try
                                build(code)
                            catch e
                                e
                            end
                            @test err isa ErrorException && occursin("names no shared build targets", err.msg)
                        end
                    end
                    # A live holder, a process that may write the tree, is still
                    # waited for, as by every build.
                    chmod(home, 0o755)
                    release = Channel{Nothing}(1)
                    holder = Threads.@spawn StanBlocks.with_toolchain_lock(() -> take!(release), home)
                    @test timedwait(() -> ispath(lockfile), 30) === :ok
                    chmod(home, 0o555)
                    waiter = Threads.@spawn build("// 3\n" * code)
                    try
                        sleep(3)
                        @test !istaskdone(waiter)
                        @test length(readlines(log)) == 6
                    finally
                        chmod(home, 0o755)
                        put!(release, nothing)
                        fetch(holder)
                    end
                    @test timedwait(() -> istaskdone(waiter), 60) === :ok
                    @test istaskdone(waiter) && read(fetch(waiter), String) == "// 3\n" * code
                finally
                    foreach(path -> chmod(path, 0o755), (home, unnamed, user))
                end
            end
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

@testitem "artifact identity: a header's quoted includes, listed dependencies and stanc arguments" tags=[:slic, :regression] begin
    # Snag raw-source-immut-2d40975e: a USER_HEADER's own includes were not in
    # the identity, so editing one reused a library built against the old one.
    using StanBlocks
    code = "parameters { real x; } model { x ~ normal(0, 1); }"
    key(context) = StanBlocks._default_build_path(code, context)
    # Every stanc word, from a STANCFLAGS make argument or stanc_args, reaches
    # BridgeStan as one element, so its compile_model emits one STANCFLAGS word.
    @test StanBlocks._split_stanc_make_args(
        ["STAN_THREADS=true", "STANCFLAGS=--O1 --warn-pedantic"], ["--allow-undefined"]) ==
        (["STAN_THREADS=true"], ["--O1", "--warn-pedantic", "--allow-undefined"])
    @test StanBlocks._compile_stanc_args(["--O1", "--allow-undefined"]) == ["--O1 --allow-undefined"]
    @test StanBlocks._compile_stanc_args(String[]) == String[]
    # No build environment variable names a path here, so only the arguments do.
    quiet = [name => nothing for name in keys(ENV) if name in ("MATH", "USER_HEADER", "CXX",
        "CC", "AR", "LD", "RANLIB", "MAKE") || startswith(name, "TBB") ||
        startswith(name, "BRIDGESTAN") || (startswith(name, "STAN") && !startswith(name, "STANBLOCKS_"))]
    mktempdir() do dir
        home = mkpath(joinpath(dir, "bridgestan"))
        withenv(quiet...) do
            plain = StanBlocks._native_build_context(home, ["STAN_THREADS=true"])
            # Builds without stanc arguments, toolchain paths or dependencies
            # keep their existing keys.
            @test !haskey(plain, :stanc_args) && !haskey(plain, :toolchain)
            @test key(StanBlocks._native_build_context(home, ["STAN_THREADS=true"];
                stanc_args=["--O1"])) != key(plain)

            header = joinpath(dir, "ext.hpp")
            included = joinpath(dir, "include", "offset.hpp")
            mkpath(dirname(included))
            write(header, "#include <ostream>\n#include \"include/offset.hpp\"\n")
            # The include cycle back to the header must terminate.
            write(included, "#define EXT_OFFSET 1.0\n#include \"../ext.hpp\"\n")
            args = ["STAN_THREADS=true", "USER_HEADER=$header"]
            context() = StanBlocks._native_build_context(home, args)
            base = context()
            @test base.toolchain.paths ==
                ["USER_HEADER" => realpath(header), "#include" => realpath(included)]
            @test key(context()) == key(base)
            write(included, "#define EXT_OFFSET 2.0\n#include \"../ext.hpp\"\n")
            edited_include = context()
            @test key(edited_include) != key(base)
            write(header, read(header, String) * "// edited\n")
            @test key(context()) != key(edited_include)

            # make resolves a relative USER_HEADER in the BridgeStan home, and
            # the preprocessor its quoted includes next to it.
            relative = joinpath(home, "relative.hpp")
            write(relative, "#include \"../include/offset.hpp\"\n")
            @test Set(StanBlocks._native_build_context(home, ["USER_HEADER=relative.hpp"]).toolchain.paths) ==
                Set(["USER_HEADER" => realpath(relative), "#include" => realpath(included),
                     "#include" => realpath(header)])
            # The environment's header is followed as a make argument's is.
            from_env = withenv("USER_HEADER" => header) do
                StanBlocks._native_build_context(home, String[])
            end
            @test from_env.toolchain.paths ==
                ["USER_HEADER" => realpath(header), "#include" => realpath(included)]
            # A listed dependency is recorded and followed the same way.
            @test StanBlocks._native_build_context(home, String[]; dependencies=[header]).toolchain.paths ==
                ["dependency" => realpath(header), "#include" => realpath(included)]
        end
    end
    missing_header = joinpath(tempdir(), "stanblocks-no-such-header-$(rand(UInt)).hpp")
    @test_throws ArgumentError stan_library(code; dependencies=[missing_header])
end

@testitem "stan_library: raw source and an edited header or include get new immutable libraries" tags=[:slic, :regression, :bridgestan] begin
    using StanBlocks
    using LogDensityProblems: logdensity
    import BridgeStan
    source = """
    functions {
      real ext_shift(data real x);
    }
    data {
      real x0;
    }
    parameters {
      real theta;
    }
    model {
      theta ~ normal(ext_shift(x0), 1);
    }
    """
    stan_json = """{"x0": 0.0}"""
    density(library) = logdensity(StanBlocks.StanLogDensityProblems.StanProblem(library, stan_json), [0.0])
    mktempdir() do dir
        withenv("STANBLOCKS_BUILD_DIR" => joinpath(dir, "build"), "USER_HEADER" => nothing) do
            header = joinpath(dir, "ext_shift.hpp")
            offset = joinpath(dir, "include", "ext_offset.hpp")
            mkpath(dirname(offset))
            write(header, """
            #include <ostream>
            #include "include/ext_offset.hpp"
            double ext_shift(const double& x, std::ostream* pstream__) { return x + EXT_OFFSET; }
            """)
            write(offset, "#define EXT_OFFSET 1.0\n")
            make_args = ["STAN_THREADS=true", "USER_HEADER=$header"]
            stanc_args = ["--allow-undefined", "--warn-pedantic"]

            # Control: BridgeStan.compile_model emits one STANCFLAGS assignment
            # per stanc argument and make keeps the last, so only
            # --warn-pedantic reaches stanc, which then rejects the declared
            # but undefined function.
            toolchain = realpath(BridgeStan.get_bridgestan_path(; download=false))
            direct = joinpath(dir, "direct.stan")
            write(direct, source)
            @test_throws ErrorException StanBlocks.with_toolchain_lock(toolchain) do
                BridgeStan.compile_model(direct; make_args, stanc_args)
            end

            one = stan_library(source; make_args, stanc_args)
            @test startswith(one, realpath(joinpath(dir, "build")))
            @test read(replace(one, r"_model\.so$" => ".stan"), String) == source
            @test density(one) == -0.5
            built = mtime(one)
            @test stan_library(source; make_args, stanc_args) == one
            # The same flags spelled as a make argument, alone or combined with
            # stanc_args, are the same build.
            @test stan_library(source;
                make_args=[make_args; "STANCFLAGS=--allow-undefined --warn-pedantic"]) == one
            @test stan_library(source; make_args=[make_args; "STANCFLAGS=--allow-undefined"],
                stanc_args=["--warn-pedantic"]) == one
            @test mtime(one) == built

            # Editing a quoted include, then the header itself, selects new
            # libraries, which this process loads beside the first one.
            write(offset, "#define EXT_OFFSET 2.0\n")
            two = stan_library(source; make_args, stanc_args)
            @test two != one
            @test density(two) == -2.0
            write(header, replace(read(header, String), "x + EXT_OFFSET" => "x + 2 * EXT_OFFSET"))
            three = stan_library(source; make_args, stanc_args)
            @test three ∉ (one, two)
            @test density(three) == -8.0
            @test density(one) == -0.5
            @test mtime(one) == built
        end
    end
end
