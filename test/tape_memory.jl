using TestItemRunner

@testitem "tape memory: every library links the release functions, keyed by their source" tags=[:slic, :regression] begin
    # Snag stan-tape-memory-288a1422: the release functions are linked into
    # every library through make's LDLIBS, which a command-line assignment
    # replaces (an earlier one, and the environment's).
    using StanBlocks
    object = "/build/key/build-1/stanblocks_tape.o"
    @test StanBlocks._link_tape_object(["STAN_THREADS=true", "LDLIBS=-lold", "O=0", "LDLIBS=-lfoo -lbar"], object) ==
        ["STAN_THREADS=true", "O=0", "LDLIBS=$object -lfoo -lbar"]
    @test withenv(() -> StanBlocks._link_tape_object(["STAN_THREADS=true"], object), "LDLIBS" => "-lenv") ==
        ["STAN_THREADS=true", "LDLIBS=$object -lenv"]
    @test withenv(() -> StanBlocks._link_tape_object(String[], object), "LDLIBS" => nothing) == ["LDLIBS=$object"]
    # A home without a Makefile compiles no object; BridgeStan reports it.
    @test StanBlocks._link_tape_object(["STAN_THREADS=true"], nothing) == ["STAN_THREADS=true"]
    mktempdir() do dir
        @test StanBlocks._compile_tape_object(dir, String[], dir) === nothing
        # The functions' source is part of every artifact's identity.
        @test StanBlocks._native_build_context(dir, String[]).tape ==
            bytes2hex(StanBlocks.SHA.sha256(read(joinpath(pkgdir(StanBlocks), "src", "slic_stan", "tape_memory.cpp"))))
    end
end

@testitem "tape memory: a release hands one library's arena back and keeps its gradients" tags=[:slic, :regression, :bridgestan] begin
    # Snag stan-tape-memory-288a1422: Stan Math keeps each thread's reverse-mode
    # arena for as long as the thread lives, one per thread and library;
    # destroying the model, collecting and `malloc_trim` return none of it.
    using StanBlocks
    import BridgeStan, Libdl
    # Several hundred thousand tape entries per gradient. The second library
    # also has a caller's C++ header (USER_HEADER), which links beside the
    # release functions.
    header = joinpath(mktempdir(), "offset_of.hpp")
    write(header, "#include <ostream>\ndouble offset_of(const double& k, std::ostream* pstream__) { return k; }\n")
    source(offset) = """
        functions { real offset_of(data real k); }
        data { int N; }
        transformed data { real k = $offset; }
        parameters { real x; }
        model {
          real s = 0;
          for (n in 1:N) s += sin(x * n + k);
          target += -0.5 * square(x) + 1e-6 * s;
        }
        """
    builds = [Threads.@spawn(stan_library(replace(source("1.0"), r"functions \{.*\}\n" => ""))),
              Threads.@spawn(stan_library(source("offset_of(2.0)");
                  make_args=["STAN_THREADS=true", "USER_HEADER=$header"], stanc_args=["--allow-undefined"]))]
    libraries = fetch.(builds)
    models = [BridgeStan.StanModel(library, """{"N": 200000}""") for library in libraries]
    theta = [0.3]
    threads = StanBlocks._tape_threads()
    on(f, tid) = fetch(StanBlocks._on_thread(f, tid))
    resident(model, tid) = on(() -> StanBlocks._tape_bytes(model; resident=true), tid)
    reserved(model, tid) = on(() -> StanBlocks._tape_bytes(model), tid)
    gradient(model, tid) = on(() -> BridgeStan.log_density_gradient(model, theta), tid)

    for library in libraries
        handle = Libdl.dlopen(library)
        @test Libdl.dlsym(handle, :stanblocks_tape_release; throw_error=false) !== nothing
        @test ccall(Libdl.dlsym(handle, :stanblocks_tape_per_thread), Cint, ()) == 1
    end
    if Sys.islinux()
        # Before a gradient a thread has no tape in a library, or an empty one.
        before = [gradient(m, tid) for m in models, tid in threads]
        held = [resident(m, tid) for m in models, tid in threads]
        sizes = [reserved(m, tid) for m in models, tid in threads]
        @test all(>(8 * 2^20), held)
        @test all(sizes .>= held)

        released = fetch(StanBlocks.release_tape_memory(first(models)))
        @test released == (; released=length(threads), busy=0, unused=0)
        # The first library's arenas are gone from memory, on every thread;
        # their blocks are kept, and the other library's arenas are untouched.
        @test all(tid -> resident(models[1], tid) == 0, threads)
        @test [reserved(models[1], tid) for tid in threads] == sizes[1, :]
        @test all(tid -> resident(models[2], tid) >= held[2, tid] ÷ 2, threads)
        # Gradients write into the same blocks again and evaluate exactly as before.
        @test [gradient(models[1], tid) for tid in threads] == before[1, :]
        @test all(tid -> resident(models[1], tid) >= held[1, tid] ÷ 2, threads)
        @test [reserved(models[1], tid) for tid in threads] == sizes[1, :]

        # A library is also named by its path, once this process has loaded it.
        @test fetch(StanBlocks.release_tape_memory(libraries[2])).released == length(threads)
        @test all(tid -> resident(models[2], tid) == 0, threads)
        @test [gradient(models[2], tid) for tid in threads] == before[2, :]
        # The problem `stan_instantiate` returns is accepted as its model.
        problem = StanBlocks.StanLogDensityProblems.StanProblem(models[2])
        @test fetch(StanBlocks.release_tape_memory(problem)).released == length(threads)

        # Reuse only answers a pending release, on the thread that released.
        reuse = Libdl.dlsym(Libdl.dlopen(libraries[2]), :stanblocks_tape_reuse)
        @test on(() -> (ccall(reuse, Cint, ()), ccall(reuse, Cint, ())), first(threads)) == (1, 0)
        @test StanBlocks.reuse_tape_memory(models[2]) === nothing
    end
    # Nothing to release: a path no loaded library has, a loaded library
    # without the functions.
    @test StanBlocks.release_tape_memory(joinpath(mktempdir(), "absent_model.so")) === nothing
    @test StanBlocks.release_tape_memory(Libdl.dlpath(Libdl.dlopen("libjulia"))) === nothing
    @test_throws ArgumentError StanBlocks.release_tape_memory(1.0)
end
