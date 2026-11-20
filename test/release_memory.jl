"""
StanBlocks compiles with the environment's BridgeStan. Built through BridgeStan's
dogfood line (`nsiccha/bridgestan`, branch `dogfood`; decision `1f3gj3e`), a
library exports `bs_release_memory`, and `BridgeStan.release_memory` hands each
Julia thread's autodiff memory back to the system. StanBlocks has no tape-memory
code of its own any more: the stopgap `release_tape_memory`, which read Stan
Math's private arena bookkeeping, is removed (snag `stan-tape-memory-288a1422`,
CHANGELOG). This item
pins that the libraries StanBlocks builds support BridgeStan's call, including
one that links a caller's C++ header, and that a release leaves gradients as
they were. The test environment takes BridgeStan.jl from that line
(`test/Project.toml` `[sources]`).
"""
@testitem "native: StanBlocks-built libraries hand autodiff memory back through BridgeStan.release_memory" tags=[:slic, :regression, :bridgestan] begin
    using StanBlocks
    import BridgeStan, Libdl, LogDensityProblems
    @test isdefined(BridgeStan, :release_memory)

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
    libraries = [stan_library(replace(source("1.0"), r"functions \{.*\}\n" => "")),
                 stan_library(source("offset_of(2.0)");
                     make_args=["STAN_THREADS=true", "USER_HEADER=$header"], stanc_args=["--allow-undefined"])]
    for library in libraries
        @test Libdl.dlsym(Libdl.dlopen(library), :bs_release_memory; throw_error=false) !== nothing
    end
    (Sys.islinux() || Sys.isapple()) || return

    models = [BridgeStan.StanModel(library, """{"N": 20000}""") for library in libraries]
    threads = 1:(Threads.nthreads(:interactive) + Threads.nthreads(:default))
    function on(f, tid)
        task = Task(f)
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1) == 1 || error("cannot pin a task to thread $tid")
        fetch(schedule(task))
    end
    gradient(model, tid) = on(() -> BridgeStan.log_density_gradient(model, [0.3]), tid)
    before = [gradient(model, tid) for model in models, tid in threads]
    for (i, model) in enumerate(models)
        # Every thread evaluated a gradient with this library, so each one has
        # memory to hand back; the next gradients evaluate exactly as before.
        @test BridgeStan.release_memory(model) == fill(:released, length(threads))
        @test [gradient(model, tid) for tid in threads] == before[i, :]
    end

    # The problem `stan_instantiate` returns carries its BridgeStan model.
    problem = stan_instantiate(@slic (; y=[0.1, -0.4, 0.7]) begin
        mu ~ normal(0, 1)
        y ~ normal(mu, 1)
    end)
    LogDensityProblems.logdensity_and_gradient(problem, [0.2])
    outcomes = BridgeStan.release_memory(problem.model)
    @test length(outcomes) == length(threads)
    @test :released in outcomes && all(in((:released, :unused)), outcomes)
end
